"""One loading contract for every analysis that reads the shared per-agent cache.

The cache at `empirical-support/cache/agents` is written by `db_source.py` and therefore by every
pipeline in this repository. A missing file is not evidence that an agent fails a criterion, so an
analysis that drops it silently measures the intersection of its roster with whatever another run
happened to extract, not the roster it declares.

Three outcomes, kept apart:

  * **input failure** — the file is absent, unreadable, unparseable, missing a column the analysis
    indexes, or structurally unusable (wrong dtype, unordered or duplicated dates). This raises: a
    cohort that silently shrinks to what is on disk is not a cohort.
  * **scientific exclusion** — the frame parses and carries the right schema but does not qualify.
    This includes a frame with **zero rows**: an extraction that found nothing for an agent is a
    fact about the corpus, not a defect in the pipeline. It is recorded with its reason.
  * **admitted** — the frame is returned.

`digest()` additionally hashes the bytes actually read, so a cohort can be pinned by content rather
than by filename. An id list alone is not reproducibility: replacing a cached parquet while keeping
its name changes the estimate without changing the register.

Usage:

    from input_contract import admit, load_admitted, merge_per_id
    reports = admit(pops, CACHE, "my_analysis", out_dir=HERE,
                    analysis_min_rows=MIN_DAYS, required_cols=(...))
    per_id = merge_per_id(reports)           # pass to workers; they verify against it
    df = load_admitted(a_id, CACHE, per_id)  # raises if the bytes are not the admitted bytes
    out["inputs"] = reports
"""
from __future__ import annotations

import hashlib
import io
import json
import re
from pathlib import Path

import pandas as pd


class InputUnavailable(Exception):
    """A required input could not be read. Distinct from a scientific exclusion."""


# --- the cross-run pin registry ------------------------------------------------------------
# A content digest is taken over ONE declared id set, so a pin can only validate a run that
# declares the same set. Reusing the headline cohort's pin for an analysis with a different
# population refuses every time, and a check that fires on correct runs is one its owner learns to
# route around. Each (analysis, population) therefore carries its own entry, recorded on first
# sight and compared on every later run. There is no caller-supplied digest and no override
# parameter: the registry is the only authority, so no call site can opt out of it.
PIN_REGISTRY = Path(__file__).resolve().parent / "input_pins.json"

# Enrolment and re-freezing happen only under an explicit command. In ordinary analysis mode a
# missing or mismatched pin is a refusal, because a run that records its own expectation is its
# own authority and pins nothing.
FREEZE_MODE = False


def _validate_registry(pins, where) -> None:
    """Refuse a registry that parses but cannot constrain anything.

    Valid JSON is not a valid registry. A record carrying `n_declared` but no digest passes the
    count check and then leaves nothing to compare against, so the run succeeds unpinned: a
    corrupt registry that looks healthy. The writer checks its candidate through this same
    function, because a writer validating with its own copy of the rules is a second
    implementation that will drift from the one the reader enforces.

    Two kinds of record live here. An INPUT pin carries `n_declared` and `content_digest_sha256`.
    An AXIS pin, keyed `axis<suffix>`, carries the digests of the cached axis and of the roster
    that trained it, plus the estimator settings that make an axis what it is. Both are validated;
    neither is waved through for being the other. The kind is decided by the key, because that is
    what the reader uses to find the record.
    """
    bad = []
    if not isinstance(pins, dict):
        bad.append("top level is not an object")
    else:
        for k, v in pins.items():
            if not isinstance(v, dict):
                bad.append(f"{k}: record is not an object")
                continue

            # Two record kinds, told apart by the KEY, not by what the record happens to carry.
            # Discriminating on contents would let a malformed input pin escape validation by
            # sprouting an axis field; the reader looks an axis up by name (`axis{suffix}`), so
            # the name is what decides which rules apply, here and there alike.
            if k.startswith("axis"):
                for field in ("npz_sha256", "roster_sha256"):
                    h = v.get(field)
                    if not isinstance(h, str) or not re.fullmatch(r"[0-9a-f]{64}", h):
                        bad.append(f"{k}: {field} is not 64 lowercase hex characters")
                md = v.get("min_days")
                if isinstance(md, bool) or not isinstance(md, int) or md < 0:
                    bad.append(f"{k}: min_days is {md!r}, not a non-negative integer")
                ft = v.get("features")
                if (not isinstance(ft, list) or not ft
                        or not all(isinstance(x, str) and x for x in ft)):
                    bad.append(f"{k}: features is not a non-empty list of names")
                continue

            n = v.get("n_declared")
            d = v.get("content_digest_sha256")
            if isinstance(n, bool) or not isinstance(n, int) or n < 0:
                bad.append(f"{k}: n_declared is {n!r}, not a non-negative integer")
            if not isinstance(d, str) or not re.fullmatch(r"[0-9a-f]{64}", d):
                bad.append(f"{k}: content_digest_sha256 is not 64 lowercase hex characters")
    if bad:
        raise SystemExit(
            f"{where} is not a valid pin registry:" + "".join(
                chr(10) + "    " + b for b in bad[:8]) + chr(10) +
            "A record that parses but carries no usable digest disables the comparison silently, "
            "which is worse than no registry at all. Restore it, or re-freeze deliberately.")


def _load_pins() -> dict:
    """Read the registry, or refuse. A corrupt registry must not become an empty one.

    Returning `{}` on a parse error removes every cross-run pin in the package without failing a
    single run: the next analysis records fresh digests over whatever is on disk and calls that
    the expectation. Corruption would therefore silently disable exactly the check the registry
    exists to provide.
    """
    if not PIN_REGISTRY.exists():
        return {}
    try:
        pins = json.loads(PIN_REGISTRY.read_text(encoding="utf-8"))
    except Exception as exc:                                # noqa: BLE001
        raise SystemExit(
            f"{PIN_REGISTRY} is unreadable ({type(exc).__name__}: {exc}). The input pin registry "
            f"is the only cross-run guarantee that a cached file has not been replaced; treating "
            f"a corrupt one as empty would silently drop every pin. Restore it from version "
            f"control, or re-freeze deliberately with "
            f"`python freeze_input_pins.py --from-artifact <artifact>.json --prefix <prefix>`.")
    _validate_registry(pins, str(PIN_REGISTRY))
    return pins


def _record_pin(key: str, digest: str, n_declared: int) -> None:
    """Write one pin, having first checked that what is written would be readable back.

    `_load_pins` refuses a malformed registry, so a write that produces one disables every
    analysis at once and the failure surfaces far from its cause. Validating the candidate here
    keeps the registry an invariant of the writer rather than a hope of the reader.
    """
    if not isinstance(key, str) or not key:
        raise SystemExit(f"refusing to record a pin under a non-string key ({key!r})")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise SystemExit(f"refusing to record {key}: {digest!r} is not a 64-character sha256")
    if isinstance(n_declared, bool) or not isinstance(n_declared, int) or n_declared < 0:
        raise SystemExit(
            f"refusing to record {key}: n_declared {n_declared!r} is not a non-negative integer")

    pins = _load_pins()
    pins[key] = {"content_digest_sha256": digest, "n_declared": n_declared}
    blob = json.dumps(dict(sorted(pins.items())), indent=2)

    # Read the candidate back through the reader's own validator before it replaces the registry.
    tmp = PIN_REGISTRY.with_suffix(".json.candidate")
    tmp.write_text(blob, encoding="utf-8")
    try:
        _validate_registry(json.loads(blob), str(tmp))
    except SystemExit:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(PIN_REGISTRY)


def _semantic_problem(df, required_cols):
    """Return a reason string if a required column is structurally unusable, else None."""
    for c in required_cols:
        col = df[c]
        if c == "date":
            d = pd.to_datetime(col, errors="coerce")
            if d.isna().all() and len(col):
                return "date-unparseable"
            if d.duplicated().any():
                return "date-duplicated"
            if not d.is_monotonic_increasing:
                return "date-non-monotone"
            continue
        if len(col) and not pd.api.types.is_numeric_dtype(col):
            return f"non-numeric:{c}"
    return None


def preflight(ids, cache_dir, label, out_dir=None, required_cols=(),
              analysis_min_rows=None):
    """Admit every declared input before any estimate is computed, or refuse to run.

    Byte-readability is not admission. This parses each declared parquet, checks the columns the
    analysis will index and that the frame is structurally usable, and records per id the sha256
    of the bytes, the parse outcome and the row count. Anything absent, unreadable, unparseable,
    schema-short or structurally unusable is an INPUT FAILURE and aborts the run: a cohort that
    silently shrinks
    to whatever happens to be parseable is not a cohort.

    A parseable, correctly-shaped frame with **zero rows** is NOT an input failure. The extractor
    found nothing for that agent, which is a fact about the corpus, and it is admitted and counted
    with the other scientific exclusions.

    `analysis_min_rows` is different in kind and never aborts. An agent with too few observed days
    is a SCIENTIFIC EXCLUSION - a result about that agent, not a defect in the pipeline - so it is
    counted and listed here for the artifact and applied by the analysis itself. Conflating the
    two makes a normal cohort look like a broken one and, worse, would let a real input failure be
    waved through as a criterion failure.

    Must be called BEFORE the analysis. The per-id manifest it returns is what workers verify
    against, so a file swapped between admission and use is detected rather than averaged in.

    The registry closes the gap that admission alone leaves open. Verifying content WITHIN a run
    proves the workers read what was admitted; it says nothing about whether what was admitted is
    what the frozen cohort was frozen ON. A cached parquet silently re-extracted between runs is
    valid, correctly shaped, admitted without complaint - and changes the estimate on a nominally
    frozen run. The pin for this (analysis, population) turns that into a refusal, and it is read
    from `input_pins.json` rather than passed in, so the expectation cannot be supplied by the run
    it is meant to constrain.
    """
    cache = Path(cache_dir)
    manifest, missing, unreadable, unparseable = {}, [], [], []
    below_criterion: list = []
    for a_id in sorted({int(i) for i in ids}):
        f = cache / f"{a_id}.parquet"
        if not f.exists():
            missing.append(a_id)
            continue
        try:
            raw = f.read_bytes()
        except Exception as exc:
            unreadable.append((a_id, type(exc).__name__))
            continue
        try:
            # Parse the BYTES THAT WERE HASHED. Re-opening the path would let a replacement
            # between the two reads produce a manifest describing one file and a schema check
            # describing another - admission would then certify bytes nobody validated.
            df = pd.read_parquet(io.BytesIO(raw))
        except Exception as exc:
            unparseable.append((a_id, type(exc).__name__))
            continue
        cols = set(df.columns)
        if required_cols and not set(required_cols) <= cols:
            unparseable.append((a_id, f"missing-columns:{sorted(set(required_cols) - cols)[:3]}"))
            continue
        # A parseable, correctly-shaped frame with zero rows is a fact about the corpus - the
        # extractor found nothing for this agent - and belongs with the below-minimum exclusions.
        # Counting it as an input failure would make "the corpus has no rows here" and "the
        # pipeline could not read this" the same event, which is the conflation this module
        # exists to prevent, in the opposite direction.
        if analysis_min_rows is not None and len(df) < analysis_min_rows:
            below_criterion.append((a_id, len(df)))
        # Names and a row count are not a schema. A column of the wrong dtype, a date column
        # that is all-missing, duplicated or non-monotone, passes a names-only check and then
        # produces silently wrong windows downstream, because the estimators index positionally.
        bad = _semantic_problem(df, required_cols)
        if bad:
            unparseable.append((a_id, bad))
            continue
        manifest[a_id] = {"sha256": hashlib.sha256(raw).hexdigest(), "rows": int(len(df))}

    digest = None
    if manifest:
        digest = hashlib.sha256(chr(10).join(
            f"{i}:{manifest[i]['sha256']}:{manifest[i]['rows']}" for i in sorted(manifest)
        ).encode()).hexdigest()
    report = {
        "label": label,
        "n_declared": len({int(i) for i in ids}),
        "n_admitted": len(manifest),
        "n_missing": len(missing),
        "n_unreadable": len(unreadable),
        "n_unparseable": len(unparseable),

        "missing_ids": missing[:50],
        "unreadable": unreadable[:50],
        "unparseable": unparseable[:50],


        "analysis_min_rows": analysis_min_rows,
        "n_below_analysis_minimum": len(below_criterion),
        "below_analysis_minimum": below_criterion[:50],
        "required_cols": sorted(required_cols),
        "content_digest_sha256": digest,
        "per_id": {str(k): v for k, v in sorted(manifest.items())},
    }
    failures = len(missing) + len(unreadable) + len(unparseable)
    if failures:
        if out_dir is not None:
            Path(out_dir, f"{label}_inputs.failed.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8")
        raise SystemExit(
            f"{label}: {failures} input failure(s) of {report['n_declared']} declared "
            f"({len(missing)} missing, {len(unreadable)} unreadable, {len(unparseable)} "
            f"unparseable). The cohort would shrink to "
            f"whatever is on disk; refusing to run. First: "
            f"{(missing[:3] or unreadable[:3] or unparseable[:3])}")
    # The registry is the sole pin authority; callers have no override. Every analysis is
    # pinned by `input_pins.json` or refuses.
    pin_key = label
    # The registry holds two record kinds and tells them apart by the key prefix. No analysis
    # label starts with `axis` today, so the two sets are disjoint by circumstance; this makes
    # them disjoint by enforcement, before a future label quietly collides with an axis record
    # and is validated under the wrong rules.
    if pin_key.startswith("axis"):
        raise SystemExit(
            f"{label}: analysis labels may not begin with 'axis'. That prefix names the botness-"
            f"axis records in input_pins.json, which are validated as axes rather than as input "
            f"pins; a label that collides with one would be checked under the wrong rules.")
    pinned_digest = None
    rec = _load_pins().get(pin_key)
    if rec is None:
        if not FREEZE_MODE:
            raise SystemExit(
                f"{label}: no input pin is recorded for this analysis, so nothing constrains "
                f"which bytes it reads across runs. Enrolling one silently would make the "
                f"first run its own authority. Freeze deliberately:\n"
                f"    python freeze_input_pins.py --from-artifact <artifact>.json --prefix <prefix>\n"
                f"  (this run's digest would be {digest})")
        _record_pin(pin_key, digest, report["n_declared"])
        print(f"  [freeze] {label}: recorded {digest[:12]} over {report['n_declared']} inputs")
    elif rec.get("n_declared") != report["n_declared"]:
        if not FREEZE_MODE:
            raise SystemExit(
                f"{label}: the declared set changed, {rec.get('n_declared')} -> "
                f"{report['n_declared']}. The recorded pin does not describe this population, "
                f"and re-recording it here would let a cohort change re-freeze itself - the "
                f"opposite of a frozen registry. Re-freeze deliberately:\n"
                f"    python freeze_input_pins.py --from-artifact <artifact>.json --prefix <prefix>")
        _record_pin(pin_key, digest, report["n_declared"])
        print(f"  [freeze] {label}: re-recorded over {report['n_declared']} inputs "
              f"(was {rec.get('n_declared')})")
    else:
        pinned_digest = rec.get("content_digest_sha256")
    if pinned_digest and digest != pinned_digest:
        if out_dir is not None:
            Path(out_dir, f"{label}_inputs.failed.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8")
        raise SystemExit(
            f"{label}: the admitted inputs are not the inputs this cohort was frozen on.\n"
            f"  expected content digest {pinned_digest}\n"
            f"  got                     {digest}\n"
            f"The id list matches, so a cached file was replaced rather than the roster changed. "
            f"Re-freeze deliberately if that was intended; refusing to run otherwise.")
    # The report is written only now, after every refusal condition has passed. Writing it before
    # the checks let a REFUSED run replace the manifest of the last successful one - destroying the
    # evidence of what was admitted, at exactly the moment that evidence matters. A failed run
    # leaves a separate `.failed.json` instead.
    if out_dir is not None:
        Path(out_dir, f"{label}_inputs.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")
    print(f"  [inputs] {label}: {report['n_declared']} declared, all parsed, "
          f"{len(below_criterion)} below the analysis minimum (a scientific exclusion), "
          f"digest {digest[:12] if digest else 'n/a'}"
          f"{' (matches the frozen pin)' if pinned_digest else ''}")
    return report


def verify_against(manifest_per_id, a_id, path):
    """Worker-side check that the bytes being read are the bytes that were admitted."""
    rec = manifest_per_id.get(str(int(a_id))) or manifest_per_id.get(int(a_id))
    if rec is None:
        raise InputUnavailable(f"agent {a_id} was not admitted by preflight")
    got = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if got != rec["sha256"]:
        raise InputUnavailable(
            f"agent {a_id}: content changed after admission "
            f"({rec['sha256'][:10]} -> {got[:10]})")
    return True


def admit(groups, cache_dir, label, out_dir=None, required_cols=(),
          analysis_min_rows=None):
    """Preflight every declared population before any estimate is computed.

    Returns {group: report}. Any input failure in any group aborts the run, so an analysis either
    sees its whole declared roster or does not run at all.
    """
    return {g: preflight(ids, cache_dir, f"{label}_{g}", out_dir=out_dir,
                         required_cols=required_cols,
                         analysis_min_rows=analysis_min_rows)
            for g, ids in sorted(groups.items())}


def merge_per_id(reports) -> dict:
    """One id -> record map across populations, for workers to verify against."""
    per: dict[str, dict] = {}
    for rep in (reports.values() if isinstance(reports, dict) else reports):
        per.update(rep["per_id"])
    return per


def load_admitted(a_id, cache_dir, manifest_per_id):
    """Read one admitted input, proving the bytes are the bytes that were admitted.

    Admission happens in the parent; the loaders run in process pools, so a worker's own register
    reaches nobody and cannot enforce anything. What a worker CAN do is refuse content that
    changed after admission, which is the only part of the contract it is in a position to check.
    The bytes are read once and parsed from memory, so verification costs no extra I/O.
    """
    a_id = int(a_id)
    rec = manifest_per_id.get(str(a_id)) or manifest_per_id.get(a_id)
    if rec is None:
        raise InputUnavailable(f"agent {a_id} was not admitted by preflight")
    raw = Path(cache_dir, f"{a_id}.parquet").read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    if got != rec["sha256"]:
        raise InputUnavailable(
            f"agent {a_id}: content changed after admission "
            f"({rec['sha256'][:10]} -> {got[:10]})")
    return pd.read_parquet(io.BytesIO(raw))
