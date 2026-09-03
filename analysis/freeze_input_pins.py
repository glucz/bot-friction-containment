"""Explicitly freeze the input pin registry. Enrolment is a decision, not a side effect.

`preflight` refuses to run an analysis whose inputs have no recorded pin, and refuses again if the
declared set has changed since the pin was taken. Both refusals exist because a run that records
its own expectation is its own authority and constrains nothing: the first execution after any
change would quietly become the new baseline, which is what a frozen registry is meant to prevent.

Freezing therefore happens here, under a command someone had to type, and it prints what it is
about to record so the decision is visible.

    python freeze_input_pins.py --show
    python freeze_input_pins.py --from-artifact event_spacing_recompute.json --prefix event_spacing
    python freeze_input_pins.py --from-artifact ... --prefix ... --apply         --confirm-artifact-sha256 <the hash the dry run printed>

Enrolment is TWO steps. The first validates the artifact's `inputs` schema, recomputes each
aggregate digest from its own `per_id` records, and prints full 64-character digests plus the
artifact's hash. The second writes, and only if the hash is repeated back. Typing a command is
authority only when the command shows what will be trusted; a one-step write that prints sixteen
characters of a digest is a rubber stamp.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import input_contract as ic  # noqa: E402
from input_contract import _load_pins  # noqa: E402


def _show() -> None:
    pins = ic._load_pins()
    if not pins:
        print("registry is empty")
        return
    # Two record kinds, and the display has to know both. Indexing one shape unconditionally is
    # how a registry that validates cleanly still breaks the command that reads it out.
    axes = {k: v for k, v in pins.items() if k.startswith("axis")}
    inputs = {k: v for k, v in pins.items() if k not in axes}
    print(f"{len(inputs)} pinned population(s), {len(axes)} pinned axis/axes:")
    for k, v in sorted(inputs.items()):
        print(f"  {k:34s} {v['n_declared']:>6,} inputs  {v['content_digest_sha256'][:16]}")
    for k, v in sorted(axes.items()):
        print(f"  {k:34s} {'axis':>6s}  npz {v['npz_sha256'][:16]}  "
              f"roster {v['roster_sha256'][:12]}  min_days={v['min_days']}  "
              f"[{', '.join(v['features'])}]")


def _read_once(path: Path) -> tuple[bytes, str, object]:
    """Read the artifact once, and hash and parse THOSE bytes.

    Reading twice - once to parse, once to hash - attests bytes that need not be the bytes that
    were parsed. A file rewritten between the two reads is then confirmed under the hash of a
    version nobody inspected, which is the failure the confirmation step exists to prevent.
    """
    raw = path.read_bytes()
    return raw, hashlib.sha256(raw).hexdigest(), json.loads(raw.decode("utf-8"))


def _plan_from_artifact(name: str, prefix: str, only=(),
                        explicit_prefix: bool = False) -> tuple[Path, str, list]:
    """What a freeze WOULD record, with full digests and the artifact's own hash.

    Nothing is written here. Typing a command is authority only when the command shows what will be
    trusted; a one-step write that prints sixteen characters of a digest is a rubber stamp.
    """
    # Resolve against the INVOKER's directory first, then against the analysis directory. Only
    # the second was tried, so any documented path with a directory component in it ("outputs/…")
    # was rewritten into `analysis/outputs/…` and could never resolve -- and a command that always
    # refuses is indistinguishable, to a gate that accepts refusals, from one that is data-gated.
    candidates = [Path(name), HERE / name]
    path = next((c for c in candidates if c.exists()), None)
    if path is None:
        raise SystemExit(
            f"{name} does not exist. Looked in: "
            + "; ".join(str(c.resolve()) for c in candidates))
    _raw, art_sha, blob = _read_once(path)
    standalone = False
    inputs = blob.get("inputs")
    if not isinstance(inputs, dict):
        # A standalone enrolment report is one input group written on its own, and it is the form
        # `preflight` actually emits (`{label}_inputs.json`). Refusing it meant the shipped tool
        # could not freeze the pin the shipped pipeline needs, which is how an artifact ends up
        # published with nothing constraining the bytes it read.
        if isinstance(blob.get("label"), str) and "content_digest_sha256" in blob:
            inputs = {blob["label"]: blob}
            # `preflight` looks this pin up by the LABEL alone (input_contract.py: `pin_key =
            # label`). Writing it under a prefixed key would leave a registry that validates and
            # displays perfectly while the reader still finds nothing, so the key here is the
            # label and `--prefix` is refused rather than silently ignored.
            standalone = True
            if explicit_prefix:
                raise SystemExit(
                    f"{name} is a standalone enrolment report, whose pin key is its label "
                    f"({blob['label']!r}) because that is what preflight looks up. --prefix would "
                    f"be ignored, so it is refused instead of quietly doing nothing.")
        else:
            raise SystemExit(
                f"{name} has no `inputs` block to freeze from, and is not a standalone enrolment "
                f"report either (those carry `label` and `content_digest_sha256` at the top level)")

    if only:
        unknown = sorted(set(only) - set(inputs))
        if unknown:
            raise SystemExit(
                f"{name} has no input group(s) {', '.join(unknown)}; it carries "
                f"{', '.join(sorted(inputs))}")
        inputs = {g: r for g, r in inputs.items() if g in set(only)}

    plan = []
    for group, rep in sorted(inputs.items()):
        if not isinstance(rep, dict):
            raise SystemExit(f"{name}: inputs.{group} is not an object")
        digest = rep.get("content_digest_sha256")
        declared = rep.get("n_declared")
        per_id = rep.get("per_id")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise SystemExit(f"{name}: inputs.{group} has no valid content digest")
        # `isinstance(True, int)` is True in Python, so a Boolean count passes a naive integer
        # check and then formats as 1 or 0. A count is a cardinality: Booleans and negatives are
        # not counts, and a count that disagrees with the records it summarises is not a count of
        # them.
        if isinstance(declared, bool) or not isinstance(declared, int) or declared < 0:
            raise SystemExit(
                f"{name}: inputs.{group}.n_declared is not a non-negative integer ({declared!r})")

        # Recompute the aggregate from the per-id records rather than trusting the stored total.
        # An artifact with no per_id block cannot be recomputed, and degrading silently to trusting
        # its stored digest turns the advertised defence into a no-op that omission can skip.
        if not isinstance(per_id, dict) or not per_id:
            raise SystemExit(
                f"{name}: inputs.{group} has no per_id block, so its aggregate digest cannot be "
                f"recomputed and there is nothing here to verify. Re-run the analysis with a "
                f"contract that emits per_id, or freeze from an artifact that has one.")
        if declared != len(per_id):
            raise SystemExit(
                f"{name}: inputs.{group}.n_declared is {declared} but per_id carries "
                f"{len(per_id)} records. The count and the records it summarises must agree, or "
                f"the pin describes a population the digest was not taken over.")

        for i, rec in sorted(per_id.items()):
            # Ids must be CANONICAL decimal integers. The aggregate digest is built by sorting
            # these keys through int(), so a non-numeric key dies with a raw ValueError from
            # inside the digest expression rather than a refusal, and "07" and "7" are two
            # spellings of one agent that produce two different digests over identical bytes.
            if not isinstance(i, str) or not re.fullmatch(r"(0|[1-9][0-9]*)", i):
                raise SystemExit(
                    f"{name}: inputs.{group}.per_id key {i!r} is not a canonical decimal integer. "
                    f"The aggregate digest is taken over these keys, so a non-numeric or "
                    f"zero-padded id either cannot be ordered or gives two digests for one agent.")
            if not isinstance(rec, dict):
                raise SystemExit(f"{name}: inputs.{group}.per_id[{i}] is not an object")
            sha = rec.get("sha256")
            rows = rec.get("rows")
            if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
                raise SystemExit(
                    f"{name}: inputs.{group}.per_id[{i}] has no valid sha256. A record without a "
                    f"digest contributes its own id to the aggregate and constrains no bytes.")
            if isinstance(rows, bool) or not isinstance(rows, int) or rows < 0:
                raise SystemExit(
                    f"{name}: inputs.{group}.per_id[{i}].rows is not a non-negative integer "
                    f"({rows!r})")

        joined = chr(10).join(
                f"{i}:{per_id[i]['sha256']}:{per_id[i]['rows']}"
            for i in sorted(per_id, key=lambda x: int(x)))
        recomputed = hashlib.sha256(joined.encode()).hexdigest()
        if recomputed != digest:
            raise SystemExit(
                f"{name}: inputs.{group} aggregate digest does not match its own per_id "
                f"records ({digest[:16]} vs {recomputed[:16]}). Do not freeze this artifact.")

        key = group if standalone else f"{prefix}_{group}"
        old = _load_pins().get(key)
        if old is None:
            verb = "record"
        elif old.get("content_digest_sha256") == digest:
            verb = "unchanged"
        else:
            verb = "REPLACE"
        plan.append((key, declared, digest, verb, old))
    return path, art_sha, plan


def _print_plan(path: Path, art_sha: str, plan: list) -> None:
    # The resolved path, not the basename: with two candidate roots the basename does not say
    # which file was read, and a plan is authority only when it names what it is a plan about.
    print(f"artifact  {path.resolve()}")
    print(f"sha256    {art_sha}")
    print()
    for key, declared, digest, verb, old in plan:
        print(f"  {verb:9s} {key}")
        print(f"            {declared:>7,} inputs")
        print(f"            {digest}")
        if verb == "REPLACE":
            print(f"            was {old.get('content_digest_sha256')}")
    print()
    print("To write these, repeat with:")
    print(f"    --apply --confirm-artifact-sha256 {art_sha}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true", help="print the registry and exit")
    ap.add_argument("--from-artifact", help="freeze from an artifact's own `inputs` block")
    ap.add_argument("--prefix", help="registry key prefix for --from-artifact")
    ap.add_argument("--only", action="append", default=[], metavar="GROUP",
                    help="freeze only this input group (repeatable); default is all of them")
    ap.add_argument("--apply", action="store_true", help="write; requires the confirmation hash")
    ap.add_argument("--confirm-artifact-sha256", help="the artifact hash printed by the dry run")
    a = ap.parse_args()

    if a.show or not a.from_artifact:
        _show()
        return

    prefix = a.prefix or Path(a.from_artifact).stem
    path, art_sha, plan = _plan_from_artifact(a.from_artifact, prefix, tuple(a.only),
                                              explicit_prefix=a.prefix is not None)

    if not a.apply:
        print("DRY RUN - nothing is written\n")
        _print_plan(path, art_sha, plan)
        return

    if a.confirm_artifact_sha256 != art_sha:
        _print_plan(path, art_sha, plan)
        raise SystemExit(
            "\nrefusing to write: --confirm-artifact-sha256 does not match this artifact.\n"
            f"  given    {a.confirm_artifact_sha256}\n"
            f"  expected {art_sha}\n"
            "The confirmation exists so that what is written is what was reviewed. Re-run the dry "
            "run, read the digests, then paste the hash it prints.")

    ic.FREEZE_MODE = True
    n = 0
    for key, declared, digest, verb, _old in plan:
        if verb == "unchanged":
            continue
        ic._record_pin(key, digest, declared)
        n += 1
        print(f"  wrote {key}  {digest[:16]}  {declared:,} inputs")
    print(f"\n{n} pin(s) written")


if __name__ == "__main__":
    main()
