"""
db_analysis_identity_rotation.py — Identity rotation / respawn as a third escape
channel (this paper, Finding 6). See PLAN_identity_rotation.md.

Question: when a honeypot-confirmed bot is blocked and its volume collapses
(currently labeled `retreat`), does the *operation* reappear under a different
a_id (= a different user-agent), time-locked to the block and linked by shared
infrastructure? If so, "retreat" overcounts true suppression and rotation is a
second observability-loss channel.

BRIDGE — what survives the UA change. The `ip` table (52M rows) cannot be looked
up by i_agent (non-leftmost in its only index -> TokuDB error 1152), so the IP
bridge is infeasible on this schema. We use the HONEYPOT-URL bridge instead:
url2agent_honeypot (194k rows, agent-ids + url-ids, NO PII) is pulled ONCE and
all linkage is computed in memory. A fresh UA hitting the SAME decoy paths is a
stronger handoff signal than a shared cloud IP anyway.

SCOPE (state plainly in this paper §7.8): successors found this way are themselves
honeypot-confirmed bots, so this measures rotation WITHIN the confirmed-bot fleet
(both endpoints are ground-truth bots -> no label-contamination ambiguity).
Handoffs to identities that never trip a honeypot need the (infeasible) IP bridge
-> the detected rate is a LOWER BOUND.

RARITY: popular honeypots are hit by thousands of unrelated bots, so shared URLs
are weighted by inverse popularity. Precision core = shares a RARE decoy;
recall envelope = shares any decoy.

CLUSTER UNIT: death-cluster (a death A + its bridged successors = one bootstrap
resample unit) — privacy-safe and removes the bridge/cluster circularity.

Stages (resumable): deaths -> bridge graph -> pairs (event-study around t_A) ->
inference (conservation + nulls + per-pair placebo).
"""
from __future__ import annotations

import datetime as _dt
import json
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import hashlib

import numpy as np
import pandas as pd
from pathlib import Path

import config
import db
import db_source as dbs
from roster import ROSTER_CSV

# ---- knobs ----
WC = 14               # +/- CALENDAR-day window around the death date t_A (= 2*tau_A)
POP_CAP = 1000        # ignore honeypot URLs hit by more agents than this for candidate generation
CORE_POP = 25         # "rare" decoy: a shared URL hit by <= this many agents = precision core
TOP_K = 40            # max candidate successors examined per death (by rarity score)
PLACEBO_OFFSET = 60   # days before t_A for the per-pair placebo event
MIN_DAYS = 15
VOL_FLOOR = 1.0
RANDOM_STATE = 42
N_BOOT = 2000

# ---- IP bridge (requires idx_ip_agent_time on the ip table) ----
TOPK_IP = 12
N_NULL_DRAWS = 200        # permutation draws for the placebo distribution               # candidate successors per death (by shared-IP count)
IP_BRIDGE_TIMEOUT_MS = 30000   # per-query cap; pathological popular-IP joins skip
IP_OVERLAP_CORE = 0.5      # precision core: successor shares >=50% of A's IP footprint
IP_SHARED_MIN = 3          # ...and >=3 shared IPs (kills single-popular-IP coincidences)

TAB = config.TAB_DIR
FIG = config.FIG_DIR
DEATHS_CSV = TAB / "DB_rotation_deaths.csv"
ARM_IDS_CSV = config.OUT_DIR / "DB_arm_of_record_ids.csv"
# The AXIS training source. It must carry `is_bot`/`is_control`; the arm-of-record ID
# file is a bare id list and cannot train an axis. This is a different question from
# WHICH DEATH COHORT to analyse, which the suffix selects, and conflating the two is
# what made the documented command depend on a stale cached axis.
AXIS_ROSTER_CSV = config.OUT_DIR / "roster_v2.csv"
PAIRS_CSV = TAB / "DB_rotation_pairs.csv"
IP_PAIRS_CSV = TAB / "DB_rotation_ip_pairs.csv"
SUFFIX = ""            # set by the entry points; suffixes every artifact of an alternative arm
_C_ROSTER = [ROSTER_CSV]   # roster the helpers below read; rebound by _set_suffix callers


def _set_suffix(sfx: str) -> None:
    """Point the death list, the pair caches and every output at a suffixed name,
    so the rotation test can be re-run on an alternative arm definition without
    overwriting the artifacts of the default run."""
    global SUFFIX, DEATHS_CSV, PAIRS_CSV, IP_PAIRS_CSV
    SUFFIX = sfx
    DEATHS_CSV = TAB / f"DB_rotation_deaths{sfx}.csv"
    PAIRS_CSV = TAB / f"DB_rotation_pairs{sfx}.csv"
    IP_PAIRS_CSV = TAB / f"DB_rotation_ip_pairs{sfx}.csv"


# ===================== botness axis =====================

# The ordered feature specification the axis is expressed in. Recorded in the provenance sidecar,
# because an axis built over different features or a different order is a different axis even when
# the roster and the estimator settings match.
_AXIS_FEATURES = ("p404", "robots_rate", "log1p(hits)")


def _behavior_matrix(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack([
        df["p404"].to_numpy(float),
        df["robots_rate"].to_numpy(float),
        np.log1p(df["hits"].to_numpy(float)),
    ])


AXIS_NPZ = TAB / "_botness_axis.npz"    # note: suffixed by build_botness_axis when SUFFIX is set


def _pin_registry_path() -> Path:
    """`input_pins.json`, in the source tree or in the public bundle.

    The two layouts put the analysis directory in different places relative to this file;
    both ship the registry alongside this script, so either resolves.
    """
    here = Path(__file__).resolve().parent
    for cand in _PIN_CANDIDATES(here):
        if cand.exists():
            return cand
    return here / "analysis" / "input_pins.json"          # for the message when it is missing


def _PIN_CANDIDATES(here: Path) -> tuple:
    return (here.parent / "papers" / "D-adversarial-friction" / "analysis" / "input_pins.json",
            here / "analysis" / "input_pins.json")


def _axis_pin(key: str) -> dict:
    """The registry entry for one axis, from outside the files it authenticates.

    Refuses a missing or unparseable registry rather than returning nothing. A check that
    degrades to a pass when its authority is absent is not a check: whoever can forge the
    npz and its sidecar can also delete the file that would have caught the forgery.
    """
    reg = _pin_registry_path()
    if not reg.exists():
        looked = "; ".join(str(c) for c in _PIN_CANDIDATES(Path(__file__).resolve().parent))
        raise SystemExit(
            f"input_pins.json is missing, so no authority outside the axis cache can establish "
            f"which bytes the pinned axis has. Looked in: {looked}. Restore the registry, or "
            f"rebuild the axis without --cache.")
    try:
        blob = json.loads(reg.read_text(encoding="utf-8"))
    except Exception as exc:                                   # noqa: BLE001
        raise SystemExit(f"{reg.name} could not be read as JSON ({type(exc).__name__}: {exc})")
    if key not in blob:
        raise SystemExit(
            f"{reg.name} has no `{key}` entry, so the cached axis is pinned only by the sidecar "
            f"beside it. A payload and a sidecar rewritten together agree with each other; the "
            f"registry is what they cannot rewrite. Add the entry, or rebuild without --cache.")
    return blob[key]


def build_botness_axis(roster: pd.DataFrame, min_days: int = 20, cache: bool = False) -> dict:
    axis_npz = AXIS_NPZ.with_name(f"_botness_axis{SUFFIX}.npz")

    # The roster is validated BEFORE the cache is consulted, and a cached axis must prove it came
    # from this roster. Returning a cached axis first makes the roster argument decorative: a
    # deliberately supplied --roster is ignored whenever the file exists, and a revised roster
    # leaves every result on the old axis with no refusal. An axis is a population object, so it
    # is pinned to the population that produced it.
    missing_cols = [c for c in ("is_bot", "is_control") if c not in roster.columns]
    if missing_cols:
        raise SystemExit(
            f"the botness axis needs {', '.join(missing_cols)} and this roster has "
            f"{', '.join(map(str, roster.columns))}. The axis is built from labelled controls and "
            f"bots, so a bare id list cannot produce one.")
    roster_digest = hashlib.sha256(
        pd.util.hash_pandas_object(roster[["a_id", "is_bot", "is_control"]],
                                   index=False).values.tobytes()).hexdigest()

    if cache and axis_npz.exists():
        # A digest stored INSIDE the file it authenticates is not an integrity boundary: anyone
        # who can rewrite the payload can rewrite the digest beside it. Provenance lives in a
        # sidecar that also records the estimator settings, and it covers the npz bytes, so an
        # altered axis, an altered min_days, or a substituted roster are all refusals.
        side = axis_npz.with_suffix(".provenance.json")
        if not side.exists():
            raise SystemExit(
                f"{axis_npz.name} has no {side.name}, so nothing outside the file establishes "
                f"what it was built from. Delete the cache and rebuild deliberately.")
        prov = json.loads(side.read_text(encoding="utf-8"))
        npz_sha = hashlib.sha256(axis_npz.read_bytes()).hexdigest()

        # The sidecar and the npz can be forged as a PAIR: rewrite the payload, regenerate the
        # sidecar coherently, and both agree. The registry is the authority neither of them can
        # rewrite, so the pair is checked against it whenever an entry exists.
        pinned = _axis_pin(f"axis{SUFFIX}")
        for field, want in (("npz_sha256", npz_sha), ("roster_sha256", roster_digest)):
            if pinned.get(field) != want:
                raise SystemExit(
                    f"{axis_npz.name} does not match the {field} pinned in "
                    f"{_pin_registry_path().name}: the registry records "
                    f"{str(pinned.get(field))[:12]!r}, these bytes give {want[:12]!r}. A payload "
                    f"and a sidecar rewritten together agree with each other, so agreement "
                    f"between them proves nothing; the registry is the authority.")
        for field, want, got in (("npz_sha256", prov.get("npz_sha256"), npz_sha),
                                 ("roster_sha256", prov.get("roster_sha256"), roster_digest),
                                 ("min_days", prov.get("min_days"), int(min_days)),
                                 ("features", prov.get("features"), list(_AXIS_FEATURES))):
            if want != got:
                raise SystemExit(
                    f"{axis_npz.name} does not match its provenance on {field}: recorded "
                    f"{str(want)[:40]!r}, this run requires {str(got)[:40]!r}. An axis is a "
                    f"property of the population, the estimator settings and the bytes that "
                    f"produced it; refusing rather than returning it.")
        d = np.load(axis_npz)
        return {"mu_H": d["mu_H"], "sigma_H": d["sigma_H"], "axis": d["axis"],
                "bot_centroid_b": float(d["bot_centroid_b"])}

    missing = [c for c in ("is_bot", "is_control") if c not in roster.columns]
    if missing:
        raise SystemExit(
            f"the botness axis needs {', '.join(missing)} and this roster has "
            f"{', '.join(roster.columns)}. The axis is built from labelled controls and bots, so "
            f"a bare id list cannot produce one; pass a roster carrying those columns, or supply "
            f"a prebuilt axis.")

    ctl, bot = [], []
    for _, r in roster.iterrows():
        cp = dbs._cache_path(int(r["a_id"]))
        if not cp.exists():
            continue
        df = pd.read_parquet(cp)
        if len(df) < min_days:
            continue
        v = _behavior_matrix(df).mean(axis=0)
        if bool(r["is_bot"]):
            bot.append(v)
        elif bool(r["is_control"]):
            ctl.append(v)
    ctl, bot = np.array(ctl), np.array(bot)
    if not len(ctl) or not len(bot):
        raise SystemExit(
            f"the botness axis has {len(ctl)} control and {len(bot)} bot agents with enough "
            f"observed days, so it cannot be built. Caching it would write an all-NaN axis that a "
            f"later run would silently load as if it were a measurement.")
    mu_H, sigma_H, mu_B = ctl.mean(0), ctl.std(0) + 1e-9, bot.mean(0)
    muB_z = (mu_B - mu_H) / sigma_H
    axis = muB_z / (np.linalg.norm(muB_z) + 1e-9)
    out = {"mu_H": mu_H, "sigma_H": sigma_H, "axis": axis,
           "bot_centroid_b": float(np.linalg.norm(muB_z))}
    if cache:
        config.ensure_dirs()
        np.savez(axis_npz, **out)
        # Provenance is written AFTER the payload and covers its bytes.
        axis_npz.with_suffix(".provenance.json").write_text(json.dumps({
            "npz_sha256": hashlib.sha256(axis_npz.read_bytes()).hexdigest(),
            "roster_sha256": roster_digest,
            "min_days": int(min_days),
            "features": list(_AXIS_FEATURES),
            "shapes": {k: list(np.asarray(v).shape) for k, v in out.items()},
        }, indent=2), encoding="utf-8")
    return out


def _b_series(df: pd.DataFrame, ax: dict) -> np.ndarray:
    return ((_behavior_matrix(df) - ax["mu_H"]) / ax["sigma_H"]) @ ax["axis"]


# ===================== honeypot bridge (in-memory, no PII) =====================

def load_honeypot_graph() -> dict:
    """Pull url2agent_honeypot once; build agent->urls, url->popularity. a_ids and
    url-ids only (no PII)."""
    print("[bridge] pulling url2agent_honeypot (once)...")
    t = db.read_sql("SELECT uh_a_id AS a, uh_u_id AS u FROM url2agent_honeypot WHERE uh_total > 0")
    t = t[t["a"] <= dbs.MAX_A_ID]
    agent_urls: dict[int, set] = {}
    for a, u in zip(t["a"].astype(int), t["u"].astype(int)):
        agent_urls.setdefault(a, set()).add(u)
    url_pop = t.groupby("u")["a"].nunique().to_dict()
    url_agents: dict[int, list] = {}
    for a, u in zip(t["a"].astype(int), t["u"].astype(int)):
        if url_pop[u] <= POP_CAP:                        # only non-mega-popular decoys link agents
            url_agents.setdefault(u, []).append(a)
    print(f"  {len(agent_urls)} honeypot agents, {len(url_pop)} decoy URLs "
          f"(median pop {int(np.median(list(url_pop.values())))})")
    return {"agent_urls": agent_urls, "url_pop": url_pop, "url_agents": url_agents}


def bridge_candidates(a_id: int, g: dict, k: int = TOP_K) -> pd.DataFrame:
    urls = g["agent_urls"].get(a_id, set())
    if not urls:
        return pd.DataFrame(columns=["a_id", "shared_hp", "rare_score", "min_pop", "jaccard"])
    cand: dict[int, list] = {}
    for u in urls:
        if g["url_pop"].get(u, 10**9) > POP_CAP:
            continue
        for b in g["url_agents"].get(u, []):
            if b != a_id:
                cand.setdefault(b, []).append(u)
    cols = ["a_id", "shared_hp", "rare_score", "min_pop", "jaccard"]
    if not cand:
        return pd.DataFrame(columns=cols)
    rows = []
    for b, shared in cand.items():
        pops = [g["url_pop"][u] for u in shared]
        union = len(urls | g["agent_urls"].get(b, set()))
        rows.append({"a_id": b, "shared_hp": len(shared),
                     "rare_score": float(sum(1.0 / p for p in pops)),
                     "min_pop": int(min(pops)),
                     "jaccard": len(shared) / union if union else 0.0})
    return pd.DataFrame(rows).sort_values("rare_score", ascending=False).head(k)


# ===================== IP bridge (server-side self-join; a_id + counts only, no PII) =====================

_IP_NA = "SELECT COUNT(DISTINCT i_name) c FROM ip WHERE i_agent = %s AND i_proxy = 0"
_IP_BRIDGE = """
SELECT b.i_agent AS a_id, COUNT(DISTINCT b.i_name) AS shared_ips
FROM ip a JOIN ip b ON a.i_name = b.i_name
WHERE a.i_agent = %s AND b.i_agent <> a.i_agent AND a.i_proxy = 0
GROUP BY b.i_agent ORDER BY shared_ips DESC LIMIT %s
"""


def bridge_candidates_ip(a_id: int, conn, k: int = TOPK_IP) -> pd.DataFrame:
    """Successors sharing non-proxy IPs with the dead bot. Reaches ANY agent (incl.
    fresh / non-honeypot identities) -> closes the channel the honeypot bridge missed.
    Strength = overlap fraction (shared / A's IP footprint). Never returns IPs."""
    cols = ["a_id", "shared_ips", "overlap_frac", "in_core"]
    try:
        nA = int(db.read_sql(_IP_NA, (a_id,), conn=conn).c.iloc[0])
        if nA == 0:
            return pd.DataFrame(columns=cols)
        # Serve the enumeration from cache when warm_ip_bridges.py has already run it. The
        # query is the pipeline's dominant cost and scales with the dead agent's IP footprint;
        # the cached frame is the same raw result, so every threshold below still applies here.
        df = None
        try:
            import warm_ip_bridges as _wib
            df = _wib.load_cached(a_id, min_cap=k * 4)
        except Exception:
            df = None
        if df is None:
            df = db.read_sql(_IP_BRIDGE, (a_id, k * 4), conn=conn)
    except Exception as e:
        # Record, do not merely print. A bridge-enumeration failure removes a death from the
        # analysis just as surely as an extraction failure does, and the 19 missing rows in the
        # v2 run came from THIS query's 30 s cap, not from successor extraction.
        _UNFETCHED[int(a_id)] = f"ip-bridge enumeration: {type(e).__name__}: {str(e)[:100]}"
        print(f"  [ip-bridge] skip {a_id}: {str(e)[:70]}")
        return pd.DataFrame(columns=cols)
    if df.empty:
        return pd.DataFrame(columns=cols)
    df["overlap_frac"] = df["shared_ips"] / nA
    df["in_core"] = ((df["overlap_frac"] >= IP_OVERLAP_CORE) & (df["shared_ips"] >= IP_SHARED_MIN)).astype(int)
    df["rank"] = df["in_core"] * 10**6 + df["shared_ips"]
    return df.sort_values("rank", ascending=False).head(k).drop(columns="rank")


# ===================== pairs: event-study around the shared death date t_A =====================

# Agents whose extraction failed. An empty frame from a failed query is indistinguishable
# downstream from an agent that genuinely has no data, so the two must not be conflated
# silently: a volume-correlated hole in the cohort is invisible unless coverage is reported.
# Written to outputs/DB_rotation_unfetched<suffix>.json at the end of the run.
_UNFETCHED: dict[int, str] = {}


def _series(a_id: int, conn, t_center=None) -> pd.DataFrame:
    """Series for a candidate successor.

    When `t_center` is given and the agent has no full cached history, only the span the
    bridges actually read is fetched: the placebo window opens at t-PLACEBO_OFFSET-WC and the
    main window closes at t+WC. A full-history extraction of a heavy agent costs minutes to
    tens of minutes because the group-by cannot use index h_1; the bounded fetch is a range
    scan on that index and is ~11x faster. `_cal_stats` treats absent days as zero volume, and
    the span below covers both windows completely, so no day the estimator reads is lost.
    """
    cp = dbs._cache_path(int(a_id))
    if cp.exists():
        return pd.read_parquet(cp)
    try:
        if t_center is not None:
            lo = t_center - _dt.timedelta(days=PLACEBO_OFFSET + WC + 1)
            hi = t_center + _dt.timedelta(days=WC + 1)
            return dbs.extract_series_rotation(int(a_id), lo, hi, conn=conn)
        return dbs.extract_series(int(a_id), conn=conn, use_cache=True)
    except Exception as exc:
        # Record rather than swallow. The heaviest agents are the ones that fail, and they are
        # the operators most capable of rotating, so an unrecorded skip biases the result
        # toward finding no successor.
        _UNFETCHED[int(a_id)] = f"{type(exc).__name__}: {str(exc)[:120]}"
        return pd.DataFrame()


def dump_unfetched(tag: str = "") -> None:
    """Write the skipped-agent register so coverage can be reported, not assumed."""
    import json as _json
    p = config.OUT_DIR / f"DB_rotation_unfetched{SUFFIX}{tag}.json"
    # Called from a `finally`, so it also runs when the analysis aborted. A coverage register from
    # a run that produced no result describes nothing, and overwriting the last good one loses the
    # coverage the published artifact was computed under.
    if not _UNFETCHED and p.exists():
        print(f"[coverage] nothing extracted this run; leaving {p.name} as it stands")
        return
    p.write_text(_json.dumps({"n_unfetched": len(_UNFETCHED),
                              "agents": {str(k): v for k, v in sorted(_UNFETCHED.items())}},
                             indent=2), encoding="utf-8")
    print(f"[coverage] {len(_UNFETCHED)} agents could not be extracted -> {p.name}")


def _cal_stats(df: pd.DataFrame, t_center, ax: dict, days: int = WC):
    """Calendar-aligned windows around t_center (missing days = 0 volume, so a
    silent-then-active successor registers a real ramp). Returns
    (log-vol pre, log-vol post, botness pre, botness post, n_pre_days, n_post_days)."""
    dates = pd.to_datetime(df["date"]).dt.date.to_numpy()
    hits = df["hits"].to_numpy(float)
    b = _b_series(df, ax)
    lo, hi = t_center - _dt.timedelta(days=days), t_center + _dt.timedelta(days=days)
    pre = (dates >= lo) & (dates < t_center)
    post = (dates >= t_center) & (dates < hi)
    return (float(np.log1p(hits[pre].sum())), float(np.log1p(hits[post].sum())),
            float(np.nanmean(b[pre])) if pre.any() else np.nan,
            float(np.nanmean(b[post])) if post.any() else np.nan,
            int(pre.sum()), int(post.sum()))


def stage_pairs(deaths: pd.DataFrame, ax: dict, g: dict, out_csv=None) -> pd.DataFrame:
    # resolved at CALL time: a default bound at def time would ignore _set_suffix
    out_csv = PAIRS_CSV if out_csv is None else out_csv
    partial = out_csv.with_suffix(".partial.csv")
    progress = out_csv.with_suffix(".progress.txt")
    rows, start = [], 0
    if partial.exists() and progress.exists():
        rows = pd.read_csv(partial).to_dict("records")
        start = int(progress.read_text().strip())
        print(f"[pairs] resuming from death {start}/{len(deaths)} ({len(rows)} pairs so far)")
    conn = db.get_connection()
    for n, (_, drow) in enumerate(deaths.iloc[start:].iterrows(), start + 1):
        a_id = int(drow["a_id"])
        t_A = pd.to_datetime(drow["t_death"]).date()
        dfa = pd.read_parquet(dbs._cache_path(a_id))
        vA_pre, vA_post, bA_pre, bA_post, _, _ = _cal_stats(dfa, t_A, ax)
        for _, c in bridge_candidates(a_id, g).iterrows():
            b_id = int(c["a_id"])
            dfb = _series(b_id, conn, t_center=t_A)
            if len(dfb) < MIN_DAYS:
                continue
            vB_pre, vB_post, bB_pre, bB_post, _, npost = _cal_stats(dfb, t_A, ax)
            if npost == 0:                                # B not observed after t_A at all
                continue
            # per-pair placebo: B's change around a date PLACEBO_OFFSET days earlier
            t_P = t_A - _dt.timedelta(days=PLACEBO_OFFSET)
            vBp_pre, vBp_post, _, _, _, _ = _cal_stats(dfb, t_P, ax)
            rows.append({
                "a_death": a_id, "a_succ": b_id, "t_A": t_A,
                "shared_hp": int(c["shared_hp"]), "rare_score": float(c["rare_score"]),
                "min_pop": int(c["min_pop"]), "jaccard": float(c["jaccard"]),
                "in_core": int(c["min_pop"] <= CORE_POP),
                "bA_pre": bA_pre, "dA_b": bA_post - bA_pre, "dA_v": vA_post - vA_pre,
                "bB_post": bB_post, "dB_b": bB_post - bB_pre, "dB_v": vB_post - vB_pre,
                "dB_v_placebo": (vBp_post - vBp_pre),
            })
        if n % 50 == 0:
            print(f"  [pairs] {n}/{len(deaths)} deaths, {len(rows)} pairs")
            pd.DataFrame(rows).to_csv(partial, index=False)
            progress.write_text(str(n))
    conn.close()
    pairs = pd.DataFrame(rows).drop_duplicates(subset=["a_death", "a_succ"], keep="first")
    pairs.to_csv(out_csv, index=False)
    partial.unlink(missing_ok=True)
    progress.unlink(missing_ok=True)
    print(f"[pairs] {len(pairs)} candidate pairs -> {out_csv.name}")
    return pairs


def stage_pairs_ip(deaths: pd.DataFrame, ax: dict, out_csv=None) -> pd.DataFrame:
    # resolved at CALL time (see stage_pairs)
    out_csv = IP_PAIRS_CSV if out_csv is None else out_csv
    """Same event-study as stage_pairs but using the IP bridge. Resumable. Successors
    may be ANY agent (extracted on demand), so this tests handoffs into fresh /
    non-honeypot identities that the honeypot bridge cannot reach."""
    partial = out_csv.with_suffix(".partial.csv")
    progress = out_csv.with_suffix(".progress.txt")
    rows, start = [], 0
    if partial.exists() and progress.exists():
        rows = pd.read_csv(partial).to_dict("records")
        start = int(progress.read_text().strip())
        print(f"[ip-pairs] resuming from death {start}/{len(deaths)} ({len(rows)} pairs so far)")
    bconn = db.get_connection(timeout_ms=IP_BRIDGE_TIMEOUT_MS)
    sconn = db.get_connection()
    for n, (_, drow) in enumerate(deaths.iloc[start:].iterrows(), start + 1):
        a_id = int(drow["a_id"])
        t_A = pd.to_datetime(drow["t_death"]).date()
        dfa = pd.read_parquet(dbs._cache_path(a_id))
        vA_pre, vA_post, bA_pre, bA_post, _, _ = _cal_stats(dfa, t_A, ax)
        for _, c in bridge_candidates_ip(a_id, bconn).iterrows():
            b_id = int(c["a_id"])
            dfb = _series(b_id, sconn, t_center=t_A)
            if len(dfb) < MIN_DAYS:
                continue
            vB_pre, vB_post, bB_pre, bB_post, _, npost = _cal_stats(dfb, t_A, ax)
            if npost == 0:
                continue
            t_P = t_A - _dt.timedelta(days=PLACEBO_OFFSET)
            vBp_pre, vBp_post, _, _, _, _ = _cal_stats(dfb, t_P, ax)
            rows.append({
                "a_death": a_id, "a_succ": b_id, "t_A": t_A,
                "shared_ips": int(c["shared_ips"]), "overlap_frac": float(c["overlap_frac"]),
                "in_core": int(c["in_core"]),
                "bA_pre": bA_pre, "dA_b": bA_post - bA_pre, "dA_v": vA_post - vA_pre,
                "bB_post": bB_post, "dB_b": bB_post - bB_pre, "dB_v": vB_post - vB_pre,
                "dB_v_placebo": (vBp_post - vBp_pre),
            })
        if n % 25 == 0:
            print(f"  [ip-pairs] {n}/{len(deaths)} deaths, {len(rows)} pairs")
            pd.DataFrame(rows).to_csv(partial, index=False)
            progress.write_text(str(n))
    bconn.close(); sconn.close()
    pairs = pd.DataFrame(rows).drop_duplicates(subset=["a_death", "a_succ"], keep="first")
    pairs.to_csv(out_csv, index=False)
    partial.unlink(missing_ok=True)
    progress.unlink(missing_ok=True)
    print(f"[ip-pairs] {len(pairs)} candidate pairs -> {out_csv.name}")
    return pairs


# ===================== inference =====================

def _boot(x: np.ndarray, stat, n: int = N_BOOT, seed: int = RANDOM_STATE):
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return (np.nan, np.nan, np.nan, len(x))
    rng = np.random.default_rng(seed)
    bs = [stat(x[rng.integers(0, len(x), len(x))]) for _ in range(n)]
    return (float(stat(x)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), len(x))


def _transfer(sub: pd.DataFrame, label: str) -> dict:
    """Does the successor's volume RAMP (dB_v>0) and its botness rise, as A dies?
    Headline = mean successor volume change around t_A; CI by death-cluster bootstrap."""
    if sub.empty:
        return {"label": label, "n": 0}
    # death-cluster bootstrap: resample deaths, keep their candidate rows together
    deaths = sub["a_death"].unique()
    rng = np.random.default_rng(RANDOM_STATE)
    groups = {d: sub[sub.a_death == d] for d in deaths}

    def cluster_mean(col):
        vals = []
        for _ in range(N_BOOT):
            chosen = rng.choice(deaths, len(deaths), replace=True)
            vals.append(np.nanmean(np.concatenate([groups[d][col].to_numpy() for d in chosen])))
        point = float(np.nanmean(sub[col]))
        return point, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

    dBv, dBv_lo, dBv_hi = cluster_mean("dB_v")
    dBb, dBb_lo, dBb_hi = cluster_mean("dB_b")
    plac = float(np.nanmean(sub["dB_v_placebo"]))

    # DATE-cluster bootstrap, reported alongside the death-cluster one.
    # Deaths are not independent: they concentrate on shared calendar dates (41 of the 261 v4
    # deaths fall on 2022-02-10, and 10 of the 18 disappearance episodes on 2021-03-25). Resampling
    # deaths treats agents that died together on one date as separate evidence, which understates
    # the interval whenever a common shock is what put them there. Resampling DATES keeps them
    # together. Where the two disagree, the date-clustered interval is the honest one.
    dates = pd.to_datetime(sub["t_A"]).dt.date.unique()
    dgroups = {d: sub[pd.to_datetime(sub["t_A"]).dt.date == d] for d in dates}
    drng = np.random.default_rng(RANDOM_STATE)

    def date_cluster_ci(col):
        vals = []
        for _ in range(N_BOOT):
            chosen = drng.choice(dates, len(dates), replace=True)
            vals.append(np.nanmean(np.concatenate([dgroups[d][col].to_numpy() for d in chosen])))
        return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]

    dBv_date_ci = date_cluster_ci("dB_v")
    n_dates = int(len(dates))
    top_date_share = float(max(len(g) for g in dgroups.values()) / len(sub))
    return {"label": label, "n_pairs": int(len(sub)), "n_deaths": int(len(deaths)),
            "n_dates": n_dates, "largest_date_share_of_pairs": round(top_date_share, 4),
            "succ_dvol_ci_date_cluster": dBv_date_ci,
            "succ_dvol_mean": dBv, "succ_dvol_ci": [dBv_lo, dBv_hi],
            "succ_dbot_mean": dBb, "succ_dbot_ci": [dBb_lo, dBb_hi],
            "succ_dvol_placebo": plac,
            "ramp_minus_placebo": dBv - plac}


def _link_permutation_null(pairs: pd.DataFrame, deaths: pd.DataFrame, ax: dict,
                           conn=None, n_draws: int = N_NULL_DRAWS,
                           seed: int = RANDOM_STATE,
                           all_pairs: pd.DataFrame | None = None,
                           dedupe_dates: bool = False) -> dict:
    """Permutation placebo: recompute each successor's volume change around dates it is NOT
    linked to, repeatedly, and report the DISTRIBUTION.

    Five requirements define this null. Each is a constraint a reimplementer must satisfy, and
    each rules out a specific way a permutation placebo can be built and still be wrong.

    1. A DRAW IS NOT A NULL. One realisation is an illustration, not a falsification: a 200-seed
       replay of a single-draw version of this estimator spans -0.467 to +0.279. `n_draws`
       assignments are run and the mean is reported with a 2.5-97.5 percentile band, so the
       observed effect is read against a spread rather than against one number.
    2. "ANOTHER DEATH" MUST BE ANOTHER. Drawing from the raw death-date list can hand a successor
       a date it is genuinely linked to, which pulls the null toward the observed effect. Each
       successor's eligible pool excludes every date it appears with.
    3. COVERAGE MUST NOT BE SELECTED ON THE OUTCOME. `_cal_stats` treats absent days as zero
       volume by design, so a successor observed before a date and silent after it has a real,
       negative change - exactly the case of interest. Converting `npost == 0` to NaN and dropping
       it would bias the null upward. A zero post-window is a measurement whenever the successor
       has a series at all. Two things do exclude, and both are decided BEFORE any draw so that
       support is identical across replicates (requirement 4): a missing series, and a date on
       which the successor is unobserved on both sides of the window, which would yield a
       mechanical 0 - 0 rather than a measurement.
    4. ELIGIBILITY MUST BE DECIDED BEFORE THE DRAW. Discarding a unit whose permuted window turns
       out dormant changes the sample on every replicate and centres the band on the units that
       survive. Each successor's pool is restricted up front to dates it was observed on; a
       successor with no such date is dropped once. `fixed_support_attrition` reports the cost.
    5. WINDOW-ONLY SUCCESSORS CANNOT BE DRAWN REPEATEDLY without a fetch per draw, so they are
       excluded and the exclusion is REPORTED rather than folded into a coverage number that
       sounds complete. Their share is stated in the output.
    """
    cols_empty = {"label": "link_permutation_distribution", "succ_dvol_mean": None,
                  "n_draws": 0, "n_pairs": 0, "n_pairs_evaluated": 0, "coverage": 0.0}
    if pairs.empty:
        return cols_empty
    all_pairs = pairs if all_pairs is None else all_pairs
    rng = np.random.default_rng(seed)
    # DATE WEIGHTING. The pool is the cohort's death RECORDS, and those concentrate: 261 deaths
    # occupy 90 calendar dates, so a crowded date enters a successor's candidate pool once per
    # death that fell on it. That is the intended null for "another death in this cohort", but it
    # is NOT a uniform draw over calendar dates, and the two answer different questions. Both are
    # computed; `unique_dates` reports the deduplicated version as a sensitivity, because a paper
    # that spends this much care on date clustering cannot leave the weighting implicit.
    pool = pd.to_datetime(deaths["t_death"]).dt.date.to_numpy()
    if dedupe_dates:
        pool = np.array(sorted(set(pool.tolist())))

    # Per successor: its full history (no SQL), and the dates it must NOT be given.
    # Built from ALL pairs, not just the stratum being tested. When the core null built its
    # exclusion list from core pairs alone, 93 of 459 core successors kept 214 envelope-only
    # linked dates eligible, so a successor could be "permuted" onto a death it is genuinely
    # linked to through a weaker pair. The bias ran toward the observed effect -- conservative
    # for the paper, wrong for the null.
    linked: dict[int, set] = {}
    for _, r in all_pairs.iterrows():
        linked.setdefault(int(r["a_succ"]), set()).add(pd.to_datetime(r["t_A"]).date())

    series: dict[int, pd.DataFrame | None] = {}
    for b in linked:
        cp = dbs._cache_path(int(b))
        series[b] = pd.read_parquet(cp) if cp.exists() else None

    usable = [b for b in linked if series[b] is not None]
    rows = pairs[pairs["a_succ"].astype(int).isin(usable)]
    if rows.empty:
        return cols_empty

    # FIXED SUPPORT. The previous version drew from the unrestricted unrelated-date pool and
    # then discarded the successor if the draw landed on a dormant window. That is post-draw
    # deletion: the successor and pair composition changed on every replicate, successors whose
    # histories cover more unrelated dates were upweighted, and the result was an available-case
    # statistic on shifting support rather than a placebo comparable to the observed mean. The
    # Eligibility is decided BEFORE the draw, so attrition cannot shape the band. Each successor's pool is restricted to unrelated
    # dates on which it was actually observed on at least one side of the window, and a successor
    # with no such date is dropped once, up front, from every replicate. Support is then identical
    # across draws and the band is comparable to the observed estimate.
    eligible: dict[int, list] = {}
    per_succ_dormant: dict[int, tuple[int, int]] = {}     # successor -> (dormant, candidates)
    for b in usable:
        ok, n_d, n_c = [], 0, 0
        for d in pool:
            if d in linked[b]:
                continue
            n_c += 1
            _, _, _, _, npre, npost = _cal_stats(series[b], d, ax)
            if npre == 0 and npost == 0:
                n_d += 1                # dormant: no observation on either side
                continue
            ok.append(d)
        per_succ_dormant[b] = (n_d, n_c)
        if ok:
            eligible[b] = ok
    support = [b for b in usable if b in eligible]
    rows_fixed = rows[rows["a_succ"].astype(int).isin(support)]
    if rows_fixed.empty:
        return cols_empty

    # Dormancy, measured three ways, because the denominator decides what the number means.
    # The previous field averaged over every usable ENVELOPE successor regardless of the stratum
    # under test, which is why the core and envelope blocks reported the same 0.2208.
    def _share(succ_ids, weights=None):
        d_tot = c_tot = 0
        for b in succ_ids:
            d, c = per_succ_dormant.get(int(b), (0, 0))
            w = 1 if weights is None else weights.get(int(b), 0)
            d_tot += d * w; c_tot += c * w
        return round(d_tot / c_tot, 4) if c_tot else None

    strat_succ = rows["a_succ"].astype(int).unique().tolist()
    row_w = rows["a_succ"].astype(int).value_counts().to_dict()

    draws = []
    for _ in range(n_draws):
        per_succ = {}
        for b in support:
            el = eligible[b]
            t = el[int(rng.integers(len(el)))]
            vpre, vpost, _, _, _, _ = _cal_stats(series[b], t, ax)
            per_succ[b] = vpost - vpre
        vals = [per_succ[int(r)] for r in rows_fixed["a_succ"].astype(int)]
        draws.append(float(np.mean(vals)))
    if not draws:
        return cols_empty
    arr = np.array(draws, float)
    return {"label": ("link_permutation_distribution_unique_dates" if dedupe_dates
                      else "link_permutation_distribution"),
            "date_pool": ("unique calendar dates" if dedupe_dates
                          else "death records, so crowded dates repeat"),
            "n_pool": int(len(pool)),
            "succ_dvol_mean": float(arr.mean()),
            "ci": [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))],
            "n_draws": int(len(arr)),
            "n_pairs": int(len(pairs)),
            "n_pairs_evaluated": int(len(rows_fixed)),
            "coverage": round(len(rows_fixed) / max(1, len(pairs)), 4),
            # Stratum-specific. `linked` and `usable` are built from all_pairs so that a
            # successor's excluded dates cover every link it has; reporting their sizes described
            # the envelope even when the core was being tested (752/622 for a 459-successor core).
            "n_successors": int(pairs["a_succ"].nunique()),
            "n_successors_evaluated": int(rows_fixed["a_succ"].nunique()),
            "n_successors_pool": len(linked),
            # successor-unweighted, over the successors of THIS stratum
            "dormant_share_of_candidate_dates": _share(strat_succ),
            # pair-weighted, matching how the point estimate weights successors: the figure the
            # manuscript quotes
            "dormant_share_pair_weighted": _share(strat_succ, row_w),
            # attrition: pair rows dropped because a successor had NO eligible active date at all
            "fixed_support_attrition": round(1 - (len(rows_fixed) / max(1, len(rows))), 4),
            "n_pairs_fixed_support": int(len(rows_fixed)),
            "n_successors_fixed_support": int(rows_fixed["a_succ"].nunique()),
            "eligibility_rule": ("fixed support: each successor's unrelated-date pool is "
                                 "restricted BEFORE drawing to dates on which it was "
                                 "observed on at least one side; support is identical "
                                 "across replicates"),
            "excluded_reason": ("successors without a full-history cache are excluded: a repeated "
                                "permutation would need one bounded fetch per successor per draw"),
            "linked_dates_excluded_per_successor": True}


def _null_conn():
    """Lazy connection for the permutation null's windowed fallbacks."""
    try:
        return db.get_connection()
    except Exception:
        return None


def stage_inference(pairs: pd.DataFrame, deaths: pd.DataFrame, ax: dict, tag: str = "") -> dict:
    core = pairs[pairs["in_core"] == 1]
    env = pairs
    res = {
        "bridge": "ip" if tag == "_ip" else "honeypot",
        "n_deaths": int(len(deaths)),
        "n_deaths_with_candidate": int(pairs["a_death"].nunique()),
        "n_candidate_pairs": int(len(pairs)),
        "transfer_core": _transfer(core, "precision_core"),
        "transfer_envelope": _transfer(env, "recall_envelope"),
        "null_link_perm_core": _link_permutation_null(core, deaths, ax, conn=_null_conn(), all_pairs=env),
        "null_link_perm_core_unique_dates": _link_permutation_null(
            core, deaths, ax, conn=_null_conn(), all_pairs=env, dedupe_dates=True),
        "null_link_perm_envelope": _link_permutation_null(env, deaths, ax, conn=_null_conn(), all_pairs=env),
        "null_link_perm_envelope_unique_dates": _link_permutation_null(
            env, deaths, ax, conn=_null_conn(), all_pairs=env, dedupe_dates=True),
        "transfer_rate_core": round(core["a_death"].nunique() / max(1, len(deaths)), 4),
        "transfer_rate_envelope": round(env["a_death"].nunique() / max(1, len(deaths)), 4),
    }
    # A null that could not be drawn is not a null of zero. `_link_permutation_null` returns an
    # empty block when it has no pairs, which is a legitimate result, and the SAME empty block
    # when its per-date activity source is unavailable, which is not. Writing the second as if it
    # were the first publishes `n_draws: 0` beside a real point estimate, and the file overwrites
    # the artifact a manuscript cites. Distinguish them by the pairs the estimate itself used.
    _hollow = [k for k, v in res.items()
               if k.startswith("null_link_perm")
               and isinstance(v, dict) and v.get("n_draws") == 0
               and len(core if "core" in k else env) > 0]
    if _hollow:
        raise SystemExit(
            f"refusing to write {config.OUT_DIR.name}/DB_rotation_summary{SUFFIX}{tag}.json: "
            f"{', '.join(_hollow)} produced 0 draws over a non-empty pair set, which means the "
            f"permutation null's per-date activity source was unavailable rather than that there "
            f"was nothing to draw. Publishing it would put a hollow null beside a real point "
            f"estimate and overwrite the cited artifact.")

    _plot_conservation(env, core, tag)
    (config.OUT_DIR / f"DB_rotation_summary{SUFFIX}{tag}.json").write_text(json.dumps(res, indent=2, default=str))
    _write_summary(res, tag)
    return res


def _plot_conservation(env: pd.DataFrame, core: pd.DataFrame, tag: str = "") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    ax = axes[0]
    if len(env):
        ax.scatter(-env["dA_v"], env["dB_v"], s=10, alpha=0.25, color="gray", label="recall envelope")
    if len(core):
        ax.scatter(-core["dA_v"], core["dB_v"], s=18, alpha=0.7, color="tab:red", label="precision core")
    ax.axhline(0, color="k", lw=0.6); ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("dying agent's volume LOSS  (-dA_v)")
    ax.set_ylabel("successor's volume change  (dB_v)")
    ax.set_title("Volume transfer around the block"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax2 = axes[1]
    if len(env):
        ax2.scatter(env["bA_pre"], env["bB_post"], s=10, alpha=0.25, color="gray")
    if len(core):
        ax2.scatter(core["bA_pre"], core["bB_post"], s=18, alpha=0.7, color="tab:red")
    ax2.set_xlabel("b_A_pre (dying agent botness)"); ax2.set_ylabel("b_B_post (successor botness)")
    ax2.set_title("Botness level at handoff"); ax2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / f"DB_rotation_conservation{SUFFIX}{tag}.png", dpi=150); plt.close(fig)


def _write_summary(res: dict, tag: str = "") -> None:
    c, e = res["transfer_core"], res["transfer_envelope"]
    nc, ne = res["null_link_perm_core"], res["null_link_perm_envelope"]
    bridge = res.get("bridge", "honeypot")
    core_desc = "rare shared IP (>=50% overlap, >=3 IPs)" if bridge == "ip" else "rare shared decoy"

    def fmt(t):
        if not t.get("n_pairs"):
            return "no pairs"
        return (f"successor dvol {t['succ_dvol_mean']:+.3f} CI {[round(x,3) for x in t['succ_dvol_ci']]}, "
                f"dbot {t['succ_dbot_mean']:+.3f} CI {[round(x,3) for x in t['succ_dbot_ci']]}, "
                f"placebo dvol {t['succ_dvol_placebo']:+.3f} (n={t['n_pairs']} pairs / {t['n_deaths']} deaths)")

    def fmtnull(t):
        if t.get("succ_dvol_mean") is None:
            return "n/a"
        ci = t.get("ci")
        band = f", 2.5-97.5% [{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else ""
        return (f"successor dvol around dates it is NOT linked to = {t['succ_dvol_mean']:+.3f}{band} "
                f"over {t.get('n_draws', 1)} draws, on {t.get('n_pairs_evaluated', t.get('n_eval'))} "
                f"of {t.get('n_pairs', '?')} pairs "
                f"({100 * t.get('coverage', 0):.0f}%; successors without a full-history cache are "
                f"excluded)")
    if bridge == "ip":
        bridge_line = ("Bridge = shared non-proxy IPs (overlap-weighted); successors may be ANY agent "
                       "incl. fresh / non-honeypot identities -> this CLOSES the channel the honeypot "
                       "bridge missed. UA+IP co-rotation still escapes -> residual lower bound.")
    else:
        bridge_line = ("Bridge = shared honeypot decoy URLs (rarity-weighted); successors are themselves "
                       "confirmed bots (intra-fleet rotation); IP bridge infeasible on this schema -> LOWER BOUND.")
    L = [f"# Identity rotation / respawn — Finding 6 ({bridge} bridge)\n",
         f"Deaths (persistent post-block collapses among honeypot bots): **{res['n_deaths']}**; "
         f"with >=1 bridged successor: {res['n_deaths_with_candidate']}; candidate pairs: {res['n_candidate_pairs']}.\n",
         bridge_line + "\n",
         "## Transfer rate (deaths with a bridged successor)\n",
         f"- precision core ({core_desc}): **{res['transfer_rate_core']}**",
         f"- recall envelope (any shared): {res['transfer_rate_envelope']}\n",
         "## Does the successor ramp up as the bot dies? (event-study around t_A)\n",
         f"- precision core: {fmt(c)}",
         f"- recall envelope: {fmt(e)}\n",
         "## Falsification\n",
         f"- link-permutation null (core): {fmtnull(nc)}",
         f"- link-permutation null (envelope): {fmtnull(ne)}",
         "  (real ramp must EXCEED the successor's ramp around a random unrelated death; "
         "if equal, the ramp is generic successor growth, not a handoff)",
         "- per-pair placebo: `placebo dvol` above is the successor's volume change around a date "
         f"{PLACEBO_OFFSET} d before the block "
         f"(core ramp-minus-placebo = {c.get('ramp_minus_placebo', float('nan')):+.3f}, "
         f"envelope = {e.get('ramp_minus_placebo', float('nan')):+.3f}).\n",
         "_Planted-handoff positive control: run with `--selftest` (separate pass)._\n"]
    (config.OUT_DIR / f"DB_rotation_SUMMARY{SUFFIX}{tag}.md").write_text("\n".join(L), encoding="utf-8")


# ===================== positive control (planted handoff) =====================

def selftest_planted(ax: dict, n: int = 300, seed: int = RANDOM_STATE) -> dict:
    """Split real confirmed-bot series at their midpoint: first half = a dying A,
    second half = a born successor B (same operator => a genuine handoff). The
    statistic must show dA_v<0 (collapse), dB_v>0 (ramp), bB_post~bA_pre
    (botness conserved), and a NULL temporal placebo. Validates sensitivity."""
    roster = pd.read_csv(_C_ROSTER[0])
    bots = roster[roster["is_bot"] == True]["a_id"].astype(int).to_numpy()   # noqa: E712
    rng = np.random.default_rng(seed)
    rows = []
    for a_id in rng.permutation(bots):
        if len(rows) >= n:
            break
        cp = dbs._cache_path(int(a_id))
        if not cp.exists():
            continue
        df = pd.read_parquet(cp)
        dates = pd.to_datetime(df["date"]).dt.date
        span = (dates.max() - dates.min()).days
        if span < 4 * WC:
            continue
        t_split = dates.min() + _dt.timedelta(days=span // 2)
        A, B = df[dates < t_split], df[dates >= t_split]
        if len(A) < MIN_DAYS or len(B) < MIN_DAYS:
            continue
        vA_pre, vA_post, bA_pre, bA_post, _, _ = _cal_stats(A, t_split, ax)
        vB_pre, vB_post, bB_pre, bB_post, _, npost = _cal_stats(B, t_split, ax)
        if npost == 0:
            continue
        t_P = t_split - _dt.timedelta(days=PLACEBO_OFFSET)
        vBp_pre, vBp_post, _, _, _, _ = _cal_stats(B, t_P, ax)
        rows.append({"dA_v": vA_post - vA_pre, "dB_v": vB_post - vB_pre,
                     "bA_pre": bA_pre, "bB_post": bB_post,
                     "dB_v_placebo": vBp_post - vBp_pre})
    p = pd.DataFrame(rows)
    m = p[["bA_pre", "bB_post"]].dropna()
    corr = float(np.corrcoef(m["bA_pre"], m["bB_post"])[0, 1]) if len(m) > 2 else float("nan")
    res = {"n_planted": int(len(p)),
           "dA_v_mean_want_neg": float(p["dA_v"].mean()),
           "dB_v_mean_want_pos": float(p["dB_v"].mean()),
           "dB_v_placebo_mean_want_0": float(p["dB_v_placebo"].mean()),
           "b_transfer_bias_want_0": float((p["bB_post"] - p["bA_pre"]).mean()),
           "pearson_bApre_bBpost_want_1": corr,
           "frac_ramp_exceeds_placebo": float((p["dB_v"] > p["dB_v_placebo"]).mean())}
    (config.OUT_DIR / f"DB_rotation_selftest{SUFFIX}.json").write_text(json.dumps(res, indent=2, default=str))
    print("[selftest] planted-handoff positive control:")
    print(json.dumps(res, indent=2, default=str))
    return res


# ===================== placebo-death pass (organic, non-block collapses) =====================

def build_placebo_deaths(seed: int = RANDOM_STATE, n_match: int | None = None) -> pd.DataFrame:
    """Deaths defined by a persistent volume collapse NOT coincident with a block
    spike (organic churn). If respawn is friction-driven, the bridged-successor
    ramp here should be WEAKER than for the real (block-driven) deaths."""
    # The unthinned rule, for the same reason as the main detector (see db_rotation_probe):
    # this uses block days to EXCLUDE candidates, so a thinned list would under-exclude and
    # let block-adjacent days into a placebo defined by the absence of blocks.
    from db_rotation_probe import _rotation_events as _block_events
    roster = pd.read_csv(_C_ROSTER[0])
    bots = roster[roster["is_bot"] == True]["a_id"].astype(int)   # noqa: E712
    rows = []
    for a_id in bots:
        cp = dbs._cache_path(int(a_id))
        if not cp.exists():
            continue
        df = pd.read_parquet(cp)
        if len(df) < 25:
            continue
        hits = df["hits"].to_numpy(float)
        cblock = df["cblock"].to_numpy(float)
        blk = set(_block_events(df).tolist())
        for i in range(5, len(df) - 1):
            if i in blk or cblock[max(0, i - 3):i + 4].sum() > 0:    # require NO block nearby
                continue
            pre = hits[i - 5:i].mean()
            if pre < 5:
                continue
            tail = hits[i + 1:].mean()
            if tail < 0.5 * pre and hits[i + 1:i + 21].max() < pre:
                rows.append({"a_id": int(a_id),
                             "t_death": pd.to_datetime(df["date"].iloc[i]).date(),
                             "pre_vol": float(pre), "tail_vol": float(tail)})
                break
    placebo = pd.DataFrame(rows)
    if n_match and len(placebo) > n_match:
        placebo = placebo.sample(n_match, random_state=seed)
    placebo.to_csv(TAB / f"DB_rotation_placebo_deaths{SUFFIX}.csv", index=False)
    print(f"[placebo] {len(placebo)} organic (non-block) deaths")
    return placebo


def run_placebo(roster_csv=None, suffix: str = "_v5") -> dict:
    roster_csv = AXIS_ROSTER_CSV if roster_csv is None else roster_csv
    if suffix:
        _set_suffix(suffix)
    _C_ROSTER[0] = roster_csv
    config.ensure_dirs()
    roster = pd.read_csv(roster_csv)
    ax = build_botness_axis(roster, cache=True)
    real = pd.read_csv(DEATHS_CSV)
    placebo = build_placebo_deaths(n_match=len(real))
    if placebo.empty:
        print("[placebo] none found"); return {}
    placebo["t_death"] = pd.to_datetime(placebo["t_death"])
    g = load_honeypot_graph()
    pairs = stage_pairs(placebo, ax, g, out_csv=TAB / f"DB_rotation_placebo_pairs{SUFFIX}.csv")
    t = _transfer(pairs[pairs["in_core"] == 1], "placebo_core")
    t["transfer_rate_core"] = round(pairs[pairs.in_core == 1]["a_death"].nunique() / max(1, len(placebo)), 4)
    (config.OUT_DIR / f"DB_rotation_placebo{SUFFIX}.json").write_text(json.dumps(t, indent=2, default=str))
    print("[placebo] core transfer:", json.dumps(t, indent=2, default=str))
    return t


# ===================== main =====================

def run(roster_csv=None, suffix: str = "_v5") -> dict:
    roster_csv = AXIS_ROSTER_CSV if roster_csv is None else roster_csv
    if suffix:
        _set_suffix(suffix)
    _C_ROSTER[0] = roster_csv
    config.ensure_dirs()
    roster = pd.read_csv(roster_csv)
    print("[axis] building botness axis...")
    ax = build_botness_axis(roster, cache=True)
    print(f"  bot centroid b = {ax['bot_centroid_b']:.3f}")
    deaths = pd.read_csv(DEATHS_CSV, parse_dates=["t_death"])
    print(f"[deaths] {len(deaths)} loaded")
    if len(deaths) < 20:
        print(f"[GATE] only {len(deaths)} deaths — stopping.")
        return {"n_deaths": int(len(deaths)), "gated": True}
    partial = PAIRS_CSV.with_suffix(".partial.csv")
    if PAIRS_CSV.exists() and not partial.exists():
        pairs = pd.read_csv(PAIRS_CSV, parse_dates=["t_A"])
        print(f"[pairs] reused complete {len(pairs)} from {PAIRS_CSV.name}")
    else:
        g = load_honeypot_graph()                 # only needed when (re)computing pairs
        pairs = stage_pairs(deaths, ax, g)
    if pairs.empty:
        print("[done] no candidate pairs — rotation undetectable via honeypot bridge.")
        return {"n_deaths": int(len(deaths)), "n_pairs": 0}
    res = stage_inference(pairs, deaths, ax)
    print(json.dumps(res, indent=2, default=str))
    return res


def run_ipbridge(roster_csv=None, suffix: str = "_v5") -> dict:
    roster_csv = AXIS_ROSTER_CSV if roster_csv is None else roster_csv
    """IP-bridge variant: tests handoffs into ANY agent (incl. fresh/non-honeypot
    identities). Requires idx_ip_agent_time on the ip table. Writes _ip outputs."""
    if suffix:
        _set_suffix(suffix)
    _C_ROSTER[0] = roster_csv
    config.ensure_dirs()
    roster = pd.read_csv(roster_csv)
    print("[axis] building botness axis...")
    ax = build_botness_axis(roster, cache=True)
    print(f"  bot centroid b = {ax['bot_centroid_b']:.3f}")
    deaths = pd.read_csv(DEATHS_CSV, parse_dates=["t_death"])
    print(f"[deaths] {len(deaths)} loaded")
    partial = IP_PAIRS_CSV.with_suffix(".partial.csv")
    if IP_PAIRS_CSV.exists() and not partial.exists():
        pairs = pd.read_csv(IP_PAIRS_CSV, parse_dates=["t_A"])
        print(f"[ip-pairs] reused complete {len(pairs)} from {IP_PAIRS_CSV.name}")
    else:
        pairs = stage_pairs_ip(deaths, ax)
    if pairs.empty:
        print("[done] no IP-bridge candidate pairs.")
        return {"n_deaths": int(len(deaths)), "n_pairs": 0}
    res = stage_inference(pairs, deaths, ax, tag="_ip")
    print(json.dumps(res, indent=2, default=str))
    return res


if __name__ == "__main__":
    import sys
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        import argparse

        # The original CLI used bare "--selftest" / "--placebo" / "--ipbridge" as a
        # mode word. argparse cannot take a "--"-prefixed token as a positional, so
        # the mode is pulled out of argv first and the rest parsed normally.
        argv = sys.argv[1:]
        mode = ""
        for m in ("--selftest", "--placebo", "--ipbridge"):
            if m in argv:
                mode = m.lstrip("-")
                argv.remove(m)

        ap = argparse.ArgumentParser()
        ap.add_argument("--mode", default=mode,
                        choices=["", "selftest", "placebo", "ipbridge"])
        # The plain documented command must reproduce the CURRENT Finding 6 cohort. The two
        # selectors are independent: --roster trains the botness axis and needs the label
        # columns, --suffix chooses the death cohort. An alternative arm stays available
        # but must be named; the default is the current one.
        ap.add_argument("--roster", default=str(AXIS_ROSTER_CSV),
                        help="botness-axis training roster; needs is_bot/is_control")
        ap.add_argument("--suffix", default="_v5",
                        help="cohort suffix (default: _v5, the current death list)")
        a = ap.parse_args(argv)
        if a.suffix:
            _set_suffix(a.suffix)
        _C_ROSTER[0] = a.roster
        try:
            if a.mode == "selftest":
                ax = build_botness_axis(pd.read_csv(a.roster), cache=True)
                selftest_planted(ax)
            elif a.mode == "placebo":
                run_placebo(a.roster, a.suffix)
            elif a.mode == "ipbridge":
                run_ipbridge(a.roster, a.suffix)
            else:
                run(a.roster, a.suffix)
        finally:
            # Always, including on an aborted run: a partial result whose coverage is unknown
            # is worse than a partial result that says what it missed.
            dump_unfetched("_" + a.mode if a.mode else "")
