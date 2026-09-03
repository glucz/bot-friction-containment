"""
Step 0 feasibility probe for the identity-rotation analysis (PLAN_identity_rotation.md).

Two cheap, independent checks — neither commits to the full pipeline:

  A. DB reachability + bridge-table shape/size (ip, url2agent_honeypot, hackip),
     using information_schema row ESTIMATES (instant) and DESCRIBE. No scans, no
     PII: we never SELECT i_name here.
  B. Death-set count from the LOCAL CACHE only (no DB): agents on the arm of record
     with a *persistent* post-block volume collapse (the candidate "deaths" that
     the per-agent design currently mislabels as retreat). Writes the death list
     to outputs/tables/DB_rotation_deaths_v5.csv for the next stage.

The death count gates the build: too small -> report the null cheaply; large
enough -> we know the per-death runtime budget for the bridge queries.

EVENT CANDIDATES ARE NOT THINNED HERE, and that is a deliberate design choice
rather than an oversight. `db_battery._block_events`
thins candidates to a minimum gap because the difference-in-differences estimators
of Findings 2, 3 and 5 difference a pre-window against a post-window, and two
events closer than that gap would share observations between windows. This
detector does none of that. It searches an agent's own series for the LAST
block-pressure day after which volume collapses and stays collapsed to the end of
observation, and it returns at most ONE death per agent, so no observation is ever
reused across events and there is nothing for a spacing rule to protect.

Thinning actively damages it: removing candidate days can delete the very day the
collapse begins, after which the detector either finds an earlier, wrong onset or
finds nothing. The detector returns at most one persistent death per agent, so there
is nothing for a spacing rule to protect. Multiple friction spikes within a few days
are the normal signature of one sustained enforcement action against one operator,
and collapsing them to a single candidate is exactly what a per-agent changepoint
search must not do.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import config
import db_source as dbs


def _rotation_events(df) -> "np.ndarray":
    """Block-pressure days with NO spacing thinning -- see the module docstring.

    Deliberately reimplemented rather than delegated: `_block_events` reads the module-level
    EVENT_SPACING, which is env-overridable, so delegating would leave this detector's cohort
    silently dependent on a variable set for an unrelated estimator.
    """
    import numpy as _np
    blk = df["cblock"].to_numpy(dtype=float)
    r429 = df["c429"].to_numpy(dtype=float)
    idx = set()
    if len(blk) >= 3 and blk.std() > 0:
        thr = blk.mean() + 1.5 * blk.std()
        idx.update(_np.flatnonzero(blk > max(thr, 0)).tolist())
    idx.update(_np.flatnonzero(r429 > 0).tolist())
    return _np.array(sorted(idx), dtype=int)
from _analysis_path import analysis_dir
from arm_ids import arm_of_record_ids

ARM_IDS_CSV = config.OUT_DIR / "DB_arm_of_record_ids.csv"
N_ARM = 7812



W = 5             # pre-event window (active days)
DEATH_W = 20      # min look-ahead to call a collapse "persistent"
MIN_DAYS = 25     # need room for a pre-window + a look-ahead
VOL_FLOOR = 5.0   # mean pre-event daily hits must exceed this (ignore trivial agents)
DROP = 0.5        # post/pre volume ratio below this = collapse


def probe_db() -> None:
    print("=" * 70)
    print("A. DB reachability + bridge-table shape")
    print("=" * 70)
    import db
    try:
        print("ping:", db.ping())
    except Exception as e:
        print("DB UNREACHABLE:", str(e)[:200])
        return
    # row ESTIMATES (instant, no scan)
    try:
        est = db.read_sql(
            "SELECT table_name, table_rows FROM information_schema.tables "
            "WHERE table_schema=DATABASE() AND table_name IN "
            "('ip','url2agent_honeypot','hackip','agent','hits')"
        )
        print("\nrow estimates (information_schema):")
        print(est.to_string(index=False))
    except Exception as e:
        print("info_schema failed:", str(e)[:160])
    for t in ("ip", "url2agent_honeypot"):
        try:
            d = db.read_sql(f"DESCRIBE {t}")
            print(f"\nDESCRIBE {t}:")
            print(d[["Field", "Type", "Key"]].to_string(index=False))
        except Exception as e:
            print(f"DESCRIBE {t} failed:", str(e)[:160])
    # one cheap, bounded bridge self-join sanity check on a SMALL known bot,
    # returning only a COUNT (never i_name). Pick the smallest honeypot bot in cache.
    print("\n(bridge self-join cost test deferred to stage 1 — needs a real death a_id)")


def _persistent_death(df: pd.DataFrame):
    """Return (t_A, pre_vol, tail_vol) for the LAST block event after which volume
    collapses and stays collapsed to the end of observation; else None."""
    if len(df) < MIN_DAYS or "hits" not in df:
        return None
    hits = df["hits"].to_numpy(float)
    events = _rotation_events(df)
    best = None
    for i in events:
        if i < W or i + 1 >= len(df):
            continue
        pre = hits[i - W:i].mean()
        if pre < VOL_FLOOR:
            continue
        tail = hits[i + 1:].mean()                 # all activity after the block
        look = hits[i + 1:i + 1 + DEATH_W]
        if len(look) < 1:
            continue
        # persistent: the whole remaining tail stays below half of pre-volume
        if tail < DROP * pre and look.max() < pre:
            best = (pd.to_datetime(df["date"].iloc[i]).date(), float(pre), float(tail))
    return best


def probe_deaths(roster_csv=None, suffix: str = "_v5") -> pd.DataFrame:
    """Death cohort on the arm of record.

    `roster_csv`/`suffix` let the cohort be rebuilt on an alternative arm definition without
    overwriting the current run's death list. The DEFAULT is the released arm-of-record ID
    file, the 7,812 agents the manuscripts report.
    """
    print(chr(10) + "=" * 70)
    print("B. Death-set count (cache only, no DB)")
    print("=" * 70)
    src = Path(ARM_IDS_CSV if roster_csv is None else roster_csv)
    if not src.exists():
        raise SystemExit(f"{src} is missing; it is the released arm-of-record authority")

    if src.name == ARM_IDS_CSV.name:
        # THE canonical arm goes through the shared reader. Converting with int(v) here would be a
        # second population path with none of its contract: `True` becomes agent 1 before any
        # count check, and a duplicate or a fractional id passes silently.
        _ids = arm_of_record_ids(src)
    else:
        # An explicitly named alternative roster is validated separately; the canonical contract
        # is not weakened to accommodate it.
        roster = pd.read_csv(src)
        if "is_bot" in roster.columns:
            sel = roster[roster["is_bot"] == True]        # noqa: E712
        elif "role" in roster.columns:
            sel = roster[roster["role"] == "abusive_share"]
        else:
            sel = roster
        col = sel["a_id"]
        if col.dtype == bool or any(isinstance(v, bool) for v in col.head(64)):
            raise SystemExit(f"{src.name}: a_id is Boolean, not an agent identifier")
        if col.isna().any() or not (col % 1 == 0).all():
            raise SystemExit(f"{src.name}: a_id has missing or non-integer values")
        _ids = sorted({int(v) for v in col})
    print(f"   population {src.name}: {len(_ids):,} agents")
    bots = pd.DataFrame({"a_id": _ids})

    # Admit the whole declared arm before counting deaths. A death set assembled from whatever is
    # on disk is not the roster's death set, and the uncached agents are not a random subset: the
    # extractor fails hardest on the highest-volume agents, which are the ones most able to
    # rotate. `analysis/input_contract.preflight` refuses to run rather than shrink.
    import sys as _sys
    _sys.path.insert(0, str(analysis_dir()))
    from input_contract import load_admitted, preflight  # noqa: E402
    _cache_dir = dbs._cache_path(1).parent
    _admitted = preflight(bots["a_id"].astype(int).tolist(), _cache_dir,
                          f"rotation_deaths{suffix or '_v5'}",
                          out_dir=config.OUT_DIR, analysis_min_rows=MIN_DAYS)

    rows, n_cached, n_with_events = [], 0, 0
    for a_id in bots["a_id"].astype(int):
        # Read through the admitted manifest, not around it. Preflighting and then opening the
        # path again admits one set of bytes and analyses another.
        df = load_admitted(a_id, _cache_dir, _admitted["per_id"])
        n_cached += 1
        if len(df) < MIN_DAYS:
            continue
        if len(_rotation_events(df)):
            n_with_events += 1
        d = _persistent_death(df)
        if d is not None:
            rows.append({"a_id": a_id, "t_death": d[0], "pre_vol": d[1], "tail_vol": d[2],
                         "drop_ratio": d[2] / d[1]})
    deaths = pd.DataFrame(rows)
    print(f"honeypot bots in roster:     {len(bots)}")
    print(f"  ...present in cache:       {n_cached}")
    print(f"  ...with >=1 block event:   {n_with_events}")
    print(f"  ...persistent post-block death (DROP<{DROP}): {len(deaths)}")
    if not deaths.empty:
        config.ensure_dirs()
        out = config.TAB_DIR / f"DB_rotation_deaths{suffix}.csv"
        # Total order, not just by time: t_death has ties, and a tie broken by whatever
        # order the upstream grouping happened to produce makes the released artifact
        # differ byte-for-byte between runs of the same cohort, which defeats the
        # byte-comparison an independent check would use.
        deaths.sort_values(["t_death", "a_id"]).to_csv(out, index=False)
        print(f"  death list -> {out}")
        print("\n  death-date span:", deaths["t_death"].min(), "->", deaths["t_death"].max())
        print("  drop_ratio quartiles:",
              np.round(deaths["drop_ratio"].quantile([.25, .5, .75]).to_numpy(), 3))
    return deaths


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    # The plain documented command must build Finding 6's cohort on the arm of record. An
    # alternative arm stays available, but it has to be named, and it writes under its own suffix
    # so it cannot overwrite the current death list.
    ap.add_argument("--roster", default=str(ARM_IDS_CSV),
                    help="population source (default: the arm-of-record ID file)")
    ap.add_argument("--suffix", default="_v5",
                    help="output suffix for the death list (default: _v5)")
    ap.add_argument("--skip-db-probe", action="store_true")
    a = ap.parse_args()
    if not a.skip_db_probe:
        probe_db()
    probe_deaths(a.roster, a.suffix)
