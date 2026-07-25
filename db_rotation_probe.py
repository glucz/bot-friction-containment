"""
Step 0 feasibility probe for the identity-rotation analysis (PLAN_identity_rotation.md).

Two cheap, independent checks — neither commits to the full pipeline:

  A. DB reachability + bridge-table shape/size (ip, url2agent_honeypot, hackip),
     using information_schema row ESTIMATES (instant) and DESCRIBE. No scans, no
     PII: we never SELECT i_name here.
  B. Death-set count from the LOCAL CACHE only (no DB): honeypot-confirmed bots
     with a *persistent* post-block volume collapse (the candidate "deaths" that
     the per-agent design currently mislabels as retreat). Writes the death list
     to outputs/tables/DB_rotation_deaths.csv for the next stage.

The death count gates the build: too small -> report the null cheaply; large
enough -> we know the per-death runtime budget for the bridge queries.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
import db_source as dbs
from db_battery import _block_events
from roster import ROSTER_CSV

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
    events = _block_events(df)
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


def probe_deaths() -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("B. Death-set count (cache only, no DB)")
    print("=" * 70)
    roster = pd.read_csv(ROSTER_CSV)
    bots = roster[roster["is_bot"] == True]            # noqa: E712
    rows, n_cached, n_with_events = [], 0, 0
    for a_id in bots["a_id"].astype(int):
        cp = dbs._cache_path(a_id)
        if not cp.exists():
            continue
        n_cached += 1
        df = pd.read_parquet(cp)
        if len(df) < MIN_DAYS:
            continue
        if len(_block_events(df)):
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
        out = config.TAB_DIR / "DB_rotation_deaths.csv"
        deaths.sort_values("t_death").to_csv(out, index=False)
        print(f"  death list -> {out}")
        print("\n  death-date span:", deaths["t_death"].min(), "->", deaths["t_death"].max())
        print("  drop_ratio quartiles:",
              np.round(deaths["drop_ratio"].quantile([.25, .5, .75]).to_numpy(), 3))
    return deaths


if __name__ == "__main__":
    probe_db()
    probe_deaths()
