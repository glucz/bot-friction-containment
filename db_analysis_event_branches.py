"""Two descriptive checks the manuscript needs for Section 7.

1. EVENT BRANCH ATTRIBUTION. The block-event rule of the paper fires when an
   agent's 401/403 count exceeds its own baseline by >=1.5 SD *or* when it
   receives at least one 429. The two branches have very different thresholds,
   so the manuscript should say how many events each contributes.

2. ROBOTS.TXT PREVALENCE. Cross-check of the reported "6.1% of confirmed bots
   ever access robots.txt" on a random sample, since an earlier non-random slice
   suggested a higher figure.

Usage: python db_analysis_event_branches.py
Writes: outputs/DB_event_branch_and_robots.json
"""
from __future__ import annotations

import json
import random
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

import config
import db_source as dbs
from roster import ROSTER_CSV

MIN_DAYS = 20
SAMPLE_N = 6000
COLS = ["hits", "c200", "c404", "cblock", "c429", "robots"]


def job(a_id: int):
    cp = dbs._cache_path(a_id)
    if not cp.exists():
        return None
    try:
        d = pd.read_parquet(cp, columns=COLS)
    except Exception:
        return None
    ever_robots = float(d["robots"].sum()) > 0
    long_enough = len(d) >= MIN_DAYS

    b = d["cblock"].to_numpy(float)
    q = d["c429"].to_numpy(float)
    spike = b > (b.mean() + 1.5 * b.std())
    return (
        bool(ever_robots),
        bool(long_enough),
        int((spike & (q <= 0)).sum()),   # 401/403 branch only
        int((~spike & (q > 0)).sum()),   # 429 branch only
        int((spike & (q > 0)).sum()),    # both on the same day
        float(d["c429"].sum()),
        float(d["cblock"].sum()),
        float(d["hits"].sum()),
    )


def run() -> dict:
    config.ensure_dirs()
    roster = pd.read_csv(ROSTER_CSV)
    bots = roster.loc[roster["is_bot"] == True, "a_id"].astype(int).tolist()  # noqa: E712
    random.seed(config.RANDOM_STATE)
    random.shuffle(bots)
    sample = bots[:SAMPLE_N]

    n = ever = longd = 0
    sp = q_only = both = 0
    c429 = cblk = hits = 0.0
    with ProcessPoolExecutor(max_workers=8) as ex:
        for r in ex.map(job, sample, chunksize=100):
            if r is None:
                continue
            n += 1
            ever += r[0]
            longd += r[1]
            sp += r[2]
            q_only += r[3]
            both += r[4]
            c429 += r[5]
            cblk += r[6]
            hits += r[7]

    total_events = sp + q_only + both
    out = {
        "sampled_bot_agents": n,
        "ever_robots": ever,
        "ever_robots_pct": round(100.0 * ever / max(n, 1), 2),
        "agents_with_min_days": longd,
        "events_401_403_branch_only": sp,
        "events_429_branch_only": q_only,
        "events_both_same_day": both,
        "events_total": total_events,
        "pct_events_from_429_branch_only": round(100.0 * q_only / max(total_events, 1), 2),
        "responses_429": c429,
        "responses_401_403": cblk,
        "hits": hits,
    }
    json.dump(out, open(config.OUT_DIR / "DB_event_branch_and_robots.json", "w"), indent=2)
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    run()
