"""Block-event branch weights on the arm of record (offline, cache-only).

Section 7.1 of the MethodsX companion states what share of block-event days arrive through the
429 branch rather than the 401/403 branch. That share is a property of a named population, so it
is computed here on the arm of record and on nothing else.

The complete arm-of-record ID file is admitted before counting. Any missing, unreadable or
changed cache file aborts the run. The artifact contains only the arm of record.

No database access. The population is read from the released ID authority
`outputs/DB_arm_of_record_ids.csv` and every number comes from `cache/agents/*.parquet`, so this
runs during a database blackout and cannot rewrite a population file as a side effect.

Usage: python db_event_branches_v2.py
Writes: outputs/DB_event_branch_v2.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _analysis_path import on_path  # noqa: E402

on_path()

import config  # noqa: E402
from arm_ids import N_ARM, arm_of_record_ids  # noqa: E402
from input_contract import admit, load_admitted, merge_per_id  # noqa: E402

MIN_DAYS = 20
COLS = ("hits", "cblock", "c429", "robots")
CACHE = HERE / "cache" / "agents"
ARM_IDS_CSV = config.OUT_DIR / "DB_arm_of_record_ids.csv"
N_ARM = 7812
OUT_JSON = config.OUT_DIR / "DB_event_branch_v2.json"


def _tally(a_id: int, per_id: dict) -> tuple:
    """Per-agent branch tally. A block-event day is a spike in 401/403; the 429 branch is
    counted on days with 429s and no spike, and days carrying both are counted separately so the
    denominator is days rather than branch hits."""
    d = load_admitted(a_id, CACHE, per_id)
    b = d["cblock"].to_numpy(float)
    q = d["c429"].to_numpy(float)
    spike = b > (b.mean() + 1.5 * b.std())
    return (
        bool(float(d["robots"].sum()) > 0),
        bool(len(d) >= MIN_DAYS),
        int((spike & (q <= 0)).sum()),
        int((~spike & (q > 0)).sum()),
        int((spike & (q > 0)).sum()),
        float(d["c429"].sum()),
        float(d["cblock"].sum()),
        float(d["hits"].sum()),
    )


def run() -> dict:
    config.ensure_dirs()
    ids = arm_of_record_ids()

    inputs = admit({"abusive": ids}, CACHE, "event_branch", out_dir=config.OUT_DIR,
                   required_cols=COLS, analysis_min_rows=MIN_DAYS)
    per_id = merge_per_id(inputs)

    n = ever = longd = sp = q_only = both = 0
    c429 = cblk = hits = 0.0
    per_agent_blocks: list[float] = []
    for a_id in ids:
        r = _tally(a_id, per_id)
        per_agent_blocks.append(r[6])
        n += 1
        ever += r[0]
        longd += r[1]
        sp += r[2]
        q_only += r[3]
        both += r[4]
        c429 += r[5]
        cblk += r[6]
        hits += r[7]

    total = sp + q_only + both
    blocked = sum(1 for v in per_agent_blocks if v > 0)
    largest = max(per_agent_blocks) if per_agent_blocks else 0.0
    arm = {
        "label": "the arm of record (7,812), every declared agent admitted",
        "agents_declared": len(ids),
        "agents_admitted": n,
        "ever_robots": ever,
        "ever_robots_pct": round(100.0 * ever / max(n, 1), 2),
        "agents_with_min_days": longd,
        "events_401_403_branch_only": sp,
        "events_429_branch_only": q_only,
        "events_both_same_day": both,
        "events_total": total,
        "pct_events_from_429_branch_only": round(100.0 * q_only / max(total, 1), 2),
        "pct_events_touching_429_branch": round(100.0 * (q_only + both) / max(total, 1), 2),
        "responses_429": c429,
        "responses_401_403": cblk,
        "hits": hits,
        # Reach and concentration of the block channel, on the same admitted population and in the
        # same pass, so the three cannot drift apart between scripts.
        "agents_ever_blocked": blocked,
        "pct_agents_ever_blocked": round(100.0 * blocked / max(n, 1), 2),
        "largest_agent_block_responses": largest,
        "largest_agent_block_share_pct": round(100.0 * largest / cblk, 2) if cblk else None,
    }
    out = {
        "population": "arm_of_record",
        "min_days": MIN_DAYS,
        "definition": ("event days are counted per agent: a 401/403 spike day with no 429, a 429 "
                       "day with no spike, or a day carrying both"),
        "arm_of_record": arm,
        "inputs": inputs,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"  declared {arm['agents_declared']:,}  admitted {arm['agents_admitted']:,}")
    print(f"  events   {arm['events_total']:,}  429-branch only {arm['events_429_branch_only']:,} "
          f"= {arm['pct_events_from_429_branch_only']:.2f}%")
    print(f"  touching the 429 branch {arm['pct_events_touching_429_branch']:.2f}%")
    print(f"  ever blocked {arm['agents_ever_blocked']:,} "
          f"= {arm['pct_agents_ever_blocked']:.2f}%; largest agent carries "
          f"{arm['largest_agent_block_share_pct']:.2f}% of block responses")
    print(f"\nwrote {OUT_JSON}")
    return out


if __name__ == "__main__":
    run()
