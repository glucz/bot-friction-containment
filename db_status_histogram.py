"""True per-status-code histogram from the AGWA database.

The Parquet cache stores only the channels the analyses need, and it merges the
two block codes into one column (`cblock = SUM(h_status IN (401,403))`). So no
local artifact can answer "how many blocks per error code", and the split of the
block channel into 401 (unauthorized) versus 403 (forbidden) has to come from
the database.

The query is per-agent and index-covered (idx h_5 = (h_a_id, h_status)), never a
scan, and returns aggregates only - no IPs, no URLs, no PII. The population is the arm of record, read from `DB_arm_of_record_ids.csv`
(see longtail/POPULATIONS-KEY.md). Draws are random: taking `sorted(ids)[:n]`
selects the lowest agent ids, which are systematically the oldest and largest
accounts rather than a representative draw.

**Sampling warning.** The pooled 403:401 ratio is not stable under
sampling, because a single agent can carry tens of thousands of responses of one
code. Run
`--n 0` for a census and quote the AGENT-LEVEL statistics (median per-agent 403
share, share of agents where 403 is the majority), which the pooled ratio cannot
substitute for.

Note on the request universe: the counts here are the agent's FULL history in the
`hits` table, whereas the `hits` totals in DB_event_branch_v2.json come from the
parquet cache, which drops confirmed collection-downtime days. The two are not
interchangeable and must not be mixed inside one sentence.

Usage: python db_status_histogram.py [--n 400] [--workers 8]
Writes: outputs/DB_status_histogram.json
"""
from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

import config
from arm_ids import arm_of_record_ids
import db

ARM_IDS_CSV = config.OUT_DIR / "DB_arm_of_record_ids.csv"   # 7,812; see arms_v2.arm_of_record
OUT_JSON = config.OUT_DIR / "DB_status_histogram.json"
SQL = "SELECT h_status AS s, COUNT(*) AS n FROM hits WHERE h_a_id=%s GROUP BY h_status"
TIMEOUT_MS = 60_000
BLOCK_CODES = (401, 403, 429)


def _worker(chunk: list[int]):
    """One connection per worker; sum the status histogram over its agents.

    Also returns per-agent block counts. The pooled 403:401 ratio is dominated
    by a handful of very large agents and is NOT stable across random samples
    (see the sampling note in the module docstring), so the agent-level
    distribution has to be reported alongside it.
    """
    local: dict[int, int] = {}
    per = []
    ok = err = 0
    conn = db.get_connection(timeout_ms=TIMEOUT_MS)
    try:
        for a_id in chunk:
            try:
                d = db.read_sql(SQL, (int(a_id),), conn=conn)
            except Exception:
                err += 1
                continue
            ok += 1
            row = {int(r.s): int(r.n) for r in d.itertuples()}
            per.append((int(a_id), row.get(401, 0), row.get(403, 0),
                        row.get(429, 0), int(sum(row.values()))))
            for k, v in row.items():
                local[k] = local.get(k, 0) + v
    finally:
        conn.close()
    return local, ok, err, per


def histogram(ids: list[int], label: str, workers: int, tag: str = "") -> dict:
    total: dict[int, int] = {}
    rows = []
    ok = err = 0
    chunks = [ids[i::workers] for i in range(workers)]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for local, o, e, per in ex.map(_worker, chunks):
            ok += o
            err += e
            rows.extend(per)
            for k, v in local.items():
                total[k] = total.get(k, 0) + v
    hits = sum(total.values())
    blocks = {str(c): total.get(c, 0) for c in BLOCK_CODES}
    blocks["401+403"] = total.get(401, 0) + total.get(403, 0)
    return {
        "label": label,
        "agents_queried": len(ids),
        "agents_ok": ok,
        "agents_failed": err,
        "hits": hits,
        "by_status": {str(k): v for k, v in sorted(total.items(), key=lambda kv: -kv[1])},
        "blocks": blocks,
        "block_share_of_hits_pct": round(100.0 * blocks["401+403"] / max(hits, 1), 4),
        "ratio_403_over_401_pooled": round(total.get(403, 0) / max(total.get(401, 0), 1), 3),
        **_agent_level(rows, tag),
    }


_HELD: dict = {}


def _agent_level(rows, tag: str) -> dict:
    """Agent-level view of the block channel, robust to the giant agents."""
    d = pd.DataFrame(rows, columns=["a_id", "c401", "c403", "c429", "hits"])
    # The frame is HANDED BACK, not written here. Writing it at this point and refusing the run
    # afterwards would leave an incomplete census on disk under a name that looks current.
    # `run()` writes it only once the run is accepted, under a name that says which it is.
    _HELD["per_agent"] = d
    blocked = d[(d.c401 + d.c403) > 0]
    if blocked.empty:
        return {"agents_with_blocks": 0}
    share403 = blocked.c403 / (blocked.c401 + blocked.c403)
    top = d.assign(b=d.c401 + d.c403).nlargest(1, "b")
    return {
        "agents_with_blocks": int(len(blocked)),
        "median_agent_403_share": round(float(share403.median()), 3),
        "pct_agents_403_majority": round(100.0 * float((share403 > 0.5).mean()), 1),
        "largest_agent_share_of_block_mass_pct": round(
            100.0 * float(top.b.iloc[0]) / max(float((d.c401 + d.c403).sum()), 1), 1),
    }


def _shuffled(ids: list[int]) -> list[int]:
    random.seed(config.RANDOM_STATE)        # same seed as db_analysis_event_branches.py
    random.shuffle(ids)
    return ids


def run(n: int = 400, workers: int = 8) -> dict:
    # FIRST executable statement: a negative sample size is neither a census nor a sample, and an
    # argument check that runs after directories are made and a population is read has already
    # acted on input it had not accepted.
    if n < 0:
        raise SystemExit(
            f"--n must be 0 for a census or a positive sample size, not {n}")

    config.ensure_dirs()
    # The canonical arm comes through the shared reader, which is where the Boolean, malformed,
    # duplicate, positivity and exact-count contract lives. Reading the CSV here would be a second
    # population path with none of it.
    arm = _shuffled(list(arm_of_record_ids()))
    if n == 0:                                   # census, no sampling
        arm_ids = arm
        arm_lab = f"DB_arm_of_record_ids.csv, all {len(arm):,} agents (census)"
    else:
        arm_ids = arm[:n]
        arm_lab = "DB_arm_of_record_ids.csv, random n of the seed-42 shuffle"
    arm_block = histogram(arm_ids, arm_lab, workers, tag="_arm")

    # A census with unresolved agents is not a census. Either every declared agent answers or
    # the run refuses, exactly as the cache-side contract behaves.
    if n == 0 and arm_block["agents_failed"]:
        raise SystemExit(
            f"{arm_block['agents_failed']} of {arm_block['agents_queried']} agents did not "
            f"return a histogram, so this is not the census it would be labelled as. Resolve "
            f"those agents against the database and re-run, or run a sample with --n rather "
            f"than publishing a partial census.")

    # Sample and census must not share a filename: a later sample would otherwise overwrite a
    # census, or be mistaken for one.
    tag = ("_arm_census" if n == 0
           else f"_arm_sample{n}_seed{config.RANDOM_STATE}")
    per_agent = _HELD.pop("per_agent", None)

    res = {
        "population": "arm_of_record",
        "sample_size_per_arm": n if n > 0 else "census",
        "request_universe": "full history in `hits`, not the cache window",
        "abusive_arm_of_record": arm_block,
    }
    json.dump(res, open(OUT_JSON, "w"), indent=2)
    if per_agent is not None:
        per_agent.to_csv(config.TAB_DIR / f"DB_status_per_agent{tag}.csv", index=False)
    print(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400,
                    help="agents per arm; 0 = census over the whole arm")
    ap.add_argument("--workers", type=int, default=8, help="parallel connections")
    a = ap.parse_args()
    run(n=a.n, workers=a.workers)
