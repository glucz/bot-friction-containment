"""Re-fetch named agents into the per-agent cache, and say exactly why any of them fails.

`warm_agent_cache._warm` swallows the diagnosis: it catches every exception and returns
`f"fail {type(exc).__name__}"`, discarding the message, the traceback and the identity of the
agent, and the caller then throws that string away and increments a counter. An agent that cannot
be extracted therefore leaves no evidence beyond `failed 1` in a progress line. That is how the
heaviest declared crawlers went missing without anyone noticing, and the same silence is what made
the cache-shaped cohort possible in the first place.

Differences from the bulk warmer, all of them about being told what happened:

  * a FRESH connection per agent with the statement timeout explicitly disabled, so a connection
    dropped during a long extraction fails that agent rather than poisoning a cached handle every
    later agent on the thread will reuse;
  * the full exception type, message and traceback, per agent;
  * retries with backoff, each one logged, because a dropped connection is worth one more try and a
    genuinely absent agent is not;
  * empty and failed reported as DIFFERENT outcomes - an agent with no rows is a fact about the
    corpus, an agent that raised is a fact about the pipeline;
  * elapsed time and row count per agent, so a slow success is distinguishable from a hang;
  * an explicit exit code: nonzero if any declared id is still missing at the end.

Writes are atomic (temp file plus `os.replace`), so a reader sees an absent file or a complete
one, never a partial.

Usage:
    python refetch_agents.py 186118 665695935 2080904040
    python refetch_agents.py --ids-from some.csv [--retries 3] [--workers 3]
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

import pandas as pd

import db
import db_source as dbs

_print_lock = threading.Lock()


def _say(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def _fetch_one(a_id: int, retries: int) -> dict:
    """Fetch one agent. Returns an outcome dict; never raises."""
    a_id = int(a_id)
    cp = dbs._cache_path(a_id)
    if cp.exists():
        return {"a_id": a_id, "outcome": "already-cached", "rows": None, "secs": 0.0}

    if a_id > dbs.MAX_A_ID:
        # extract_series returns an empty frame for these WITHOUT querying, which the bulk warmer
        # records as "empty" - indistinguishable from an agent the corpus genuinely lacks.
        return {"a_id": a_id, "outcome": "over-max-a-id", "rows": 0, "secs": 0.0,
                "detail": f"id exceeds MAX_A_ID={dbs.MAX_A_ID}; no hits can exist for it"}

    last = None
    for attempt in range(1, retries + 1):
        t0 = time.time()
        conn = None
        try:
            # A fresh connection per attempt. The bulk warmer caches one per thread, so a
            # connection dropped mid-extraction breaks every later agent on that thread too.
            conn = db.get_connection(timeout_ms=0)
            df = dbs.extract_series(a_id, conn=conn, use_cache=False)
            secs = time.time() - t0
            if df is None or df.empty:
                return {"a_id": a_id, "outcome": "empty", "rows": 0, "secs": secs,
                        "detail": "query returned no rows - a fact about the corpus, not a failure"}
            cp.parent.mkdir(parents=True, exist_ok=True)
            tmp = cp.with_suffix(f".tmp{os.getpid()}_{threading.get_ident()}")
            df.to_parquet(tmp, index=False)
            os.replace(tmp, cp)
            return {"a_id": a_id, "outcome": "fetched", "rows": int(len(df)), "secs": secs}
        except Exception as exc:                      # noqa: BLE001 - reporting is the point
            secs = time.time() - t0
            last = {"a_id": a_id, "outcome": "failed", "rows": None, "secs": secs,
                    "detail": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc()}
            _say(f"  [{a_id}] attempt {attempt}/{retries} failed after {secs:,.0f}s -> "
                 f"{type(exc).__name__}: {exc}")
            if attempt < retries:
                back = 5 * attempt
                _say(f"  [{a_id}] retrying in {back}s")
                time.sleep(back)
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:                     # noqa: BLE001
                    pass
    return last


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*", type=int)
    ap.add_argument("--ids-from", default=None, help="CSV with an a_id column")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()

    ids = list(a.ids)
    if a.ids_from:
        ids += pd.read_csv(a.ids_from)["a_id"].astype(int).tolist()
    ids = sorted(set(int(i) for i in ids))
    if not ids:
        ap.error("give ids on the command line or with --ids-from")

    _say(f"re-fetching {len(ids)} agent(s) with no statement timeout, "
         f"{a.retries} attempt(s) each, {a.workers} worker(s)")
    for i in ids:
        _say(f"   {i}")
    _say("")

    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(_fetch_one, i, a.retries): i for i in ids}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            tail = f" {r['rows']:,} rows" if r.get("rows") is not None else ""
            _say(f"  [{r['a_id']}] {r['outcome']}{tail} in {r['secs']:,.0f}s"
                 + (f" - {r['detail']}" if r.get("detail") else ""))

    _say(f"\n{'=' * 62}\ndone in {(time.time() - t0) / 60:.1f} min")
    by = {}
    for r in results:
        by.setdefault(r["outcome"], []).append(r["a_id"])
    for outcome in sorted(by):
        _say(f"  {outcome:16s} {len(by[outcome])}  {by[outcome]}")

    for r in results:
        if r.get("traceback"):
            _say(f"\n--- traceback for {r['a_id']} ---\n{r['traceback']}")

    still = [i for i in ids if not dbs._cache_path(i).exists()]
    if still:
        _say(f"\nSTILL MISSING: {still}")
        _say("These are not 'empty' unless reported so above. An agent that raised is a pipeline")
        _say("fact and must be either fetched or DECLARED excluded with its reason - never left to")
        _say("shrink a cohort silently.")
        raise SystemExit(1)
    _say("\nall requested agents are now cached")


if __name__ == "__main__":
    main()
