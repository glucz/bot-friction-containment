"""
Resilient, resumable, parallel extraction of every agent in the roster.

This is the long pole of the overnight run (~50k agents). Design for unattended
operation across service-limit blackouts:
  - per-agent Parquet cache; already-cached agents are SKIPPED -> safe to re-run
  - thread pool, one reused DB connection per worker (reconnect on drop)
  - per-agent try/except: one bad agent never stops the run; failures logged
  - per-query statement timeout (set in db.get_connection) kills any runaway
  - progress + ETA logged to outputs/extract_progress.log every PROGRESS_EVERY

    python extract_roster.py --workers 8            # full run
    python extract_roster.py --workers 8 --limit 100  # small test slice

Re-running after an interruption resumes from the cache automatically.
"""
from __future__ import annotations

import argparse
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

import config
import db
import db_source as dbs
from roster import ROSTER_CSV

PROGRESS_EVERY = 200
EXTRACT_TIMEOUT_MS = 25_000   # giants abort at 25s and the worker moves on
RETRIES = 3                   # transient TokuDB/connection errors under concurrency are retryable
GIANT_CAP = 500_000           # skip agents whose a_totalhit exceeds this (would just time out)
LOG = config.OUT_DIR / "extract_progress.log"
FAILURES = config.OUT_DIR / "extract_failures.csv"
ROSTER_SIZES = config.OUT_DIR / "roster_sizes.csv"

_local = threading.local()
_lock = threading.Lock()


def _conn():
    c = getattr(_local, "conn", None)
    if c is None:
        c = db.get_connection(timeout_ms=EXTRACT_TIMEOUT_MS)
        _local.conn = c
    return c


def _drop_conn():
    try:
        _local.conn.close()
    except Exception:
        pass
    _local.conn = None


def _extract_one(a_id: int) -> tuple[int, str, int]:
    """Extract+cache one agent. Returns (a_id, status, n_days). Retries transient
    errors with backoff (small agents that fail are concurrency blips, not size)."""
    if dbs._cache_path(a_id).exists():
        return (a_id, "cached", -1)
    last = "unknown"
    for attempt in range(1, RETRIES + 1):
        try:
            df = dbs.extract_series(a_id, conn=_conn(), use_cache=True)
            return (a_id, "ok" if not df.empty else "empty", len(df))
        except Exception as e:
            last = type(e).__name__
            _drop_conn()                       # force a fresh connection
            if attempt < RETRIES:
                time.sleep(1.5 * attempt)      # backoff to let contention clear
    return (a_id, f"error:{last}", 0)


def _log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with _lock:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="only first N agents (testing)")
    args = ap.parse_args()
    config.ensure_dirs()

    # Prefer roster_sizes.csv (has a_totalhit) so we can skip known giants.
    src = ROSTER_SIZES if ROSTER_SIZES.exists() else ROSTER_CSV
    roster = pd.read_csv(src)
    n_all = len(roster)
    n_giant = 0
    if "a_totalhit" in roster.columns:
        giant = roster["a_totalhit"] > GIANT_CAP        # NaN (unknown size) is kept
        n_giant = int(giant.sum())
        roster = roster[~giant]
    ids = roster["a_id"].astype(int).tolist()
    # Shuffle so giant/slow agents don't cluster and jam all workers at once.
    ids = pd.Series(ids).sample(frac=1.0, random_state=config.RANDOM_STATE).tolist()
    if args.limit:
        ids = ids[: args.limit]
    todo = [a for a in ids if not dbs._cache_path(a).exists()]
    _log(f"roster={n_all} (skipped {n_giant} giants > {GIANT_CAP:,} hits)  "
         f"already_cached={len(ids)-len(todo)}  todo={len(todo)}  workers={args.workers} timeout={EXTRACT_TIMEOUT_MS}ms")

    counts = {"ok": 0, "empty": 0, "cached": 0, "error": 0}
    failures: list[tuple[int, str]] = []
    t0 = time.time()
    done = 0

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_extract_one, a): a for a in todo}
        for fut in as_completed(futs):
            a_id, status, _ = fut.result()
            done += 1
            key = "error" if status.startswith("error") else status
            counts[key] = counts.get(key, 0) + 1
            if key == "error":
                failures.append((a_id, status))
            if done % PROGRESS_EVERY == 0 or done == len(todo):
                rate = done / max(1e-9, time.time() - t0)
                eta_h = (len(todo) - done) / max(1e-9, rate) / 3600
                _log(f"{done}/{len(todo)}  ok={counts['ok']} empty={counts['empty']} "
                     f"err={counts['error']}  {rate:.1f}/s  ETA {eta_h:.1f}h")

    if failures:
        pd.DataFrame(failures, columns=["a_id", "status"]).to_csv(FAILURES, index=False)
    _log(f"DONE in {(time.time()-t0)/3600:.2f}h  {counts}  failures->{FAILURES if failures else 'none'}")


if __name__ == "__main__":
    main()
