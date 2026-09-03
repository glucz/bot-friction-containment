"""Pre-warm the windows the link-permutation null will ask for, in parallel.

The null evaluates each successor around a RANDOM OTHER death's date, so the windows it needs
are not the ones any death-date warm produces. Serially that is minutes per uncached successor,
and the 404 sub-query on a heavy agent runs four to five minutes on the current indexes.

The draws are reproducible: `_link_permutation_null` seeds `np.random.default_rng(RANDOM_STATE)`
and, since the per-successor fix, draws once per successor on first encounter while iterating the
pair frame in order. Replaying that exact sequence here names the same windows, so the stage
afterwards finds them cached and does no SQL at all.

It is replayed rather than imported because the null does the drawing inline; if that loop ever
changes shape, this file must change with it. The assertion to keep in mind: a warm that misses
simply leaves work for the stage, it never produces a wrong number.

Usage: python warm_null_windows.py --pairs outputs/tables/DB_rotation_ip_pairs_v4.csv \
                                   --deaths outputs/tables/DB_rotation_deaths_v4.csv [--workers 6]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

import db
import db_source as dbs
import db_analysis_identity_rotation as R

_local = threading.local()
_lock = threading.Lock()
_stat = {"fetched": 0, "failed": 0}


def _conn():
    if not hasattr(_local, "c"):
        _local.c = db.get_connection()
    return _local.c


def permuted_windows(pairs: pd.DataFrame, deaths: pd.DataFrame) -> set:
    """Replay the null's draw sequence for both the core and the envelope call."""
    pool = pd.to_datetime(deaths["t_death"]).dt.date.to_numpy()
    need = set()
    for subset in (pairs[pairs["in_core"] == 1], pairs):       # same order as stage_inference
        rng = np.random.default_rng(R.RANDOM_STATE)            # fresh seed per call, as in the null
        seen: dict[int, object] = {}
        for _, r in subset.iterrows():
            b = int(r["a_succ"])
            if b not in seen:
                seen[b] = rng.choice(pool)
            need.add((b, seen[b]))
    return need


def _warm(job):
    b, t = job
    if dbs._cache_path(int(b)).exists():        # full history: the null uses it directly
        return
    lo = t - _dt.timedelta(days=R.PLACEBO_OFFSET + R.WC + 1)
    hi = t + _dt.timedelta(days=R.WC + 1)
    try:
        dbs.extract_series_rotation(int(b), lo, hi, conn=_conn())
        with _lock:
            _stat["fetched"] += 1
    except Exception:
        with _lock:
            _stat["failed"] += 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--deaths", required=True)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()

    pairs = pd.read_csv(a.pairs)
    deaths = pd.read_csv(a.deaths)
    need = permuted_windows(pairs, deaths)
    print(f"null needs {len(need):,} (successor, permuted-date) windows", flush=True)

    todo = []
    for b, t in need:
        if dbs._cache_path(int(b)).exists():
            continue
        lo = t - _dt.timedelta(days=R.PLACEBO_OFFSET + R.WC + 1)
        hi = t + _dt.timedelta(days=R.WC + 1)
        if (dbs.WINDOW_CACHE_DIR / f"rot_{int(b)}_{lo}_{hi}.parquet").exists():
            continue
        todo.append((b, t))
    print(f"  already satisfied: {len(need) - len(todo):,}   to fetch: {len(todo):,} "
          f"(workers {a.workers})", flush=True)
    if not todo:
        print("nothing to do")
        return

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(_warm, j) for j in todo]
        for n, _ in enumerate(as_completed(futs), 1):
            if n % 10 == 0 or n == len(todo):
                el = time.time() - t0
                rate = n / max(el, 1e-9)
                print(f"  [warm] {n}/{len(todo)}  fetched {_stat['fetched']}  "
                      f"failed {_stat['failed']}  {rate*60:.1f}/min  "
                      f"eta {((len(todo)-n)/max(rate,1e-9))/60:.1f} min", flush=True)
    print(f"done in {(time.time()-t0)/60:.1f} min: fetched {_stat['fetched']}, "
          f"failed {_stat['failed']}")


if __name__ == "__main__":
    main()
