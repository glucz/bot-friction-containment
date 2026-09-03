"""Parallel pre-fetch of the IP-bridge candidate frames, one per death, into a parquet cache.

The rotation pipeline enumerates a dead agent's IP-bridge candidates with a self-join on `ip`
(52.1M rows), once per death, on a single connection. That query is the pipeline's dominant
cost and it scales with the dead agent's IP footprint:

    agent with    191 non-proxy IPs ->   1.6 s
    agent with 20,433 non-proxy IPs ->  18.8 s
    agent with 102,605 non-proxy IPs -> 58 s for the footprint count alone

The reason is a missing index column, not the join. `EXPLAIN` reports
`key=idx_ip_agent_time ... Using where` rather than `Using index`, because `i_proxy` is in no
index: every one of the agent's `ip` rows has to be read off disk just to test the proxy flag.
The same query re-run against a warm buffer pool returns in ~0 s, so the cost is cold random
I/O. `ALTER TABLE ip ADD INDEX (i_agent, i_proxy, i_name)` would make both the footprint count
and the join's driving side covering; until that exists, this script hides the latency by
running the deaths concurrently and by never asking the same question twice.

Two properties make it safe to run beside a live pipeline, matching warm_agent_cache.py:

  * an existing cache file is never rewritten, so a frame the pipeline already has is left alone;
  * a new frame is written to a temporary name in the same directory and moved into place with
    `os.replace`, which is atomic, so a concurrent reader sees either no file or a complete one.

The cached frame is the RAW query result (agent id and shared-IP count) before any threshold, so
the pipeline keeps applying its own `IP_OVERLAP_CORE` / `IP_SHARED_MIN` / `TOPK_IP`. The row cap
used at fetch time is recorded in the file's metadata sidecar; a request for a larger cap
refuses the cache rather than silently serving a truncated frame.

Usage:
    python warm_ip_bridges.py --deaths outputs/tables/DB_rotation_deaths_v4.csv --workers 6
    python warm_ip_bridges.py --ids 95648,129172 --workers 6
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

import config
import db

CACHE = Path(__file__).resolve().parent / "cache" / "ip_bridges"

# The pipeline's own query, verbatim from db_analysis_identity_rotation.bridge_candidates_ip.
SQL = ("SELECT b.i_agent AS a_id, COUNT(DISTINCT b.i_name) AS shared_ips "
       "FROM ip a JOIN ip b ON a.i_name = b.i_name "
       "WHERE a.i_agent = %s AND b.i_agent <> a.i_agent AND a.i_proxy = 0 "
       "GROUP BY b.i_agent ORDER BY shared_ips DESC LIMIT %s")

_local = threading.local()
_lock = threading.Lock()
_done = {"n": 0, "fetched": 0, "cached": 0, "failed": 0, "rows": 0}


def _conn():
    """One connection per worker thread, with the statement timeout disabled.

    `db_config.ini` already sets `max_execution_time_ms = 0`, so no cap is applied; passing it
    explicitly documents the intent, because a cap here is what removed the IP-rich operators
    from the v2 run -- and those are exactly the operators most able to rotate identity.
    """
    if not hasattr(_local, "c"):
        _local.c = db.get_connection(timeout_ms=0)
    return _local.c


def cache_path(a_id: int) -> Path:
    return CACHE / f"{int(a_id)}.parquet"


def load_cached(a_id: int, min_cap: int | None = None):
    """Return the cached raw frame for a death, or None.

    `min_cap` refuses a frame fetched under a smaller row cap than the caller needs, so a
    truncated enumeration is never mistaken for a complete one.
    """
    p = cache_path(a_id)
    if not p.exists():
        return None
    if min_cap is not None:
        meta = p.with_suffix(".meta.json")
        if meta.exists():
            try:
                if int(json.loads(meta.read_text())["cap"]) < int(min_cap):
                    return None
            except Exception:
                return None
    try:
        return pd.read_parquet(p)
    except Exception:
        return None


def _warm(a_id: int, cap: int) -> str:
    a_id = int(a_id)
    if cache_path(a_id).exists():
        with _lock:
            _done["n"] += 1
            _done["cached"] += 1
        return "cached"
    t0 = time.time()
    try:
        df = db.read_sql(SQL, (a_id, cap), conn=_conn())
    except Exception as exc:
        with _lock:
            _done["n"] += 1
            _done["failed"] += 1
        return f"fail {type(exc).__name__}: {str(exc)[:80]}"
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = CACHE / f".{a_id}.{os.getpid()}.{threading.get_ident()}.tmp"
    df.to_parquet(tmp)
    os.replace(tmp, cache_path(a_id))
    cache_path(a_id).with_suffix(".meta.json").write_text(
        json.dumps({"cap": int(cap), "rows": int(len(df)), "seconds": round(time.time() - t0, 1)}),
        encoding="utf-8")
    with _lock:
        _done["n"] += 1
        _done["fetched"] += 1
        _done["rows"] += len(df)
    return f"ok {len(df)} rows in {time.time() - t0:.1f}s"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deaths", default=str(config.TAB_DIR / "DB_rotation_deaths_v4.csv"),
                    help="CSV with an a_id column: the death list to enumerate")
    ap.add_argument("--ids", default=None, help="comma-separated agent ids instead of --deaths")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--cap", type=int, default=48,
                    help="row cap per death; the pipeline asks for TOPK_IP*4 = 48")
    a = ap.parse_args()

    if a.ids:
        ids = [int(x) for x in a.ids.split(",")]
    else:
        ids = pd.read_csv(a.deaths)["a_id"].astype(int).tolist()
    todo = [i for i in ids if not cache_path(i).exists()]
    print(f"deaths {len(ids)}   already enumerated {len(ids) - len(todo)}   to fetch {len(todo)}")
    if not todo:
        return

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(_warm, i, a.cap): i for i in todo}
        for k, f in enumerate(as_completed(futs), 1):
            r = f.result()
            if r.startswith("fail"):
                print(f"  [ip-bridge] {futs[f]}: {r}", flush=True)
            if k % 10 == 0 or k == len(todo):
                el = (time.time() - t0) / 60
                rate = k / el if el else 0
                print(f"  [ip-bridge] {k}/{len(todo)}  fetched {_done['fetched']}  "
                      f"failed {_done['failed']}  {rate:.1f}/min  "
                      f"eta {(len(todo) - k) / rate if rate else 0:.1f} min", flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min: fetched {_done['fetched']}, "
          f"failed {_done['failed']}, candidate rows {_done['rows']:,}")


if __name__ == "__main__":
    main()
