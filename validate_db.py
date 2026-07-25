"""
Small end-to-end validation of the DB extraction (run before the full battery).

Builds a mixed set of ~16 agents (the original local examples + a sample of
moderate honeypot-labeled bots), extracts each enriched series, times it, and
prints a sanity summary. Confirms: correctness, that block/robots signals appear,
caching works, and per-agent timing is acceptable for an overnight run.

    python validate_db.py
"""
from __future__ import annotations

import time

import pandas as pd

import db
import db_source as dbs

# Local labeled examples (filename a_id -> label), validated to exist in `agent`.
LOCAL = [(1006025140, "bot"), (1032111909, "human"),
         (1002085156, "chrome"), (100013, "pothuman")]


def moderate_honeypot_bots(n: int = 12) -> list[int]:
    """Honeypot-labeled bots of moderate total activity (avoid giant agents
    whose extraction would be slow). Small-table scan (194k rows) is fine."""
    df = db.read_sql(
        """
        SELECT uh_a_id, SUM(uh_total) AS t
        FROM url2agent_honeypot
        WHERE uh_total>0 AND uh_a_id<=%s
        GROUP BY uh_a_id HAVING t BETWEEN 200 AND 8000
        ORDER BY uh_a_id LIMIT %s
        """,
        (dbs.MAX_A_ID, n),
    )
    return df["uh_a_id"].astype(int).tolist()


def main() -> None:
    print("emptydays loaded:", len(dbs.empty_days()), "downtime dates")
    print("global date range:", dbs.date_floor_ceiling())

    agents = list(LOCAL) + [(aid, "honeypot_bot") for aid in moderate_honeypot_bots()]
    print(f"\nValidating extraction on {len(agents)} agents...\n")

    rows, t0 = [], time.time()
    for a_id, label in agents:
        t = time.time()
        ag = dbs.make_agent(a_id, label, use_cache=False)  # force DB hit for timing
        dt = time.time() - t
        if ag is None:
            print(f"  {a_id} ({label}): NO DATA"); continue
        f = ag.features
        rows.append({
            "a_id": a_id, "label": label, "sec": round(dt, 1),
            "ua": (ag.meta.get("a_name") or "")[:42],
            "days": len(f), "hits": int(f.hits.sum()),
            "blocks(401/403)": int(f.cblock.sum()), "rate_limited(429)": int(f.c429.sum()),
            "robots_hits": int(f.robots.sum()),
            "max_ip/day": (int(f.n_ip.max()) if f.n_ip.notna().any() else None),
            "date_lo": str(f.date.min()), "date_hi": str(f.date.max()),
        })
    rep = pd.DataFrame(rows)
    pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 44)
    print(rep.to_string(index=False))
    print(f"\nTotal {time.time()-t0:.1f}s for {len(rows)} agents "
          f"({(time.time()-t0)/max(1,len(rows)):.2f}s/agent, cold).")

    # confirm cache round-trips instantly
    t = time.time()
    _ = dbs.extract_series(LOCAL[0][0], use_cache=True)
    print(f"cache read: {time.time()-t:.3f}s")


if __name__ == "__main__":
    main()
