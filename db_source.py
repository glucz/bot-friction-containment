"""
Enriched per-agent series from the live AGWA database.

This is the real implementation of the DBSource concept stubbed in
datasource.py. It re-derives, for one agent, a CALENDAR-DATED daily series with
the signals the local derived files never had:

  - real block codes (401/403) and rate-limit (429), not just 200/404,
  - robots.txt access (h_u_id = 64) -> RFC-9309 compliance,
  - network spread as distinct IP / domain / country / URL counts per day.

All constraints from the data descriptor + live probing are enforced:
  * a_id <= 2_147_483_647  (agents above this have NO hits — int/bigint overflow)
  * `emptydays` (confirmed collection downtime) rows are dropped
  * only INDEXED per-agent queries (WHERE h_a_id=%s, idx h_1/h_5) — never a scan
  * aggregates only: we never SELECT raw IPs (i_name) or export PII
  * per-agent Parquet cache -> resumable across service-limit blackouts

`h_ts` is the processing date (≈ request date + up to 1 day; scheduler conflicts
can merge two days into one inflated-volume day), so downstream analysis keys
filter-pressure events on the STATUS RATIO, not on volume.
"""
from __future__ import annotations

import datetime as _dt
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

import config
import db
from datasource import Agent

MAX_A_ID = 2_147_483_647
ROBOTS_U_ID = 64
CACHE_DIR = config.HERE / "cache" / "agents"

# Validated, index-covered extraction (uses h_1=(h_a_id,h_ts) / h_5=(h_a_id,h_status)).
# FULL adds per-day network-spread via COUNT(DISTINCT ...); on very large agents
# the DISTINCT + GROUP BY filesort can trip TokuDB (error 1152), so we fall back
# to BASIC (the primary signals: volume/status/block/robots) and mark `degraded`.
_EXTRACT_FULL = """
SELECT DATE(h_ts) AS date, COUNT(*) AS hits,
       SUM(h_status=200) AS c200, SUM(h_status=404) AS c404,
       SUM(h_status IN (401,403)) AS cblock, SUM(h_status=429) AS c429,
       SUM(h_u_id=%s) AS robots,
       COUNT(DISTINCT h_i_id) AS n_ip, COUNT(DISTINCT h_d_id) AS n_dom,
       COUNT(DISTINCT h_ccode) AS n_ctry, COUNT(DISTINCT h_u_id) AS n_url
FROM hits WHERE h_a_id=%s GROUP BY DATE(h_ts) ORDER BY date
"""

_EXTRACT_BASIC = """
SELECT DATE(h_ts) AS date, COUNT(*) AS hits,
       SUM(h_status=200) AS c200, SUM(h_status=404) AS c404,
       SUM(h_status IN (401,403)) AS cblock, SUM(h_status=429) AS c429,
       SUM(h_u_id=%s) AS robots
FROM hits WHERE h_a_id=%s GROUP BY DATE(h_ts) ORDER BY date
"""


# Per-day COUNT(DISTINCT) (FULL) is 3-46s/agent and can trip TokuDB; BASIC is
# 0.3-2s and never fails. So BASIC is primary. FULL spread is opt-in and only
# attempted for agents below this hit count (above it, it's too slow anyway).
SPREAD_HITCAP = 20_000


def _query_with_fallback(a_id: int, conn, with_spread: bool, total_hits: int | None) -> tuple[pd.DataFrame, bool]:
    """Return (daily frame, has_spread?). Uses BASIC unless spread is requested
    AND the agent is small enough for FULL to be fast & safe."""
    if with_spread and (total_hits is None or total_hits <= SPREAD_HITCAP):
        try:
            return db.read_sql(_EXTRACT_FULL, (ROBOTS_U_ID, a_id), conn=conn), True
        except Exception:
            pass  # fall through to BASIC
    df = db.read_sql(_EXTRACT_BASIC, (ROBOTS_U_ID, a_id), conn=conn)
    for c in ("n_ip", "n_dom", "n_ctry", "n_url"):
        df[c] = np.nan
    return df, False


@lru_cache(maxsize=1)
def empty_days() -> frozenset:
    """Confirmed collection-downtime dates (to drop from every series)."""
    df = db.read_sql("SELECT e_day FROM emptydays")
    return frozenset(pd.to_datetime(df["e_day"]).dt.date)


@lru_cache(maxsize=1)
def date_floor_ceiling() -> tuple:
    df = db.read_sql("SELECT MIN(a_start) lo, MAX(a_last) hi FROM agent")
    return (df.lo.iloc[0], df.hi.iloc[0])


def agent_meta(a_id: int, conn=None) -> dict | None:
    """User-agent string + first/last seen, via the agent PK (indexed)."""
    df = db.read_sql(
        "SELECT a_id,a_name,a_start,a_last,a_totalhit FROM agent WHERE a_id=%s",
        (int(a_id),), conn=conn,
    )
    return None if df.empty else df.iloc[0].to_dict()


def _cache_path(a_id: int) -> Path:
    return CACHE_DIR / f"{a_id}.parquet"


def extract_series(a_id: int, conn=None, use_cache: bool = True, with_spread: bool = False) -> pd.DataFrame:
    """Calendar-dated enriched daily series for one agent (empty if none).

    Primary signals (always, fast): volume, 200/404, block (401/403), 429,
    robots.txt. Network-spread (distinct IP/dom/country) only if `with_spread`
    and the agent is small (see SPREAD_HITCAP) -- otherwise those columns are NaN.
    Adds derived rates and a `day_index` (days since first observation).
    """
    a_id = int(a_id)
    if a_id > MAX_A_ID:
        return pd.DataFrame()  # no hits exist for these agents (int/bigint overflow)
    cp = _cache_path(a_id)
    if use_cache and cp.exists():
        return pd.read_parquet(cp)

    total_hits = None
    if with_spread:
        total_hits = int(db.read_sql("SELECT COUNT(*) c FROM hits WHERE h_a_id=%s",
                                      (a_id,), conn=conn).c.iloc[0])
    df, has_spread = _query_with_fallback(a_id, conn, with_spread, total_hits)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df = df[~df["date"].isin(empty_days())].reset_index(drop=True)
    if not df.empty:
        df = _add_derived(df)
        df["has_spread"] = has_spread
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cp, index=False)
    return df


def _add_derived(df: pd.DataFrame) -> pd.DataFrame:
    h = df["hits"].clip(lower=1)
    df["p200"] = df["c200"] / h
    df["p404"] = df["c404"] / h
    df["p_block"] = df["cblock"] / h          # 401/403 — explicit "you're blocked"
    df["p_429"] = df["c429"] / h              # rate-limited
    df["robots_rate"] = df["robots"] / h      # RFC-9309 compliance proxy
    df["resistance"] = (1.0 - df["p200"]).clip(lower=0.0)
    # network spread (evasion signals): per-day distinct-identity richness
    df["spread_ip"] = df["n_ip"]
    df["spread_dom"] = df["n_dom"]
    df["spread_ctry"] = df["n_ctry"]
    df["evasion_index"] = df[["n_ip", "n_dom", "n_ctry"]].mean(axis=1)
    d = pd.to_datetime(df["date"])
    df["day_index"] = (d - d.min()).dt.days
    return df


def make_agent(a_id: int, label: str, conn=None, use_cache: bool = True) -> Agent | None:
    """Build an Agent (datasource.Agent contract) from the DB, or None."""
    feats = extract_series(a_id, conn=conn, use_cache=use_cache)
    if feats.empty:
        return None
    meta = agent_meta(a_id, conn=conn) or {}
    # Keep only the UA string in meta; never carry raw IPs.
    return Agent(agent_id=str(a_id), label=label, features=feats,
                 meta={"a_name": meta.get("a_name"), "a_start": str(meta.get("a_start")),
                       "a_last": str(meta.get("a_last")), "source": "db"})
