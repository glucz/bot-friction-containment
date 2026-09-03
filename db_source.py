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

import os
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
FROM hits WHERE h_a_id=%s GROUP BY DATE(h_ts)
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
        # The query no longer sorts server-side, so order it here. Downstream code indexes this
        # frame positionally (db_battery._block_events, _winmean), so an unsorted frame would
        # silently produce wrong windows rather than an error.
        df = df.sort_values("date", kind="mergesort")
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


WINDOW_CACHE_DIR = config.HERE / "cache" / "windows"

_EXTRACT_WINDOW = """
SELECT DATE(h_ts) AS date, COUNT(*) AS hits,
       SUM(h_status=200) AS c200, SUM(h_status=404) AS c404,
       SUM(h_status IN (401,403)) AS cblock, SUM(h_status=429) AS c429,
       SUM(h_u_id=%s) AS robots
FROM hits WHERE h_a_id=%s AND h_ts >= %s AND h_ts < %s GROUP BY DATE(h_ts)
"""


def extract_series_rotation(a_id: int, lo, hi, conn=None, use_cache: bool = True) -> pd.DataFrame:
    """Minimal windowed series for the rotation bridges: date, hits, c404, robots.

    `db_analysis_identity_rotation._behavior_matrix` reads exactly three signals -- p404,
    robots_rate and log1p(hits) -- and `_cal_stats` reads hits. Nothing in the bridges touches
    c200, cblock or c429, and fetching them is what makes the query expensive: they force the
    engine to read every row in the window, 18 M+ of them for the largest agents.

    Split into three cheap pieces instead:
      * hits/day   -- index-only on h_1 (h_a_id, h_ts)                 ~0.7 s
      * 404/day    -- h_5 (h_a_id, h_status) then rows for 404s only   ~7 s
      * robots/day -- index-only on h_6 (h_u_id, h_a_id, h_ts)         ~0.03 s
    Measured 8.1 s against 18.1 s for the combined windowed query, and the gap widens sharply
    on high-volume agents because the dominant term is rows-read, not days-returned.

    The columns this does NOT fetch are set to NaN, never 0: a NaN propagates visibly if some
    future caller reads them, whereas a 0 would look like a real measurement of "no blocks".
    """
    a_id = int(a_id)
    lo_s, hi_s = str(lo), str(hi)
    cp = WINDOW_CACHE_DIR / f"rot_{a_id}_{lo_s}_{hi_s}.parquet"
    if use_cache and cp.exists():
        return pd.read_parquet(cp)

    q_hits = ("SELECT DATE(h_ts) AS date, COUNT(*) AS hits FROM hits "
              "WHERE h_a_id=%s AND h_ts>=%s AND h_ts<%s GROUP BY DATE(h_ts)")
    q_404 = ("SELECT DATE(h_ts) AS date, COUNT(*) AS c404 FROM hits "
             "WHERE h_a_id=%s AND h_status=404 AND h_ts>=%s AND h_ts<%s GROUP BY DATE(h_ts)")
    q_rob = ("SELECT DATE(h_ts) AS date, COUNT(*) AS robots FROM hits "
             "WHERE h_u_id=%s AND h_a_id=%s AND h_ts>=%s AND h_ts<%s GROUP BY DATE(h_ts)")

    df = db.read_sql(q_hits, (a_id, lo_s, hi_s), conn=conn)
    if df.empty:
        return pd.DataFrame()
    d404 = db.read_sql(q_404, (a_id, lo_s, hi_s), conn=conn)
    drob = db.read_sql(q_rob, (ROBOTS_U_ID, a_id, lo_s, hi_s), conn=conn)
    for extra in (d404, drob):
        if not extra.empty:
            df = df.merge(extra, on="date", how="left")
    for c, default in (("c404", 0.0), ("robots", 0.0)):
        df[c] = df[c].fillna(default) if c in df.columns else default
    # Deliberately absent, see docstring.
    for c in ("c200", "cblock", "c429", "n_ip", "n_dom", "n_ctry", "n_url"):
        df[c] = np.nan

    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values("date", kind="mergesort")
    df = df[~df["date"].isin(empty_days())].reset_index(drop=True)
    if not df.empty:
        df = _add_derived(df)
        df["has_spread"] = False
        df["rotation_minimal"] = True        # marks the frame as partial, for any later reader
    if use_cache and not df.empty:
        WINDOW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = cp.with_suffix(".tmp")
        df.to_parquet(tmp, index=False)
        os.replace(tmp, cp)
    return df


def extract_series_window(a_id: int, lo, hi, conn=None, use_cache: bool = True) -> pd.DataFrame:
    """Daily series for ONE agent restricted to [lo, hi).

    Why this exists: the unrestricted extraction groups by DATE(h_ts) over an agent's whole
    history, which cannot use index h_1 (h_a_id, h_ts) for the grouping and so reads every row
    of the agent -- 193 s for a 1.07 M-row agent, and 20+ minutes for the 90 M-row ones. Adding
    the date range turns it into a narrow range scan on that index: the same agent over a
    73-day window takes 18 s, an 11x reduction, and the status sums come free because the row
    set is small.

    Use ONLY where a bounded window is genuinely sufficient -- the rotation bridges, which look
    at +/-14 days around a death and a placebo 60 days earlier. Anything needing a full history
    (the death detector, the botness axis) must use `extract_series`.

    Cached separately from the full series so a truncated frame can never be mistaken for one.
    """
    a_id = int(a_id)
    lo_s, hi_s = str(lo), str(hi)
    cp = WINDOW_CACHE_DIR / f"{a_id}_{lo_s}_{hi_s}.parquet"
    if use_cache and cp.exists():
        return pd.read_parquet(cp)
    df = db.read_sql(_EXTRACT_WINDOW, (ROBOTS_U_ID, a_id, lo_s, hi_s), conn=conn)
    for c in ("n_ip", "n_dom", "n_ctry", "n_url"):
        df[c] = np.nan
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df = df.sort_values("date", kind="mergesort")
        df = df[~df["date"].isin(empty_days())].reset_index(drop=True)
    if not df.empty:
        df = _add_derived(df)
        df["has_spread"] = False
    if use_cache:
        WINDOW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = cp.with_suffix(".tmp")
        df.to_parquet(tmp, index=False)
        os.replace(tmp, cp)
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
