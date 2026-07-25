"""
Derived behavioral signals shared across analyses.

These turn the 14 raw features into the constructs the claim is stated in:
  - evasion_index   : network-identity spread (IP rotation + multi-domain +
                      multi-country fan-out). Rising => the visible->evasive
                      (s_V -> s_E) tactic shift of Papers D/F.
  - resistance      : fraction of requests NOT served 200 (1 - p200). Proxy for
                      "filter is biting" (includes 404/403/429/5xx in the derived
                      data; 403/429 become explicit once re-derived from the DB).
  - compliance      : robots.txt access rate (RFC 9309 filter awareness).
  - request_diversity: url entropy (scanning breadth).

A *filter-pressure event* is an active day on which resistance is anomalously
high relative to the agent's own baseline -- the agent hitting a wall. Analysis B
measures how behavior changes after such events.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `df` with derived signal columns added."""
    out = df.copy()
    out["evasion_index"] = out[["ip_ent", "dom_ent", "ctry_ent"]].mean(axis=1)
    out["resistance"] = (1.0 - out["p200"]).clip(lower=0.0)   # non-200 share
    out["compliance"] = out["robots"]
    out["request_diversity"] = out["url_ent"]
    return out


def filter_pressure_events(
    active: pd.DataFrame,
    z: float = config.FILTER_PRESSURE_Z,
    min_gap: int | None = None,
) -> np.ndarray:
    """Positional indices (into `active`) of filter-pressure events.

    An event is a day whose `resistance` exceeds (mean + z * std) of the agent's
    own active-day resistance series. `min_gap` (defaults to the event window)
    enforces spacing so overlapping pre/post windows are not double-counted.
    """
    if min_gap is None:
        min_gap = config.EVENT_WINDOW
    if "resistance" not in active.columns:
        active = add_derived(active)
    r = active["resistance"].to_numpy()
    if len(r) < 3 or np.allclose(r, r[0]):
        return np.array([], dtype=int)
    thr = r.mean() + z * r.std(ddof=0)
    if not np.isfinite(thr) or thr <= 0:
        return np.array([], dtype=int)
    cand = np.flatnonzero(r > thr)
    # Enforce a minimum gap (keep the first of any tightly-clustered run).
    kept: list[int] = []
    last = -(10**9)
    for i in cand:
        if i - last >= min_gap:
            kept.append(int(i))
            last = i
    return np.array(kept, dtype=int)


def window_mean(
    active: pd.DataFrame, center: int, feature: str, side: str, w: int = config.EVENT_WINDOW
) -> float:
    """Mean of `feature` in the pre- or post-event window around positional `center`."""
    if side == "pre":
        lo, hi = max(0, center - w), center                      # [center-w, center-1]
    else:
        lo, hi = center + 1, min(len(active), center + 1 + w)     # [center+1, center+w]
    seg = active[feature].to_numpy()[lo:hi]
    return float(np.nanmean(seg)) if len(seg) else np.nan


def lifecycle_thirds(active: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split an agent's active days into early / mid / late thirds by row order."""
    n = len(active)
    a, b = n // 3, 2 * n // 3
    return active.iloc[:a], active.iloc[a:b], active.iloc[b:]


def autocorr_time(x: np.ndarray) -> float:
    """Decorrelation time: first lag where the autocorrelation drops below 1/e.

    Used to estimate the natural timescale of behavioral adaptation, which
    calibrates the population-dynamics companion's replicator speed relative to the defender's response
    delay. Returns NaN for degenerate series.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 8 or np.allclose(x, x[0]):
        return np.nan
    x = x - x.mean()
    var = np.dot(x, x)
    if var <= 0:
        return np.nan
    max_lag = min(n - 1, 100)
    for lag in range(1, max_lag + 1):
        ac = np.dot(x[:-lag], x[lag:]) / var
        if ac < np.e**-1:
            return float(lag)
    return float(max_lag)


def centroid_shift(active: pd.DataFrame, features: list[str], frac: float = 0.2) -> float:
    """Euclidean distance between early-life and late-life feature centroids.

    Quantifies how far an agent's behavior drifts over its lifecycle (C1).
    """
    n = len(active)
    k = max(1, int(n * frac))
    early = active[features].to_numpy()[:k]
    late = active[features].to_numpy()[-k:]
    if len(early) == 0 or len(late) == 0:
        return np.nan
    return float(np.linalg.norm(np.nanmean(late, axis=0) - np.nanmean(early, axis=0)))
