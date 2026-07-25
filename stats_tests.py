"""
Nonparametric tests for non-stationarity (Analysis A) and helpers.

- Mann-Kendall via Kendall's tau against the time index: detects gradual,
  monotonic drift (does the feature trend up or down over the lifecycle?).
- Pettitt's test (rank formulation, O(n log n)): detects a single abrupt
  regime shift and its location (does the agent switch tactics at some point?).

Both are distribution-free, appropriate for the bounded, non-Gaussian features.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class TrendResult:
    tau: float
    p_value: float
    direction: int          # +1 increasing, -1 decreasing, 0 none
    n: int


@dataclass
class ChangePointResult:
    location: int           # positional index of the change point
    p_value: float
    significant: bool
    n: int


def mann_kendall(x: np.ndarray, alpha: float = 0.05) -> TrendResult:
    """Monotonic-trend test using Kendall's tau vs. observation order."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 8 or np.allclose(x, x[0]):
        return TrendResult(tau=0.0, p_value=1.0, direction=0, n=n)
    t = np.arange(n)
    tau, p = stats.kendalltau(t, x)
    if not np.isfinite(tau):
        return TrendResult(tau=0.0, p_value=1.0, direction=0, n=n)
    direction = 0
    if p < alpha:
        direction = int(np.sign(tau))
    return TrendResult(tau=float(tau), p_value=float(p), direction=direction, n=n)


def pettitt(x: np.ndarray, alpha: float = 0.05) -> ChangePointResult:
    """Pettitt single-change-point test (rank-based).

    U_t = 2 * sum_{i<=t} rank(x_i) - t (N+1);  K = max_t |U_t|.
    Approximate significance: p ~= 2 exp(-6 K^2 / (N^3 + N^2)).
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 10 or np.allclose(x, x[0]):
        return ChangePointResult(location=-1, p_value=1.0, significant=False, n=n)
    ranks = stats.rankdata(x)
    u = 2.0 * np.cumsum(ranks) - np.arange(1, n + 1) * (n + 1)
    k_idx = int(np.argmax(np.abs(u)))
    k = abs(u[k_idx])
    p = 2.0 * np.exp(-6.0 * k**2 / (n**3 + n**2))
    p = float(min(1.0, p))
    return ChangePointResult(
        location=k_idx, p_value=p, significant=p < alpha, n=n
    )


def bootstrap_ci(
    values: np.ndarray, n_boot: int = 2000, ci: float = 95.0, seed: int = 42
) -> tuple[float, float, float]:
    """Bootstrap mean and percentile CI. Returns (mean, lo, hi)."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    boot = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    lo = float(np.percentile(boot, (100 - ci) / 2))
    hi = float(np.percentile(boot, 100 - (100 - ci) / 2))
    return (float(values.mean()), lo, hi)
