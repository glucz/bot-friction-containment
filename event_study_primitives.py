"""Event-study primitives shared by the current scripts.

Extracted so that a current analysis does not import a historical whole-analysis module to reach
three functions. The definitions are unchanged.

  * `_block_events(df, spacing)` - kept block events, thinned to a minimum separation
  * `_winmean(a, c, side, w)`    - pre/post window mean around an event centre
  * `_did(a, b, n)`              - difference in differences with a bootstrap interval

Window geometry: windows are [i-w, i+w], so two kept events have disjoint windows only when their
centres differ by at least 2w+1. The spacing an analysis passes is therefore part of its
specification, not a tuning knob.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

EVENT_W = 5
EVENT_SPACING = int(os.environ.get("AGWA_EVENT_SPACING", EVENT_W))


def _block_events(df: pd.DataFrame) -> np.ndarray:
    """Positional indices of real-block-pressure days: cblock spikes above the
    agent's own baseline (z>=1.5), OR any day with >=1 rate-limit (429)."""
    blk = df["cblock"].to_numpy(dtype=float)
    r429 = df["c429"].to_numpy(dtype=float)
    idx = set()
    if len(blk) >= 3 and blk.std() > 0:
        thr = blk.mean() + 1.5 * blk.std()
        idx.update(np.flatnonzero(blk > max(thr, 0)).tolist())
    idx.update(np.flatnonzero(r429 > 0).tolist())
    ev = sorted(idx)
    # enforce spacing so pre/post windows don't overlap (see EVENT_SPACING above: at the
    # default this does NOT achieve that, and is kept only so prior artifacts reproduce)
    kept, last = [], -10**9
    for i in ev:
        if i - last >= EVENT_SPACING:
            kept.append(i); last = i
    return np.array(kept, dtype=int)


def _winmean(a: np.ndarray, c: int, side: str, w: int) -> float:
    lo, hi = (max(0, c - w), c) if side == "pre" else (c + 1, min(len(a), c + 1 + w))
    seg = a[lo:hi]
    return float(np.nanmean(seg)) if len(seg) else np.nan


def _did(a: np.ndarray, b: np.ndarray, n=2000) -> dict:
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 5 or len(b) < 5:
        return {"did": None, "ci": [None, None], "sig": False, "bot": None, "ctrl": None}
    rng = np.random.default_rng(42)
    ba = rng.choice(a, (n, len(a)), True).mean(1)
    bb = rng.choice(b, (n, len(b)), True).mean(1)
    d = ba - bb
    lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    return {"did": float(a.mean() - b.mean()), "ci": [round(lo, 4), round(hi, 4)],
            "sig": bool(lo > 0 or hi < 0), "bot": round(float(a.mean()), 4), "ctrl": round(float(b.mean()), 4)}
