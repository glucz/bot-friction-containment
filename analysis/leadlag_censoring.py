"""Response-lag censoring on the arm of record: boundary shares and Kaplan-Meier medians.

The peak-lag search runs over +/-14 days, so a responder whose true lag exceeds the window is
recorded AT the boundary rather than beyond it. Whether that truncation moves the median is a
question about the estimator, and it has to be answered on the cohort the article reports - a
censoring *method* transfers across cohorts, its *result* does not.

Inputs: `leadlag_calendar_per_agent.csv`, emitted by `leadlag_calendar_recompute.py`, on the
primary calendar indexing. Responders only, since a leader's peak is on the other side.

Emits, per population: the responder count, how many sit at the boundary, that share, the naive
median and the Kaplan-Meier median with boundary peaks treated as right-censored, plus the survival
curve the figure draws.

Read-only apart from the JSON it writes.
Usage: python leadlag_censoring.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = HERE / "leadlag_calendar_per_agent.csv"
OUT = HERE / "leadlag_censoring.json"

LAG_MAX = 14
ARMS = [("abusive", "abusive automation"), ("verified", "verified humans")]


def _km(lags: np.ndarray, censored: np.ndarray):
    """Kaplan-Meier survival over observed lags, boundary peaks right-censored."""
    curve, S = [], 1.0
    for t in np.sort(np.unique(lags[~censored])):
        n_risk = int((lags >= t).sum())
        d = int(((lags == t) & ~censored).sum())
        if n_risk:
            S *= 1 - d / n_risk
        curve.append({"t": float(t), "survival": float(S), "n_risk": n_risk, "n_event": d})
    median = next((c["t"] for c in curve if c["survival"] <= 0.5), None)
    return curve, median


def main() -> None:
    df = pd.read_csv(SRC)
    out: dict = {
        "source": SRC.name,
        "indexing": "calendar",
        "lag_max": LAG_MAX,
        "definition": ("responders only; a peak at |lag| >= lag_max is treated as right-censored "
                       "rather than as an observed lag"),
        "populations": {},
    }
    for key, label in ARMS:
        g = df[(df.group == key) & (df["calendar_responder"] == True)]  # noqa: E712
        lags = g["calendar_post_lag"].dropna().to_numpy(float)
        cens = np.abs(lags) >= LAG_MAX
        curve, km_median = _km(lags, cens)
        out["populations"][key] = {
            "label": label,
            "n_responders": int(len(lags)),
            "n_boundary_responders": int(cens.sum()),
            "boundary_share_pct": round(100.0 * float(cens.mean()), 2),
            "median_naive": float(np.median(lags)),
            "median_km": km_median,
            "km_curve": curve,
        }
        p = out["populations"][key]
        print(f"  {label:22s} n={p['n_responders']:4d}  boundary={p['n_boundary_responders']:3d} "
              f"({p['boundary_share_pct']:.2f}%)  median naive {p['median_naive']:.0f} / "
              f"KM {p['median_km']:.0f}")

    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
