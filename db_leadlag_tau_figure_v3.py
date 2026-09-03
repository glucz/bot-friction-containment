"""db_leadlag_tau_figure_v3.py -- response-lag distribution on the 999-replicate run.

Source: `analysis/leadlag_calendar_per_agent.csv`, emitted by
`leadlag_calendar_recompute.py` (999-replicate null, roster_v2 `abusive_share`).

Draws BOTH indexings, since the human arm's median differs between them; calendar is primary
and is what the manuscripts quote.

Read-only apart from the figure. Usage: python db_leadlag_tau_figure_v3.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

SRC = (Path(__file__).resolve().parent.parent / "papers" / "D-adversarial-friction" /
       "analysis" / "leadlag_calendar_per_agent.csv")
FIG_OUT = config.FIG_DIR / "DB_leadlag_tau_distribution_v2_arm.png"

ARMS = [("abusive", "abusive automation", "#c0392b"),
        ("verified", "verified humans", "#2980b9")]
LAG_MAX = 14


def main() -> None:
    df = pd.read_csv(SRC)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)

    summary = []
    for ax, idx in zip(axes, ("calendar", "active")):
        rcol, lcol = f"{idx}_responder", f"{idx}_post_lag"
        for key, label, colour in ARMS:
            g = df[(df.group == key) & (df[rcol] == True)]  # noqa: E712
            lags = g[lcol].dropna().to_numpy(float)
            if not lags.size:
                continue
            med = float(np.median(lags))
            at_bound = float((np.abs(lags) >= LAG_MAX).mean())
            ax.hist(lags, bins=np.arange(0.5, LAG_MAX + 1.5, 1.0), alpha=0.55,
                    color=colour, label=f"{label} (n={len(lags)}, median {med:.0f} d)",
                    density=True, edgecolor="white", linewidth=0.4)
            ax.axvline(med, color=colour, linestyle="--", linewidth=1.4)
            summary.append((idx, label, len(lags), med, 100 * at_bound))
        ax.set_title(f"{idx} indexing" + ("  (primary)" if idx == "calendar" else ""), fontsize=10)
        ax.set_xlabel("peak response lag (days)", fontsize=9)
        ax.legend(fontsize=8, frameon=False)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("density of responder agents", fontsize=9)

    fig.tight_layout()
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT, dpi=150)
    print(f"wrote {FIG_OUT}")
    for idx, label, n, med, pct in summary:
        print(f"  {idx:<9} {label:<20} n={n:<4} median {med:.0f} d   at boundary {pct:.1f}%")


if __name__ == "__main__":
    main()
