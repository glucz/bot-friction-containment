"""Re-render E_botness_density.png from cached outputs (no database needed).

The full pipeline (`analysis_e_calibration.py`) needs live DB access to
recompute the scored-agent botness distribution. When only the *presentation*
of that figure changes -- axis labels, title, styling -- this script rebuilds
the PNG byte-for-byte equivalently from the cached bin table written by that
pipeline run:

    outputs/tables/E_botness_density.csv   (40 bin centres + densities)
    outputs/summary.json                   (operating threshold, mass above it)

Usage:  python replot_botness_density.py
"""
from __future__ import annotations

import json

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import config  # noqa: E402

TITLE = "Empirical botness density (AGWA-derived) -- containment-map input"


def main() -> None:
    tab = pd.read_csv(config.TAB_DIR / "E_botness_density.csv")
    summary = json.loads((config.OUT_DIR / "summary.json").read_text(encoding="utf-8"))
    cal = summary["E_calibration"]["botness_density"]
    thr = float(cal["threshold"])

    centres = tab["botness"].to_numpy()
    density = tab["density"].to_numpy()
    width = float(centres[1] - centres[0]) if len(centres) > 1 else 0.025

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    # Same visual as the pipeline's ax.hist(..., bins=40, range=(0,1), density=True).
    ax.bar(centres, density, width=width, color="tab:gray", alpha=0.8, align="center")
    ax.axvline(thr, color="tab:red", ls="--", lw=2,
               label=f"operating threshold = {thr:.3f}")
    ax.set_xlabel("botness score b")
    ax.set_ylabel("empirical density")
    ax.set_title(TITLE)
    ax.set_xlim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = config.FIG_DIR / "E_botness_density.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}  (threshold {thr:.3f}, "
          f"{cal['frac_above_threshold']:.1%} of mass above it)")


if __name__ == "__main__":
    main()
