"""db_leadlag_shuffle_figure_v3.py -- circular-shift null figure on the 999-replicate run.

Drawn from `analysis/leadlag_calendar_recompute.json`, which is the lead-lag run
of record (999 replicates, roster_v2 `abusive_share`, calendar indexing). It stores the full
`null_distribution`, since a summary mean and max cannot be plotted.

Two panels, matching what the caption describes:
  left   responder share -- the null distribution with the observed value marked, per arm
  right  coupling strength -- observed median peak |ccf| against its null, per arm

Read-only apart from the figure it writes. Usage: python db_leadlag_shuffle_figure_v3.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config

SRC = (Path(__file__).resolve().parent.parent / "papers" / "D-adversarial-friction" /
       "analysis" / "leadlag_calendar_recompute.json")
FIG_OUT = config.FIG_DIR / "DB_leadlag_shuffle_null_v2.png"

ARMS = [("abusive", "abusive automation", "#c0392b"),
        ("verified", "verified humans", "#2980b9"),
        ("declared", "declared crawlers", "#27ae60"),
        ("pothuman", "sensitivity arm", "#7f8c8d")]


def main() -> None:
    d = json.loads(SRC.read_text())
    pops = d["populations"]
    n_null = d.get("n_null")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2))

    # ---- left: responder-share null distributions -----------------------------------
    for i, (key, label, colour) in enumerate(ARMS):
        nb = pops[key]["null"]["responder_share"]
        dist = np.asarray(nb.get("null_distribution", []), dtype=float)
        if not dist.size:
            continue
        y = i + np.zeros_like(dist)
        axL.scatter(100 * dist, y + np.random.default_rng(0).normal(0, 0.055, dist.size),
                    s=3, alpha=0.25, color=colour, linewidths=0)
        axL.scatter([100 * nb["obs"]], [i], marker="D", s=70, color=colour,
                    edgecolor="black", zorder=5)
        axL.text(100 * nb["obs"], i + 0.26, f"{100*nb['obs']:.1f}%  p={nb['p']:.3f}",
                 ha="center", fontsize=8.5, color="black")
    axL.set_yticks(range(len(ARMS)))
    axL.set_yticklabels([a[1] for a in ARMS], fontsize=9)
    axL.set_xlabel("responder share (%)", fontsize=9)
    axL.set_title(f"Responder share against its own null ({n_null} shifts)", fontsize=10)
    axL.grid(axis="x", alpha=0.3)
    axL.invert_yaxis()

    # ---- right: coupling strength, observed vs null ----------------------------------
    xs = np.arange(len(ARMS))
    obs = [pops[k]["calendar"].get("median_abs_ccf") or 0.0 for k, _, _ in ARMS]
    nul = [pops[k]["null"]["median_abs_ccf"]["null_mean"] for k, _, _ in ARMS]
    axR.bar(xs - 0.19, obs, width=0.36, label="observed",
            color=[a[2] for a in ARMS], edgecolor="black", linewidth=0.5)
    axR.bar(xs + 0.19, nul, width=0.36, label="null mean",
            color="white", edgecolor="black", linewidth=0.5, hatch="///")
    for x, o in zip(xs, obs):
        axR.text(x - 0.19, o + 0.004, f"{o:.3f}", ha="center", fontsize=8)
    axR.set_xticks(xs)
    axR.set_xticklabels([a[1].replace(" ", "\n") for a in ARMS], fontsize=8.5)
    axR.set_ylabel("median peak |cross-correlation|", fontsize=9)
    axR.set_title("Coupling strength: the discriminator that survives", fontsize=10)
    axR.legend(fontsize=8.5, frameon=False)
    axR.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT, dpi=150)
    print(f"wrote {FIG_OUT}")
    for key, label, _ in ARMS:
        nb = pops[key]["null"]["responder_share"]
        cc = pops[key]["null"]["median_abs_ccf"]
        print(f"  {label:<20} share {100*nb['obs']:5.1f}% (null {100*nb['null_mean']:5.1f}%, "
              f"p={nb['p']:.3f})   |ccf| {cc['obs']:.4f} (null {cc['null_mean']:.4f})")


if __name__ == "__main__":
    main()
