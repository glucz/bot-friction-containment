"""MethodsX Fig. 1: pre/post observability by dose, from the specification of record.

Input: `dose_events_corrected.csv`, written by `dose_split_recompute.py` from the same in-memory
frame that produces `dose_split_of_record.json`. Rendering the panels from that one frame is what
keeps figure and artifact on one specification - the block-insensitive 404 coordinate, a
`[-10,-6]` baseline and a 16-observation spacing. A render built by any other pipeline would carry
a different event set behind an identical-looking caption.

Contract: `numbers_manifest.py` hashes the released render against this output, and the caption's
quoted values are checked against the same JSON fields separately, since the caption text is
handwritten and the hash cannot see it.

Read-only apart from its own PNG. Usage: python dose_balance_figure.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EVENTS = HERE / "dose_events_corrected.csv"
ART = HERE / "dose_split_recompute.json"
OUT = HERE / "dose_balance_corrected.png"
PUBLISHED = ROOT / "empirical-support" / "outputs" / "figures" / "DB_dose_balance_v3.png"

LIGHT, HEAVY = "#4C72B0", "#C44E52"


def main() -> None:
    if not EVENTS.exists():
        raise SystemExit(f"{EVENTS.name} not found -- run dose_split_recompute.py first")
    d = pd.read_csv(EVENTS)
    blob = json.loads(ART.read_text(encoding="utf-8"))
    key = next(k for k in blob if k.startswith("corrected"))
    rec = blob[key]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, grp, title in zip(axes, ("abusive", "verified"),
                              ("Abusive automation", "Verified humans")):
        G = d[d.group == grp]
        if G.empty:
            continue
        med = G.intensity.median()
        heavy = G.intensity > med
        r = rec[grp]

        # Pre and post observability, split at the within-group median dose.
        pairs = [("light", G[~heavy], LIGHT), ("heavy", G[heavy], HEAVY)]
        x = np.arange(2)
        w = 0.36
        for k, (lab, sub, col) in enumerate(pairs):
            # The frame stores the pre level and the change, not the post level.
            pre = sub.pre_O.mean()
            post = pre + sub.dO.mean()
            ax.bar(x + (k - 0.5) * w, [pre, post], width=w, color=col,
                   label=f"{lab} dose (n = {len(sub):,})")
            for xi, v in zip(x + (k - 0.5) * w, [pre, post]):
                ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(["pre-event $O$", "post-event $O$"])
        ax.set_ylabel("observability $O$")
        ax.set_title(f"{title}\nSMD pre-$O$ {r['smd_pre_O']:+.2f}, "
                     f"pre-log-volume {r['smd_pre_logvol']:+.2f}", fontsize=10)
        # Headroom so the legend cannot sit on top of the value labels.
        tops = [sub.pre_O.mean() + max(sub.dO.mean(), 0.0)
                for _, sub, _ in pairs if not sub.empty]
        tops += [sub.pre_O.mean() for _, sub, _ in pairs if not sub.empty]
        ax.set_ylim(0, max(tops) * 1.34)
        ax.legend(fontsize=8, frameon=False, loc="upper center", ncol=2)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"Dose balance at the specification of record ({key})", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT, dpi=160)
    PUBLISHED.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PUBLISHED, dpi=160)
    print(f"wrote {OUT}")
    print(f"wrote {PUBLISHED}")

    ab = rec["abusive"]["pre_levels"]
    print(f"  abusive pre-O light {ab['pre_O']['light']:.3f}  heavy {ab['pre_O']['heavy']:.3f}  "
          f"diff {ab['pre_O']['diff']:+.3f} CI [{ab['pre_O']['diff_ci'][0]:.2f}, "
          f"{ab['pre_O']['diff_ci'][1]:.2f}]")
    print(f"  abusive SMD pre-logvol {rec['abusive']['smd_pre_logvol']:+.3f}")


if __name__ == "__main__":
    main()
