"""db_robots_response_v3.py -- robots.txt response to real blocks, on the event set of record.

Source: `analysis/event_spacing_recompute.json:figure_data.robots_response.corrected_11`, emitted
by the same admitted pass that produces the Section 7.7 estimates. The figure therefore cannot
carry a different cohort from the text: the means, the uncertainty and the event counts are read
from that block rather than recomputed here.

Read-only apart from the figure. Usage: python db_robots_response_v3.py
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config

SRC = (Path(__file__).resolve().parent.parent / "papers" / "D-adversarial-friction" /
       "analysis" / "event_spacing_recompute.json")
FIG_OUT = config.FIG_DIR / "DB_robots_response_to_block_v3.png"

ARMS = [("abusive", "abusive automation", "#c0392b"),
        ("verified", "verified humans", "#2980b9")]


def main() -> None:
    blk = json.loads(SRC.read_text(encoding="utf-8"))["figure_data"]["robots_response"]["corrected_11"]
    off = np.array(blk["offsets"], float)

    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
    for key, label, colour in ARMS:
        p = blk["populations"][key]
        mean = np.array(p["mean"], float)
        sem = np.array(p["sem"], float)
        ax.plot(off, mean, color=colour, lw=1.8, marker="o", ms=4,
                label=f"{label} (n_events={p['n_events']})")
        ax.fill_between(off, mean - sem, mean + sem, color=colour, alpha=0.18, linewidth=0)
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("days from block event", fontsize=9)
    ax.set_ylabel("baseline-subtracted robots.txt rate", fontsize=9)
    ax.set_title(f"robots.txt response to real blocks, spacing {blk['spacing']}", fontsize=10)
    ax.legend(fontsize=9, frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT, dpi=150)
    print(f"wrote {FIG_OUT}")
    for key, label, _ in ARMS:
        p = blk["populations"][key]
        print(f"  {label:<22} n_agents={p['n_agents']:<6} n_events={p['n_events']}")
    print(f"  input digests: {blk['input_digests']}")


if __name__ == "__main__":
    main()
