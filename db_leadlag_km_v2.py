"""db_leadlag_km_v2.py -- response-lag survival on the arm of record.

Source: `analysis/leadlag_censoring.json`, computed from the same per-agent table that produces the
lead-lag estimates, so the curves and the text cannot describe different cohorts.

Read-only apart from the figure. Usage: python db_leadlag_km_v2.py
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

import config

SRC = (Path(__file__).resolve().parent.parent / "papers" / "D-adversarial-friction" /
       "analysis" / "leadlag_censoring.json")
FIG_OUT = config.FIG_DIR / "DB_leadlag_KM_v2_arm.png"
COLOURS = {"abusive": "#c0392b", "verified": "#2980b9"}


def main() -> None:
    blob = json.loads(SRC.read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
    for key, p in blob["populations"].items():
        t = [0.0] + [c["t"] for c in p["km_curve"]]
        s = [1.0] + [c["survival"] for c in p["km_curve"]]
        ax.step(t, s, where="post", lw=1.9, color=COLOURS.get(key),
                label=f"{p['label']} (n={p['n_responders']}, censored {p['n_boundary_responders']}, "
                      f"median {p['median_km']:.0f} d)")
    ax.axhline(0.5, color="k", lw=0.6, ls="--")
    ax.set_xlabel("response lag (days)", fontsize=9)
    ax.set_ylabel("share of responders with a longer lag", fontsize=9)
    ax.set_title("Response-lag survival, boundary peaks right-censored "
                 f"(|lag| >= {blob['lag_max']} d)", fontsize=10)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=9, frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT, dpi=150)
    print(f"wrote {FIG_OUT}")
    for key, p in blob["populations"].items():
        print(f"  {p['label']:22s} n={p['n_responders']} censored={p['n_boundary_responders']} "
              f"median={p['median_km']}")


if __name__ == "__main__":
    main()
