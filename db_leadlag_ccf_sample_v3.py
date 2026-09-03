"""db_leadlag_ccf_sample_v3.py -- sample cross-correlation curves on the run of record.

Sources, both current:
  * `analysis/leadlag_calendar_per_agent.csv` - the cohort and per-agent peaks, emitted by
    `leadlag_calendar_recompute.py`;
  * `analysis/leadlag_calendar_recompute.json` - the population median |post-peak CCF| printed on
    each panel, so the number on the figure is the field the text cites.

The plotted curves are computed exactly as the estimator computes them: calendar-indexed,
zero-filled, friction = `p_block`, response = `p404`, lags restricted to +/-14.

Read-only apart from the figure. Usage: python db_leadlag_ccf_sample_v3.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import db_source as dbs
from db_analysis_leadlag import _ccf

ANA = (Path(__file__).resolve().parent.parent / "papers" / "D-adversarial-friction" / "analysis")
SRC = ANA / "leadlag_calendar_per_agent.csv"
JSON_SRC = ANA / "leadlag_calendar_recompute.json"
FIG_OUT = config.FIG_DIR / "DB_leadlag_ccf_sample_v2_arm.png"

ARMS = [("abusive", "abusive automation", "#c0392b"),
        ("verified", "verified humans", "#2980b9")]
LAG_MAX = 14
# The curves drawn are the N_CURVES responders with the LARGEST |post-peak CCF|, which is a
# selection rule and not a representative sample. The caption must say so.
N_CURVES = 6
INDEXING = "calendar"      # the indexing the manuscripts report throughout


def _calendar_series(a_id: int):
    """The zero-filled calendar series the estimator uses, from the cached active-day frame."""
    df = pd.read_parquet(dbs._cache_path(int(a_id)))
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    full = pd.date_range(d["date"].min(), d["date"].max(), freq="D")
    d = d.set_index("date").reindex(full)
    for c in ("p_block", "p404"):
        d[c] = d[c].fillna(0.0)
    return d["p404"].to_numpy(float), d["p_block"].to_numpy(float)


def main() -> None:
    df = pd.read_csv(SRC)
    art = json.loads(JSON_SRC.read_text(encoding="utf-8"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150, sharey=True)
    summary = []
    for ax, (key, label, colour) in zip(axes, ARMS):
        g = df[(df.group == key) & (df[f"{INDEXING}_responder"] == True)]  # noqa: E712
        g = g.reindex(g[f"{INDEXING}_post_ccf"].abs().sort_values(ascending=False).index)
        drawn = 0
        for _, r in g.iterrows():
            if drawn >= N_CURVES:
                break
            if not dbs._cache_path(int(r["a_id"])).exists():
                continue
            resp, frict = _calendar_series(int(r["a_id"]))
            if np.nanstd(frict) < 1e-9 or np.nanstd(resp) < 1e-9:
                continue
            lags, ccf = _ccf(resp, frict)
            m = (lags >= -LAG_MAX) & (lags <= LAG_MAX)
            ax.plot(lags[m], ccf[m], color=colour, alpha=0.55, lw=1.2)
            drawn += 1

        med = float(art["populations"][key][INDEXING]["median_abs_ccf"])
        n = int(art["populations"][key]["n"])
        ax.axvline(0, color="k", lw=0.6)
        ax.set_title(f"{label}\nmedian |post-peak CCF| = {med:.3f} "
                     f"({INDEXING} indexing, cohort n={n})", fontsize=10)
        ax.set_xlabel("lag (days; positive = behavior follows friction)", fontsize=9)
        ax.grid(alpha=0.3)
        summary.append((label, drawn, med, n))
    axes[0].set_ylabel("cross-correlation", fontsize=9)

    fig.tight_layout()
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT, dpi=150)
    print(f"wrote {FIG_OUT}")
    for label, drawn, med, n in summary:
        print(f"  {label:<22} {drawn} curves   median |CCF| {med:.4f}   cohort n={n}")


if __name__ == "__main__":
    main()
