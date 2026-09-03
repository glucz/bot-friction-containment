"""Rank the submitted estimates within the specification grid, and fit the monthly series.

Reads `spec_grid_and_window_free.json` and `combined_numbers_of_record.json` and reports where
the balanced-panel analogue and the full-event headline fall in the grid distribution. The
headline is NOT a member of the grid, which requires a balanced +/-10 panel, so its rank says
where it would fall rather than where it sits. Must be run AFTER the grid it ranks against.

Read-only apart from its JSON.
Usage: python derived_ranks_and_monthly.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
import sys as _sys
_sys.path.insert(0, str(HERE))
from _layout import data_root  # noqa: E402

# Resolved, not assumed: the table ships beside this package in the bundle and lives under
# empirical-support in the source tree.
MONTHLY = data_root() / "outputs" / "tables" / "DB_observability_monthly_v2.csv"

# The submitted specification, as named in D:398.
CHOSEN = dict(coord="nonblocked", spacing=16, W=5, gap=5)   # preferred spec: [-10,-6] baseline needs 16


def main() -> None:
    grid = json.loads((HERE / "spec_grid_and_window_free.json").read_text())["grid"]
    record = json.loads((HERE / "combined_numbers_of_record.json").read_text())

    dids = np.array([c["did"] for c in grid])
    cell = next(c for c in grid
                if all(c[k] == v for k, v in CHOSEN.items()))
    headline = record["did_combined"]

    def rank(x: float) -> float:
        """Percentile among the grid, as 'fraction of cells at least as negative'."""
        return float(100.0 * (dids <= x).mean())

    out = {
        "grid_n": len(grid),
        "balanced_analogue": {
            "did": cell["did"], "n_bot": cell["n_bot"], "n_ctrl": cell["n_ctrl"],
            "percentile_in_grid": rank(cell["did"]),
            "note": "the grid cell matching the submitted choices; a member of the grid",
        },
        "full_event_headline": {
            "did": headline["did"], "n_bot": headline["n_bot"], "n_ctrl": headline["n_ctrl"],
            "percentile_in_grid": rank(headline["did"]),
            "note": ("the submitted estimate, from combined_numbers_of_record.json. It is NOT a "
                     "member of this grid -- the grid requires a balanced +/-10 panel -- so this "
                     "rank says where it would fall, not where it sits."),
        },
    }

    print(f"grid cells: {out['grid_n']}")
    for key in ("balanced_analogue", "full_event_headline"):
        e = out[key]
        print(f"  {key:22s} {e['did']:+.4f} on {e['n_bot']}/{e['n_ctrl']}"
              f"  -> {e['percentile_in_grid']:.1f}th percentile")

    # ---- monthly trend, with and without the trailing partial month ----------
    if MONTHLY.exists():
        m = pd.read_csv(MONTHLY)
        out["monthly"] = {}
        for grp, g in m.groupby("group"):
            g = g.sort_values("month").reset_index(drop=True)
            x = np.arange(len(g), dtype=float)
            y = g["mean_O"].to_numpy(dtype=float)
            full = float(np.polyfit(x, y, 1)[0])
            drop = float(np.polyfit(x[:-1], y[:-1], 1)[0])
            out["monthly"][str(grp)] = {
                "n_months": int(len(g)),
                "slope_per_month": full,
                "slope_per_month_excluding_last": drop,
                "sign_reverses": bool(np.sign(full) != np.sign(drop)),
                "last_month": str(g["month"].iloc[-1]),
            }
            print(f"  monthly {grp:<10} slope {full:+.4f}/month, "
                  f"{drop:+.4f} without the last month"
                  f"{'  SIGN REVERSES' if np.sign(full) != np.sign(drop) else ''}")
    else:
        # An error string written into the artifact is not a result. Publishing one replaces the
        # monthly slopes with a message and leaves an exit code of 0 behind it.
        raise SystemExit(
            f"the monthly observability table is not available, so the monthly block cannot be "
            f"computed and this artifact must not be rewritten without it.\n"
            f"  expected at {MONTHLY}")

    p = HERE / "derived_ranks_and_monthly.json"
    json.dump(out, open(p, "w"), indent=2)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
