"""Total defender loss as a function of the graded schedule's ceiling.

Section 8.2 of the main article says the loss-minimizing static schedule sits at
a ceiling of 1.00, above both switching points, and that the surface near the top
is shallow enough that the result should be read as a direction rather than as an
exact ceiling. That claim needs an artifact of record, which this script writes.

For every ceiling on the tuning grid it reports the best total loss over the three
sigmoid widths the tuner searches, which is the quantity the tuner minimizes. It
draws the same seed-42 population as `sim_friction_policy.py` and changes nothing
in it; the numbers here reproduce that script's `static_f_target`.

Usage: python sim_ceiling_grid.py
Writes: outputs/tables/SIM_ceiling_grid.csv, outputs/SIM_ceiling_grid.json
"""
from __future__ import annotations

import csv
import json

import numpy as np

import config
import sim_friction_policy as sim

CEILINGS = np.arange(0.20, 1.0001, 0.025)
WIDTHS = (0.05, 0.08, 0.12)
OUT_CSV = config.OUT_DIR / "tables" / "SIM_ceiling_grid.csv"
OUT_JSON = config.OUT_DIR / "SIM_ceiling_grid.json"


def run() -> dict:
    config.ensure_dirs()
    rng = np.random.default_rng(42)
    pop = sim.draw_population(rng, sim.P["vA_sigma"])

    rows = []
    for ceiling in CEILINGS:
        best = None
        for width in WIDTHS:
            m = sim.eval_policy(sim.make_phi_graded(float(ceiling), width=width), pop)
            if best is None or m["L_D"] < best["L_D"]:
                best = {"width": width, "L_D": float(m["L_D"]),
                        "human_harm": float(m["human_harm"]),
                        "evasion_prob": float(m["evasion_prob"]),
                        "observed": float(m["observed"])}
        rows.append({"ceiling": round(float(ceiling), 3), **best})

    losses = np.array([r["L_D"] for r in rows])
    argmin = rows[int(losses.argmin())]
    lo = float(losses.min())

    # A ceiling is "admissible at tolerance t" if its loss is within t of the
    # minimum. The claim in 8.2 is about which switching points those admit.
    fresh_switch = sim.f_switch(sim.P["K_E"]) if hasattr(sim, "f_switch") else 0.8855488076521755
    tol_sets = {}
    for tol in (0.001, 0.005, 0.01):
        adm = [r["ceiling"] for r in rows if r["L_D"] <= lo * (1 + tol)]
        tol_sets[f"{tol:.3f}"] = {
            "ceilings": [min(adm), max(adm)], "n": len(adm),
            "all_above_fresh_switch": bool(all(c > fresh_switch for c in adm))}

    runner_up = float(np.sort(losses)[1])
    out = {"argmin_ceiling": argmin["ceiling"], "min_L_D": lo,
           "runner_up_L_D": runner_up,
           "runner_up_gap_pct": (runner_up - lo) / lo * 100,
           "fresh_switch": fresh_switch,
           "monotone_in_ceiling": bool(all(a >= b for a, b in zip(losses, losses[1:]))),
           "tolerance_sets": tol_sets, "grid": rows}

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    OUT_JSON.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


if __name__ == "__main__":
    o = run()
    print(f"argmin ceiling {o['argmin_ceiling']}  L_D {o['min_L_D']:.1f}")
    print(f"runner-up gap {o['runner_up_gap_pct']:.4f}%   monotone: {o['monotone_in_ceiling']}")
    for t, s in o["tolerance_sets"].items():
        print(f"  within {float(t)*100:.1f}%: ceilings {s['ceilings'][0]:.3f}-{s['ceilings'][1]:.3f} "
              f"(n={s['n']}), all above fresh switch: {s['all_above_fresh_switch']}")
    print(f"written: {OUT_CSV.name}, {OUT_JSON.name}")
