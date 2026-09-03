"""Sensitivity of the five-policy comparison to the initial sunk-evasion share.

Why
---
`sim_friction_policy.P["mix_evasive"]` sets what fraction of simulated agents
start with their evasion setup cost already sunk (`draw_population`, the `sunk`
draw), which drives the ratchet of Section 6.1 and therefore how much of the
population sits on the stealth side before the defender does anything.

Its value, 0.78, came from a Gaussian mixture fitted to the *curated declared*
crawler set - agents that advertise themselves as automated - and the cluster it
counts was assigned by highest network spread (IP / domain / country fan-out).
Neither the population nor the construct supports reading it as the initial
evasive share of abusive automation, and no equivalent measurement is available:
the abusive population's derived series carry spread as raw daily counts rather
than entropies, and three of the ten profile features do not exist there at all.

So the parameter is reclassified as assumed, and this sweep reports what the
Section 8 conclusions depend on across its plausible range. Note that
`obs_collapse_did` is NOT swept because it is never read by the simulation - it
is recorded in the calibration block but does not enter any computation.

Usage: python sim_sweep_mix_evasive.py
Writes: outputs/SIM_mix_evasive_sweep.json (+ markdown)
"""
from __future__ import annotations

import json

import numpy as np

import config
import sim_friction_policy as sim

GRID = [0.20, 0.40, 0.60, 0.78, 0.90]
OUT_JSON = config.OUT_DIR / "SIM_mix_evasive_sweep.json"
OUT_MD = config.OUT_DIR / "SIM_mix_evasive_sweep.md"


def one(mix: float) -> dict:
    sim.P["mix_evasive"] = mix
    sim.P["mix_visible"] = 1.0 - mix
    rng = np.random.default_rng(config.RANDOM_STATE)
    pop = sim.draw_population(rng, sim.P["vA_sigma"])

    f_target, width = sim.tune_static(pop)          # returns (f_target, width)
    phi_graded = sim.make_phi_graded(f_target, width=width)
    tuned = {"f_target": float(f_target), "width": float(width)}
    policies = {
        "none": sim.phi_none,
        "binary": sim.phi_binary,
        "throttle": sim.phi_throttle,
        "graded": phi_graded,
    }
    base = sim.eval_policy(sim.phi_none, pop)
    out = {"mix_evasive": mix, "tuned": tuned, "policies": {}}
    for name, phi in policies.items():
        r = sim.eval_policy(phi, pop)
        out["policies"][name] = {
            "extraction_idx": 100.0 * r["extraction"] / base["extraction"],
            "observed_idx": 100.0 * r["observed"] / base["observed"],
            "L_D_idx": 100.0 * r["L_D"] / base["L_D"],
            "human_harm": r["human_harm"],
            "evasion_value_share": r["evasion_prob"],
            "retreat_share": r["retreat_share"],
        }
    p = out["policies"]
    active = {k: v for k, v in p.items() if k != "none"}
    best_dash = min(active, key=lambda k: active[k]["observed_idx"])
    best_loss = min(active, key=lambda k: active[k]["L_D_idx"])
    out["best_dashboard_policy"] = best_dash
    out["best_loss_policy"] = best_loss
    out["best_dashboard_is_not_best_loss"] = best_dash != best_loss
    out["graded_beats_binary_on_harm"] = (
        p["graded"]["human_harm"] < p["binary"]["human_harm"])
    out["harm_ratio_graded_over_binary"] = (
        p["graded"]["human_harm"] / p["binary"]["human_harm"])
    return out


def run() -> dict:
    config.ensure_dirs()
    original = sim.P["mix_evasive"]
    rows = [one(m) for m in GRID]
    sim.P["mix_evasive"] = original
    res = {"grid": GRID, "rows": rows,
           "robust_best_dashboard_never_best_loss": all(
               r["best_dashboard_is_not_best_loss"] for r in rows),
           "robust_graded_beats_binary_on_harm": all(
               r["graded_beats_binary_on_harm"] for r in rows)}
    json.dump(res, open(OUT_JSON, "w"), indent=2, default=float)

    L = ["# Sensitivity to the initial sunk-evasion share\n\n",
         "| mix_evasive | best dashboard | best total loss | differ | graded/binary harm | evasion value share (graded) |\n",
         "|---|---|---|---|---|---|\n"]
    for r in rows:
        L.append(f"| {r['mix_evasive']:.2f} | {r['best_dashboard_policy']} | "
                 f"{r['best_loss_policy']} | {r['best_dashboard_is_not_best_loss']} | "
                 f"{r['harm_ratio_graded_over_binary']:.3f} | "
                 f"{r['policies']['graded']['evasion_value_share']:.3f} |\n")
    L.append(f"\n- best-dashboard policy is never the best-loss policy: "
             f"**{res['robust_best_dashboard_never_best_loss']}**\n")
    L.append(f"- graded schedule beats binary blocking on human harm: "
             f"**{res['robust_graded_beats_binary_on_harm']}**\n")
    open(OUT_MD, "w", encoding="utf-8").writelines(L)
    print("".join(L))
    return res


if __name__ == "__main__":
    run()
