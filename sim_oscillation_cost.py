"""Cost of running the friction loop above its critical gain.

Section 8.3 quotes the penalty an unstable update loop pays: higher total loss
and much higher human harm, at an essentially unchanged time-averaged dashboard.
Those numbers depend on the loop delay tau_A + tau_D, which changed when the
treatment arm was redefined by honeypot share (tau_A 7 -> 4 days), so they are
recomputed here rather than carried over.

`sim_friction_policy.py` does not itself write the summary file its docstring
names, so this script produces the stability numbers as an output of record.

Usage: python sim_oscillation_cost.py
Writes: outputs/SIM_oscillation_cost.json
"""
from __future__ import annotations

import json
import warnings

import numpy as np

import config
import sim_friction_policy as sim

warnings.filterwarnings("ignore")


def run() -> dict:
    config.ensure_dirs()
    rng = np.random.default_rng(config.RANDOM_STATE)
    pop = sim.draw_population(rng, sim.P["vA_sigma"])
    ft, w = sim.tune_static(pop)
    phi = sim.make_phi_graded(ft, width=w)
    V_target = sim.eval_policy(phi, pop)["observed"]

    k_crit, _ = sim.find_k_crit(pop, phi, V_target)
    out = {"tau_A": sim.P["tau_A"], "tau_D": sim.P["tau_D"],
           "loop_delay_days": sim.P["tau_A"] + sim.P["tau_D"],
           "k_crit": float(k_crit), "tuned_f_target": float(ft)}

    for label, k in (("stable", 0.5 * k_crit), ("unstable", 2.0 * k_crit)):
        traj = sim.run_loop(pop, phi, k, V_target=V_target)
        burn = traj["daily"].iloc[60:]          # run_loop returns dict with a daily frame
        out[label] = {"k": float(k),
                      "L_D_mean": float(burn["L_D"].mean()),
                      "human_harm_mean": float(burn["human_harm"].mean()),
                      "observed_mean": float(burn["observed"].mean()),
                      "amplitude": float(sim.osc_amplitude(traj)),
                      "period_days": float(sim.osc_period(traj))}

    out["loss_penalty_pct"] = round(
        100 * (out["unstable"]["L_D_mean"] / out["stable"]["L_D_mean"] - 1), 1)
    out["harm_penalty_pct"] = round(
        100 * (out["unstable"]["human_harm_mean"] / out["stable"]["human_harm_mean"] - 1), 1)
    out["dashboard_ratio"] = round(
        out["unstable"]["observed_mean"] / out["stable"]["observed_mean"], 4)
    out["period_over_loop_delay"] = round(
        out["unstable"]["period_days"] / out["loop_delay_days"], 2)

    json.dump(out, open(config.OUT_DIR / "SIM_oscillation_cost.json", "w"), indent=2)
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    run()
