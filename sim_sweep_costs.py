"""
sim_sweep_costs.py — Cost-parameter sensitivity of the five-policy ranking
(this paper §8.6), addressing the reviewer request to sweep the assumed attacker
economics beyond K_E and the value spread.

One-at-a-time deviations from the declared base calibration of sim_friction_policy.P:
  c_E  in {0.25, 0.35*, 0.50}   (evasion variable unit cost)
  rho  in {0.20, 0.30*, 0.45}   (residual friction exposure; c2 = 1, so c2*rho = rho)
  K_E  in {0.25, 0.50*, 1.00}   (evasion setup cost — analytic sweep in §8.2, ranking check here)
  q    in {1.0, 1.5, 2.0*, 3.0} (human-harm exponent, L_H = h * f^q)

For each setting: same seed-42 population, re-tune the static graded schedule
by the same grid search (ceiling 0.20–1.00 step 0.025, width {0.05,0.08,0.12})
minimizing L_D, then evaluate no-defense / binary block / aggressive throttle /
tuned graded. The adaptive loop is excluded (its base-case L_D is within 0.5 of
the static graded schedule; the ranking question concerns the static policies).

Outputs: SIM_cost_sweep.json, SIM_cost_sweep_SUMMARY.md in OUT_DIR.
"""
from __future__ import annotations

import json

import numpy as np

import config
import sim_friction_policy as sim

BASE = {"c_E": 0.35, "rho": 0.30, "K_E": 0.50, "q": 2.0}
GRID = {
    "c_E": (0.25, 0.35, 0.50),
    "rho": (0.20, 0.30, 0.45),
    "K_E": (0.25, 0.50, 1.00),
    "q": (1.0, 1.5, 2.0, 3.0),
}


def eval_policy_q(phi, pop, q):
    """sim.eval_policy with a parameterized human-harm exponent."""
    P = sim.P
    f_bot = phi(pop["b_bot"])
    f_hum = phi(pop["b_hum"])
    s = sim.best_response(f_bot, pop["vA"], pop["X"], pop["sunk"])
    X = pop["X"]
    extr = float(X[(s == sim.S_V) | (s == sim.S_E)].sum())
    obs = float(X[s == sim.S_V].sum())
    val = pop["vA"] * X
    evas_p = float(val[s == sim.S_E].sum() / val.sum())
    served_f = np.where(s == sim.S_V, f_bot,
                        np.where(s == sim.S_E, P["rho"] * f_bot, 0.0))
    c_D = P["gamma_D"] * (float((served_f * X).sum()) + float(f_hum.sum()))
    l_H = P["h"] * float((f_hum ** q).sum())
    e_D = P["e_D"] * extr
    l_E = P["L_E"] * float((s == sim.S_E).sum())
    return dict(extraction=extr, human_harm=l_H, evasion_share=evas_p,
                observed=obs, L_D=e_D + c_D + l_H + l_E)


def tune_static_q(pop, q):
    best = None
    for ft in np.arange(0.20, 1.001, 0.025):
        for w in (0.05, 0.08, 0.12):
            r = eval_policy_q(sim.make_phi_graded(float(ft), width=w), pop, q)
            if best is None or r["L_D"] < best[0]:
                best = (r["L_D"], float(ft), w)
    return best[1], best[2]


def run_setting(pop, q):
    ft, w = tune_static_q(pop, q)
    pols = {
        "none": sim.phi_none,
        "binary": sim.phi_binary,
        "throttle": sim.phi_throttle,
        "graded": sim.make_phi_graded(ft, width=w),
    }
    out = {name: eval_policy_q(phi, pop, q) for name, phi in pols.items()}
    base_ex, base_ld = out["none"]["extraction"], out["none"]["L_D"]
    for name, r in out.items():
        r["extraction_idx"] = round(100 * r["extraction"] / base_ex, 1)
        r["observed_idx"] = round(100 * r["observed"] / base_ex, 1)
        r["L_D_idx"] = round(100 * r["L_D"] / base_ld, 1)
        r["human_harm"] = round(r["human_harm"], 1)
    active = ("binary", "throttle", "graded")
    flags = {
        "tuned_schedule": {"ceiling": ft, "width": w},
        "throttle_worst_active_LD": max(active, key=lambda p: out[p]["L_D"]) == "throttle",
        "graded_best_LD": min(active, key=lambda p: out[p]["L_D"]) == "graded",
        "best_dashboard_is_throttle": min(active, key=lambda p: out[p]["observed"]) == "throttle",
        "graded_vs_binary_harm_ratio": round(
            out["graded"]["human_harm"] / out["binary"]["human_harm"], 3)
        if out["binary"]["human_harm"] > 0 else None,
    }
    return out, flags


def run() -> dict:
    config.ensure_dirs()
    rng = np.random.default_rng(42)
    pop = sim.draw_population(rng, sim.P["vA_sigma"])

    results = {}
    saved = {k: sim.P[k] for k in ("c_E", "rho", "K_E")}
    for param, values in GRID.items():
        for v in values:
            # reset to base, then apply the single deviation
            for k, b in saved.items():
                sim.P[k] = b
            q = BASE["q"]
            if param == "q":
                q = v
            else:
                sim.P[param] = v
            label = f"{param}={v}" + (" (base)" if v == BASE[param] else "")
            out, flags = run_setting(pop, q)
            results[label] = {"policies": out, **flags}
            print(label, "->", {p: results[label]["policies"][p]["L_D_idx"]
                                for p in ("binary", "throttle", "graded")},
                  "| throttle worst:", flags["throttle_worst_active_LD"],
                  "graded best:", flags["graded_best_LD"])
    for k, b in saved.items():
        sim.P[k] = b

    (config.OUT_DIR / "SIM_cost_sweep.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")

    L = ["# Cost-parameter sweep of the policy ranking (one-at-a-time from base)\n",
         "| Setting | Binary L_D | Throttle L_D | Graded L_D | Throttle worst? | Graded best? | Best dashboard = throttle? | Graded/binary harm |",
         "|---|---|---|---|---|---|---|---|"]
    for label, r in results.items():
        p = r["policies"]
        L.append(f"| {label} | {p['binary']['L_D_idx']} | {p['throttle']['L_D_idx']} | "
                 f"{p['graded']['L_D_idx']} | {r['throttle_worst_active_LD']} | "
                 f"{r['graded_best_LD']} | {r['best_dashboard_is_throttle']} | "
                 f"{r['graded_vs_binary_harm_ratio']} |")
    (config.OUT_DIR / "SIM_cost_sweep_SUMMARY.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    return results


if __name__ == "__main__":
    run()
