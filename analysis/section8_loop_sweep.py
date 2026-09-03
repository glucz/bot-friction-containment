"""Section 8 across the credible loop-delay range.

`sim_friction_policy.py:57-59` hard-codes `tau_A = 4` and `tau_D = 5`, both flagged
"measured". They drive `run_loop` (the attacker best-responds to friction tau_A days old, the
defender observes with delay tau_D), the critical-gain search, and the stability figure. So
the critical gain, the oscillation period, the instability costs and the dashboard ratio are
all downstream of the delay.

The delay is **not identified**: 9 days pooled, but 4 days on the only density
band where the circular-shift null is calibrated, and 15 days on the middle band. So Section 8
should be reported across that range rather than at a point.

The sweep also tests a standing cross-paper claim recorded in RESEARCH-STATE.md, that
`k_crit * (tau_A + tau_D) ~ 11` holds between 10.5 and 11.2 over a fourfold range of loop
length. If the loop halves, that regularity predicts k_crit roughly doubles.

The published configuration (4, 5) must reproduce k_crit = 1.20 before the sweep is believed.

Read-only; imports the shipped simulator rather than reimplementing it.
Usage: python section8_loop_sweep.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from _layout import data_root, on_path, require  # noqa: E402

ES = on_path()
# `sim_friction_policy` lives beside empirical-support in the source tree and at the bundle root
# after synchronisation. Resolving only one layout leaves the shipped copy unable to start.
for _cand in (ES, Path(__file__).resolve().parent.parent, Path(__file__).resolve().parent):
    if (_cand / "sim_friction_policy.py").exists():
        sys.path.insert(0, str(_cand))
        break
else:
    sys.path.insert(0, str(ES))

import sim_friction_policy as S  # noqa: E402

PUBLISHED = {"k_crit": 1.20, "osc_period": 26.7, "loss_pct": 19.1,
             "harm_pct": 49.8, "dashboard_ratio": 1.486}
BURN = 60


def evaluate(pop, phi_static, V_target, tau_A, tau_D):
    S.P["tau_A"], S.P["tau_D"] = int(tau_A), int(tau_D)
    k_crit, _ = S.find_k_crit(pop, phi_static, V_target)
    if k_crit is None:
        return {"tau_A": tau_A, "tau_D": tau_D, "loop": tau_A + tau_D, "k_crit": None}
    k_lo, k_hi = 0.5 * k_crit, 2.0 * k_crit
    tr_lo = S.run_loop(pop, phi_static, k_lo, T=420, V_target=V_target)
    tr_hi = S.run_loop(pop, phi_static, k_hi, T=420, V_target=V_target)
    d_lo = tr_lo["daily"].iloc[BURN:]
    d_hi = tr_hi["daily"].iloc[BURN:]
    per = S.osc_period(tr_hi)

    def ratio(col):
        a, b = float(d_hi[col].mean()), float(d_lo[col].mean())
        return a / b if b else float("nan")

    return {"tau_A": int(tau_A), "tau_D": int(tau_D), "loop": int(tau_A + tau_D),
            "k_crit": float(k_crit), "k_crit_x_loop": float(k_crit * (tau_A + tau_D)),
            "osc_period": float(per), "period_over_loop": float(per / (tau_A + tau_D)),
            "loss_pct": 100 * (ratio("L_D") - 1),
            "harm_pct": 100 * (ratio("human_harm") - 1),
            "dashboard_ratio": ratio("observed"),
            "amp_unstable": float(S.osc_amplitude(tr_hi))}


def main():
    rng = np.random.default_rng(S.SEED)
    pop = S.draw_population(rng, S.P["vA_sigma"])
    ft, fw = S.tune_static(pop)
    phi_static = S.make_phi_graded(ft, width=fw)
    V_target = float(S.eval_policy(phi_static, pop)["observed"])
    print(f"tuned static menu: f_target={ft:.3f}, width={fw}, V_target={V_target:.5f}")

    # published configuration first - this is the replication gate
    base = evaluate(pop, phi_static, V_target, 4, 5)
    print("\n" + "=" * 84)
    print("REPLICATION GATE at the published (tau_A, tau_D) = (4, 5)")
    for lbl, got, exp, tol in (("k_crit", base["k_crit"], PUBLISHED["k_crit"], 0.06),
                               ("oscillation period", base["osc_period"], PUBLISHED["osc_period"], 2.0),
                               ("loss %", base["loss_pct"], PUBLISHED["loss_pct"], 2.0),
                               ("human harm %", base["harm_pct"], PUBLISHED["harm_pct"], 3.0),
                               ("dashboard ratio", base["dashboard_ratio"],
                                PUBLISHED["dashboard_ratio"], 0.05)):
        ok = got is not None and abs(got - exp) <= tol
        print(f"  {lbl:<20} got {got:>8.3f}   published {exp:>8.3f}   "
              f"{'PASS' if ok else 'CHECK'}")
    print("=" * 84)

    # the sweep: the credible range from the lead-lag estimator, plus intermediate points
    grid = [(1, 1), (2, 2), (2, 3), (3, 3), (4, 4), (4, 5), (5, 6), (7, 8), (8, 8), (10, 10)]
    rows = [evaluate(pop, phi_static, V_target, a, b) for a, b in grid]

    print(f"\n{'loop':>5}{'tauA':>6}{'tauD':>6}{'k_crit':>9}{'k*loop':>9}"
          f"{'period':>9}{'per/loop':>10}{'loss%':>9}{'harm%':>9}{'dash':>8}")
    for r in rows:
        if r.get("k_crit") is None:
            print(f"{r['loop']:>5}{r['tau_A']:>6}{r['tau_D']:>6}{'none':>9}"
                  f"   (no instability found on the gain grid)")
            continue
        print(f"{r['loop']:>5}{r['tau_A']:>6}{r['tau_D']:>6}{r['k_crit']:>9.2f}"
              f"{r['k_crit_x_loop']:>9.2f}{r['osc_period']:>9.1f}{r['period_over_loop']:>10.2f}"
              f"{r['loss_pct']:>9.1f}{r['harm_pct']:>9.1f}{r['dashboard_ratio']:>8.3f}")

    ks = [r for r in rows if r.get("k_crit")]
    prod = np.array([r["k_crit_x_loop"] for r in ks])
    print(f"\nCROSS-PAPER INVARIANT  k_crit * (tau_A + tau_D)")
    print(f"  RESEARCH-STATE records ~11, held between 10.5 and 11.2 over a fourfold range")
    print(f"  observed here: min {prod.min():.2f}, max {prod.max():.2f}, "
          f"median {np.median(prod):.2f}, spread {prod.max() - prod.min():.2f}")
    holds = bool(prod.min() >= 9.0 and prod.max() <= 13.0)
    print(f"  -> {'HOLDS (loosely)' if holds else 'DOES NOT HOLD across this range'}")

    # what does the paper's conclusion depend on?
    print(f"\nSENSITIVITY OF THE PUBLISHED SECTION 8 CLAIMS")
    for r in ks:
        if r["loop"] in (4, 9, 15):
            print(f"  loop {r['loop']:>2} d: k_crit {r['k_crit']:.2f}, period {r['osc_period']:.1f} d "
                  f"({r['period_over_loop']:.2f}x loop), instability costs "
                  f"+{r['loss_pct']:.1f}% loss / +{r['harm_pct']:.1f}% harm, "
                  f"dashboard {r['dashboard_ratio']:.3f}")

    p = Path(__file__).with_name("section8_loop_sweep.json")
    json.dump({"published": PUBLISHED, "tuned": {"f_target": ft, "width": fw,
                                                 "V_target": V_target},
               "rows": rows, "invariant_holds": holds}, open(p, "w"), indent=2, default=float)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
