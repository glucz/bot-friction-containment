"""Is k_crit * T a structural constant, or a property of one schedule?

An earlier sweep found k_crit * (tau_A + tau_D) in 10.20-11.25 across loop lengths, which invites
the operational rule "use 11/T". That rule does not transfer, for two reasons: the constant is
schedule-dependent, and Theorem 5 governs the NORMALIZED loop gain k_phi * s * T, where s is the
local slope of the signal the loop controls. Dropping s drops the quantity the theorem bounds.

This tests both directly rather than accepting either claim:

  1. sweep the schedule (ceiling and width) at fixed T and see whether k_crit * T moves;
  2. estimate the local slope s of the controlled signal at each schedule's operating point,
     and see whether k_crit * s * T is stabler than k_crit * T. s is measured in the
     coordinate the loop actually moves (the schedule multiplier m), normalized by V0.

Read-only; imports the shipped simulator. Usage: python kcrit_schedule_sensitivity.py
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

BURN = 60


def local_slope(pop, ceiling, width, h=0.02):
    """|d (V_obs / V0) / dm| at m = 1 on the STATIC, PRE-RATCHET population.

    Read the caveat below before using this number. It is a static normalization, not the
    loop's operating-state sensitivity.

    The derivative is taken with respect to the schedule MULTIPLIER, not the ceiling. The
    simulated controller never moves the ceiling; it scales the whole menu by a multiplier m,

        f_i(t) = min(1, m_t * phi_base(b_i)),
        m_{t+1} = clip(m_t + k_phi * (V_obs(t - tau_D) - V*) / V0, 0, 2.5),

    (sim_friction_policy.run_loop:276-324) so Theorem 5's sensitivity s is the response of
    the V0-normalized dashboard signal to m, not to the ceiling. This function evaluates that
    response at m = 1, which is where find_k_crit's target sits (V* = eval_policy(phi_base,
    pop), the static observed volume there) -- but see the CAVEAT below before reading that
    as the loop's operating point, because it is not.

    The normalization is V0 = sum(X), the no-defense visible volume the loop itself divides
    by, rather than a hard-coded constant.

    CAVEAT, and it is fatal to the stronger reading. One might argue
    that m = 1 is the loop's operating point "by construction", because find_k_crit passes
    V* = eval_policy(phi_base, pop), the static observed volume at m = 1. That argument is
    wrong. run_loop starts at m0 = 0, recomputes best responses along the path, and
    PERMANENTLY sets `sunk` whenever a type first enters evasion, so the response function
    the loop ends up on is not the one differentiated here. `replay_operating_state` below
    MEASURES both halves and writes them into the JSON, rather than asserting them here.

    Two labelling points that must not be elided. The multipliers it
    reports are FINAL-WINDOW MEANS, not settled fixed points: at and above the critical gain
    the loop oscillates by the very criterion that located that gain, so there is no fixed
    point to quote. And the difference between the static slope at m = 1 and at a nearby
    multiplier is CURVATURE OF THE STATIC RESPONSE SURFACE, not sensitivity to the ratcheted
    state, because both evaluations use the same pre-ratchet population. It is a reason to
    distrust a single static slope, not evidence about the ratcheted one.

    So what this function returns is a reproducible pre-ratchet static normalization. It is
    NOT the normalized gain at the state that generated k_crit, and the manuscript no longer
    claims that normalizing by it puts the empirical threshold on Theorem 5's coordinate.
    Computing the honest version means capturing each cell's settled sunk state and settled
    multiplier -- and at the critical gain the multiplier oscillates by construction, so a
    single operating point is not even well defined there. Left undone deliberately.
    """
    phi_base = S.make_phi_graded(ceiling, width=width)
    V0 = float(pop["X"].sum())

    def obs(m):
        return float(S.eval_policy(lambda b: np.minimum(1.0, m * phi_base(b)), pop)["observed"])

    return abs(obs(1.0 + h) - obs(1.0 - h)) / (2.0 * h) / V0


def replay_operating_state(pop, ceiling, width, k_crit, multiples=(0.5, 1.0, 2.0)):
    """Measure what the loop actually does, instead of asserting it in prose.

    The counts and multipliers the caveat rests on are emitted as fields rather than written into
    a docstring, so a rerun reproduces them. This runs the reference schedule at the given
    multiples of its critical gain and reports, per run:

      * ``K_paid_total`` / ``entry_days`` -- whether the evasion-entry ratchet fires at all;
      * ``m_final_window_mean`` and its min/max -- the mean multiplier over the last 150 days
        and the swing around it. The mean is NOT a settled fixed point at or above k_crit,
        where the loop oscillates, which is exactly why no operating-state slope is quoted;
      * ``s_static_at_1`` and ``s_static_at_that_mean`` -- both evaluated on the pre-ratchet
        population, so their gap is curvature of the static surface and nothing more.
    """
    phi_base = S.make_phi_graded(ceiling, width=width)
    V0 = float(pop["X"].sum())
    V_target = float(S.eval_policy(phi_base, pop)["observed"])

    def obs(m):
        return float(S.eval_policy(lambda b: np.minimum(1.0, m * phi_base(b)), pop)["observed"])

    def static_slope_at(m0, h=0.02):
        return abs(obs(m0 + h) - obs(m0 - h)) / (2.0 * h) / V0

    out = []
    for mult in multiples:
        tr = S.run_loop(pop, phi_base, mult * k_crit, T=420, V_target=V_target)
        paid = tr["daily"]["K_paid"].to_numpy()
        tail = tr["m"][-150:]
        mean_m = float(tail.mean())
        out.append({
            # The delays actually in force. The sweep above mutates these module-level values
            # and once left them at (7, 8), so this block silently measured a 15-day loop and
            # reported multipliers pinned to the clip bounds. Nothing in the JSON could have
            # revealed that. Emitting them makes the failure visible forever.
            "tau_A_used": float(S.P["tau_A"]),
            "tau_D_used": float(S.P["tau_D"]),
            "T_used": float(S.P["tau_A"] + S.P["tau_D"]),
            "gain_multiple_of_k_crit": float(mult),
            "K_paid_total": float(paid.sum()),
            "entry_days": int((paid > 0).sum()),
            "m_final_window_mean": mean_m,
            "m_final_window_min": float(tail.min()),
            "m_final_window_max": float(tail.max()),
            "m_is_a_fixed_point": bool(tail.max() - tail.min() < 1e-3),
            "s_static_at_1": static_slope_at(1.0),
            "s_static_at_that_mean": static_slope_at(mean_m),
        })
    return out


def main():
    rng = np.random.default_rng(S.SEED)
    pop = S.draw_population(rng, S.P["vA_sigma"])
    ft, fw = S.tune_static(pop)
    print(f"tuned reference schedule: ceiling {ft:.3f}, width {fw}")

    rows = []
    grid = [(ft, fw, "reference (tuned)"),
            (ft, fw * 2, "width doubled"),
            (ft, fw / 2, "width halved"),
            (0.85, fw, "ceiling 0.85"),
            (0.70, fw, "ceiling 0.70"),
            (0.55, fw, "ceiling 0.55")]
    for tA, tD in ((4, 5), (2, 2), (7, 8)):
        S.P["tau_A"], S.P["tau_D"] = tA, tD
        T = tA + tD
        for ceiling, width, label in grid:
            phi = S.make_phi_graded(ceiling, width=width)
            V = float(S.eval_policy(phi, pop)["observed"])
            k, _ = S.find_k_crit(pop, phi, V)
            if k is None:
                rows.append({"T": T, "schedule": label, "ceiling": ceiling,
                             "width": width, "k_crit": None})
                continue
            s = local_slope(pop, ceiling, width)
            rows.append({"T": T, "schedule": label, "ceiling": round(ceiling, 3),
                         "width": round(width, 4), "k_crit": float(k),
                         "k_times_T": float(k * T), "s": float(s),
                         "k_s_T": float(k * s * T)})

    ok = [r for r in rows if r.get("k_crit")]
    print(f"\n{'T':>4}{'schedule':<20}{'ceiling':>9}{'width':>8}{'k_crit':>9}"
          f"{'k*T':>9}{'s':>12}{'k*s*T':>12}")
    for r in rows:
        if r.get("k_crit") is None:
            print(f"{r['T']:>4}{r['schedule']:<20}{r['ceiling']:>9}{r['width']:>8}"
                  f"{'none':>9}")
            continue
        print(f"{r['T']:>4}{r['schedule']:<20}{r['ceiling']:>9.3f}{r['width']:>8.3f}"
              f"{r['k_crit']:>9.2f}{r['k_times_T']:>9.2f}{r['s']:>12.4f}{r['k_s_T']:>12.2f}")

    kT = np.array([r["k_times_T"] for r in ok])
    ksT = np.array([r["k_s_T"] for r in ok])

    def cv(x):
        return float(np.std(x) / np.mean(x))

    print(f"\nacross all {len(ok)} (T, schedule) cells:")
    print(f"  k_crit * T      min {kT.min():7.2f}  max {kT.max():7.2f}  "
          f"spread {kT.max()/kT.min():.2f}x  CV {cv(kT):.3f}")
    print(f"  k_crit * s * T  min {ksT.min():7.2f}  max {ksT.max():7.2f}  "
          f"spread {ksT.max()/ksT.min():.2f}x  CV {cv(ksT):.3f}")

    ref = [r for r in ok if r["schedule"] == "reference (tuned)"]
    kTr = np.array([r["k_times_T"] for r in ref])
    print(f"\n  holding the schedule FIXED at the tuned one, across T:")
    print(f"    k_crit * T  min {kTr.min():.2f}  max {kTr.max():.2f}  CV {cv(kTr):.3f}")

    verdict = ("k_crit * T is stable only at a fixed schedule"
               if cv(kT) > 2 * cv(kTr) else "k_crit * T looks schedule-robust")
    print(f"\n  -> {verdict}")

    p = Path(__file__).with_name("kcrit_schedule_sensitivity.json")
    # The sweep above mutates the module-level delays and leaves them at the LAST cell
    # (7, 8). Replaying without restoring them runs the reference schedule at T = 15 and
    # reports multipliers that slam into both clip bounds -- which is a property of that
    # delay, not of the reference loop. Restore 4/5 explicitly before measuring.
    S.P["tau_A"], S.P["tau_D"] = 4, 5
    ref = [r for r in ok if r["schedule"] == "reference (tuned)" and r["T"] == 9]
    replay = replay_operating_state(pop, ft, fw, ref[0]["k_crit"]) if ref else []
    if replay:
        print("\n  replay of the reference loop (the caveat's claims, measured):")
        for r in replay:
            swing = r["m_final_window_max"] - r["m_final_window_min"]
            print(f"    {r['gain_multiple_of_k_crit']:>4}x k_crit  "
                  f"K_paid {r['K_paid_total']:>7.1f} on {r['entry_days']:>3} d  "
                  f"m mean {r['m_final_window_mean']:.4f} "
                  f"[{r['m_final_window_min']:.4f}, {r['m_final_window_max']:.4f}]  "
                  f"swing {swing:.4f}")

    json.dump({"replay_operating_state": replay,
               "caveat": (
                   "s is computed at m=1 on the static pre-ratchet population. It is NOT the "
                   "loop's operating-state sensitivity: run_loop starts at m0=0 and ratchets "
                   "evasion entry, so the surface it settles on is not the one differentiated. "
                   "replay_operating_state holds the measured evidence. Note that its "
                   "multipliers are FINAL-WINDOW MEANS and not fixed points, since the loop "
                   "oscillates at and above k_crit, and that the gap between s_static_at_1 and "
                   "s_static_at_that_mean is curvature of the STATIC surface, not the ratcheted "
                   "state's slope. ksT is therefore a static normalization and does not put "
                   "k_crit on Theorem 5's coordinate."),
               "rows": rows,
               "kT": {"min": float(kT.min()), "max": float(kT.max()), "cv": cv(kT)},
               "ksT": {"min": float(ksT.min()), "max": float(ksT.max()), "cv": cv(ksT)},
               "fixed_schedule_kT": {"min": float(kTr.min()), "max": float(kTr.max()),
                                     "cv": cv(kTr)}},
              open(p, "w"), indent=2, default=float)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
