"""The containment shares of record, emitted as an artifact.

Every share the manuscript quotes is computed here and written to `containment_shares.json`;
the manuscript cites the artifact rather than restating a derivation.

A type is **uncontainable at ceiling f** when some unauthorized option still pays there:

    visible :  v_A > c0 + c1 f**r
    evasion :  X (v_A - c_E - c2 rho f**r) > K,     K = K_E if fresh, 0 if already sunk

Containment is the complement, so the uncontainable set is the **union** of the two branches.
Three quantities differ and must never be conflated:

  * kappa            - whether an evasion window opens around f_min (Theorem 2's threshold)
  * v_collapse       - the value at which the connected band from f_min breaks
  * uncontainable(f) - whether any friction up to f reaches retreat

Writes `containment_shares.json`. Usage: python containment_shares.py
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


def uncontainable(vA, X, f, sunk, P):
    """Union of the two unauthorized branches at ceiling ``f``."""
    fr = f ** P["r"]
    visible = vA > P["c0"] + P["c1"] * fr
    K = np.where(sunk, 0.0, P["K_E"])
    evasion = X * (vA - P["c_E"] - P["c2"] * P["rho"] * fr) > K
    return visible | evasion


def shares(vA, X, mask):
    val = vA * X
    return {"agents_pct": round(100 * float(mask.mean()), 2),
            "value_pct": round(100 * float(val[mask].sum() / val.sum()), 2)}


def main():
    P = S.P
    rng = np.random.default_rng(S.SEED)
    pop = S.draw_population(rng, P["vA_sigma"])
    vA, X, sunk = pop["vA"], pop["X"], pop["sunk"]
    fH = float(np.sqrt(P["H_budget"] / P["h"]))
    out = {"seed": S.SEED, "n_bots": len(vA), "sunk_share": round(float(sunk.mean()), 4),
           "f_H": fH, "vA_sigma": P["vA_sigma"], "shares": {}}

    scenarios = {"fresh": np.zeros_like(sunk, bool),
                 "sunk": np.ones_like(sunk, bool),
                 "as_drawn": sunk}
    for xlab, Xv in (("X_eq_1", np.ones_like(X)), ("per_agent_X", X)):
        for slab, sk in scenarios.items():
            for flab, f in (("f_le_1", 1.0), ("f_le_fH", fH)):
                m = uncontainable(vA, Xv, f, sk, P)
                out["shares"][f"{xlab}|{slab}|{flab}"] = shares(vA, Xv, m)

    # branch decomposition at the two ceilings, fresh, per-agent X
    for flab, f in (("f_le_1", 1.0), ("f_le_fH", fH)):
        fr = f ** P["r"]
        vis = vA > P["c0"] + P["c1"] * fr
        eva = X * (vA - P["c_E"] - P["c2"] * P["rho"] * fr) > P["K_E"]
        out["shares"][f"branch|fresh|{flab}|visible_only"] = shares(vA, X, vis & ~eva)
        out["shares"][f"branch|fresh|{flab}|evasion_only"] = shares(vA, X, eva & ~vis)
        out["shares"][f"branch|fresh|{flab}|either"] = shares(vA, X, vis | eva)

    # the connected-band threshold, for contrast
    vstar = (P["K_E"] / P["X"] + P["c_E"] - P["c2"] * P["rho"] * P["c0"] / P["c1"]) \
        / (1 - P["c2"] * P["rho"] / P["c1"])
    out["v_collapse_X1"] = round(float(vstar), 4)
    out["v_collapse_share"] = shares(vA, X, vA > vstar)
    out["v_uncontainable_X1_fresh"] = round(
        float(P["c_E"] + P["c2"] * P["rho"] + P["K_E"] / P["X"]), 4)

    # The analytic mass above the classifier threshold, and the realized share in the seeded
    # draw the shipped UI displays. They are different estimands, not different vintages.
    import tomllib
    from scipy.stats import beta as _beta
    _toml = (Path(__file__).resolve().parents[1] / "frictionlab" / "frictionlab" /
             "calibrations" / "agwa_d.toml")
    # Calibration is required; if it is unavailable, refuse before writing so the artifact
    # cannot be incomplete.
    if not _toml.exists():
        raise SystemExit(
            f"the FrictionLab calibration is not available, so the human score density and the "
            f"harm-boundary share cannot be computed and this artifact must not be "
            f"rewritten without them.\n"
            f"  expected at {_toml}")
    _pp = tomllib.load(open(_toml, "rb"))["params"]
    _a = _pp["human_beta_a"]["value"]
    _b = _pp["human_beta_b"]["value"]
    _th = _pp["beta"]["value"]
    out["human_score_density"] = {"family": "beta", "a": _a, "b": _b, "threshold_beta": _th}

    # TWO DISTINCT ESTIMANDS, recorded separately because they are not a stale/current pair.
    # The analytic mass is a property of the assumed density; the seeded draw is one realized
    # simulated population, and the shipped UI displays that one. Collapsing them into a
    # single "human mass" field is how a difference in kind reads as a difference in vintage.
    import hashlib as _hl
    _n = int(_pp["n_humans"]["value"])
    _seed = int(_pp["seed"]["value"]) if "seed" in _pp else None
    out["human_mass_above_beta_pct_analytic"] = round(100.0 * float(1.0 - _beta.cdf(_th, _a, _b)), 2)
    out["human_beta_a"] = _a
    out["human_beta_b"] = _b
    out["beta"] = _th
    out["human_draw_n"] = _n
    out["human_draw_seed"] = _seed
    out["calibration_path"] = _toml.name
    out["calibration_sha256"] = _hl.sha256(_toml.read_bytes()).hexdigest()
    try:
        import sys as _s2
        _s2.path.insert(0, str(_toml.resolve().parents[2]))
        import frictionlab as _fl
        from frictionlab.population import draw as _draw
        _pop = _draw(_fl.Calibration.load("agwa_d"))
        _k = int((_pop.b_human > _th).sum())
        out["human_draw_above_beta_count"] = _k
        out["human_draw_above_beta_pct"] = round(100.0 * _k / len(_pop.b_human), 4)
        # Volume-weighted share of the simulated bot stream scoring below the threshold: the
        # traffic a graded schedule leaves alone, quoted alongside the agent-weighted shares.
        _below = _pop.b_bot <= _th
        _vol = _pop.X
        out["bot_volume_below_beta_pct"] = round(
            100.0 * float(_vol[_below].sum()) / float(_vol.sum()), 2)
    except Exception as _exc:                                   # noqa: BLE001
        raise SystemExit(
            f"the seeded human draw could not be reproduced ({type(_exc).__name__}: {_exc}). "
            f"It is a published quantity, so this artifact must not be written without it.")
    # kept under its historical name so existing bindings resolve to the analytic value
    out["human_mass_above_beta_pct"] = out["human_mass_above_beta_pct_analytic"]
    print(f"  human mass above beta={_th}: "
          f"{out['human_mass_above_beta_pct_analytic']:.2f}% analytic, "
          f"{out['human_draw_above_beta_pct']:.2f}% in the seed-{_seed} draw "
          f"({out['human_draw_above_beta_count']:,}/{_n:,})")

    p = Path(__file__).with_name("containment_shares.json")
    json.dump(out, open(p, "w"), indent=2)

    print(f"population {out['n_bots']:,} bots, {100*out['sunk_share']:.1f}% already sunk, "
          f"f_H = {fH:.2f}")
    print(f"connected-band threshold v* = {out['v_collapse_X1']}  "
          f"(share above it: {out['v_collapse_share']['agents_pct']}% agents / "
          f"{out['v_collapse_share']['value_pct']}% value)")
    print(f"fresh uncontainable boundary at X=1: v_A > {out['v_uncontainable_X1_fresh']}")
    print(f"\n{'convention':<34}{'ceiling':<10}{'agents':>9}{'value':>9}")
    for k in sorted(out["shares"]):
        if k.startswith("branch"):
            continue
        x, s, f = k.split("|")
        v = out["shares"][k]
        print(f"{x + ', ' + s:<34}{f:<10}{v['agents_pct']:>8.1f}%{v['value_pct']:>8.1f}%")
    print(f"\nbranch decomposition (fresh, per-agent X):")
    for k in sorted(x for x in out["shares"] if x.startswith("branch")):
        v = out["shares"][k]
        print(f"  {k:<44}{v['agents_pct']:>8.1f}%{v['value_pct']:>8.1f}%")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
