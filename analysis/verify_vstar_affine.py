"""Verify the closed-form comparative statics that Sec. 4.5 and S.12 publish.

Covers three things, all read against the calibration of record in
empirical-support/sim_friction_policy.py rather than restating it:

  1. v* (the critical attacker value) is exactly affine in the evasion entry
     cost K_E, and the K_E at which v* crosses the calibration's reference
     attacker value (S.6.2).
  2. Which defender-side quantities move kappa and in which direction: r
     cancels (R1), c1 enters with the unhelpful sign, c2*rho is the lever that
     works (S.12).
  3. The landmark ordering f_sunk < f_switch < f_min for an evasion-capable
     type (Sec. 4.5, S.12).

Usage: python verify_vstar_affine.py
"""
import importlib.util
import pathlib
import sys

_here = pathlib.Path(__file__).resolve()
for _c in (_here.parents[3] / "empirical-support" / "sim_friction_policy.py",   # source tree
           _here.parent.parent / "sim_friction_policy.py",                      # public bundle
           _here.parent / "sim_friction_policy.py"):
    if _c.exists():
        SIM = _c
        break
else:
    raise SystemExit(f"sim_friction_policy.py not found from {_here}; looked in the source tree "
                     f"and in the public bundle")
spec = importlib.util.spec_from_file_location("simfp", SIM)
m = importlib.util.module_from_spec(spec)
sys.modules["simfp"] = m
spec.loader.exec_module(m)
P = m.P

c0, c1, c2, rho, cE, KE, X, r = (P["c0"], P["c1"], P["c2"], P["rho"],
                                 P["c_E"], P["K_E"], P["X"], P["r"])
v_ref = P["vA_median"]


def kappa(vA, c1_=None, c2rho=None):
    """Containment threshold (Theorem 2), with c1 and the product c2*rho free."""
    c1_ = c1 if c1_ is None else c1_
    c2rho = c2 * rho if c2rho is None else c2rho
    return X * (vA - cE - c2rho * (vA - c0) / c1_)


def v_star_root(K=KE):
    """v* by bisection on kappa(v) = K, assuming no closed form."""
    lo, hi = c0, 100.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if kappa(mid) < K:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def v_star_closed(K=KE):
    """Closed form: v* = (K/X + cE - c2 rho c0/c1) / (1 - c2 rho/c1)."""
    return (K / X + cE - c2 * rho * c0 / c1) / (1 - c2 * rho / c1)


def f_switch(K=KE):
    return ((K / X + cE - c0) / (c1 - c2 * rho)) ** (1.0 / r)


def f_sunk():
    return ((cE - c0) / (c1 - c2 * rho)) ** (1.0 / r)


def f_min(vA):
    return ((vA - c0) / c1) ** (1.0 / r)


print("calibration: c0=%s c1=%s c2=%s rho=%s c_E=%s K_E=%s X=%s r=%s"
      % (c0, c1, c2, rho, cE, KE, X, r))
print("reference attacker value v_ref = %s (median of an ASSUMED lognormal)" % v_ref)

print("")
print("-- 1a. published landmarks --")
print("v*(K_E=%s)   numeric=%.4f  closed=%.4f   [paper: 1.100]"
      % (KE, v_star_root(KE), v_star_closed(KE)))
print("v*(K_E=%s)  numeric=%.4f  closed=%.4f   [paper: 0.767]"
      % (KE / 2, v_star_root(KE / 2), v_star_closed(KE / 2)))
print("f_switch fresh      %.4f                  [paper: 0.886]" % f_switch(KE))
print("f_switch sunk       %.4f                  [paper: 0.426]" % f_switch(0.0))

print("")
print("-- 1b. affinity of v* in K_E, and the crossing --")
slope = 1.0 / (X * (1 - c2 * rho / c1))
icept = (cE - c2 * rho * c0 / c1) / (1 - c2 * rho / c1)
print("v*(K_E) = %.5f + %.5f * K_E   (exact, from the closed form)" % (icept, slope))
maxerr = max(abs(v_star_root(K) - v_star_closed(K))
             for K in (0.0, 0.05, 0.125, 0.25, 0.375, 0.5, 0.75, 1.0))
print("max |numeric - closed| over K_E in [0,1]: %.2e" % maxerr)
K_cross = X * ((1 - c2 * rho / c1) * v_ref - cE + c2 * rho * c0 / c1)
print("v* = v_ref = %s at K_E = %.6f  (= %.4f x the calibrated K_E)"
      % (v_ref, K_cross, K_cross / KE))
print("ILLUSTRATION ONLY, tier [S]: the roundness follows from round calibration")
print("choices (K_E=0.50, v_ref=0.60, c2*rho/c1=0.25 exactly).")

print("")
print("-- 2. S.12: which defender-side quantities move kappa (at v_A = 1.2) --")
vA = 1.2
print("d kappa/d c1       = %+.4f   (>0: a sharper visible instrument HURTS)"
      % (X * c2 * rho * (vA - c0) / c1 ** 2))
print("d kappa/d (c2 rho) = %+.4f   (<0: un-sheddable friction HELPS)"
      % (-X * (vA - c0) / c1))
for cc in (1.10, 1.20, 3.00):
    print("   c1=%4.2f    -> kappa=%.4f" % (cc, kappa(vA, c1_=cc)))
for cr in (0.30, 0.60, 1.19):
    k = kappa(vA, c2rho=cr)
    print("   c2rho=%4.2f -> kappa=%+.4f   containable (K_E=%s > kappa)? %s"
          % (cr, k, KE, KE > k))
print("   attainable floor at c1 = v_A-c0 = %.2f (so f_min=1): kappa=%.4f = X(v_A-c_E-c2rho)=%.4f"
      % (vA - c0, kappa(vA, c1_=vA - c0), X * (vA - cE - c2 * rho)))
print("   r-invariance (R1): kappa does not contain r, so kappa=%.6f for every r"
      % kappa(vA))

print("")
print("-- 3. Sec. 4.5 / S.12: landmark ordering, evasion-capable type --")
print("f_sunk=%.4f < f_switch=%.4f : %s  (differ by K_E/X > 0)"
      % (f_sunk(), f_switch(KE), f_sunk() < f_switch(KE)))
for vA in (1.1001, 1.2, 1.5, 2.0):
    print("   v_A=%-7.4f f_min=%.4f > f_switch=%.4f ? %s"
          % (vA, f_min(vA), f_switch(KE), f_min(vA) > f_switch(KE)))
print("   f_switch carries no v_A; f_min rises in v_A; they coincide at v*, so every")
print("   type strictly above v* has f_min > f_switch.")
