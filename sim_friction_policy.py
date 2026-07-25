"""
sim_friction_policy.py -- Calibrated policy simulation for this paper, Section 8.

"Adversarial Friction Menus for Bot Containment: Graded Defense and the
 Overdeterrence Trap"

Implements the Section 8 simulation with all measured parameters from
Section 7.9 (AGWA corpus) and explicitly flagged assumed parameters.

HONESTY NOTE (also in SIM_SUMMARY.md): this simulation demonstrates the
MODEL'S internal logic under measured delays and densities. It is NOT an
independent empirical finding. Measured quantities (delays, densities,
strategy mix, retreat fractions, volume CV) discipline the inputs; the
attacker cost parameters are assumed and the outputs are model outputs.

Deliverables (SIM_ prefix, under outputs/figures and outputs/tables):
  1. SIM_containment_map.png      -- the paper's central figure
  2. SIM_policy_comparison.csv    -- five-policy comparison table
  3. SIM_stability_timedomain.png -- stable vs unstable gain, daily loop
  4. K_E sensitivity (band collapse) + v_A-spread robustness -> SIM_SUMMARY.md
  5. outputs/SIM_SUMMARY.md       -- parameter table + all headline numbers

Run from empirical-support/.  Seed: 42 everywhere.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
FIG = OUT / "figures"
TAB = OUT / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

SEED = 42

# ----------------------------------------------------------------------------
# Parameters.  flag: "measured" = from Section 7.9 / AGWA tables,
#                    "assumed"  = modeling choice, documented.
# ----------------------------------------------------------------------------
P = {
    # --- measured (Section 7.9) ---
    "tau_A": 7,        # attacker adaptation delay, days (measured, median)
    "tau_D": 5,        # defender reaction delay, days (measured)
    "retreat_light": 0.141,   # measured retreat fraction, light friction
    "retreat_heavy": 0.073,   # measured retreat fraction, heavy friction
    "mix_visible": 0.22,      # measured initial strategy mix (bot class)
    "mix_evasive": 0.78,
    "vol_cv": 1.22,    # measured bot daily-volume CV
    "tau_op": 0.234,   # measured operating threshold of the classifier
    "obs_collapse_did": -0.23,  # measured observability collapse (anchor for L_E)
    # --- attacker economics (assumed) ---
    "c0": 0.10,    # baseline visible-extraction unit cost
    "c1": 1.20,    # friction cost scale, visible
    "r": 1.5,      # friction cost curvature (power law c_f = c1 * f^r)
    "c_E": 0.35,   # evasion variable unit cost
    "c2": 1.00,    # residual-friction cost scale under evasion
    "rho": 0.30,   # residual friction exposure under evasion, in (0,1)
    "K_E": 0.50,   # evasion fixed setup cost
    "p_C": 1.25,   # compliant-channel unit price (priced above evasion's
                   # effective cost for all feasible f; consistent with the
                   # measured non-adoption of compliance, Section 7.7)
    "X": 1.0,      # per-period volume cap (normalized; per-agent cap drawn around it)
    # --- attacker value distribution (assumed) ---
    "vA_median": 0.60,
    "vA_sigma": 0.60,       # lognormal sigma (alt spread 1.0 for robustness)
    "vA_sigma_alt": 1.00,
    # --- defender loss (assumed scales; structure from Section 3.4) ---
    "gamma_D": 0.05,  # infrastructure cost per unit friction-volume served
    "h": 1.00,        # human-harm scale, L_H = h * f^2 per unit human volume
    "e_D": 1.00,      # extraction loss per unit successfully extracted
    "L_E": 0.25,      # fixed per-agent evasion-adaptation loss (anchored, see note)
    "H_budget": 0.25,  # per-user human-harm budget -> human-harm boundary f_H
    # --- population (assumed sizes; densities measured/assumed as flagged) ---
    "n_bots": 12000,
    "n_humans": 12000,
    "human_beta_a": 1.5,   # ASSUMED human score density Beta(1.5, 8)
    "human_beta_b": 8.0,
}
P["f_H"] = float(np.sqrt(P["H_budget"] / P["h"]))  # human-harm boundary

S_V, S_E, S_C, S_R = 0, 1, 2, 3
STRAT_NAMES = ["visible", "evasion", "comply", "retreat"]

# ----------------------------------------------------------------------------
# Empirical botness density (measured): outputs/tables/E_botness_density.csv
# ----------------------------------------------------------------------------
def botness_sampler(rng: np.random.Generator):
    dens = pd.read_csv(TAB / "E_botness_density.csv")
    centers = dens["botness"].to_numpy()
    w = dens["density"].to_numpy()
    w = w / w.sum()
    width = centers[1] - centers[0]

    def sample(n: int) -> np.ndarray:
        idx = rng.choice(len(centers), size=n, p=w)
        return np.clip(centers[idx] + rng.uniform(-width / 2, width / 2, n), 0, 1)

    return sample, centers, w / width  # also return density for the marginal strip


# ----------------------------------------------------------------------------
# Analytic curves
# ----------------------------------------------------------------------------
def f_min(vA: np.ndarray) -> np.ndarray:
    """Zero-profit friction for visible extraction: v_A = c0 + c1 f^r."""
    x = np.maximum(np.asarray(vA, dtype=float) - P["c0"], 0.0) / P["c1"]
    return x ** (1.0 / P["r"])


def f_switch(K_E: float) -> float:
    """U_E = U_V indifference at the volume cap:
       c1 f^r - c2 rho f = K_E/X + c_E - c0.
    NOTE: with utilities linear in volume and a common cap, the indifference
    friction is independent of v_A (both utilities share the v_A*x term);
    the v_A dependence of the band's effective upper edge comes from the
    evasion-profitability boundary U_E = 0 (see containment map).
    Returns np.inf if no solution in (0, 1] (evasion never preferred)."""
    rhs = K_E / P["X"] + P["c_E"] - P["c0"]
    g = lambda f: P["c1"] * f ** P["r"] - P["c2"] * P["rho"] * f - rhs
    if g(1.0) < 0:
        return np.inf
    return float(brentq(g, 1e-9, 1.0))


def utilities(f, vA, X, sunk):
    """Per-agent utilities of (s_V, s_E, s_C, s_R). Volume at the cap when the
    unit margin is positive, else zero. sunk=True -> K_E already paid."""
    f = np.asarray(f, float)
    uV = np.maximum(vA - P["c0"] - P["c1"] * f ** P["r"], 0.0) * X
    K = np.where(sunk, 0.0, P["K_E"])
    uE = np.maximum(vA - P["c_E"] - P["c2"] * P["rho"] * f, 0.0) * X - K
    uC = np.maximum(vA - P["p_C"], 0.0) * X
    uR = np.zeros_like(uV)
    return np.stack([uV, uE, uC, uR], axis=0)


def best_response(f, vA, X, sunk):
    u = utilities(f, vA, X, sunk)
    s = np.argmax(u, axis=0)
    # an active strategy is only chosen at strictly positive utility;
    # u <= 0 means the optimal volume is zero -> retreat
    s = np.where(np.take_along_axis(u, s[None, :], axis=0)[0] <= 0.0, S_R, s)
    return s


# ----------------------------------------------------------------------------
# Population
# ----------------------------------------------------------------------------
def draw_population(rng: np.random.Generator, vA_sigma: float):
    sample_b, centers, dens = botness_sampler(rng)
    nb, nh = P["n_bots"], P["n_humans"]
    b_bot = sample_b(nb)
    b_hum = rng.beta(P["human_beta_a"], P["human_beta_b"], nh)
    vA = rng.lognormal(np.log(P["vA_median"]), vA_sigma, nb)
    # per-agent volume cap with measured CV 1.22, mean 1
    s2 = np.log(1 + P["vol_cv"] ** 2)
    X = rng.lognormal(-s2 / 2, np.sqrt(s2), nb)
    X = X / X.mean()  # normalize total bot volume to n_bots
    sunk = rng.random(nb) < P["mix_evasive"]  # measured 78% already evasive
    return dict(b_bot=b_bot, b_hum=b_hum, vA=vA, X=X, sunk=sunk,
                dens_centers=centers, dens=dens)


# ----------------------------------------------------------------------------
# Policies
# ----------------------------------------------------------------------------
def phi_none(b):
    return np.zeros_like(b)


def phi_binary(b):
    return (b >= P["tau_op"]).astype(float)


def phi_throttle(b):
    return np.minimum(1.0, 3.0 * b)


def make_phi_graded(f_target, center=None, width=0.08):
    c = P["tau_op"] if center is None else center

    def phi(b):
        return f_target / (1.0 + np.exp(-(b - c) / width))

    return phi


# ----------------------------------------------------------------------------
# Static policy evaluation
# ----------------------------------------------------------------------------
def eval_policy(phi, pop):
    f_bot = phi(pop["b_bot"])
    f_hum = phi(pop["b_hum"])
    s = best_response(f_bot, pop["vA"], pop["X"], pop["sunk"])
    X = pop["X"]
    extr = float(X[(s == S_V) | (s == S_E)].sum())              # successful extraction
    obs = float(X[s == S_V].sum())                               # dashboard metric
    val = pop["vA"] * X
    evas_p = float(val[s == S_E].sum() / val.sum())              # value-mass share s_E
    retreat_share = float((s == S_R).mean())
    comply_share = float((s == S_C).mean())
    # infrastructure cost: friction actually delivered (evaders see rho*f)
    served_f = np.where(s == S_V, f_bot, np.where(s == S_E, P["rho"] * f_bot, 0.0))
    c_D = P["gamma_D"] * (float((served_f * X).sum()) + float(f_hum.sum()))
    l_H = P["h"] * float((f_hum ** 2).sum())
    e_D = P["e_D"] * extr
    l_E = P["L_E"] * float((s == S_E).sum())
    sla = float(f_hum.sum())  # harm-weighted displaced human volume (x_h = 1)
    return dict(extraction=extr, infra_cost=c_D, human_harm=l_H,
                evasion_prob=evas_p, observed=obs,
                L_D=e_D + c_D + l_H + l_E, sla=sla,
                retreat_share=retreat_share, comply_share=comply_share,
                evasion_count_share=float((s == S_E).mean()))


def tune_static(pop):
    """Grid-search the graded menu (f_target, width) minimizing total L_D."""
    best = None
    for ft in np.arange(0.20, 1.001, 0.025):
        for w in (0.05, 0.08, 0.12):
            m = eval_policy(make_phi_graded(ft, width=w), pop)
            if best is None or m["L_D"] < best[0]:
                best = (m["L_D"], ft, w)
    return best[1], best[2]


# ----------------------------------------------------------------------------
# Adaptive loop (daily, delays tau_D=5, tau_A=7, gain k_phi)
# ----------------------------------------------------------------------------
def run_loop(pop, phi_base, k_phi, T=420, V_target=None, m0=0.0):
    """Defender scales the graded menu by m_t:
         f_i(t) = min(1, m_t * phi_base(b_i)).
       Update on visible bot volume observed with delay tau_D:
         m_{t+1} = clip(m_t + k_phi * (V_obs(t - tau_D) - V*) / V0, 0, 2.5)
       Attacker best-responds to friction lagged tau_A; evasion setup is a
       ratchet (K_E sunk once paid)."""
    rng = np.random.default_rng(SEED)
    nb = len(pop["b_bot"])
    phib = phi_base(pop["b_bot"])
    phih = phi_base(pop["b_hum"])
    X, vA = pop["X"], pop["vA"]
    sunk = pop["sunk"].copy()
    V0 = float(X.sum())  # no-defense visible volume scale
    if V_target is None:
        V_target = 0.25 * V0
    m_hist = np.empty(T)
    f_hist = np.empty((T,))           # median bot friction
    vis_hist = np.empty(T)
    ev_hist = np.empty(T)
    daily = []
    m = m0
    m_traj = []
    vis_buf = []
    f_buf = []
    val = vA * X
    for t in range(T):
        f_bot = np.minimum(1.0, m * phib)
        f_buf.append(f_bot)
        # attacker responds to friction tau_A days old
        f_eff = f_buf[max(0, t - P["tau_A"])]
        s = best_response(f_eff, vA, X, sunk)
        newly = (s == S_E) & (~sunk)
        sunk = sunk | newly  # ratchet: setup cost paid once
        vis = float(X[s == S_V].sum())
        vis_buf.append(vis)
        # defender loss components this day
        f_hum = np.minimum(1.0, m * phih)
        served_f = np.where(s == S_V, f_bot, np.where(s == S_E, P["rho"] * f_bot, 0.0))
        extr = float(X[(s == S_V) | (s == S_E)].sum())
        c_D = P["gamma_D"] * (float((served_f * X).sum()) + float(f_hum.sum()))
        l_H = P["h"] * float((f_hum ** 2).sum())
        l_E = P["L_E"] * float((s == S_E).sum())
        daily.append(dict(extraction=extr, infra_cost=c_D, human_harm=l_H,
                          observed=vis, L_D=P["e_D"] * extr + c_D + l_H + l_E,
                          sla=float(f_hum.sum()),
                          evasion_prob=float(val[s == S_E].sum() / val.sum()),
                          retreat_share=float((s == S_R).mean()),
                          K_paid=float(newly.sum() * P["K_E"])))
        m_hist[t] = m
        f_hist[t] = float(np.median(f_bot))
        vis_hist[t] = vis / V0
        ev_hist[t] = float(val[s == S_E].sum() / val.sum())
        # defender observes with delay tau_D and updates
        v_obs = vis_buf[max(0, t - P["tau_D"])]
        m = float(np.clip(m + k_phi * (v_obs - V_target) / V0, 0.0, 2.5))
    return dict(m=m_hist, f=f_hist, vis=vis_hist, ev=ev_hist,
                daily=pd.DataFrame(daily))


def osc_amplitude(traj, last=150):
    """Peak-to-peak swing of the observed visible bot volume (dashboard
    signal, fraction of no-defense volume) over the final window."""
    v = traj["vis"][-last:]
    return float(v.max() - v.min())


def osc_period(traj, last=240):
    """Dominant oscillation period (days) of the dashboard signal via FFT."""
    f = traj["vis"][-last:]
    f = f - f.mean()
    if np.allclose(f, 0):
        return np.nan
    spec = np.abs(np.fft.rfft(f))
    freqs = np.fft.rfftfreq(len(f), d=1.0)
    spec[0] = 0
    k = int(np.argmax(spec))
    return float(1.0 / freqs[k]) if freqs[k] > 0 else np.nan


def find_k_crit(pop, phi_base, V_target, ks=None, thresh=0.15):
    if ks is None:
        ks = np.round(np.arange(0.05, 3.01, 0.05), 3)
    amp = {}
    k_crit = None
    for k in ks:
        tr = run_loop(pop, phi_base, k, T=420, V_target=V_target)
        amp[k] = osc_amplitude(tr)
        if k_crit is None and amp[k] > thresh:
            k_crit = k
    return k_crit, amp


# ----------------------------------------------------------------------------
# Figure 1: containment map
# ----------------------------------------------------------------------------
def containment_map(pop, phi_static, fname):
    nf, nv = 400, 400
    fg = np.linspace(0, 1, nf)
    vg = np.linspace(0.0, 2.2, nv)
    FF, VV = np.meshgrid(fg, vg)
    # regimes on the grid for a FRESH attacker (K_E unpaid) at the mean cap X=1
    br = best_response(FF.ravel(), VV.ravel(), 1.0,
                       np.zeros(FF.size, bool)).reshape(FF.shape)
    # region codes: 0 under-friction (s_V), 1 containment (s_R), 2 comply, 3 evasion
    region = np.full(FF.shape, 1)
    region[br == S_V] = 0
    region[br == S_C] = 2
    region[br == S_E] = 3

    cmap = ListedColormap(["#f4a582", "#a6dba0", "#92c5de", "#9970ab"])
    fig = plt.figure(figsize=(3.5, 4.4), dpi=300)
    gs = gridspec.GridSpec(2, 1, height_ratios=[4.2, 1.0], hspace=0.08)
    ax = fig.add_subplot(gs[0])
    axm = fig.add_subplot(gs[1], sharex=ax)

    ax.pcolormesh(fg, vg, region, cmap=cmap, vmin=0, vmax=3, shading="auto",
                  rasterized=True)
    # analytic curves
    vline = np.linspace(P["c0"] + 1e-4, vg[-1], 300)
    ax.plot(np.clip(f_min(vline), 0, 1), vline, "k-", lw=1.3,
            label=r"$f_{\min}(v_A)$")
    fsw = f_switch(P["K_E"])
    if np.isfinite(fsw):
        ax.axvline(fsw, color="k", ls="--", lw=1.1,
                   label=r"$f_{\mathrm{switch}}$ ($K_E$ unpaid)")
    fsw_sunk = f_switch(0.0)
    if np.isfinite(fsw_sunk):
        ax.axvline(fsw_sunk, color="0.35", ls=":", lw=1.1,
                   label=r"$f_{\mathrm{switch}}$ ($K_E$ sunk)")
    # human-harm boundary
    ax.axvline(P["f_H"], color="#b2182b", ls="-.", lw=1.1)
    ax.annotate("human-harm\nboundary $f_H$\n($h f^2 > " + f"{P['H_budget']}$)",
                xy=(P["f_H"], 0.30), xytext=(P["f_H"] + 0.06, 0.10),
                fontsize=5.5, color="#b2182b",
                arrowprops=dict(arrowstyle="->", color="#b2182b", lw=0.7))
    ax.set_ylabel(r"attacker value $v_A$", fontsize=7)
    ax.tick_params(labelsize=6, labelbottom=False)
    handles = [
        Patch(fc="#f4a582", label="under-friction ($s_V$ profitable)"),
        Patch(fc="#a6dba0", label="containment (retreat/comply)"),
        Patch(fc="#9970ab", label="evasion ($s_E$)"),
    ]
    if (region == 2).any():
        handles.insert(2, Patch(fc="#92c5de", label="comply ($s_C$)"))
    h2, l2 = ax.get_legend_handles_labels()
    ax.legend(handles=handles + h2, fontsize=4.6, loc="upper left",
              framealpha=0.92, borderpad=0.4, labelspacing=0.3)
    ax.set_title("Containment map (calibrated)", fontsize=8)

    # marginal strip: where BOT vs HUMAN score mass lands on the f axis under phi
    f_bot = phi_static(pop["b_bot"])
    f_hum = phi_static(pop["b_hum"])
    bins = np.linspace(0, 1, 51)
    hb, _ = np.histogram(f_bot, bins=bins, weights=pop["X"], density=True)
    hh, _ = np.histogram(f_hum, bins=bins, density=True)
    cc = 0.5 * (bins[:-1] + bins[1:])
    axm.fill_between(cc, hb, color="#762a83", alpha=0.55, lw=0,
                     label="BOT mass under $\\varphi$")
    axm.fill_between(cc, hh, color="#1b7837", alpha=0.45, lw=0,
                     label="HUMAN mass under $\\varphi$")
    # phi(b) mapped onto the f axis: mark phi at quantiles of b
    axm.axvline(P["f_H"], color="#b2182b", ls="-.", lw=0.9)
    if np.isfinite(fsw):
        axm.axvline(fsw, color="k", ls="--", lw=0.9)
    axm.set_yscale("log")
    axm.set_ylim(1e-2, max(hb.max(), hh.max()) * 2)
    axm.set_xlabel(r"friction intensity $f$", fontsize=7)
    axm.set_ylabel("score mass\n(log)", fontsize=5.5)
    axm.tick_params(labelsize=6)
    axm.legend(fontsize=4.6, loc="upper center", framealpha=0.9)
    fig.savefig(FIG / fname, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Figure 3: time-domain stability
# ----------------------------------------------------------------------------
def stability_figure(pop, phi_base, V_target, k_lo, k_hi, fname):
    tr_lo = run_loop(pop, phi_base, k_lo, T=420, V_target=V_target)
    tr_hi = run_loop(pop, phi_base, k_hi, T=420, V_target=V_target)
    per = osc_period(tr_hi)
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 4.2), dpi=300, sharex=True)
    for ax, tr, k, lab in [(axes[0], tr_lo, k_lo, "stable"),
                           (axes[1], tr_hi, k_hi, "oscillatory")]:
        t = np.arange(len(tr["f"]))
        ax.plot(t, tr["f"], color="#2166ac", lw=1.0, label="median bot friction $f$")
        ax.plot(t, tr["vis"], color="#b2182b", lw=0.9,
                label="observed visible bot volume")
        ax.plot(t, tr["ev"], color="#762a83", lw=0.9, ls="--",
                label="evasion share (value mass)")
        ax.axhline(V_target / pop["X"].sum(), color="0.5", lw=0.7, ls=":")
        ax.set_ylabel("level", fontsize=7)
        ax.tick_params(labelsize=6)
        ttl = f"$k_\\varphi={k:.2f}$ ({lab})"
        if lab == "oscillatory" and np.isfinite(per):
            ttl += f", period $\\approx$ {per:.0f} d"
        ax.set_title(ttl, fontsize=7.5)
        ax.set_ylim(-0.02, 1.05)
    axes[0].legend(fontsize=5, loc="upper right", framealpha=0.9)
    axes[1].set_xlabel("day", fontsize=7)
    fig.suptitle("Friction-update loop: $\\tau_D=5$ d, $\\tau_A=7$ d (measured)",
                 fontsize=8, y=0.995)
    fig.tight_layout()
    fig.savefig(FIG / fname, bbox_inches="tight")
    plt.close(fig)
    return per, tr_lo, tr_hi


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(SEED)
    pop = draw_population(rng, P["vA_sigma"])

    # --- analytic band ------------------------------------------------------
    fsw = f_switch(P["K_E"])
    fsw_h = f_switch(P["K_E"] / 2)
    fsw_d = f_switch(P["K_E"] * 2)
    v_ref = P["vA_median"]
    fmin_ref = float(f_min(v_ref))

    def width(fs):
        return max(0.0, min(fs, 1.0) - fmin_ref)

    def v_collapse(fs):
        # band empty when f_min(v) >= f_switch -> v >= c0 + c1*fs^r
        return P["c0"] + P["c1"] * min(fs, 1.0) ** P["r"]

    sens = {
        "K_E_base": dict(K_E=P["K_E"], f_switch=fsw, band_at_vref=width(fsw),
                         v_collapse=v_collapse(fsw)),
        "K_E_half": dict(K_E=P["K_E"] / 2, f_switch=fsw_h, band_at_vref=width(fsw_h),
                         v_collapse=v_collapse(fsw_h)),
        "K_E_double": dict(K_E=P["K_E"] * 2,
                           f_switch=fsw_d if np.isfinite(fsw_d) else float("inf"),
                           band_at_vref=width(fsw_d if np.isfinite(fsw_d) else 1.0),
                           v_collapse=v_collapse(fsw_d) if np.isfinite(fsw_d)
                           else P["c0"] + P["c1"]),
    }

    # --- static graded tuning -----------------------------------------------
    ft, fw = tune_static(pop)
    phi_static = make_phi_graded(ft, width=fw)

    # --- adaptive: critical gain --------------------------------------------
    static_metrics = eval_policy(phi_static, pop)
    # dashboard target: drive observed visible bot volume to 40% of no-defense.
    # (Targets below ~28% are unreachable: 38% of bot volume scores b < tau and
    # receives ~zero friction under any monotone menu -- noted in the summary.)
    V_target = 0.40 * float(pop["X"].sum())
    k_crit, amps = find_k_crit(pop, phi_static, V_target)
    k_stable = 0.5 * k_crit
    k_unstable = 2.0 * k_crit
    per, tr_lo, tr_hi = stability_figure(pop, phi_static, V_target,
                                         k_stable, k_unstable,
                                         "SIM_stability_timedomain.png")

    # --- five policies --------------------------------------------------------
    policies = {
        "no_defense": phi_none,
        "binary_block_tau0.234": phi_binary,
        "aggressive_throttle_3b": phi_throttle,
        f"static_graded_ft{ft:.2f}": phi_static,
    }
    rows = {}
    for name, phi in policies.items():
        rows[name] = eval_policy(phi, pop)
    # adaptive: time-average of the stable loop after burn-in (>= 60 d)
    d = tr_lo["daily"].iloc[60:]
    rows[f"adaptive_graded_k{k_stable:.2f}"] = dict(
        extraction=float(d["extraction"].mean()),
        infra_cost=float(d["infra_cost"].mean()),
        human_harm=float(d["human_harm"].mean()),
        evasion_prob=float(d["evasion_prob"].mean()),
        observed=float(d["observed"].mean()),
        L_D=float(d["L_D"].mean()),  # K_E is attacker-side, not part of L_D
        sla=float(d["sla"].mean()),
        retreat_share=float(d["retreat_share"].mean()),
        comply_share=float("nan"),
        evasion_count_share=float("nan"),
    )

    tab = pd.DataFrame(rows).T
    base = tab.loc["no_defense"]
    for col in ("extraction", "observed", "L_D"):
        tab[col + "_idx"] = 100.0 * tab[col] / base[col]
    tab.index.name = "policy"
    tab.round(4).to_csv(TAB / "SIM_policy_comparison.csv")

    # --- containment map -------------------------------------------------------
    containment_map(pop, phi_static, "SIM_containment_map.png")

    # --- retreat-fraction consistency check (vs measured 0.141 / 0.073) --------
    light = eval_policy(make_phi_graded(0.45, width=fw), pop)
    heavy = eval_policy(make_phi_graded(0.95, width=fw), pop)
    retreat_check = dict(sim_light=light["retreat_share"],
                         sim_heavy=heavy["retreat_share"],
                         sim_evasion_light=light["evasion_count_share"],
                         sim_evasion_heavy=heavy["evasion_count_share"],
                         measured_light=P["retreat_light"],
                         measured_heavy=P["retreat_heavy"])

    # --- v_A robustness ---------------------------------------------------------
    pop2 = draw_population(np.random.default_rng(SEED), P["vA_sigma_alt"])
    ft2, fw2 = tune_static(pop2)
    rows2 = {
        "no_defense": eval_policy(phi_none, pop2),
        "binary_block_tau0.234": eval_policy(phi_binary, pop2),
        "aggressive_throttle_3b": eval_policy(phi_throttle, pop2),
        f"static_graded_ft{ft2:.2f}": eval_policy(make_phi_graded(ft2, width=fw2), pop2),
    }
    tab2 = pd.DataFrame(rows2).T.sort_values("L_D")

    out = dict(
        f_switch=fsw, f_min_at_vref=fmin_ref, sens=sens,
        static_f_target=ft, static_width=fw,
        k_crit=k_crit, k_stable=k_stable, k_unstable=k_unstable,
        osc_period=per, V_target=V_target,
        retreat_check=retreat_check,
        ranking_main=list(tab.sort_values("L_D").index),
        ranking_alt=list(tab2.index),
        tab=tab, tab2=tab2, amps=amps,
    )
    with open(OUT / "SIM_headline.json", "w") as fh:
        json.dump({k: v for k, v in out.items()
                   if k not in ("tab", "tab2", "amps")}, fh, indent=2, default=str)
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("tab", "tab2", "amps")}, indent=2, default=str))
    print(tab.round(3).to_string())
    print(tab2.round(3).to_string())
    return out


if __name__ == "__main__":
    main()
