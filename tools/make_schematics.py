"""Generate explanatory schematic figures for this paper (conceptual, not data).

Deterministic (no RNG). Outputs vector + raster into outputs/figures/ so they
sit alongside the empirical figures.

Usage (from the repository root):
    python tools/make_schematics.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "savefig.bbox": "tight",
})

# ---- Illustrative (schematic) regime boundaries — NOT calibrated values ----
f_min, f_switch, f_exit = 0.35, 0.55, 0.80


def overdeterrence_regime():
    fig, ax = plt.subplots(figsize=(6.4, 3.6))

    # Regime shading
    ax.axvspan(0.0, f_switch, color="#e8f4ea", zorder=0)          # visible extraction
    ax.axvspan(f_switch, f_exit, color="#fdeceA".replace("A", "a"), zorder=0)  # evasion
    ax.axvspan(f_exit, 1.0, color="#eeeeee", zorder=0)            # retreat

    # Total defender loss L_D(f): rises, JUMPS up at f_switch, rises in evasion, drops at f_exit
    ax.plot([0.0, f_switch], [0.30, 0.45], color="#b00020", lw=2.2, zorder=3)
    ax.plot([f_switch, f_exit], [0.78, 0.86], color="#b00020", lw=2.2, zorder=3,
            label="total defender loss $L_D(f)$")
    ax.plot([f_exit, 1.0], [0.52, 0.56], color="#b00020", lw=2.2, zorder=3)
    # discontinuity markers at f_switch
    ax.plot([f_switch], [0.45], "o", mfc="white", mec="#b00020", ms=5, zorder=4)
    ax.plot([f_switch], [0.78], "o", color="#b00020", ms=5, zorder=4)
    # the jump J
    ax.annotate("", xy=(f_switch, 0.78), xytext=(f_switch, 0.45),
                arrowprops=dict(arrowstyle="<->", color="#b00020", lw=1.4), zorder=5)
    ax.text(f_switch + 0.012, 0.615, "loss jump $J$", color="#b00020", fontsize=9, va="center")

    # Observed visible volume x_vis(f): high, then drops to 0 at f_switch
    ax.plot([0.0, f_switch], [0.92, 0.92], color="#1f4e9b", lw=2.0, ls="--", zorder=3,
            label="observed scraping $x_{vis}(f)$")
    ax.plot([f_switch, f_switch], [0.92, 0.0], color="#1f4e9b", lw=2.0, ls="--", zorder=3)
    ax.plot([f_switch, 1.0], [0.0, 0.0], color="#1f4e9b", lw=2.0, ls="--", zorder=3)

    # Boundary lines + labels
    for x, lab in [(f_min, "$f_{min}$"), (f_switch, "$f_{switch}$"), (f_exit, "$f_{exit}$")]:
        ax.axvline(x, color="#666666", lw=0.8, ls=":", zorder=1)
        ax.text(x, -0.10, lab, ha="center", va="top", fontsize=9)

    # Regime labels inside the axes near the top (below the title — avoids collision)
    ax.text(f_switch / 2, 0.965, "visible extraction", ha="center", va="top", fontsize=8.5, color="#2e7d32")
    ax.text((f_switch + f_exit) / 2, 0.965, "evasion", ha="center", va="top", fontsize=8.5, color="#b00020")
    ax.text((f_exit + 1.0) / 2, 0.965, "retreat", ha="center", va="top", fontsize=8.5, color="#555555")

    # The trap annotation
    ax.annotate("the overdeterrence trap:\nobserved scraping ↓ while total loss ↑",
                xy=(f_switch, 0.30), xytext=(0.62, 0.30),
                fontsize=8.5, va="center",
                arrowprops=dict(arrowstyle="->", color="#333333", lw=1.0))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("friction intensity $f$")
    ax.set_ylabel("normalized level (schematic)")
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title("Overdeterrence: total loss jumps where observed scraping drops (schematic)", fontsize=10)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"SCHEMATIC_overdeterrence_regime.{ext}"),
                    dpi=200 if ext == "png" else None)
    plt.close(fig)
    print("wrote SCHEMATIC_overdeterrence_regime.pdf/.png to", OUT)


def evasion_advantage():
    """Two panels: (top) the single-peaked evasion advantage Δ(f) for a containable
    and an evasion-capable type; (bottom) the mixed-population squeeze — one friction
    level must serve types on both sides of κ. Schematic parameters (r = 1, X = 1)."""
    c0, cE, c1, c2rho, vA = 0.10, 0.20, 1.20, 0.30, 0.65
    fmin = (vA - c0) / c1                      # 0.458
    kappa = vA - cE - c2rho * (vA - c0) / c1   # 0.3125
    KE_B, KE_A = 0.18, 0.45                    # evasion-capable / containable
    fsw = (cE - c0 + KE_B) / (c1 - c2rho)      # 0.311
    fex = (vA - cE - KE_B) / c2rho             # 0.900

    def delta(f, KE):
        lower = (c0 - cE) + (c1 - c2rho) * f - KE
        upper = vA - cE - c2rho * f - KE
        return lower if f <= fmin else upper

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.4, 5.4),
                                   gridspec_kw={"height_ratios": [2.1, 1.0]})

    fs = [i / 400 for i in range(401)]
    dB = [delta(f, KE_B) for f in fs]
    dA = [delta(f, KE_A) for f in fs]
    ax1.axhline(0, color="#444444", lw=0.9)
    ax1.plot(fs, dB, color="#b00020", lw=2.2,
             label="evasion-capable type ($K_E < \\kappa$)")
    ax1.plot(fs, dA, color="#2e7d32", lw=2.2, ls="--",
             label="containable type ($K_E > \\kappa$)")
    ax1.fill_between(fs, 0, dB, where=[d > 0 for d in dB], color="#fdecea", zorder=0)
    for x, lab in [(fsw, "$f_{switch}$"), (fmin, "$f_{min}$"), (fex, "$f_{exit}$")]:
        ax1.axvline(x, color="#666666", lw=0.8, ls=":", zorder=1)
        ax1.text(x, -0.44, lab, ha="center", va="top", fontsize=9)
    # peak annotation: the peak height is kappa - K_E, always at f_min
    ax1.annotate("peak $= \\kappa - K_E$, always at $f_{min}$",
                 xy=(fmin, kappa - KE_B), xytext=(0.60, 0.24), fontsize=8.5,
                 arrowprops=dict(arrowstyle="->", color="#333333", lw=1.0))
    ax1.text((fsw + fex) / 2, 0.035, "evasion pays here", ha="center",
             fontsize=8.5, color="#b00020")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(-0.42, 0.30)
    ax1.set_ylabel("evasion advantage $\\Delta(f)$")
    ax1.set_xticks([])
    ax1.set_title("The evasion advantage is single-peaked at the deterrence level (schematic)",
                  fontsize=10)
    ax1.legend(loc="lower left", fontsize=8, framealpha=0.9)

    # Bottom: the mixed-population squeeze
    yA, yB, bh = 1.0, 0.0, 0.30
    ax2.broken_barh([(fmin, 1 - fmin)], (yA - bh / 2, bh), color="#c9e5cc")
    ax2.broken_barh([(0, fsw)], (yB - bh / 2, bh), color="#dddddd")
    ax2.broken_barh([(fsw, fex - fsw)], (yB - bh / 2, bh), color="#f6c6c0")
    ax2.broken_barh([(fex, 1 - fex)], (yB - bh / 2, bh), color="#dddddd")
    ax2.text(fmin + (1 - fmin) / 2, yA, "containment band $[f_{min}, 1]$",
             ha="center", va="center", fontsize=8.5, color="#1b5e20")
    ax2.text(fsw + (fex - fsw) / 2, yB, "evasion window $(f_{switch}, f_{exit})$",
             ha="center", va="center", fontsize=8.5, color="#8b0000")
    ax2.text(fex + (1 - fex) / 2, yB - 0.02, "retreat", ha="center", va="center",
             fontsize=7.5, color="#555555")
    f_choice = 0.60
    ax2.axvline(f_choice, color="#1f4e9b", lw=1.6)
    ax2.annotate("one shared $f$: deters A,\nbut pushes B into stealth",
                 xy=(f_choice, 0.5), xytext=(0.03, 0.42), fontsize=8.5, va="center",
                 color="#1f4e9b",
                 arrowprops=dict(arrowstyle="->", color="#1f4e9b", lw=1.0))
    ax2.set_xlim(0, 1)
    ax2.set_ylim(-0.45, 1.45)
    ax2.set_yticks([yB, yA])
    ax2.set_yticklabels(["type B\n($K_E < \\kappa$)", "type A\n($K_E > \\kappa$)"],
                        fontsize=8.5)
    ax2.set_xlabel("friction intensity $f$")
    ax2.set_xticks([])

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"SCHEMATIC_evasion_advantage.{ext}"),
                    dpi=200 if ext == "png" else None)
    plt.close(fig)
    print("wrote SCHEMATIC_evasion_advantage.pdf/.png to", OUT)


if __name__ == "__main__":
    overdeterrence_regime()
    evasion_advantage()
