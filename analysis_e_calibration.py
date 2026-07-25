"""
Analysis E -- Calibration outputs for Papers D and B

E1: empirical botness density. The central containment-map figure
   needs "empirical botness-density contours from AGWA-derived scores". We build
   that density from the classifier's real on-disk outputs (avg_bot_prob in
   chromes_ranked..csv and bot_score_area in bot_scores_summary.json) and overlay
   the operating threshold (best_threshold.txt = 0.234).

E2: bot traffic as a time-varying perturbation. The SLA-market companion must stay
   viable against bot-induced perturbations to the client type distribution
   F(theta) (its P3 type-drift / P4 demand-shock). We quantify:
     - perturbation magnitude: per-bot daily-volume volatility (coefficient of
       variation) and burst frequency;
     - type-distribution distortion: how far the bot population shifts a
       behavioral distribution away from the human baseline (Wasserstein);
     - perturbation timescale: autocorrelation time of per-bot volume.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

import config
import signals


def _load_threshold() -> float:
    try:
        return float(config.BEST_THRESHOLD_TXT.read_text().strip())
    except Exception:
        return 0.5


def _empirical_botness() -> np.ndarray:
    """Real per-agent P(bot) from the classifier (avg_bot_prob, on [0,1]).

    We use only avg_bot_prob -- the model's mean P(bot) -- as the botness density.
    bot_score_area in the JSON is a different-scale diagnostic and must not be
    pooled with a probability.
    """
    if config.CHROME_RANK_CSV.exists():
        df = pd.read_csv(config.CHROME_RANK_CSV)
        if "avg_bot_prob" in df.columns:
            return df["avg_bot_prob"].to_numpy(dtype=float)
    return np.array([])


def run(agents_by_label: dict[str, list]) -> dict:
    config.ensure_dirs()
    headline: dict = {}

    # --- E1: empirical botness density -----------------------------------
    b = _empirical_botness()
    thr = _load_threshold()
    if b.size:
        b = np.clip(b, 0, 1)
        hist, edges = np.histogram(b, bins=40, range=(0, 1), density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        pd.DataFrame({"botness": centers, "density": hist}).to_csv(
            config.TAB_DIR / "E_botness_density.csv", index=False)
        _plot_botness_density(b, thr)
        headline["botness_density"] = {
            "n": int(b.size),
            "mean": round(float(b.mean()), 3),
            "frac_above_threshold": round(float((b >= thr).mean()), 3),
            "threshold": thr,
        }

    # --- E2: bot-perturbation calibration ---------------------------------
    vol_rows = []
    feat_pools: dict[str, list[np.ndarray]] = {}
    for label, agents in agents_by_label.items():
        feat_pools[label] = []
        for ag in agents:
            act = signals.add_derived(ag.active)
            if len(act) < config.MIN_ACTIVE_DAYS:
                continue
            hits = act["hits"].to_numpy(dtype=float)
            cv = float(hits.std() / (hits.mean() + 1e-9))
            burst = float((hits > hits.mean() + 2 * hits.std()).mean())
            vol_rows.append({
                "agent_id": ag.agent_id, "label": label,
                "volume_cv": cv, "burst_frac": burst,
                "volume_autocorr_time": signals.autocorr_time(np.log1p(hits)),
            })
            feat_pools[label].append(act["evasion_index"].to_numpy())

    vol = pd.DataFrame(vol_rows)
    vol.to_csv(config.TAB_DIR / "E_perturbation_per_agent.csv", index=False)

    pert_summary = (
        vol.groupby("label")
        .agg(median_volume_cv=("volume_cv", "median"),
             median_burst_frac=("burst_frac", "median"),
             median_volume_autocorr_time=("volume_autocorr_time", "median"))
        .reset_index()
    )
    pert_summary.to_csv(config.TAB_DIR / "E_perturbation_summary.csv", index=False)

    # type-distribution distortion: bot vs human behavioral distribution distance
    distortion = np.nan
    if feat_pools.get("bot") and feat_pools.get("human"):
        bot_vals = np.concatenate(feat_pools["bot"])
        hum_vals = np.concatenate(feat_pools["human"])
        distortion = float(wasserstein_distance(bot_vals, hum_vals))

    _plot_perturbation(vol)

    for label in ("bot", "human"):
        r = pert_summary[pert_summary.label == label]
        if not r.empty:
            headline.setdefault("perturbation", {})[label] = {
                "median_volume_cv": round(float(r.median_volume_cv.iloc[0]), 2),
                "median_burst_frac": round(float(r.median_burst_frac.iloc[0]), 3),
                "median_volume_autocorr_days": round(float(r.median_volume_autocorr_time.iloc[0]), 1),
            }
    headline["type_distribution_distortion_wasserstein_bot_vs_human"] = (
        round(distortion, 4) if np.isfinite(distortion) else None
    )
    return {"headline": headline}


def _plot_botness_density(b: np.ndarray, thr: float) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.hist(b, bins=40, range=(0, 1), density=True, color="tab:gray", alpha=0.8)
    ax.axvline(thr, color="tab:red", ls="--", lw=2, label=f"operating threshold = {thr:.3f}")
    ax.set_xlabel("botness score b")
    ax.set_ylabel("empirical density")
    ax.set_title("Empirical botness density (AGWA-derived) -- containment-map input")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "E_botness_density.png", dpi=150)
    plt.close(fig)


def _plot_perturbation(vol: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    for label, color in [("bot", "tab:red"), ("human", "tab:blue"), ("chrome", "tab:green")]:
        sub = vol[vol.label == label]["volume_cv"].dropna().clip(0, 10)
        if len(sub) == 0:
            continue
        ax.hist(sub, bins=30, alpha=0.5, density=True, label=f"{label} (med={sub.median():.2f})", color=color)
    ax.set_xlabel("daily-volume coefficient of variation")
    ax.set_ylabel("density")
    ax.set_title("Bot traffic volatility = magnitude of the SLA demand perturbation")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "E_perturbation_volatility.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    import datasource as ds

    src = ds.get_source()
    agents = {lbl: src.load_all([lbl]) for lbl in config.CLASSES}
    print(json.dumps(run(agents)["headline"], indent=2))
