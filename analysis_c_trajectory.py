"""
Analysis C -- Trajectory dynamics  (population-dynamics calibration)

Three trajectory statistics per agent, on the active-day behavioral series:

  1. lifecycle_drift   -- multivariate distance between the early-life and
                          late-life behavior centroids (standardized feature
                          space). How far the agent's tactics move over its life.
  2. adaptation_time   -- autocorrelation (1/e) time of the evasion signal. The
                          natural timescale of behavioral adaptation. This is the
                          empirical anchor for the population-dynamics companion's replicator speed relative
                          to the controller delay tau / gain eta.
  3. oscillation        -- spectral peak fraction of the detrended evasion signal.
                          Individual-agent behavioral periodicity -- suggestive of
                          the arms-race limit cycle the population-dynamics companion predicts (NOT proof of
                          the coupled system's cycle; that lives at the population
                          level, Analysis D).

If PyTorch + the trained weights are present, the same three statistics are also
computed on the true botness curve b(t) from the classifier; otherwise the
model-free behavioral trajectory stands in (see botness_inference.py).
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import botness_inference as bi
import config
import signals


def _build_scaler(agents_by_label: dict[str, list]) -> tuple[np.ndarray, np.ndarray]:
    """Global mean/std per behavior feature over pooled active days."""
    chunks = []
    for agents in agents_by_label.values():
        for ag in agents:
            act = ag.active
            if len(act):
                chunks.append(act[config.BEHAVIOR_FEATURES].to_numpy())
    X = np.vstack(chunks)
    mean = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0)
    std[std == 0] = 1.0
    return mean, std


def _periodicity(x: np.ndarray) -> tuple[float, float]:
    """(spectral peak fraction, dominant period in days) of a detrended series."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 30 or np.allclose(x, x[0]):
        return (0.0, np.nan)
    # linear detrend
    t = np.arange(n)
    x = x - np.polyval(np.polyfit(t, x, 1), t)
    fft = np.fft.rfft(x - x.mean())
    power = np.abs(fft) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0)
    if len(power) < 3 or power[1:].sum() <= 0:
        return (0.0, np.nan)
    k = 1 + int(np.argmax(power[1:]))         # skip DC
    strength = float(power[k] / power[1:].sum())
    period = float(1.0 / freqs[k]) if freqs[k] > 0 else np.nan
    return (strength, period)


def _traj_metrics(series_multi: np.ndarray, evasion: np.ndarray, mean, std) -> dict:
    """Drift (multivariate), adaptation time, oscillation -- from given arrays."""
    z = (series_multi - mean) / std
    n = len(z)
    k = max(1, int(0.2 * n))
    drift = float(np.linalg.norm(np.nanmean(z[-k:], axis=0) - np.nanmean(z[:k], axis=0)))
    tau = signals.autocorr_time(evasion)
    osc, period = _periodicity(evasion)
    return {"lifecycle_drift": drift, "adaptation_time": tau,
            "oscillation": osc, "osc_period": period}


def run(agents_by_label: dict[str, list]) -> dict:
    config.ensure_dirs()
    mean, std = _build_scaler(agents_by_label)
    rows = []
    botness_rows = []
    torch_on = bi.available()

    for label, agents in agents_by_label.items():
        for ag in agents:
            act = signals.add_derived(ag.active)
            if len(act) < 30:
                continue
            m = _traj_metrics(
                act[config.BEHAVIOR_FEATURES].to_numpy(),
                act["evasion_index"].to_numpy(),
                mean, std,
            )
            m.update({"agent_id": ag.agent_id, "label": label, "n_active": len(act)})
            rows.append(m)

            if torch_on:
                b = bi.botness_curve(ag.meta.get("path", ""))
                if b is not None and len(b) >= 30:
                    osc, period = _periodicity(b)
                    botness_rows.append({
                        "agent_id": ag.agent_id, "label": label,
                        "botness_drift": float(abs(b[-len(b)//5:].mean() - b[:len(b)//5].mean())),
                        "botness_adaptation_time": signals.autocorr_time(b),
                        "botness_oscillation": osc, "botness_osc_period": period,
                        "botness_mean": float(b.mean()),
                    })

    per_agent = pd.DataFrame(rows)
    per_agent.to_csv(config.TAB_DIR / "C_per_agent_trajectory.csv", index=False)
    if botness_rows:
        pd.DataFrame(botness_rows).to_csv(config.TAB_DIR / "C_botness_trajectory.csv", index=False)

    # --- aggregate ----------------------------------------------------------
    summary = (
        per_agent.groupby("label")
        .agg(
            n_agents=("agent_id", "count"),
            median_drift=("lifecycle_drift", "median"),
            median_adaptation_time=("adaptation_time", "median"),
            pct_oscillatory=("oscillation", lambda s: 100.0 * (s > 0.25).mean()),
            median_osc_period=("osc_period", "median"),
        )
        .reset_index()
    )
    summary.to_csv(config.TAB_DIR / "C_class_summary.csv", index=False)

    _plot_distribution(per_agent, "lifecycle_drift", "Lifecycle behavioral drift", "C_drift_distribution.png")
    _plot_distribution(per_agent, "adaptation_time", "Adaptation timescale (autocorr 1/e, days)", "C_adaptation_time.png", clip=(0, 60))

    headline = {"torch_available": torch_on}
    for label in ("bot", "human"):
        r = summary[summary.label == label]
        if not r.empty:
            headline[label] = {
                "median_drift": round(float(r.median_drift.iloc[0]), 3),
                "median_adaptation_time_days": round(float(r.median_adaptation_time.iloc[0]), 1),
                "pct_oscillatory": round(float(r.pct_oscillatory.iloc[0]), 1),
            }
    if not torch_on:
        headline["note"] = "model-free trajectory used; install torch to add true b(t) curves"
    return {"headline": headline, "summary_table": str(config.TAB_DIR / "C_class_summary.csv")}


def _plot_distribution(per_agent, col, title, fname, clip=None):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    for label, color in [("bot", "tab:red"), ("human", "tab:blue"), ("chrome", "tab:green")]:
        sub = per_agent[per_agent.label == label][col].dropna()
        if clip:
            sub = sub.clip(*clip)
        if len(sub) == 0:
            continue
        ax.hist(sub, bins=30, alpha=0.5, density=True, label=f"{label} (med={sub.median():.2f})", color=color)
    ax.set_xlabel(col)
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / fname, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    import datasource as ds
    import json

    src = ds.get_source()
    agents = {lbl: src.load_all([lbl]) for lbl in ["bot", "human", "chrome"]}
    print(json.dumps(run(agents)["headline"], indent=2))
