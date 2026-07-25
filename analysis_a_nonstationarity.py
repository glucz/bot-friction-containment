"""
Analysis A -- Per-agent non-stationarity  (supports sub-claim C1: "change over time")

For every agent we test each behavioral feature (on active days) for:
  - a monotonic lifecycle trend (Mann-Kendall), and
  - an abrupt regime shift (Pettitt change-point).

We then aggregate by class. Two honest findings drive the framing:
  (1) Non-stationarity is near-universal -- abrupt change-points appear in ~95%
      of *every* class at n~500 -- so C1 ("features change over time") holds, but
      change-point *significance* is not by itself discriminating. We therefore
      report change-point *effect size* (magnitude of the mean shift), not just
      its p-value.
  (2) Raw trend direction is confounded by tenure (longer histories accumulate
      more IP/location diversity, for humans too). The bot-specific signal is the
      *differential* against the human control: bots distinctively shift request
      composition (html/php up), probe more (404 up), check robots.txt more
      (compliance up), and *concentrate* network origin (country/url/domain
      entropy down) -- active retooling, versus humans' passive diversity gain.

The causal "respond to filters" claim is carried by Analysis B (event-conditioned
difference-in-differences), which controls for this baseline drift.

Supports: the population-dynamics companion (adaptation is real and common
-> replicator dynamics is warranted) and this paper (observed tactic
drift, not just synthetic strategies).
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import signals
import stats_tests as st

# Features whose lifecycle behaviour we characterise.
TEST_FEATURES = [
    "evasion_index", "ip_ent", "dom_ent", "ctry_ent", "url_ent",
    "resistance", "compliance", "p404", "img", "html", "js", "php", "folder",
]


def _safe_nanmedian(s) -> float:
    vals = s.to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    return float(np.median(vals)) if len(vals) else float("nan")


def run(agents_by_label: dict[str, list]) -> dict:
    config.ensure_dirs()
    per_agent_rows = []

    for label, agents in agents_by_label.items():
        for ag in agents:
            act = signals.add_derived(ag.active)
            if len(act) < config.MIN_ACTIVE_DAYS:
                continue
            row = {"agent_id": ag.agent_id, "label": label, "n_active": len(act)}
            for feat in TEST_FEATURES:
                if feat not in act.columns:
                    continue
                x = act[feat].to_numpy()
                tr = st.mann_kendall(x)
                cp = st.pettitt(x)
                row[f"{feat}__trend_dir"] = tr.direction
                row[f"{feat}__tau"] = tr.tau
                row[f"{feat}__cp_sig"] = int(cp.significant)
                # change-point location as a fraction of lifecycle (when it happens)
                row[f"{feat}__cp_frac"] = (cp.location / len(act)) if cp.location >= 0 else np.nan
                # change-point EFFECT SIZE: |mean_after - mean_before|. At n~500 the
                # Pettitt p-value is near-always significant, so magnitude -- not
                # significance -- is what distinguishes a real regime shift.
                if cp.location > 0:
                    row[f"{feat}__cp_effect"] = float(abs(x[cp.location:].mean() - x[:cp.location].mean()))
                else:
                    row[f"{feat}__cp_effect"] = np.nan
            per_agent_rows.append(row)

    per_agent = pd.DataFrame(per_agent_rows)
    per_agent.to_csv(config.TAB_DIR / "A_per_agent_nonstationarity.csv", index=False)

    # --- Aggregate by class -------------------------------------------------
    summary_rows = []
    for label in agents_by_label:
        sub = per_agent[per_agent["label"] == label]
        if sub.empty:
            continue
        for feat in TEST_FEATURES:
            dcol, ccol = f"{feat}__trend_dir", f"{feat}__cp_sig"
            if dcol not in sub.columns:
                continue
            n = len(sub)
            summary_rows.append({
                "label": label,
                "feature": feat,
                "n_agents": n,
                "pct_trend_up": 100.0 * (sub[dcol] == 1).mean(),
                "pct_trend_down": 100.0 * (sub[dcol] == -1).mean(),
                "pct_any_trend": 100.0 * (sub[dcol] != 0).mean(),
                "pct_changepoint": 100.0 * (sub[ccol] == 1).mean(),
                "median_abs_tau": float(np.nanmedian(np.abs(sub[f"{feat}__tau"]))),
                "median_cp_effect": _safe_nanmedian(sub[f"{feat}__cp_effect"]),
            })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(config.TAB_DIR / "A_class_summary.csv", index=False)

    _plot_trend_differential(summary)
    _plot_changepoint_timing(per_agent)

    # --- Headline numbers ---------------------------------------------------
    # C1 is established by near-universal non-stationarity; the bot signature is
    # in WHICH features drift, contrasted against the human control.
    headline: dict = {"nonstationarity_rate": {}, "bot_vs_human_trend_fingerprint": {}}
    for label in agents_by_label:
        sub = per_agent[per_agent["label"] == label]
        if sub.empty:
            continue
        any_trend_cols = [f"{f}__trend_dir" for f in TEST_FEATURES if f"{f}__trend_dir" in sub.columns]
        # fraction of agents with at least one significant feature trend
        has_trend = (sub[any_trend_cols].abs().sum(axis=1) > 0).mean()
        headline["nonstationarity_rate"][label] = {
            "pct_with_any_feature_trend": round(100.0 * float(has_trend), 1),
            "n_agents": int(len(sub)),
        }

    # Top bot-distinctive trends (signed bot-minus-human in % trending up).
    up = summary.pivot_table(index="feature", columns="label", values="pct_trend_up")
    if "bot" in up.columns and "human" in up.columns:
        diff = (up["bot"] - up["human"]).dropna().sort_values()
        fingerprint = {}
        for feat in list(diff.index[-4:][::-1]) + list(diff.index[:4]):
            fingerprint[feat] = round(float(diff[feat]), 1)  # +ve: bots trend up more
        headline["bot_vs_human_trend_fingerprint"] = fingerprint

    return {"headline": headline, "summary_table": str(config.TAB_DIR / "A_class_summary.csv")}


def _plot_trend_differential(summary: pd.DataFrame) -> None:
    """Diverging bars: which features do bots trend toward more than humans?

    Net trend per feature = %(up) - %(down). Plot bot-net minus human-net so the
    accumulation drift that affects both classes cancels out, isolating the
    bot-specific lifecycle signature.
    """
    up = summary.pivot_table(index="feature", columns="label", values="pct_trend_up")
    dn = summary.pivot_table(index="feature", columns="label", values="pct_trend_down")
    if not {"bot", "human"}.issubset(up.columns):
        return
    bot_net = up["bot"] - dn["bot"]
    hum_net = up["human"] - dn["human"]
    diff = (bot_net - hum_net).dropna().sort_values()
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    colors = ["tab:red" if v > 0 else "tab:blue" for v in diff.values]
    ax.barh(diff.index, diff.values, color=colors)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("bot net-trend  -  human net-trend  (percentage points)")
    ax.set_title("Bot-specific lifecycle drift signature\n(red = bots trend up more; blue = bots trend down more)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "A_trend_differential_bot_vs_human.png", dpi=150)
    plt.close(fig)


def _plot_changepoint_timing(per_agent: pd.DataFrame) -> None:
    """When in the lifecycle do bots vs humans switch tactics (evasion change-point)?"""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    for label, color in [("bot", "tab:red"), ("human", "tab:blue")]:
        sub = per_agent[per_agent["label"] == label]
        col = "evasion_index__cp_frac"
        if sub.empty or col not in sub.columns:
            continue
        sig = sub[sub.get("evasion_index__cp_sig", 0) == 1][col].dropna()
        if len(sig) == 0:
            continue
        ax.hist(sig, bins=20, range=(0, 1), alpha=0.5, label=f"{label} (n={len(sig)})", color=color, density=True)
    ax.set_xlabel("Change-point location (fraction of lifecycle)")
    ax.set_ylabel("density")
    ax.set_title("When agents abruptly shift evasion tactics")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "A_changepoint_timing.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    import datasource as ds

    src = ds.get_source()
    agents = {lbl: src.load_all([lbl]) for lbl in config.CLASSES}
    print(run(agents)["headline"])
