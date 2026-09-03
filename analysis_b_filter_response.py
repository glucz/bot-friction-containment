"""
Analysis B -- Filter-response event study  (supports sub-claim C2: "respond to filters")

This is the central contrast of the empirical case, and it is an association rather
than an identified cause: events are agent-local, not a dated policy change. For each
agent we locate
*filter-pressure events* (active days where the non-200 share spikes above the
agent's own baseline -- the agent hitting a wall) and measure how its behavior
changes in the window AFTER the event versus BEFORE.

The identifying contrast is a difference-in-differences against the verified-human
control group:

    DiD(feature) = E[ post - pre | bot ]  -  E[ post - pre | human ]

Humans encountering a 404/blocked page do not rotate IPs or fan out across
domains; bots that *respond to filters* do. A positive DiD on the evasion / IP-
rotation features is direct evidence that the behavioral change is a *response to
the filter*, not the baseline accumulation drift seen in Analysis A.

Supports: this paper (empirical f_switch: visible->evasive strategy switch triggered
by friction; Theorem 5 attacker adaptation) and the population-dynamics
companion (the feedback that drives
the replicator dynamics is real).
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

# Behavioral responses we measure around a filter-pressure event.
RESPONSE_FEATURES = [
    "evasion_index", "ip_ent", "dom_ent", "ctry_ent",
    "compliance", "request_diversity", "log_hits", "resistance",
]
W = config.EVENT_WINDOW


def _agent_volatility(act: pd.DataFrame) -> dict[str, float]:
    """Each feature's typical day-to-day movement (std of first differences).

    Used to normalize the post-event shift so a 'response' means the agent moved
    MORE than its own baseline noise -- this captures directed responses even when
    their sign differs across agents (the heterogeneous-population case of the population-dynamics companion).
    """
    vol = {}
    for feat in RESPONSE_FEATURES:
        diffs = np.diff(act[feat].to_numpy())
        vol[feat] = float(np.std(diffs)) if len(diffs) else 0.0
    return vol


def _agent_event_deltas(act: pd.DataFrame, vol: dict[str, float]) -> list[dict]:
    """Per-event signed delta and normalized |response| for one agent."""
    events = signals.filter_pressure_events(act)
    rows = []
    for i in events:
        # Require a full pre- and post-window of active days around the event.
        if i < W or i + W >= len(act):
            continue
        row = {"event_pos_frac": i / len(act)}
        for feat in RESPONSE_FEATURES:
            pre = signals.window_mean(act, i, feat, "pre", W)
            post = signals.window_mean(act, i, feat, "post", W)
            row[f"d_{feat}"] = post - pre
            # normalized response magnitude (|shift| in units of the agent's own noise)
            row[f"z_{feat}"] = abs(post - pre) / (vol[feat] + 1e-9)
        rows.append(row)
    return rows


def _agent_event_curves(act: pd.DataFrame) -> dict[str, list[np.ndarray]]:
    """Baseline-subtracted feature traces on offsets [-W, +W] for each event."""
    act = act.copy()
    act["log_hits"] = np.log1p(act["hits"].to_numpy())
    events = signals.filter_pressure_events(act)
    curves: dict[str, list[np.ndarray]] = {f: [] for f in RESPONSE_FEATURES}
    for i in events:
        if i < W or i + W >= len(act):
            continue
        for feat in RESPONSE_FEATURES:
            seg = act[feat].to_numpy()[i - W : i + W + 1]            # length 2W+1
            baseline = act[feat].to_numpy()[i - W : i].mean()        # pre-event mean
            curves[feat].append(seg - baseline)
    return curves


def run(agents_by_label: dict[str, list]) -> dict:
    config.ensure_dirs()
    per_agent_rows = []
    # accumulate event-aligned curves per class
    class_curves: dict[str, dict[str, list[np.ndarray]]] = {}

    for label, agents in agents_by_label.items():
        class_curves[label] = {f: [] for f in RESPONSE_FEATURES}
        for ag in agents:
            act = signals.add_derived(ag.active)
            if len(act) < max(config.MIN_ACTIVE_DAYS, 2 * W + 2):
                continue
            act = act.copy()
            act["log_hits"] = np.log1p(act["hits"].to_numpy())
            vol = _agent_volatility(act)
            deltas = _agent_event_deltas(act, vol)
            if not deltas:
                continue
            d = pd.DataFrame(deltas)
            agg = {"agent_id": ag.agent_id, "label": label, "n_events": len(d)}
            for feat in RESPONSE_FEATURES:
                agg[f"d_{feat}"] = float(d[f"d_{feat}"].mean())
                agg[f"z_{feat}"] = float(d[f"z_{feat}"].mean())
            per_agent_rows.append(agg)
            for feat, segs in _agent_event_curves(act).items():
                class_curves[label][feat].extend(segs)

    per_agent = pd.DataFrame(per_agent_rows)
    per_agent.to_csv(config.TAB_DIR / "B_per_agent_event_deltas.csv", index=False)

    # --- Per-class mean response (post - pre) with bootstrap CIs ------------
    resp_rows = []
    for label in agents_by_label:
        sub = per_agent[per_agent["label"] == label]
        if sub.empty:
            continue
        for feat in RESPONSE_FEATURES:
            mean, lo, hi = st.bootstrap_ci(sub[f"d_{feat}"].to_numpy(), config.BOOTSTRAP_N)
            resp_rows.append({
                "label": label, "feature": feat, "n_agents": len(sub),
                "mean_delta": mean, "ci_lo": lo, "ci_hi": hi,
                "sig": int(np.isfinite(lo) and (lo > 0 or hi < 0)),
            })
    response = pd.DataFrame(resp_rows)
    response.to_csv(config.TAB_DIR / "B_class_response.csv", index=False)

    # --- Difference-in-differences: bot minus human ------------------------
    did_rows = []
    bot = per_agent[per_agent["label"] == "bot"]
    hum = per_agent[per_agent["label"] == "human"]
    for feat in RESPONSE_FEATURES:
        if bot.empty or hum.empty:
            continue
        did, lo, hi = _did_bootstrap(bot[f"d_{feat}"].to_numpy(), hum[f"d_{feat}"].to_numpy())
        did_rows.append({
            "feature": feat,
            "did_bot_minus_human": did, "ci_lo": lo, "ci_hi": hi,
            "bot_mean": float(bot[f"d_{feat}"].mean()),
            "human_mean": float(hum[f"d_{feat}"].mean()),
            "sig": int(np.isfinite(lo) and (lo > 0 or hi < 0)),
        })
    did = pd.DataFrame(did_rows)
    did.to_csv(config.TAB_DIR / "B_difference_in_differences.csv", index=False)

    # --- Normalized response magnitude: do bots move MORE than their own noise
    #     after filter pressure than humans do? (sign-agnostic; handles the
    #     heterogeneous-population case where directions cancel.) -------------
    z_rows = []
    for feat in RESPONSE_FEATURES:
        rec = {"feature": feat}
        for label in agents_by_label:
            sub = per_agent[per_agent["label"] == label]
            if sub.empty or f"z_{feat}" not in sub.columns:
                continue
            rec[f"{label}_mean_z"] = float(sub[f"z_{feat}"].mean())
        if not bot.empty and not hum.empty:
            d, lo, hi = _did_bootstrap(bot[f"z_{feat}"].to_numpy(), hum[f"z_{feat}"].to_numpy())
            rec.update({"did_z": d, "ci_lo": lo, "ci_hi": hi,
                        "sig": int(np.isfinite(lo) and (lo > 0 or hi < 0))})
        z_rows.append(rec)
    zresp = pd.DataFrame(z_rows)
    zresp.to_csv(config.TAB_DIR / "B_normalized_response.csv", index=False)

    _plot_response_curves(class_curves, "evasion_index")
    _plot_response_curves(class_curves, "ip_ent")
    _plot_response_curves(class_curves, "compliance")
    _plot_did(did)
    _plot_normalized_response(zresp)

    # Headline
    headline = {"signed_did": {}, "normalized_response_did": {}}
    for feat in ["evasion_index", "ip_ent", "compliance", "log_hits", "resistance"]:
        r = did[did.feature == feat]
        if not r.empty:
            headline["signed_did"][feat] = {
                "bot": round(float(r.bot_mean.iloc[0]), 4),
                "human": round(float(r.human_mean.iloc[0]), 4),
                "did": round(float(r.did_bot_minus_human.iloc[0]), 4),
                "sig": bool(r.sig.iloc[0]),
            }
    for feat in ["evasion_index", "ip_ent", "compliance", "log_hits"]:
        r = zresp[zresp.feature == feat]
        if not r.empty and "did_z" in r.columns:
            headline["normalized_response_did"][feat] = {
                "bot_mean_z": round(float(r.bot_mean_z.iloc[0]), 3),
                "human_mean_z": round(float(r.human_mean_z.iloc[0]), 3),
                "did_z": round(float(r.did_z.iloc[0]), 3),
                "ci": [round(float(r.ci_lo.iloc[0]), 3), round(float(r.ci_hi.iloc[0]), 3)],
                "sig": bool(r.sig.iloc[0]),
            }
    headline["n_bot_agents_with_events"] = int(len(bot))
    headline["n_human_agents_with_events"] = int(len(hum))
    return {"headline": headline, "did_table": str(config.TAB_DIR / "B_difference_in_differences.csv")}


def _did_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = config.BOOTSTRAP_N) -> tuple[float, float, float]:
    """Bootstrap CI for mean(a) - mean(b)."""
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(config.RANDOM_STATE)
    ba = rng.choice(a, size=(n_boot, len(a)), replace=True).mean(axis=1)
    bb = rng.choice(b, size=(n_boot, len(b)), replace=True).mean(axis=1)
    diff = ba - bb
    return (float(a.mean() - b.mean()), float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5)))


def _mean_curve(segs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if not segs:
        return np.array([]), np.array([])
    M = np.vstack(segs)
    return M.mean(axis=0), M.std(axis=0) / np.sqrt(len(segs))


def _plot_response_curves(class_curves: dict, feature: str) -> None:
    offsets = np.arange(-W, W + 1)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    for label, color in [("bot", "tab:red"), ("human", "tab:blue"), ("chrome", "tab:green")]:
        if label not in class_curves:
            continue
        m, se = _mean_curve(class_curves[label][feature])
        if m.size == 0:
            continue
        ax.plot(offsets, m, color=color, marker="o", ms=3, label=f"{label} (n_events={len(class_curves[label][feature])})")
        ax.fill_between(offsets, m - 1.96 * se, m + 1.96 * se, color=color, alpha=0.15)
    ax.axvline(0, color="k", ls="--", lw=1, label="filter-pressure event")
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_xlabel("days relative to filter-pressure event")
    ax.set_ylabel(f"{feature}  (baseline-subtracted)")
    ax.set_title(f"Behavioral response to filter pressure: {feature}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / f"B_response_curve_{feature}.png", dpi=150)
    plt.close(fig)


def _plot_did(did: pd.DataFrame) -> None:
    if did.empty:
        return
    d = did.set_index("feature").reindex(RESPONSE_FEATURES).dropna(subset=["did_bot_minus_human"])
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    yerr = np.vstack([d.did_bot_minus_human - d.ci_lo, d.ci_hi - d.did_bot_minus_human])
    colors = ["tab:red" if s else "lightgray" for s in d.sig]
    ax.bar(d.index, d.did_bot_minus_human, yerr=yerr, color=colors, capsize=3)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("DiD: (bot post-pre) - (human post-pre)")
    ax.set_title("Filter-response difference-in-differences\n(red = 95% CI excludes 0)")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "B_difference_in_differences.png", dpi=150)
    plt.close(fig)


def _plot_normalized_response(zresp: pd.DataFrame) -> None:
    """Grouped bars: normalized post-filter response magnitude, bot vs human vs chrome."""
    if zresp.empty:
        return
    feats = [f for f in RESPONSE_FEATURES if f != "resistance"]
    z = zresp.set_index("feature").reindex(feats)
    labels = [l for l in ["bot", "human", "chrome"] if f"{l}_mean_z" in z.columns]
    if not labels:
        return
    x = np.arange(len(feats))
    width = 0.8 / len(labels)
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    colors = {"bot": "tab:red", "human": "tab:blue", "chrome": "tab:green"}
    for k, lab in enumerate(labels):
        ax.bar(x + (k - (len(labels) - 1) / 2) * width, z[f"{lab}_mean_z"], width,
               label=lab, color=colors.get(lab, "gray"))
    ax.set_xticks(x)
    ax.set_xticklabels(feats, rotation=45, ha="right")
    ax.set_ylabel("normalized response  |post-pre| / own daily volatility")
    ax.set_title("Magnitude of behavioral response to filter pressure (sign-agnostic)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "B_normalized_response.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    import datasource as ds

    src = ds.get_source()
    agents = {lbl: src.load_all([lbl]) for lbl in config.CLASSES}
    import json

    print(json.dumps(run(agents)["headline"], indent=2))
