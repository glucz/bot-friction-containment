"""
Analysis D -- Population strategy mix and its lifecycle evolution
              (three-strategy replicator calibration)

the population-dynamics companion models the bot population as a mix of three strategies -- visible (s_V),
evasive (s_E), mimicry (s_M) -- whose proportions x_V, x_E, x_M evolve by
replicator dynamics. Its draft (Section 6.1) proposes fitting a 3-component
mixture to AGWA and mapping the components to the three strategies. We do exactly
that, on the standardized agent-level behavioral profiles, then:

  1. Report the empirical population mix per class -> the population-dynamics companion's initial
     conditions (x_V^0, x_E^0, x_M^0) for the bot population.
  2. Re-classify each agent's early / mid / late lifecycle behavior and track how
     the mix shifts -> direct evidence that strategies evolve over time
     (replicator dynamics is empirically warranted, not just assumed).

Cluster -> strategy mapping is by centroid signature:
  visible  = high volume, high success (200), low network spread (easily seen)
  evasive  = high network spread (IP/domain/country fan-out)
  mimicry  = low volume, human-like, low detectability

Caveat: lifecycle here is agent age (days since first observation), so the mix
shift is within-agent maturation. Calendar-time population co-evolution needs the
DB re-derivation (see datasource.DBSource).
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

import config
import signals

# Agent-level features spanning the three strategy axes.
PROFILE_FEATURES = [
    "log_hits", "p200", "evasion_index", "ip_ent", "url_ent",
    "robots", "html", "php", "activity_rate", "husource",
]


def _agent_profile(act: pd.DataFrame, lifespan: int) -> dict | None:
    if len(act) == 0:
        return None
    a = signals.add_derived(act)
    prof = {
        "log_hits": float(np.log1p(a["hits"].mean())),
        "p200": float(a["p200"].mean()),
        "evasion_index": float(a["evasion_index"].mean()),
        "ip_ent": float(a["ip_ent"].mean()),
        "url_ent": float(a["url_ent"].mean()),
        "robots": float(a["robots"].mean()),
        "html": float(a["html"].mean()),
        "php": float(a["php"].mean()),
        "activity_rate": float(len(a) / max(1, lifespan)),
        "husource": float(a["husource"].mean()),
    }
    return prof


def _map_clusters(centroids: pd.DataFrame) -> dict[int, str]:
    """Assign each GMM component to visible / evasive / mimicry by signature."""
    c = centroids.copy()
    # z-score columns across the 3 clusters so signatures are comparable
    cz = (c - c.mean()) / (c.std(ddof=0) + 1e-9)
    visible_score = cz["log_hits"] + cz["p200"] - cz["evasion_index"]
    evasive_score = cz["evasion_index"] + cz["ip_ent"] + cz["url_ent"]
    mimicry_score = -cz["log_hits"] + cz["husource"] - cz["p200"]
    mapping: dict[int, str] = {}
    # Greedy assignment so each label is used once.
    scores = {"visible": visible_score, "evasive": evasive_score, "mimicry": mimicry_score}
    taken: set[int] = set()
    for strat in ["evasive", "visible", "mimicry"]:  # assign most distinctive first
        order = scores[strat].sort_values(ascending=False).index.tolist()
        for idx in order:
            if idx not in taken:
                mapping[idx] = strat
                taken.add(idx)
                break
    # any leftover (shouldn't happen with 3 clusters)
    for idx in centroids.index:
        mapping.setdefault(idx, "visible")
    return mapping


def run(agents_by_label: dict[str, list]) -> dict:
    config.ensure_dirs()

    # --- agent-level profiles ----------------------------------------------
    prof_rows = []
    for label, agents in agents_by_label.items():
        for ag in agents:
            if ag.n_active < config.MIN_ACTIVE_DAYS:
                continue
            p = _agent_profile(ag.active, ag.lifespan_days)
            if p is None:
                continue
            p.update({"agent_id": ag.agent_id, "label": label})
            prof_rows.append(p)
    prof = pd.DataFrame(prof_rows)

    scaler = StandardScaler()
    X = scaler.fit_transform(prof[PROFILE_FEATURES].to_numpy())
    gmm = GaussianMixture(n_components=config.N_STRATEGY_CLUSTERS,
                          covariance_type="full", random_state=config.RANDOM_STATE, n_init=5)
    comp = gmm.fit_predict(X)
    prof["component"] = comp

    centroids = prof.groupby("component")[PROFILE_FEATURES].mean()
    cmap = _map_clusters(centroids)
    prof["strategy"] = prof["component"].map(cmap)
    prof.to_csv(config.TAB_DIR / "D_agent_strategy.csv", index=False)
    centroids.assign(strategy=[cmap[i] for i in centroids.index]).to_csv(
        config.TAB_DIR / "D_cluster_centroids.csv")

    # --- population mix per class ------------------------------------------
    mix = (
        prof.groupby("label")["strategy"]
        .value_counts(normalize=True)
        .unstack()
        .reindex(columns=["visible", "evasive", "mimicry"])
        .fillna(0.0)
    )
    mix.to_csv(config.TAB_DIR / "D_class_mix.csv")

    # --- lifecycle migration (within-agent) --------------------------------
    life_rows = []
    for label in ["bot", "chrome"]:
        agents = agents_by_label.get(label, [])
        stage_counts = {s: {"visible": 0, "evasive": 0, "mimicry": 0} for s in ["early", "mid", "late"]}
        for ag in agents:
            if ag.n_active < 3 * config.MIN_ACTIVE_DAYS:
                continue
            act = ag.active
            thirds = signals.lifecycle_thirds(act)
            for stage, seg in zip(["early", "mid", "late"], thirds):
                p = _agent_profile(seg, max(1, int(seg["day"].iloc[-1] - seg["day"].iloc[0] + 1)))
                if p is None:
                    continue
                xz = scaler.transform([[p[f] for f in PROFILE_FEATURES]])
                strat = cmap[int(gmm.predict(xz)[0])]
                stage_counts[stage][strat] += 1
        for stage in ["early", "mid", "late"]:
            tot = sum(stage_counts[stage].values()) or 1
            life_rows.append({
                "label": label, "stage": stage, "n": tot,
                **{s: stage_counts[stage][s] / tot for s in ["visible", "evasive", "mimicry"]},
            })
    lifecycle = pd.DataFrame(life_rows)
    lifecycle.to_csv(config.TAB_DIR / "D_lifecycle_mix.csv", index=False)

    _plot_class_mix(mix)
    _plot_lifecycle(lifecycle)

    # --- headline -----------------------------------------------------------
    headline = {"cluster_to_strategy": {int(k): v for k, v in cmap.items()}}
    for label in ("bot", "chrome", "human"):
        if label in mix.index:
            headline[f"{label}_mix"] = {s: round(float(mix.loc[label, s]), 3) for s in ["visible", "evasive", "mimicry"]}
    bl = lifecycle[lifecycle.label == "bot"]
    if not bl.empty:
        early = bl[bl.stage == "early"].iloc[0]
        late = bl[bl.stage == "late"].iloc[0]
        headline["bot_lifecycle_shift"] = {
            "x0_initial_conditions_early": {s: round(float(early[s]), 3) for s in ["visible", "evasive", "mimicry"]},
            "late": {s: round(float(late[s]), 3) for s in ["visible", "evasive", "mimicry"]},
            "delta_visible": round(float(late["visible"] - early["visible"]), 3),
            "delta_evasive": round(float(late["evasive"] - early["evasive"]), 3),
        }
    return {"headline": headline, "mix_table": str(config.TAB_DIR / "D_class_mix.csv")}


def _plot_class_mix(mix: pd.DataFrame) -> None:
    order = [l for l in ["bot", "chrome", "pothuman", "human"] if l in mix.index]
    m = mix.reindex(order)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    bottom = np.zeros(len(order))
    for strat, color in [("visible", "tab:red"), ("evasive", "tab:orange"), ("mimicry", "tab:purple")]:
        ax.bar(order, m[strat], bottom=bottom, label=strat, color=color)
        bottom += m[strat].to_numpy()
    ax.set_ylabel("population fraction")
    ax.set_title("Strategy mix by class (visible / evasive / mimicry)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "D_class_strategy_mix.png", dpi=150)
    plt.close(fig)


def _plot_lifecycle(lifecycle: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150, sharey=True)
    for ax, label in zip(axes, ["bot", "chrome"]):
        sub = lifecycle[lifecycle.label == label].set_index("stage").reindex(["early", "mid", "late"])
        if sub.dropna(how="all").empty:
            continue
        for strat, color in [("visible", "tab:red"), ("evasive", "tab:orange"), ("mimicry", "tab:purple")]:
            ax.plot(["early", "mid", "late"], sub[strat], marker="o", label=strat, color=color)
        ax.set_title(f"{label}: strategy mix over lifecycle")
        ax.set_ylabel("fraction")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "D_lifecycle_strategy_shift.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    import datasource as ds
    import json

    src = ds.get_source()
    agents = {lbl: src.load_all([lbl]) for lbl in config.CLASSES}
    print(json.dumps(run(agents)["headline"], indent=2))
