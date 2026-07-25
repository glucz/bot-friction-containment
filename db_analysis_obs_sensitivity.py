"""
db_analysis_obs_sensitivity.py — Robustness battery for the observability
collapse finding (this paper §7.4), addressing three reviewer requests:

  1. METRIC sensitivity: is the post-block DiD robust to the distance metric?
       - base:   Euclidean norm of per-feature z-scores (published pipeline)
       - maha:   full-covariance Mahalanobis on the same 3 features
       - novol:  z-score Euclidean WITHOUT the log-volume coordinate
                 (404 share + robots rate only — closes the "volume drop
                 reads as stealth" loophole at the metric level)
  2. RETREAT-THRESHOLD sensitivity: retreat fraction and stealth-only DiD
     at 30% / 50% (published) / 70% post-block volume-drop cuts.
  3. CONTROL-COMPOSITION sensitivity: DiD with the comparison arm restricted
     to (a) whitelisted-infrastructure controls only, (b) manually labeled
     human/browser controls only, (c) strict pool (whitelist+labeled) with
     the centroid ALSO re-estimated on that strict pool.

Reads only the local parquet cache (no DB access). Reuses the exact event
definition, windows, and bootstrap DiD of the published pipeline
(db_battery._block_events / _winmean / _did; db_analysis_observability
constants MIN_DAYS_OBS=20, EVENT_W=5).

Outputs: DB_obs_sensitivity.json, DB_obs_sensitivity_SUMMARY.md in OUT_DIR.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import config
import db_source as dbs
from db_battery import _block_events, _did, _winmean
from roster import ROSTER_CSV

MIN_DAYS_OBS = 20
EVENT_W = 5
RETREAT_DROPS = (0.3, 0.5, 0.7)

STRICT_ROLES = {"ctrl_white", "labeled_pothuman", "labeled_chrome"}
WHITE_ROLES = {"ctrl_white"}
LABELED_ROLES = {"labeled_pothuman", "labeled_chrome"}


def _behavior_matrix(df: pd.DataFrame) -> np.ndarray:
    """Agent-chosen behavior only — identical to the published pipeline."""
    return np.column_stack([
        df["p404"].to_numpy(float),
        df["robots_rate"].to_numpy(float),
        np.log1p(df["hits"].to_numpy(float)),
    ])


def run() -> dict:
    config.ensure_dirs()
    roster = pd.read_csv(ROSTER_CSV)
    role_of = dict(zip(roster["a_id"].astype(int), roster["role"]))

    # ---- pass 1: control centroids (full pool and strict pool) -------------
    ctl_means, strict_means = [], []
    for _, r in roster[roster["is_control"] == True].iterrows():   # noqa: E712
        cp = dbs._cache_path(int(r["a_id"]))
        if not cp.exists():
            continue
        df = pd.read_parquet(cp)
        if len(df) >= MIN_DAYS_OBS:
            m = _behavior_matrix(df).mean(axis=0)
            ctl_means.append(m)
            if r["role"] in STRICT_ROLES:
                strict_means.append(m)
    ctl_means = np.array(ctl_means)
    strict_means = np.array(strict_means)

    mu_H = ctl_means.mean(axis=0)
    sigma_H = ctl_means.std(axis=0) + 1e-9
    cov_H = np.cov(ctl_means, rowvar=False)
    cov_inv = np.linalg.pinv(cov_H)
    mu_S = strict_means.mean(axis=0)
    sigma_S = strict_means.std(axis=0) + 1e-9
    print(f"centroids: full pool n={len(ctl_means)}, strict pool n={len(strict_means)}")

    def O_base(B):
        return np.linalg.norm((B - mu_H) / sigma_H, axis=1)

    def O_maha(B):
        d = B - mu_H
        return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", d, cov_inv, d), 0.0))

    def O_novol(B):
        d = (B[:, :2] - mu_H[:2]) / sigma_H[:2]
        return np.linalg.norm(d, axis=1)

    def O_strict(B):
        return np.linalg.norm((B - mu_S) / sigma_S, axis=1)

    VARIANTS = {"base": O_base, "maha": O_maha, "novol": O_novol, "strict_centroid": O_strict}

    # ---- pass 2: per-agent event deltas under every variant ----------------
    rows = []
    n_bot_events = 0
    bot_event_retreats = {d: [] for d in RETREAT_DROPS}
    for _, r in roster.iterrows():
        a_id = int(r["a_id"])
        cp = dbs._cache_path(a_id)
        if not cp.exists():
            continue
        df = pd.read_parquet(cp)
        if len(df) < MIN_DAYS_OBS:
            continue
        grp = "bot" if bool(r["is_bot"]) else ("control" if bool(r["is_control"]) else "other")
        if grp == "other":
            continue

        B = _behavior_matrix(df)
        O = {k: fn(B) for k, fn in VARIANTS.items()}
        vol = np.log1p(df["hits"].to_numpy(float))

        d_obs = {k: [] for k in VARIANTS}
        retreat_flags = {d: [] for d in RETREAT_DROPS}
        for i in _block_events(df):
            if i < EVENT_W or i + EVENT_W >= len(df):
                continue
            pre_v = _winmean(vol, i, "pre", EVENT_W)
            post_v = _winmean(vol, i, "post", EVENT_W)
            for k in VARIANTS:
                d_obs[k].append(_winmean(O[k], i, "post", EVENT_W)
                                - _winmean(O[k], i, "pre", EVENT_W))
            for drop in RETREAT_DROPS:
                retreated = bool(np.expm1(post_v) < (1 - drop) * np.expm1(pre_v))
                retreat_flags[drop].append(retreated)
                if grp == "bot":
                    bot_event_retreats[drop].append(retreated)
        if not d_obs["base"]:
            continue
        if grp == "bot":
            n_bot_events += len(d_obs["base"])

        rec = {"a_id": a_id, "group": grp, "role": role_of.get(a_id, "?"),
               "n_events": len(d_obs["base"])}
        for k in VARIANTS:
            rec[f"dO_{k}"] = float(np.mean(d_obs[k]))
        # stealth-only per threshold (base metric): mean dO over non-retreat episodes
        for drop in RETREAT_DROPS:
            stealth = [d for d, rt in zip(d_obs["base"], retreat_flags[drop]) if not rt]
            rec[f"dO_stealth_{int(drop*100)}"] = float(np.mean(stealth)) if stealth else np.nan
        rows.append(rec)

    pa = pd.DataFrame(rows)
    bot = pa[pa.group == "bot"]
    ctl = pa[pa.group == "control"]
    print(f"agents with events: bot={len(bot)} control={len(ctl)}; bot events={n_bot_events}")

    res = {"n_bot_agents": int(len(bot)), "n_ctl_agents": int(len(ctl)),
           "n_bot_events": int(n_bot_events)}

    # 1. metric variants (all-control arm)
    res["metric"] = {k: _did(bot[f"dO_{k}"].to_numpy(), ctl[f"dO_{k}"].to_numpy())
                     for k in VARIANTS}

    # 2. retreat thresholds: fraction + stealth-only DiD (base metric)
    res["retreat"] = {}
    for drop in RETREAT_DROPS:
        key = int(drop * 100)
        res["retreat"][key] = {
            "frac_bot_events_retreat": round(float(np.mean(bot_event_retreats[drop])), 4),
            "stealth_only_did": _did(bot[f"dO_stealth_{key}"].to_numpy(),
                                     ctl[f"dO_stealth_{key}"].to_numpy()),
        }

    # 3. control-composition arms (base metric, published centroid)
    res["control_arm"] = {
        "all": res["metric"]["base"],
        "whitelist_only": _did(bot["dO_base"].to_numpy(),
                               ctl[ctl.role.isin(WHITE_ROLES)]["dO_base"].to_numpy()),
        "labeled_only": _did(bot["dO_base"].to_numpy(),
                             ctl[ctl.role.isin(LABELED_ROLES)]["dO_base"].to_numpy()),
        "strict_arm_and_centroid": _did(
            bot["dO_strict_centroid"].to_numpy(),
            ctl[ctl.role.isin(STRICT_ROLES)]["dO_strict_centroid"].to_numpy()),
    }
    for k in ("whitelist_only", "labeled_only", "strict_arm_and_centroid"):
        arm_roles = {"whitelist_only": WHITE_ROLES, "labeled_only": LABELED_ROLES,
                     "strict_arm_and_centroid": STRICT_ROLES}[k]
        res["control_arm"][k]["n_ctrl_agents"] = int(ctl.role.isin(arm_roles).sum())

    (config.OUT_DIR / "DB_obs_sensitivity.json").write_text(
        json.dumps(res, indent=2), encoding="utf-8")

    # ---- summary ------------------------------------------------------------
    def fmt(d):
        if d["did"] is None:
            return "n/a (too few agents)"
        return f"{d['did']:+.3f} CI [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}] (bot {d['bot']:+.3f} vs ctrl {d['ctrl']:+.3f})"

    L = ["# Observability-collapse robustness battery (metric / threshold / control arm)\n",
         f"- agents with usable events: {len(bot)} bots, {len(ctl)} controls; {n_bot_events} bot events\n",
         "## 1. Distance-metric variants (DiD of post-pre ΔO, bots vs all controls)"]
    for k in VARIANTS:
        L.append(f"- {k}: {fmt(res['metric'][k])}")
    L.append("\n## 2. Retreat-threshold sensitivity")
    for key, v in res["retreat"].items():
        L.append(f"- drop >{key}%: retreat fraction {v['frac_bot_events_retreat']:.3f}; "
                 f"stealth-only DiD {fmt(v['stealth_only_did'])}")
    L.append("\n## 3. Control-arm composition (base metric)")
    for k, v in res["control_arm"].items():
        n = v.get("n_ctrl_agents", len(ctl))
        L.append(f"- {k} (n={n}): {fmt(v)}")
    (config.OUT_DIR / "DB_obs_sensitivity_SUMMARY.md").write_text("\n".join(L), encoding="utf-8")
    try:
        print("\n".join(L))
    except UnicodeEncodeError:   # cp125x consoles: outputs are already on disk
        print("\n".join(L).encode("ascii", "replace").decode())
    return res


if __name__ == "__main__":
    run()
