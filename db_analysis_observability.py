"""
db_analysis_observability.py — Observability (b_obs) around real friction events.

Observability O(t) = || (v(t) - mu_H) / sigma_H ||, the daily distance of an
agent's BEHAVIOR vector from the control-population centroid. Operationalizes
b_obs, the signal Papers D and F assume the defender's loop depends on.

CRITICAL design choice (fixes a circularity in the first version): the feature
vector contains ONLY agent-chosen behavior — [p404, robots_rate, log1p(hits)].
It must NOT contain p_block / resistance / p200: those are the defender's
response, so a block event would inflate O(t) mechanically and the "post-block
observability" would be measuring the block itself.

Analyses:
  1. DiD of post-pre O around block events: bots vs controls.
  2. Retreat vs stealth split: post-block activity drop >50% = "retreated";
     active but with lower O = "went stealthy". Only the latter is b_obs decay.
  3. Calendar trend of monthly O (interpretation derived FROM the slopes).
  4. Light vs heavy friction frontier (f_switch): events split at the
     within-group median of event-day p_block; compare dO and d(log-volume).

Outputs: tables/DB_observability_per_agent.csv, tables/DB_observability_monthly.csv,
figures/DB_observability_decay.png, figures/DB_observability_trend.png,
DB_observability_SUMMARY.md
"""
from __future__ import annotations

import json
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import db_source as dbs
from db_battery import _block_events, _did, _winmean
from roster import ROSTER_CSV

MIN_DAYS_OBS = 20
EVENT_W = 5
RETREAT_DROP = 0.5         # >50% post-block volume drop = retreated, not stealthy
OUT_TAB = config.TAB_DIR
OUT_FIG = config.FIG_DIR


def _behavior_matrix(df: pd.DataFrame) -> np.ndarray:
    """Agent-chosen behavior only — no defender-response channels."""
    return np.column_stack([
        df["p404"].to_numpy(float),
        df["robots_rate"].to_numpy(float),
        np.log1p(df["hits"].to_numpy(float)),
    ])


def run() -> dict:
    config.ensure_dirs()
    roster = pd.read_csv(ROSTER_CSV)

    # ---- pass 1: control-population centroid in behavior space ----
    ctl_means = []
    for _, r in roster[roster["is_control"] == True].iterrows():     # noqa: E712
        cp = dbs._cache_path(int(r["a_id"]))
        if not cp.exists():
            continue
        df = pd.read_parquet(cp)
        if len(df) >= MIN_DAYS_OBS:
            ctl_means.append(_behavior_matrix(df).mean(axis=0))
    ctl_means = np.array(ctl_means)
    mu_H = ctl_means.mean(axis=0)
    sigma_H = ctl_means.std(axis=0) + 1e-9
    print(f"control centroid from {len(ctl_means)} agents: mu={np.round(mu_H,4)} sigma={np.round(sigma_H,4)}")

    # ---- pass 2: per-agent O(t), event DiD, monthly trend, frontier ----
    rows, monthly = [], {}
    events_pool = []        # per-event records for the light/heavy frontier
    for _, r in roster.iterrows():
        cp = dbs._cache_path(int(r["a_id"]))
        if not cp.exists():
            continue
        df = pd.read_parquet(cp)
        if len(df) < MIN_DAYS_OBS:
            continue
        grp = "bot" if bool(r["is_bot"]) else ("control" if bool(r["is_control"]) else "other")

        O = np.linalg.norm((_behavior_matrix(df) - mu_H) / sigma_H, axis=1)
        vol = np.log1p(df["hits"].to_numpy(float))
        pblk = df["p_block"].to_numpy(float)

        rec = {"a_id": int(r["a_id"]), "group": grp, "n_days": len(df),
               "mean_O": float(np.mean(O))}

        d_obs, kinds = [], []
        for i in _block_events(df):
            if i < EVENT_W or i + EVENT_W >= len(df):
                continue
            dO = _winmean(O, i, "post", EVENT_W) - _winmean(O, i, "pre", EVENT_W)
            pre_v = _winmean(vol, i, "pre", EVENT_W)
            post_v = _winmean(vol, i, "post", EVENT_W)
            retreated = np.expm1(post_v) < (1 - RETREAT_DROP) * np.expm1(pre_v)
            d_obs.append(dO)
            kinds.append("retreat" if retreated else "stealth")
            events_pool.append({"group": grp, "intensity": float(pblk[i]),
                                "dO": float(dO), "dvol": float(post_v - pre_v),
                                "kind": kinds[-1]})
        rec["n_events"] = len(d_obs)
        rec["d_O"] = float(np.mean(d_obs)) if d_obs else np.nan
        stealth = [d for d, k in zip(d_obs, kinds) if k == "stealth"]
        rec["d_O_stealth"] = float(np.mean(stealth)) if stealth else np.nan
        rec["frac_retreat"] = (kinds.count("retreat") / len(kinds)) if kinds else np.nan

        mser = pd.to_datetime(df["date"]).dt.to_period("M").astype(str).to_numpy()
        for m in np.unique(mser):
            monthly.setdefault((grp, m), []).append(float(np.mean(O[mser == m])))
        rows.append(rec)

    pa = pd.DataFrame(rows)
    pa.to_csv(OUT_TAB / "DB_observability_per_agent.csv", index=False)

    bot, ctl = pa[pa.group == "bot"], pa[pa.group == "control"]
    did_all = _did(bot["d_O"].to_numpy(float), ctl["d_O"].to_numpy(float))
    did_stealth = _did(bot["d_O_stealth"].to_numpy(float), ctl["d_O_stealth"].to_numpy(float))

    # ---- calendar trend (interpretation FROM the data) ----
    mrows = [{"group": g, "month": m, "mean_O": float(np.mean(v)), "n_agents": len(v)}
             for (g, m), v in monthly.items()]
    mdf = pd.DataFrame(mrows).sort_values(["group", "month"])
    mdf.to_csv(OUT_TAB / "DB_observability_monthly.csv", index=False)
    slopes = {}
    for g in ("bot", "control"):
        s = mdf[(mdf.group == g) & (mdf.n_agents >= 20)].reset_index(drop=True)
        slopes[g] = float(np.polyfit(np.arange(len(s)), s["mean_O"], 1)[0]) if len(s) > 12 else np.nan

    # ---- light vs heavy frontier (within-group median split on intensity) ----
    ev = pd.DataFrame(events_pool)
    frontier = {}
    for g in ("bot", "control"):
        e = ev[ev.group == g]
        if len(e) < 20:
            continue
        med = e["intensity"].median()
        light, heavy = e[e.intensity <= med], e[e.intensity > med]
        frontier[g] = {
            "light": {"dO": round(float(light["dO"].mean()), 4),
                      "dvol": round(float(light["dvol"].mean()), 4),
                      "frac_retreat": round(float((light.kind == "retreat").mean()), 3)},
            "heavy": {"dO": round(float(heavy["dO"].mean()), 4),
                      "dvol": round(float(heavy["dvol"].mean()), 4),
                      "frac_retreat": round(float((heavy.kind == "retreat").mean()), 3)},
            "n_events": len(e),
        }

    trend_txt = (f"bot slope {slopes.get('bot'):+.5f}/mo, control slope {slopes.get('control'):+.5f}/mo "
                 f"-> bots {'diverge from' if (slopes.get('bot') or 0) > (slopes.get('control') or 0) else 'converge toward'} "
                 "controls relative to the control drift")

    res = {"n_agents": len(pa), "n_bot": len(bot), "n_control": len(ctl),
           "dO_DiD_all_events": did_all,
           "dO_DiD_stealth_only": did_stealth,
           "frac_bot_events_retreat": round(float(ev[ev.group == 'bot'].kind.eq('retreat').mean()), 3) if len(ev) else None,
           "calendar_slopes_per_month": {k: (round(v, 5) if np.isfinite(v) else None) for k, v in slopes.items()},
           "calendar_interpretation": trend_txt,
           "frontier_light_vs_heavy": frontier}

    # ---- figures ----
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    labels = ["all events", "stealth only"]
    bvals = [did_all["bot"], did_stealth["bot"]]
    cvals = [did_all["ctrl"], did_stealth["ctrl"]]
    x = np.arange(2)
    ax.bar(x - 0.18, bvals, 0.36, color="tab:red", label="bots")
    ax.bar(x + 0.18, cvals, 0.36, color="tab:blue", label="controls")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("dO (post - pre block), behavior-only features")
    ax.set_title("Observability change after real block events")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(OUT_FIG / "DB_observability_decay.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    for g, color in (("bot", "tab:red"), ("control", "tab:blue")):
        s = mdf[(mdf.group == g) & (mdf.n_agents >= 20)].sort_values("month")
        ax.plot(range(len(s)), s["mean_O"], color=color, marker="o", ms=3, label=g)
    ax.set_xlabel("month index (2019-2023)"); ax.set_ylabel("mean O")
    ax.set_title("Observability over calendar time (behavior-only features)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT_FIG / "DB_observability_trend.png", dpi=150)
    plt.close(fig)

    # ---- summary ----
    def _fmt(d):
        if d["did"] is None:
            return "insufficient data"
        return (f"bot {d['bot']:+.4f} vs ctrl {d['ctrl']:+.4f}, DiD {d['did']:+.4f} "
                f"CI {d['ci']} {'SIGNIFICANT' if d['sig'] else '(ns)'}")
    L = ["# Observability (b_obs) around real friction — behavior-only features\n",
         f"Agents: {len(pa)} (bots {len(bot)}, controls {len(ctl)}). Features: p404, "
         "robots_rate, log-volume; defender-response channels EXCLUDED by design "
         "(no circularity).\n",
         f"\n## dO after block events\n- all events: {_fmt(did_all)}\n"
         f"- stealth-only (volume kept within 50%): {_fmt(did_stealth)}\n"
         f"- bot events classified retreat: {res['frac_bot_events_retreat']}\n",
         f"\n## Calendar trend\n- {trend_txt}\n",
         "\n## Light vs heavy friction (f_switch frontier)\n"]
    for g, fr in frontier.items():
        L.append(f"- {g} (n={fr['n_events']} events): light dO {fr['light']['dO']:+.4f}, "
                 f"dvol {fr['light']['dvol']:+.4f}, retreat {fr['light']['frac_retreat']}  |  "
                 f"heavy dO {fr['heavy']['dO']:+.4f}, dvol {fr['heavy']['dvol']:+.4f}, "
                 f"retreat {fr['heavy']['frac_retreat']}")
    (config.OUT_DIR / "DB_observability_SUMMARY.md").write_text("\n".join(L), encoding="utf-8")

    print(json.dumps(res, indent=2, default=str))
    return res


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        run()
