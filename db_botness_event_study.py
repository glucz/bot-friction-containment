"""Does the CLASSIFIER'S OWN score fall after friction events?

Paper D measures observability collapse in a hand-built three-feature space
(404 share, robots.txt rate, log volume). The transformer classifier of the
sibling repo -- the instrument that actually assigns the botness score b that the
theory's friction schedule phi(b) consumes -- has never been run as an event
outcome. The L_E mechanism predicts b should fall too. This tests it.

DESIGN. Finding 1's configuration, because it is the only one the classifier's
input files support: the per-agent TSVs carry no 401/403/429 column, so events
are agent-local spikes in the non-200 share (resistance > mean + 1.5 sd of the
agent's own active-day series, min_gap = 5 so windows cannot overlap), exactly
signals.filter_pressure_events. The outcome is the classifier's P(bot) curve,
window-aligned to each window's last day. DiD is 177 confirmed bots against the
200 hand-verified humans, bootstrap 95% CI clustered by agent.

SCOPE. This is NOT the 401/403/429 rule of Findings 2-6, and these are the
curated sets, not the 7,812-agent arm. It answers "does b move around friction
in the same direction O does", not "reproduce Finding 2 in b".

Usage: python db_botness_event_study.py
Outputs: outputs/DB_botness_event_study.json
         outputs/figures/DB_botness_event_study.png
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import botness_inference as bi
import config
import signals

SEQ_LEN = 30
W = config.EVENT_WINDOW
THR = 0.234

RENAME = {
    "day": "day", "traffic_hits": "hits", "404_pct": "p404", "200_pct": "p200",
    "robots_pct": "robots", "img_pct": "img", "html_pct": "html", "js_pct": "js",
    "php_pct": "php", "folder_pct": "folder", "ip_entropy": "ip_ent",
    "domain_entropy": "dom_ent", "country_entropy": "ctry_ent",
    "url_entropy": "url_ent", "husource": "husource",
}


def agent_frame(path):
    """Parse with the PUBLISHED parser, rename to pipeline schema, attach b(t)."""
    repo = str(config.CLASSIFIER_REPO)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    import bot_training_and_inference9 as clf

    df = clf.parse_user_agent_file(path).rename(columns=RENAME)
    probs = bi.botness_curve(path)
    if probs is None or len(df) < SEQ_LEN:
        return None
    expected = len(df) - SEQ_LEN + 1
    if len(probs) != expected:
        raise RuntimeError("alignment mismatch on %s: %d probs vs %d windows"
                           % (path, len(probs), expected))
    b = np.full(len(df), np.nan)
    b[SEQ_LEN - 1:] = probs      # window i covers rows [i, i+29] -> its LAST day
    df["botness"] = b
    return signals.add_derived(df)


def agent_events(path):
    df = agent_frame(path)
    if df is None:
        return None
    act = df[df["hits"] > 0]
    if len(act) < 3:
        return None
    rows = []
    for i in signals.filter_pressure_events(act):
        if i < W or i + W >= len(act):
            continue
        pre_b = signals.window_mean(act, i, "botness", "pre", W)
        post_b = signals.window_mean(act, i, "botness", "post", W)
        if not np.isfinite(pre_b) or not np.isfinite(post_b):
            continue          # window lands in the first 29 days, b undefined there
        pre_r = signals.window_mean(act, i, "resistance", "pre", W)
        post_r = signals.window_mean(act, i, "resistance", "post", W)
        pre_v = signals.window_mean(act, i, "hits", "pre", W)
        post_v = signals.window_mean(act, i, "hits", "post", W)
        rows.append({
            "d_botness": post_b - pre_b,
            "d_resistance": post_r - pre_r,
            "vol_ratio": (post_v / pre_v) if pre_v else np.nan,
        })
    if not rows:
        return None
    return {"events": rows,
            "mean_b": float(np.nanmean(act["botness"])),
            "frac_above_thr": float(np.nanmean(act["botness"] >= THR))}


def collect(pattern):
    per_agent, mean_b, frac = [], [], []
    for p in sorted(glob.glob(str(Path(config.DATA_DIR) / pattern))):
        r = agent_events(p)
        if r is None:
            continue
        per_agent.append(r["events"])
        mean_b.append(r["mean_b"])
        frac.append(r["frac_above_thr"])
    return per_agent, mean_b, frac


def _mean(per_agent, key, mask=None):
    vals = [e[key] for ev in per_agent for e in ev if mask is None or mask(e)]
    return float(np.nanmean(vals)) if vals else float("nan")


def did_boot(bots, humans, key, mask=None, n=4000, seed=42):
    rng = np.random.default_rng(seed)
    obs = _mean(bots, key, mask) - _mean(humans, key, mask)
    draws = []
    for _ in range(n):
        b = [bots[i] for i in rng.integers(0, len(bots), len(bots))]
        h = [humans[i] for i in rng.integers(0, len(humans), len(humans))]
        d = _mean(b, key, mask) - _mean(h, key, mask)
        if np.isfinite(d):
            draws.append(d)
    return {
        "did": round(float(obs), 4),
        "ci95": [round(float(np.percentile(draws, 2.5)), 4),
                 round(float(np.percentile(draws, 97.5)), 4)],
        "bot_mean": round(_mean(bots, key, mask), 4),
        "human_mean": round(_mean(humans, key, mask), 4),
        "n_bot_events": sum(1 for ev in bots for e in ev if mask is None or mask(e)),
        "n_human_events": sum(1 for ev in humans for e in ev if mask is None or mask(e)),
    }


def main():
    if not bi.available():
        raise SystemExit("torch or botness_model.pt unavailable -- cannot run")

    bots, bot_mb, bot_fr = collect("*.bot")
    hums, hum_mb, hum_fr = collect("*.human")

    def stealth(e):
        return np.isfinite(e["vol_ratio"]) and e["vol_ratio"] >= 0.5

    out = {
        "design": ("Finding 1 configuration: events = agent-local non-200 share spike "
                   "(resistance > mean + 1.5sd, min_gap 5); outcome = classifier P(bot); "
                   "DiD = confirmed bots minus hand-verified humans; bootstrap by agent."),
        "scope": ("Curated sets (177 .bot, 200 .human), NOT the 7,812 arm. The TSVs carry "
                  "no 401/403/429 column, so this is NOT the rule of Findings 2-6."),
        "n_bot_agents": len(bots),
        "n_human_agents": len(hums),
        "validation_model_wired_correctly": {
            "mean_b_bots": round(float(np.mean(bot_mb)), 4),
            "mean_b_humans": round(float(np.mean(hum_mb)), 4),
            "frac_windows_above_thr_bots": round(float(np.mean(bot_fr)), 4),
            "frac_windows_above_thr_humans": round(float(np.mean(hum_fr)), 4),
            "threshold": THR,
        },
        "botness_DiD": did_boot(bots, hums, "d_botness"),
        "botness_DiD_stealth_only": did_boot(bots, hums, "d_botness", stealth),
        "resistance_DiD_sanity": did_boot(bots, hums, "d_resistance"),
    }

    config.ensure_dirs()
    (config.OUT_DIR / "DB_botness_event_study.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=150)
    ax = axes[0]
    ax.hist([e["d_botness"] for ev in bots for e in ev], bins=40, alpha=0.6, density=True,
            color="tab:red", label="confirmed bots (%d events)" % out["botness_DiD"]["n_bot_events"])
    ax.hist([e["d_botness"] for ev in hums for e in ev], bins=40, alpha=0.6, density=True,
            color="tab:blue", label="verified humans (%d)" % out["botness_DiD"]["n_human_events"])
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("change in classifier P(bot), post minus pre")
    ax.set_ylabel("density")
    ax.set_title("Per-event change in the classifier's own score\nDiD = %.4f [%.4f, %.4f]"
                 % (out["botness_DiD"]["did"], out["botness_DiD"]["ci95"][0],
                    out["botness_DiD"]["ci95"][1]))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    keys = ["botness_DiD", "botness_DiD_stealth_only"]
    ax.errorbar(range(2), [out[k]["did"] for k in keys],
                yerr=[[out[k]["did"] - out[k]["ci95"][0] for k in keys],
                      [out[k]["ci95"][1] - out[k]["did"] for k in keys]],
                fmt="o", capsize=5, color="tab:purple")
    ax.axhline(0, color="k", lw=1, ls="--")
    ax.set_xticks(range(2))
    ax.set_xticklabels(["all events", "stealth only\n(vol >= 50% of pre)"], fontsize=9)
    ax.set_ylabel("DiD in classifier P(bot)")
    ax.set_title("Does the classifier's score fall after friction?")
    ax.grid(alpha=0.3)

    fig.suptitle("Classifier-score event study (curated sets, non-200 spike rule): "
                 "the test Paper D's observability collapse never ran", fontsize=10)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "DB_botness_event_study.png", dpi=150)
    plt.close(fig)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
