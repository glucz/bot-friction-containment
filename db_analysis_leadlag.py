"""
db_analysis_leadlag.py — Empirical lead-lag between friction and bot behavior.

Sign convention (VERIFIED with synthetic known-lag test):
    ccf = correlate(RESPONSE, FRICTION); positive lag  => response FOLLOWS friction
    (correlate(in1,in2) peaks at k where in1[n] ~ in2[n-k])

For each agent we compute the normalized CCF between p_block (friction the agent
experiences) and each response channel (p404 probing, log-volume), then split
the lag axis:
    POST side (+1..+14): behavior follows friction  -> "responder", tau = peak lag
    PRE  side (-14..-1): behavior precedes friction -> "leader" (e.g. probing
                          triggers blocks; measures the DEFENDER's reaction time)
Lag 0 is excluded from the peak search: p404 and p_block share the same daily
denominator (hits), which builds in a compositional same-day correlation.

Outputs:
  - outputs/tables/DB_leadlag_per_agent.csv
  - outputs/figures/DB_leadlag_tau_distribution.png
  - outputs/DB_leadlag_SUMMARY.md
"""
from __future__ import annotations

import json
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

import config
import db_source as dbs
from roster import ROSTER_CSV

MIN_DAYS_LL = 30
MIN_BLOCK_DAYS = 3        # need real friction variation to correlate against
LAG_MAX = 14
OUT_TAB = config.TAB_DIR
OUT_FIG = config.FIG_DIR


def _selftest() -> None:
    """Verify the sign convention on a synthetic series with known +3 lag."""
    rng = np.random.default_rng(0)
    frict = rng.normal(0, 1, 300)
    resp = np.roll(frict, 3)                      # response follows friction by 3
    lags, ccf = _ccf(resp, frict)
    peak = int(lags[np.argmax(np.abs(ccf))])
    assert peak == 3, f"sign convention broken: expected +3, got {peak}"


def _ccf(resp: np.ndarray, frict: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r = (resp - np.nanmean(resp)) / (np.nanstd(resp) + 1e-9)
    f = (frict - np.nanmean(frict)) / (np.nanstd(frict) + 1e-9)
    ccf = signal.correlate(r, f, mode="full") / len(r)
    lags = signal.correlation_lags(len(r), len(f), mode="full")
    m = np.abs(lags) <= LAG_MAX
    return lags[m], ccf[m]


def _peak_sides(lags: np.ndarray, ccf: np.ndarray) -> dict:
    """Peak |ccf| on the post (+1..+LAG_MAX) and pre (-LAG_MAX..-1) sides."""
    out = {}
    for side, mask in (("post", lags >= 1), ("pre", lags <= -1)):
        if not mask.any():
            out[f"{side}_lag"], out[f"{side}_ccf"] = np.nan, np.nan
            continue
        i = np.argmax(np.abs(ccf[mask]))
        out[f"{side}_lag"] = int(lags[mask][i])
        out[f"{side}_ccf"] = float(ccf[mask][i])
    return out


def _boot_median(x: np.ndarray, n: int = 2000, seed: int = 42) -> tuple[float, float, float]:
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(x, (n, len(x)), True), axis=1)
    return (float(np.median(x)), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5)))


def run() -> dict:
    _selftest()
    config.ensure_dirs()
    roster = pd.read_csv(ROSTER_CSV)

    rows = []
    for _, r in roster.iterrows():
        cp = dbs._cache_path(int(r["a_id"]))
        if not cp.exists():
            continue
        df = pd.read_parquet(cp)
        if len(df) < MIN_DAYS_LL or int((df["cblock"] > 0).sum()) < MIN_BLOCK_DAYS:
            continue
        frict = df["p_block"].to_numpy(float)
        if np.nanstd(frict) < 1e-9:
            continue
        grp = "bot" if bool(r["is_bot"]) else ("control" if bool(r["is_control"]) else "other")

        rec = {"a_id": int(r["a_id"]), "group": grp, "n_days": len(df),
               "n_block_days": int((df["cblock"] > 0).sum())}
        for name, series in (("p404", df["p404"].to_numpy(float)),
                             ("vol", np.log1p(df["hits"].to_numpy(float)))):
            if np.nanstd(series) < 1e-9:
                continue
            lags, ccf = _ccf(series, frict)
            s = _peak_sides(lags, ccf)
            rec[f"{name}_post_lag"] = s["post_lag"]; rec[f"{name}_post_ccf"] = s["post_ccf"]
            rec[f"{name}_pre_lag"] = s["pre_lag"];   rec[f"{name}_pre_ccf"] = s["pre_ccf"]
            rec[f"{name}_responder"] = bool(abs(s["post_ccf"]) > abs(s["pre_ccf"]))
        rows.append(rec)

    pa = pd.DataFrame(rows)
    pa.to_csv(OUT_TAB / "DB_leadlag_per_agent.csv", index=False)

    res = {"n_agents": len(pa),
           "n_bot": int((pa.group == "bot").sum()),
           "n_control": int((pa.group == "control").sum())}

    for grp in ("bot", "control"):
        g = pa[pa.group == grp]
        if g.empty or "p404_responder" not in g:
            continue
        responders = g[g["p404_responder"] == True]          # noqa: E712
        tau_med, lo, hi = _boot_median(responders["p404_post_lag"].to_numpy(float))
        res[grp] = {
            "frac_responders": round(float(g["p404_responder"].mean()), 3),
            "tau_response_days": {"median": tau_med, "ci": [round(lo, 2), round(hi, 2)]},
            "defender_reaction_days": _boot_median(
                -g.loc[g["p404_responder"] == False, "p404_pre_lag"].to_numpy(float))[0],  # noqa: E712
            "median_post_ccf_sign": round(float(np.median(responders["p404_post_ccf"])), 3),
        }

    # ---- figure: tau distributions for responders ----
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    for grp, color in (("bot", "tab:red"), ("control", "tab:blue")):
        g = pa[(pa.group == grp) & (pa.get("p404_responder") == True)]  # noqa: E712
        if g.empty:
            continue
        ax.hist(g["p404_post_lag"], bins=np.arange(0.5, LAG_MAX + 1.5), alpha=0.55,
                label=f"{grp} responders (n={len(g)})", color=color, density=True)
    ax.set_xlabel("response lag after friction (active days)")
    ax.set_ylabel("density")
    ax.set_title("Lead-lag: probing response AFTER friction (verified sign convention)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT_FIG / "DB_leadlag_tau_distribution.png", dpi=150)
    plt.close(fig)

    # ---- summary ----
    L = ["# Lead-lag: friction <-> behavior (sign convention verified by self-test)\n",
         f"Agents: {res['n_agents']} (bots {res['n_bot']}, controls {res['n_control']}); "
         f"filters: >={MIN_DAYS_LL} days, >={MIN_BLOCK_DAYS} block days, lag 0 excluded "
         "(compositional same-day coupling).\n"]
    for grp in ("bot", "control"):
        if grp not in res:
            continue
        v = res[grp]
        L.append(f"## {grp}\n"
                 f"- responders (behavior follows friction): {v['frac_responders']*100:.1f}%\n"
                 f"- tau_response median {v['tau_response_days']['median']} d, "
                 f"CI {v['tau_response_days']['ci']}\n"
                 f"- leaders' implied defender reaction time: {v['defender_reaction_days']} d\n"
                 f"- median post-peak CCF sign: {v['median_post_ccf_sign']} "
                 "(negative = probing is SUPPRESSED after friction)\n")
    L.append("\n## Paper mapping\n"
             "- tau_response = bot adaptation delay tau in the SLA-market companion's Theorem 4 / this paper's Theorem 5.\n"
             "- The pre-side (probing precedes blocks) measures the DEFENDER's reaction "
             "time — the other delay in the same loop. Both delays enter the Nyquist "
             "denominator; the loop period is bounded below by their sum.\n")
    (config.OUT_DIR / "DB_leadlag_SUMMARY.md").write_text("\n".join(L), encoding="utf-8")

    print(json.dumps(res, indent=2, default=str))
    return res


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        run()
