"""
db_analysis_leadlag_robustness.py — Two robustness checks for db_analysis_leadlag.py
(this paper, Section 7.6). DOES NOT modify any pipeline file; reads its outputs/cache.

Headline aggregation mirrored from db_analysis_leadlag.py:
    - channel: p404 vs p_block (vol channel is computed there but NOT used in the
      headline responder-share / tau numbers)
    - responder  = p404_responder  (|post-side peak CCF| > |pre-side peak CCF|,
      lags +1..+14 vs -14..-1, lag 0 excluded)
    - tau        = p404_post_lag among responders
    - "median post-peak CCF" = SIGNED median of p404_post_ccf among responders

Test 1 (Kaplan-Meier): tau values piled at the +14-day search boundary are
right-censored, not true 14-day lags. Hand-rolled product-limit estimator with
Greenwood variance and log-log 95% CI; median CI by Brookmeyer-Crowley.

Test 2 (shuffle falsification): for a seeded subsample of the cohort, circularly
shift p_block by a random offset in [15, n-15] (preserves the autocorrelation of
both series, destroys their alignment) and recompute the post-side peak CCF and
responder flag 10 times per agent. Real structure should vanish under the null.

Outputs (all NEW files):
  - outputs/tables/DB_leadlag_KM_curves.csv
  - outputs/tables/DB_leadlag_shuffle_per_agent.csv   (checkpoint, resumable)
  - outputs/figures/DB_leadlag_KM.png
  - outputs/figures/DB_leadlag_shuffle_null.png
  - outputs/DB_leadlag_robustness_SUMMARY.md
  - outputs/leadlag_shuffle_progress.log
"""
from __future__ import annotations

import datetime as _dt
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

LAG_MAX = 14                      # mirrors db_analysis_leadlag.LAG_MAX
N_NULL = 10                       # null replicates per agent
N_BOT_SAMPLE = 3000
N_CTRL_SAMPLE = 1500
SHIFT_MIN = 15                    # circular shift offset range [15, n-15]
CHUNK = 250                       # agents per checkpoint flush

PER_AGENT_CSV = config.TAB_DIR / "DB_leadlag_per_agent.csv"
SHUFFLE_CSV = config.TAB_DIR / "DB_leadlag_shuffle_per_agent.csv"
KM_CSV = config.TAB_DIR / "DB_leadlag_KM_curves.csv"
FIG_KM = config.FIG_DIR / "DB_leadlag_KM.png"
FIG_NULL = config.FIG_DIR / "DB_leadlag_shuffle_null.png"
SUMMARY_MD = config.OUT_DIR / "DB_leadlag_robustness_SUMMARY.md"
PROGRESS_LOG = config.OUT_DIR / "leadlag_shuffle_progress.log"

Z95 = 1.959963984540054


# --------------------------------------------------------------------------
# CCF machinery: byte-for-byte mirror of db_analysis_leadlag.py conventions
# --------------------------------------------------------------------------
def _ccf(resp: np.ndarray, frict: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r = (resp - np.nanmean(resp)) / (np.nanstd(resp) + 1e-9)
    f = (frict - np.nanmean(frict)) / (np.nanstd(frict) + 1e-9)
    ccf = signal.correlate(r, f, mode="full") / len(r)
    lags = signal.correlation_lags(len(r), len(f), mode="full")
    m = np.abs(lags) <= LAG_MAX
    return lags[m], ccf[m]


def _peak_sides(lags: np.ndarray, ccf: np.ndarray) -> dict:
    out = {}
    for side, mask in (("post", lags >= 1), ("pre", lags <= -1)):
        if not mask.any():
            out[f"{side}_lag"], out[f"{side}_ccf"] = np.nan, np.nan
            continue
        i = np.argmax(np.abs(ccf[mask]))
        out[f"{side}_lag"] = int(lags[mask][i])
        out[f"{side}_ccf"] = float(ccf[mask][i])
    return out


def _selftest() -> None:
    rng = np.random.default_rng(0)
    frict = rng.normal(0, 1, 300)
    resp = np.roll(frict, 3)
    lags, ccf = _ccf(resp, frict)
    assert int(lags[np.argmax(np.abs(ccf))]) == 3, "sign convention broken"


# --------------------------------------------------------------------------
# Test 1: Kaplan-Meier with right-censoring at the 14-day search boundary
# --------------------------------------------------------------------------
def km_estimate(taus: np.ndarray) -> pd.DataFrame:
    """Product-limit estimator. Event if tau < LAG_MAX; censored if tau == LAG_MAX.

    Returns one row per distinct event time: t, n_risk, d, S, var (Greenwood),
    lo/hi (log-log 95% CI for S).
    """
    taus = np.asarray(taus, float)
    taus = taus[np.isfinite(taus)].astype(int)
    n0 = len(taus)
    event_times = np.sort(np.unique(taus[taus < LAG_MAX]))
    rows, S, gw_sum = [], 1.0, 0.0
    for t in event_times:
        n_risk = int((taus >= t).sum())          # censoring only at LAG_MAX
        d = int((taus == t).sum())
        S *= 1.0 - d / n_risk
        gw_sum += d / (n_risk * (n_risk - d)) if n_risk > d else np.inf
        var = S**2 * gw_sum
        if 0.0 < S < 1.0 and np.isfinite(var):   # log-log (stays in [0,1])
            se_ll = np.sqrt(var) / (abs(S * np.log(S)) + 1e-300)
            lo = S ** np.exp(Z95 * se_ll)
            hi = S ** np.exp(-Z95 * se_ll)
        else:
            lo = max(S - Z95 * np.sqrt(max(var, 0.0)), 0.0)
            hi = min(S + Z95 * np.sqrt(max(var, 0.0)), 1.0)
        rows.append({"t": int(t), "n_risk": n_risk, "d": d, "S": S,
                     "var_greenwood": var, "lo": lo, "hi": hi})
    df = pd.DataFrame(rows)
    df.attrs["n_total"] = n0
    df.attrs["n_censored"] = int((taus == LAG_MAX).sum())
    return df


def km_median_ci(km: pd.DataFrame) -> tuple[float, float, float]:
    """KM median = smallest t with S(t) <= 0.5; 95% CI by Brookmeyer-Crowley
    (times whose log-log CI for S contains 0.5)."""
    def first_t(cond: pd.Series) -> float:
        idx = km.index[cond]
        return float(km.loc[idx[0], "t"]) if len(idx) else np.nan
    med = first_t(km["S"] <= 0.5)
    lo = first_t(km["lo"] <= 0.5)        # earliest t where CI reaches 0.5
    hi = first_t(km["hi"] < 0.5)         # earliest t where even hi is below 0.5
    return med, lo, hi


def run_km(pa: pd.DataFrame) -> dict:
    res = {}
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    km_rows = []
    for grp, color in (("bot", "tab:red"), ("control", "tab:blue")):
        g = pa[(pa.group == grp) & (pa["p404_responder"] == True)]   # noqa: E712
        taus = g["p404_post_lag"].to_numpy(float)
        km = km_estimate(taus)
        med, lo, hi = km_median_ci(km)
        naive = float(np.median(taus[np.isfinite(taus)]))
        n_cens = km.attrs["n_censored"]
        res[grp] = {"n_responders": km.attrs["n_total"],
                    "n_censored_at_14": n_cens,
                    "pct_censored": round(100 * n_cens / km.attrs["n_total"], 2),
                    "naive_median": naive,
                    "km_median": med, "km_median_ci95": [lo, hi]}
        km.insert(0, "group", grp)
        km_rows.append(km)
        # step curve from S(0)=1
        ts = np.concatenate([[0], km["t"].to_numpy(float)])
        Ss = np.concatenate([[1.0], km["S"].to_numpy(float)])
        los = np.concatenate([[1.0], km["lo"].to_numpy(float)])
        his = np.concatenate([[1.0], km["hi"].to_numpy(float)])
        ax.step(ts, Ss, where="post", color=color, lw=2,
                label=f"{grp} responders (n={km.attrs['n_total']}, "
                      f"{n_cens} censored at +14)")
        ax.fill_between(ts, los, his, step="post", color=color, alpha=0.15)
        # censoring ticks at the boundary
        S_end = Ss[-1]
        ax.plot([LAG_MAX], [S_end], marker="|", ms=14, mew=2.5, color=color)
        ax.hlines(S_end, ts[-1], LAG_MAX, color=color, lw=2)
        if np.isfinite(med):
            ax.vlines(med, 0, 0.5, color=color, ls=":", lw=1.2)
    ax.axhline(0.5, color="gray", ls="--", lw=1, alpha=0.7)
    ax.set_xlabel("response lag after friction (active days)")
    ax.set_ylabel("S(t) = P(response lag > t)")
    ax.set_title("Kaplan-Meier response lag, lags at the +14 d search boundary "
                 "right-censored\n(p404 channel, responders only; "
                 "log-log Greenwood 95% bands; '|' = censoring at boundary)")
    ax.set_xlim(0, LAG_MAX + 0.5); ax.set_ylim(0, 1.02)
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG_KM, dpi=150); plt.close(fig)
    pd.concat(km_rows, ignore_index=True).to_csv(KM_CSV, index=False)
    return res


# --------------------------------------------------------------------------
# Test 2: circular-shift shuffle null
# --------------------------------------------------------------------------
def _log(msg: str) -> None:
    line = f"{_dt.datetime.now().isoformat(timespec='seconds')}  {msg}"
    with open(PROGRESS_LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    print(line, flush=True)


def _sample_cohort(pa: pd.DataFrame) -> pd.DataFrame:
    """Seeded (42) subsample of the lead-lag cohort with a valid p404 channel."""
    ok = pa[pa["p404_post_ccf"].notna()].sort_values("a_id")
    rng = np.random.default_rng(42)
    parts = []
    for grp, n_want in (("bot", N_BOT_SAMPLE), ("control", N_CTRL_SAMPLE)):
        g = ok[ok.group == grp]
        if len(g) > n_want:
            idx = rng.choice(len(g), size=n_want, replace=False)
            g = g.iloc[np.sort(idx)]
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def _shuffle_one(a_id: int, grp: str) -> dict | None:
    cp = dbs._cache_path(int(a_id))
    if not cp.exists():
        return None
    df = pd.read_parquet(cp)
    frict = df["p_block"].to_numpy(float)
    resp = df["p404"].to_numpy(float)
    n = len(df)
    if n < 2 * SHIFT_MIN or np.nanstd(frict) < 1e-9 or np.nanstd(resp) < 1e-9:
        return None
    lags, ccf = _ccf(resp, frict)
    s = _peak_sides(lags, ccf)
    rec = {"a_id": int(a_id), "group": grp, "n_days": n,
           "real_post_lag": s["post_lag"], "real_post_ccf": s["post_ccf"],
           "real_pre_ccf": s["pre_ccf"],
           "real_responder": bool(abs(s["post_ccf"]) > abs(s["pre_ccf"]))}
    rng = np.random.default_rng([42, int(a_id)])     # resume-stable per agent
    for j in range(N_NULL):
        off = int(rng.integers(SHIFT_MIN, n - SHIFT_MIN + 1))  # [15, n-15]
        lags, ccf = _ccf(resp, np.roll(frict, off))
        s = _peak_sides(lags, ccf)
        rec[f"null_post_ccf_{j}"] = s["post_ccf"]
        rec[f"null_responder_{j}"] = bool(abs(s["post_ccf"]) > abs(s["pre_ccf"]))
    return rec


def run_shuffle(pa: pd.DataFrame) -> pd.DataFrame:
    sample = _sample_cohort(pa)
    done: set[int] = set()
    if SHUFFLE_CSV.exists():
        done = set(pd.read_csv(SHUFFLE_CSV, usecols=["a_id"])["a_id"].astype(int))
    todo = sample[~sample["a_id"].astype(int).isin(done)]
    _log(f"shuffle null: sample={len(sample)} "
         f"(bots={int((sample.group == 'bot').sum())}, "
         f"controls={int((sample.group == 'control').sum())}); "
         f"done={len(done)}, todo={len(todo)}")
    buf = []
    for k, (_, r) in enumerate(todo.iterrows(), 1):
        rec = _shuffle_one(int(r["a_id"]), str(r["group"]))
        if rec is not None:
            buf.append(rec)
        if k % CHUNK == 0 or k == len(todo):
            if buf:
                pd.DataFrame(buf).to_csv(SHUFFLE_CSV, mode="a", index=False,
                                         header=not SHUFFLE_CSV.exists())
                buf = []
            _log(f"shuffle null: processed {k}/{len(todo)} this run")
    return pd.read_csv(SHUFFLE_CSV).drop_duplicates("a_id", keep="first")


def aggregate_shuffle(sh: pd.DataFrame, pa: pd.DataFrame) -> dict:
    res = {}
    # QA: recomputed real values must match the published per-agent CSV
    m = sh.merge(pa[["a_id", "p404_post_ccf", "p404_responder"]], on="a_id")
    res["qa_real_ccf_match"] = round(float(
        (np.abs(m["real_post_ccf"] - m["p404_post_ccf"]) < 1e-9).mean()), 4)
    res["qa_real_responder_match"] = round(float(
        (m["real_responder"] == m["p404_responder"]).mean()), 4)

    null_resp_cols = [f"null_responder_{j}" for j in range(N_NULL)]
    null_ccf_cols = [f"null_post_ccf_{j}" for j in range(N_NULL)]
    for grp in ("bot", "control"):
        g = sh[sh.group == grp]
        real_share = float(g["real_responder"].mean())
        null_shares = np.array([float(g[c].mean()) for c in null_resp_cols])
        pct = 100.0 * (float((null_shares < real_share).sum())
                       + 0.5 * float((null_shares == real_share).sum())) / N_NULL
        real_med_ccf = float(np.median(
            g.loc[g["real_responder"], "real_post_ccf"]))
        null_med_ccf = np.array([
            float(np.median(g.loc[g[rc], cc]))
            for rc, cc in zip(null_resp_cols, null_ccf_cols)])
        real_med_abs = float(np.median(np.abs(g["real_post_ccf"])))
        null_abs_pool = np.abs(g[null_ccf_cols].to_numpy(float)).ravel()
        res[grp] = {
            "n_agents": int(len(g)),
            "real_responder_share": round(real_share, 4),
            "null_responder_share_mean": round(float(null_shares.mean()), 4),
            "null_responder_share_range": [round(float(null_shares.min()), 4),
                                           round(float(null_shares.max()), 4)],
            "real_share_pctile_in_null": pct,
            "real_median_post_ccf_responders": round(real_med_ccf, 4),
            "null_median_post_ccf_responders_mean": round(float(null_med_ccf.mean()), 4),
            "null_median_post_ccf_responders_range": [
                round(float(null_med_ccf.min()), 4),
                round(float(null_med_ccf.max()), 4)],
            "real_median_abs_post_ccf_all": round(real_med_abs, 4),
            "null_median_abs_post_ccf_all": round(float(np.median(null_abs_pool)), 4),
        }
    # figure: real vs null peak post CCF, bots
    gb = sh[sh.group == "bot"]
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    bins = np.linspace(-1, 1, 81)
    ax.hist(gb["real_post_ccf"], bins=bins, density=True, alpha=0.6,
            color="tab:red", label=f"real (n={len(gb)})")
    ax.hist(gb[null_ccf_cols].to_numpy(float).ravel(), bins=bins, density=True,
            alpha=0.55, color="tab:gray",
            label=f"circular-shift null (n={len(gb) * N_NULL})")
    ax.set_xlabel("post-side peak CCF (p404 vs p_block, lags +1..+14)")
    ax.set_ylabel("density")
    ax.set_title("Shuffle falsification (bots): post-side peak CCF, real vs "
                 "circularly time-shifted p_block\n(shift offset ~ U[15, n-15], "
                 "10 replicates/agent, seed 42)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG_NULL, dpi=150); plt.close(fig)
    return res


# --------------------------------------------------------------------------
def write_summary(km_res: dict, sh_res: dict) -> None:
    L = ["# Lead-lag robustness (this paper, Section 7.6): KM censoring + shuffle null\n",
         "Conventions mirrored from `db_analysis_leadlag.py`: cohort >=30 active "
         "days and >=3 block days; p404 vs p_block channel (the channel used by "
         "the published headline numbers); normalized CCF via "
         "`scipy.signal.correlate`; lag window +/-14 with lag 0 excluded; "
         "responder = post-side (+1..+14) |peak| > pre-side |peak|; "
         "tau = post-side peak lag.\n",
         "## Test 1 — Kaplan-Meier median response lag "
         "(tau = 14 right-censored)\n"]
    for grp in ("bot", "control"):
        v = km_res[grp]
        lo, hi = v["km_median_ci95"]
        hi_s = f"{hi:g}" if np.isfinite(hi) else ">14 (censored)"
        L.append(f"### {grp}\n"
                 f"- responders n={v['n_responders']}, censored at +14: "
                 f"{v['n_censored_at_14']} ({v['pct_censored']}%)\n"
                 f"- naive median tau: {v['naive_median']:.1f} d\n"
                 f"- KM median tau: {v['km_median']:.1f} d "
                 f"(95% CI [{lo:g}, {hi_s}], Brookmeyer-Crowley on log-log "
                 f"Greenwood bands)\n")
    L.append("**Verdict (KM):** see figure `figures/DB_leadlag_KM.png`; "
             "curve table `tables/DB_leadlag_KM_curves.csv`.\n")
    L.append("\n## Test 2 — Circular-shift shuffle falsification "
             f"(seed 42; {N_NULL} replicates/agent; offset ~ U[15, n-15])\n")
    L.append(f"QA: recomputed real post-CCF matches published per-agent CSV for "
             f"{sh_res['qa_real_ccf_match']*100:.1f}% of sampled agents; "
             f"responder flags match {sh_res['qa_real_responder_match']*100:.1f}%.\n")
    for grp in ("bot", "control"):
        v = sh_res[grp]
        L.append(
            f"### {grp} (n={v['n_agents']})\n"
            f"- responder share: real {v['real_responder_share']*100:.1f}% vs "
            f"null {v['null_responder_share_mean']*100:.1f}% "
            f"(range {v['null_responder_share_range'][0]*100:.1f}-"
            f"{v['null_responder_share_range'][1]*100:.1f}% over {N_NULL} "
            f"replicates); real value sits at the "
            f"{v['real_share_pctile_in_null']:.0f}th percentile of the null\n"
            f"- median post-peak CCF among responders (signed, headline "
            f"convention): real {v['real_median_post_ccf_responders']:.3f} vs "
            f"null {v['null_median_post_ccf_responders_mean']:.3f} "
            f"(range {v['null_median_post_ccf_responders_range'][0]:.3f} to "
            f"{v['null_median_post_ccf_responders_range'][1]:.3f})\n"
            f"- median |post-peak CCF|, all sampled agents: real "
            f"{v['real_median_abs_post_ccf_all']:.3f} vs null "
            f"{v['null_median_abs_post_ccf_all']:.3f}\n")
    L.append("**Verdict (shuffle):** see figure "
             "`figures/DB_leadlag_shuffle_null.png`; per-agent table "
             "`tables/DB_leadlag_shuffle_per_agent.csv`.\n")
    SUMMARY_MD.write_text("\n".join(L), encoding="utf-8")


def run() -> dict:
    _selftest()
    config.ensure_dirs()
    pa = pd.read_csv(PER_AGENT_CSV)
    km_res = run_km(pa)
    _log("KM done: " + json.dumps(km_res, default=str))
    sh = run_shuffle(pa)
    sh_res = aggregate_shuffle(sh, pa)
    _log("shuffle aggregate done")
    write_summary(km_res, sh_res)
    out = {"km": km_res, "shuffle": sh_res}
    print(json.dumps(out, indent=2, default=str))
    return out


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        run()
