"""
db_analysis_dose_robustness.py — Selection-vs-dose robustness for the Section 7.5
dose-response frontier (this paper).

Reviewer objection: the light/heavy friction split is OBSERVATIONAL — sites may
apply heavy friction to bots that were already stealthier / more aggressive, so
the deeper observability collapse under heavy friction could be selection, not
dose response. This script answers with three standard designs, all on the SAME
event definition as db_analysis_observability.py:

  1. BALANCE: pre-event observability (pre_O) and pre-event log-volume
     (pre_logvol) compared light vs heavy, with standardized mean differences
     (SMD, pooled SD) and 95% CIs from a CLUSTER bootstrap that resamples
     AGENTS (events within an agent are correlated).
  2. ADJUSTED ESTIMATE (bots): OLS of dO on heavy + pre_O + pre_logvol, and the
     same linear-probability model for retreat; heavy coefficient with a
     cluster-by-agent bootstrap CI, compared with the raw dose gap.
  3. MATCHING (bots): 1-NN matching WITH replacement of each heavy event to a
     light event on standardized (pre_O, pre_logvol), caliper 0.2 pooled SD;
     matched heavy-light differences in dO and retreat with cluster bootstrap
     CIs (matching re-run inside every bootstrap replicate).

CRITICAL design choice inherited from db_analysis_observability.py: the
observability feature vector contains ONLY agent-chosen behavior —
[p404, robots_rate, log1p(hits)] — never p_block / resistance / p200 (those are
the defender's response; including them would make O measure the block itself).

Event extraction is identical to the original frontier: _block_events from
db_battery (cblock z>=1.5 spikes or any 429 day, spacing >= EVENT_W),
+/-EVENT_W active-day windows, intensity = event-day p_block, retreat = post
volume < (1 - RETREAT_DROP) of pre volume, light/heavy split at the
WITHIN-GROUP median of intensity.

Resumable: the streaming pass appends events to tables/DB_dose_events.csv and
processed agent ids to tables/DB_dose_processed.csv every CHECKPOINT_EVERY
agents; on restart, agents present in either file are skipped. Progress goes to
outputs/dose_progress.log. The control centroid is cached in
tables/DB_dose_centroid.json. Re-running after completion skips straight to the
analysis stage.

Outputs: tables/DB_dose_events.csv, figures/DB_dose_balance.png,
DB_dose_robustness_SUMMARY.md (+ DB_dose_robustness.json with every number).
"""
from __future__ import annotations

import argparse
import json
import time
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import config
import db_source as dbs
from db_analysis_observability import _behavior_matrix
from db_battery import _block_events, _did, _winmean  # noqa: F401  (_did kept for convention parity)
from roster import ROSTER_CSV

MIN_DAYS_OBS = 20          # same cohort filter as db_analysis_observability.py
EVENT_W = 5                # +/- active-day window around a block event
RETREAT_DROP = 0.5         # >50% post-block volume drop = retreated, not stealthy
CALIPER = 0.2              # matching caliper, in pooled-SD (standardized) units
N_BOOT = config.BOOTSTRAP_N            # 2000 cluster-bootstrap replicates
CHECKPOINT_EVERY = 2000    # flush buffers + log every N agents
SEED = config.RANDOM_STATE             # 42

OUT_TAB = config.TAB_DIR
OUT_FIG = config.FIG_DIR
EVENTS_CSV = OUT_TAB / "DB_dose_events.csv"
PROCESSED_CSV = OUT_TAB / "DB_dose_processed.csv"
CENTROID_JSON = OUT_TAB / "DB_dose_centroid.json"
PROGRESS_LOG = config.OUT_DIR / "dose_progress.log"
SUMMARY_MD = config.OUT_DIR / "DB_dose_robustness_SUMMARY.md"
RESULTS_JSON = config.OUT_DIR / "DB_dose_robustness.json"

EVENT_COLS = ["a_id", "group", "intensity", "pre_O", "post_O", "dO",
              "pre_logvol", "dvol", "kind"]


def _log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    with open(PROGRESS_LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


# --------------------------------------------------------------------------- #
# Stage 1 — control-population behavior centroid (identical to observability) #
# --------------------------------------------------------------------------- #
def _centroid(roster: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, int]:
    if CENTROID_JSON.exists():
        d = json.loads(CENTROID_JSON.read_text())
        _log(f"centroid loaded from cache ({d['n_ctl']} control agents)")
        return np.array(d["mu_H"]), np.array(d["sigma_H"]), int(d["n_ctl"])
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
    CENTROID_JSON.write_text(json.dumps(
        {"mu_H": mu_H.tolist(), "sigma_H": sigma_H.tolist(), "n_ctl": len(ctl_means)}))
    _log(f"control centroid from {len(ctl_means)} agents: "
         f"mu={np.round(mu_H, 4)} sigma={np.round(sigma_H, 4)}")
    return mu_H, sigma_H, len(ctl_means)


# --------------------------------------------------------------------------- #
# Stage 2 — streaming, resumable event collection                             #
# --------------------------------------------------------------------------- #
def _flush(ev_buf: list[dict], proc_buf: list[int]) -> None:
    if ev_buf:
        pd.DataFrame(ev_buf, columns=EVENT_COLS).to_csv(
            EVENTS_CSV, mode="a", header=not EVENTS_CSV.exists(), index=False)
        ev_buf.clear()
    if proc_buf:
        pd.DataFrame({"a_id": proc_buf}).to_csv(
            PROCESSED_CSV, mode="a", header=not PROCESSED_CSV.exists(), index=False)
        proc_buf.clear()


def collect(roster: pd.DataFrame, mu_H: np.ndarray, sigma_H: np.ndarray,
            max_agents: int = 0) -> bool:
    """Stream all roster agents; append one row per block event. Returns True
    when the full roster has been processed (analysis may proceed)."""
    done: set[int] = set()
    if PROCESSED_CSV.exists():
        done.update(pd.read_csv(PROCESSED_CSV)["a_id"].astype(int).tolist())
    if EVENTS_CSV.exists():
        done.update(pd.read_csv(EVENTS_CSV)["a_id"].astype(int).tolist())
    todo = roster[~roster["a_id"].astype(int).isin(done)]
    _log(f"collect: {len(done)} agents already processed, {len(todo)} to go"
         + (f" (budget {max_agents})" if max_agents else ""))
    if todo.empty:
        return True

    ev_buf: list[dict] = []
    proc_buf: list[int] = []
    n_proc, n_ev, t0 = 0, 0, time.time()
    for _, r in todo.iterrows():
        a_id = int(r["a_id"])
        cp = dbs._cache_path(a_id)
        if cp.exists():
            df = pd.read_parquet(cp)
            if len(df) >= MIN_DAYS_OBS:
                grp = "bot" if bool(r["is_bot"]) else (
                    "control" if bool(r["is_control"]) else "other")
                O = np.linalg.norm((_behavior_matrix(df) - mu_H) / sigma_H, axis=1)
                vol = np.log1p(df["hits"].to_numpy(float))
                pblk = df["p_block"].to_numpy(float)
                for i in _block_events(df):
                    if i < EVENT_W or i + EVENT_W >= len(df):
                        continue
                    pre_O = _winmean(O, i, "pre", EVENT_W)
                    post_O = _winmean(O, i, "post", EVENT_W)
                    pre_v = _winmean(vol, i, "pre", EVENT_W)
                    post_v = _winmean(vol, i, "post", EVENT_W)
                    retreated = np.expm1(post_v) < (1 - RETREAT_DROP) * np.expm1(pre_v)
                    ev_buf.append({"a_id": a_id, "group": grp,
                                   "intensity": float(pblk[i]),
                                   "pre_O": float(pre_O), "post_O": float(post_O),
                                   "dO": float(post_O - pre_O),
                                   "pre_logvol": float(pre_v),
                                   "dvol": float(post_v - pre_v),
                                   "kind": "retreat" if retreated else "stealth"})
                    n_ev += 1
        proc_buf.append(a_id)
        n_proc += 1
        if n_proc % CHECKPOINT_EVERY == 0:
            _flush(ev_buf, proc_buf)
            rate = n_proc / max(time.time() - t0, 1e-9)
            _log(f"collect: {n_proc}/{len(todo)} agents this run "
                 f"({n_ev} events appended, {rate:.0f} agents/s)")
        if max_agents and n_proc >= max_agents:
            _flush(ev_buf, proc_buf)
            _log(f"collect: agent budget {max_agents} reached — resumable, rerun to continue")
            return False
    _flush(ev_buf, proc_buf)
    _log(f"collect: COMPLETE — {n_proc} agents this run, {n_ev} events appended")
    return True


# --------------------------------------------------------------------------- #
# Cluster bootstrap (resamples AGENTS, not events)                            #
# --------------------------------------------------------------------------- #
def _cluster_index(ids: np.ndarray):
    """Pre-compute row indices sorted by cluster + offsets for fast resampling."""
    order = np.argsort(ids, kind="stable")
    sid = ids[order]
    cuts = np.flatnonzero(np.diff(sid)) + 1
    starts = np.concatenate([[0], cuts])
    sizes = np.diff(np.concatenate([starts, [len(sid)]]))
    return order.astype(np.int64), starts.astype(np.int64), sizes.astype(np.int64)


def _resample_rows(rows_sorted, starts, sizes, rng):
    """One cluster-bootstrap replicate: row indices of resampled agents."""
    k = len(starts)
    pick = rng.integers(0, k, k)
    s = sizes[pick]
    total = int(s.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64)
    ends = np.cumsum(s)
    step = np.ones(total, dtype=np.int64)
    step[0] = 0
    step[ends[:-1]] -= s[:-1]
    intra = np.cumsum(step)                       # 0..s_i-1 within each block
    return rows_sorted[np.repeat(starts[pick], s) + intra]


def _cluster_boot(ids: np.ndarray, stat_fn, n: int = N_BOOT, seed: int = SEED) -> np.ndarray:
    """stat_fn(idx) -> 1-D stat vector; returns (n_reps, n_stats) array."""
    rows_sorted, starts, sizes = _cluster_index(ids)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        out.append(stat_fn(_resample_rows(rows_sorted, starts, sizes, rng)))
    return np.asarray(out, float)


def _ci(samples: np.ndarray) -> list[float]:
    s = samples[np.isfinite(samples)]
    if len(s) == 0:
        return [float("nan"), float("nan")]
    return [round(float(np.percentile(s, 2.5)), 4), round(float(np.percentile(s, 97.5)), 4)]


def _smd(x_h: np.ndarray, x_l: np.ndarray) -> float:
    """Standardized mean difference, pooled SD = sqrt((var_h + var_l)/2)."""
    sd = np.sqrt((np.var(x_h, ddof=1) + np.var(x_l, ddof=1)) / 2.0)
    return float((x_h.mean() - x_l.mean()) / (sd + 1e-12))


def _ols_coef(X: np.ndarray, y: np.ndarray) -> float:
    """Coefficient on column 1 (heavy dummy); column 0 is the intercept."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(beta[1])


def _match_diff(z: np.ndarray, heavy: np.ndarray, y: np.ndarray,
                caliper: float = CALIPER) -> tuple[float, int, int]:
    """1-NN with replacement: each heavy event -> nearest light event on
    standardized (pre_O, pre_logvol); pairs beyond `caliper` discarded.
    Returns (mean heavy-light diff in y, n matched, n heavy total)."""
    zh, zl = z[heavy], z[~heavy]
    if len(zh) == 0 or len(zl) == 0:
        return float("nan"), 0, int(heavy.sum())
    dist, j = cKDTree(zl).query(zh, k=1)
    ok = dist <= caliper
    if not ok.any():
        return float("nan"), 0, len(zh)
    diff = float(y[heavy][ok].mean() - y[~heavy][j[ok]].mean())
    return diff, int(ok.sum()), len(zh)


# --------------------------------------------------------------------------- #
# Stage 3 — analysis                                                          #
# --------------------------------------------------------------------------- #
def analyze() -> dict:
    ev = pd.read_csv(EVENTS_CSV)
    n_raw = len(ev)
    num_cols = ["intensity", "pre_O", "post_O", "dO", "pre_logvol", "dvol"]
    ev = ev[np.isfinite(ev[num_cols].to_numpy(float)).all(axis=1)].reset_index(drop=True)
    n_dropped = n_raw - len(ev)
    _log(f"analyze: {len(ev)} events ({n_dropped} non-finite rows dropped), "
         f"{ev['a_id'].nunique()} agents with >=1 event")

    res: dict = {"n_events_total": int(len(ev)), "n_nonfinite_dropped": int(n_dropped),
                 "groups": {}}

    for g in ("bot", "control"):
        e = ev[ev["group"] == g].reset_index(drop=True)
        if len(e) < 20:
            continue
        ids = e["a_id"].to_numpy(np.int64)
        med = float(e["intensity"].median())
        heavy = (e["intensity"] > med).to_numpy()          # same convention: light = intensity <= median
        pre_O = e["pre_O"].to_numpy(float)
        post_O = e["post_O"].to_numpy(float)
        dO = e["dO"].to_numpy(float)
        pre_lv = e["pre_logvol"].to_numpy(float)
        dvol = e["dvol"].to_numpy(float)
        ret = (e["kind"] == "retreat").to_numpy(float)

        # ---- raw frontier (sanity check against the original Section 7.5) ----
        raw = {arm: {"n": int(m.sum()),
                     "dO": round(float(dO[m].mean()), 4),
                     "dvol": round(float(dvol[m].mean()), 4),
                     "frac_retreat": round(float(ret[m].mean()), 3)}
               for arm, m in (("light", ~heavy), ("heavy", heavy))}
        raw_gap_dO = float(dO[heavy].mean() - dO[~heavy].mean())
        raw_gap_ret = float(ret[heavy].mean() - ret[~heavy].mean())

        # ---- point estimates: balance, arm means, (bots) OLS + matching ----
        Xadj = np.column_stack([np.ones(len(e)), heavy.astype(float), pre_O, pre_lv])
        # standardization for matching: pooled SD over this group's events
        zsd = np.array([np.sqrt((np.var(pre_O[heavy], ddof=1) + np.var(pre_O[~heavy], ddof=1)) / 2),
                        np.sqrt((np.var(pre_lv[heavy], ddof=1) + np.var(pre_lv[~heavy], ddof=1)) / 2)])
        zmu = np.array([pre_O.mean(), pre_lv.mean()])
        Z = (np.column_stack([pre_O, pre_lv]) - zmu) / (zsd + 1e-12)

        def stats_vec(idx: np.ndarray) -> list[float]:
            h, nh = heavy[idx], ~heavy[idx]
            if h.sum() < 2 or nh.sum() < 2:
                return [np.nan] * 16
            pOh, pOl = pre_O[idx][h], pre_O[idx][nh]
            pVh, pVl = pre_lv[idx][h], pre_lv[idx][nh]
            out = [pOl.mean(), pOh.mean(), pOh.mean() - pOl.mean(), _smd(pOh, pOl),
                   pVl.mean(), pVh.mean(), pVh.mean() - pVl.mean(), _smd(pVh, pVl),
                   post_O[idx][nh].mean(), post_O[idx][h].mean(),
                   dO[idx][h].mean() - dO[idx][nh].mean(),
                   ret[idx][h].mean() - ret[idx][nh].mean()]
            if g == "bot":
                out += [_ols_coef(Xadj[idx], dO[idx]), _ols_coef(Xadj[idx], ret[idx]),
                        _match_diff(Z[idx], h, dO[idx])[0],
                        _match_diff(Z[idx], h, ret[idx])[0]]
            else:
                out += [np.nan] * 4
            return out

        _log(f"analyze: cluster bootstrap for {g} "
             f"({len(e)} events, {len(np.unique(ids))} agents, {N_BOOT} reps)")
        B = _cluster_boot(ids, stats_vec)

        gres = {
            "n_events": int(len(e)),
            "n_agents": int(len(np.unique(ids))),
            "intensity_median_split": round(med, 4),
            "raw_frontier": raw,
            "raw_gap": {"dO_heavy_minus_light": round(raw_gap_dO, 4),
                        "retreat_heavy_minus_light": round(raw_gap_ret, 4)},
            "balance": {
                "pre_O": {"light_mean": round(float(pre_O[~heavy].mean()), 4),
                          "heavy_mean": round(float(pre_O[heavy].mean()), 4),
                          "diff": round(float(pre_O[heavy].mean() - pre_O[~heavy].mean()), 4),
                          "diff_ci95": _ci(B[:, 2]),
                          "smd": round(_smd(pre_O[heavy], pre_O[~heavy]), 4),
                          "smd_ci95": _ci(B[:, 3])},
                "pre_logvol": {"light_mean": round(float(pre_lv[~heavy].mean()), 4),
                               "heavy_mean": round(float(pre_lv[heavy].mean()), 4),
                               "diff": round(float(pre_lv[heavy].mean() - pre_lv[~heavy].mean()), 4),
                               "diff_ci95": _ci(B[:, 6]),
                               "smd": round(_smd(pre_lv[heavy], pre_lv[~heavy]), 4),
                               "smd_ci95": _ci(B[:, 7])},
            },
            "fig_means": {"pre_O_light": [round(float(pre_O[~heavy].mean()), 4), _ci(B[:, 0])],
                          "pre_O_heavy": [round(float(pre_O[heavy].mean()), 4), _ci(B[:, 1])],
                          "post_O_light": [round(float(post_O[~heavy].mean()), 4), _ci(B[:, 8])],
                          "post_O_heavy": [round(float(post_O[heavy].mean()), 4), _ci(B[:, 9])]},
            "raw_gap_ci": {"dO_ci95": _ci(B[:, 10]), "retreat_ci95": _ci(B[:, 11])},
        }

        if g == "bot":
            # adjusted (OLS) point estimates
            adj_dO = _ols_coef(Xadj, dO)
            adj_ret = _ols_coef(Xadj, ret)
            # matched point estimates
            m_dO, n_m, n_h = _match_diff(Z, heavy, dO)
            m_ret, _, _ = _match_diff(Z, heavy, ret)
            gres["adjusted_ols"] = {
                "controls": ["pre_O", "pre_logvol"],
                "heavy_coef_dO": round(adj_dO, 4), "heavy_coef_dO_ci95": _ci(B[:, 12]),
                "heavy_coef_retreat": round(adj_ret, 4), "heavy_coef_retreat_ci95": _ci(B[:, 13]),
                "raw_vs_adjusted_dO": [round(raw_gap_dO, 4), round(adj_dO, 4)],
                "raw_vs_adjusted_retreat": [round(raw_gap_ret, 4), round(adj_ret, 4)],
            }
            gres["matching"] = {
                "spec": "1-NN with replacement on standardized (pre_O, pre_logvol), "
                        f"caliper {CALIPER} pooled SD",
                "n_heavy": int(n_h), "n_matched": int(n_m),
                "pct_discarded": round(100 * (1 - n_m / max(n_h, 1)), 2),
                "matched_diff_dO": round(m_dO, 4), "matched_diff_dO_ci95": _ci(B[:, 14]),
                "matched_diff_retreat": round(m_ret, 4),
                "matched_diff_retreat_ci95": _ci(B[:, 15]),
            }
        res["groups"][g] = gres

    res["verdict"] = _verdict(res)
    _figure(res)
    _summary(res)
    RESULTS_JSON.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    return res


def _verdict(res: dict) -> dict:
    b = res["groups"]["bot"]
    smd_O = b["balance"]["pre_O"]["smd"]
    smd_V = b["balance"]["pre_logvol"]["smd"]
    raw_dO, adj_dO = b["adjusted_ols"]["raw_vs_adjusted_dO"]
    m_dO = b["matching"]["matched_diff_dO"]
    m_dO_ci = b["matching"]["matched_diff_dO_ci95"]
    adj_ci = b["adjusted_ols"]["heavy_coef_dO_ci95"]
    raw_ret, adj_ret = b["adjusted_ols"]["raw_vs_adjusted_retreat"]
    m_ret, m_ret_ci = b["matching"]["matched_diff_retreat"], b["matching"]["matched_diff_retreat_ci95"]

    imbalance = max(abs(smd_O), abs(smd_V))
    selection_present = imbalance >= 0.10           # conventional SMD threshold
    selection_strong = imbalance >= 0.25
    dose_survives_dO = (adj_ci[1] < 0) and (m_dO_ci[1] < 0)
    dose_survives_ret = (adj_ci is not None) and (m_ret_ci[1] < 0 or m_ret_ci[0] > 0 or
                                                  b["adjusted_ols"]["heavy_coef_retreat_ci95"][1] < 0)
    # honest text assembled from the numbers
    if selection_strong:
        bal_txt = (f"Pre-event covariates are STRONGLY imbalanced across doses "
                   f"(max |SMD| = {imbalance:.3f} >= 0.25): selection into heavy friction is real, "
                   f"and the matched/adjusted estimates are the paper numbers.")
    elif selection_present:
        bal_txt = (f"Pre-event covariates show non-negligible imbalance "
                   f"(max |SMD| = {imbalance:.3f}, above the 0.10 convention but below 0.25): "
                   f"some selection into heavy friction exists, so the matched/adjusted "
                   f"estimates are the defensible paper numbers.")
    else:
        bal_txt = (f"Pre-event covariates are balanced across doses "
                   f"(max |SMD| = {imbalance:.3f} < 0.10): little evidence that heavy friction "
                   f"targeted pre-different bots.")
    if dose_survives_dO:
        eff_txt = (f"The heavy-dose deepening of the observability collapse survives adjustment: "
                   f"raw gap {raw_dO:+.4f}, covariate-adjusted {adj_dO:+.4f} "
                   f"(95% CI {adj_ci}), matched {m_dO:+.4f} (95% CI {m_dO_ci}) — "
                   f"all negative and bounded away from zero.")
    else:
        eff_txt = (f"The heavy-dose gap does NOT clearly survive adjustment: raw {raw_dO:+.4f}, "
                   f"adjusted {adj_dO:+.4f} (95% CI {adj_ci}), matched {m_dO:+.4f} "
                   f"(95% CI {m_dO_ci}) — treat the frontier as descriptive, not causal.")
    ret_txt = (f"Retreat dose gap: raw {raw_ret:+.4f}, adjusted {adj_ret:+.4f} "
               f"(95% CI {b['adjusted_ols']['heavy_coef_retreat_ci95']}), matched {m_ret:+.4f} "
               f"(95% CI {m_ret_ci}).")
    return {"max_abs_smd_bots": round(imbalance, 4),
            "selection_present_smd_ge_0.10": bool(selection_present),
            "selection_strong_smd_ge_0.25": bool(selection_strong),
            "dose_effect_on_dO_survives_adjustment_and_matching": bool(dose_survives_dO),
            "dose_effect_on_retreat_survives": bool(dose_survives_ret),
            "text": " ".join([bal_txt, eff_txt, ret_txt])}


# --------------------------------------------------------------------------- #
# Figure + summary                                                            #
# --------------------------------------------------------------------------- #
def _figure(res: dict) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    labels, pre_v, pre_ci, post_v, post_ci = [], [], [], [], []
    for g in ("bot", "control"):
        if g not in res["groups"]:
            continue
        fm = res["groups"][g]["fig_means"]
        for arm in ("light", "heavy"):
            labels.append(f"{g}\n{arm}")
            pre_v.append(fm[f"pre_O_{arm}"][0]);  pre_ci.append(fm[f"pre_O_{arm}"][1])
            post_v.append(fm[f"post_O_{arm}"][0]); post_ci.append(fm[f"post_O_{arm}"][1])
    x = np.arange(len(labels))
    pre_err = np.abs(np.array(pre_ci).T - np.array(pre_v))
    post_err = np.abs(np.array(post_ci).T - np.array(post_v))
    ax.bar(x - 0.18, pre_v, 0.36, yerr=pre_err, capsize=3,
           color="tab:gray", label="pre-event O (5-day mean)")
    ax.bar(x + 0.18, post_v, 0.36, yerr=post_err, capsize=3,
           color="tab:red", label="post-event O (5-day mean)")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("observability O (behavior-only features)")
    ax.set_title("Dose balance: pre- vs post-event observability by friction dose\n"
                 "(cluster-bootstrap 95% CIs, agents resampled)")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(OUT_FIG / "DB_dose_balance.png", dpi=150)
    plt.close(fig)


def _summary(res: dict) -> None:
    L = ["# Dose-response frontier robustness — selection vs dose (this paper, Section 7.5)\n",
         f"Events: {res['n_events_total']} ({res['n_nonfinite_dropped']} non-finite dropped). "
         f"Same event definition, windows (+/-{EVENT_W} active days), MIN_DAYS_OBS={MIN_DAYS_OBS}, "
         f"retreat threshold {RETREAT_DROP}, and within-group median intensity split as the original "
         "frontier. All CIs are 95% cluster bootstraps resampling AGENTS "
         f"(n={N_BOOT}, seed {SEED}).\n"]
    for g in ("bot", "control"):
        if g not in res["groups"]:
            continue
        r = res["groups"][g]
        L.append(f"\n## {g}s — {r['n_events']} events from {r['n_agents']} agents "
                 f"(median split at event-day p_block = {r['intensity_median_split']})\n")
        L.append("### Raw frontier (reproduction)\n")
        for arm in ("light", "heavy"):
            a = r["raw_frontier"][arm]
            L.append(f"- {arm} (n={a['n']}): dO {a['dO']:+.4f}, dvol {a['dvol']:+.4f}, "
                     f"retreat {a['frac_retreat']}")
        L.append(f"- raw heavy-light gap: dO {r['raw_gap']['dO_heavy_minus_light']:+.4f} "
                 f"CI {r['raw_gap_ci']['dO_ci95']}, retreat "
                 f"{r['raw_gap']['retreat_heavy_minus_light']:+.4f} CI {r['raw_gap_ci']['retreat_ci95']}\n")
        L.append("### Pre-event balance (light vs heavy)\n")
        for var in ("pre_O", "pre_logvol"):
            b = r["balance"][var]
            L.append(f"- {var}: light {b['light_mean']:.4f} vs heavy {b['heavy_mean']:.4f}, "
                     f"diff {b['diff']:+.4f} CI {b['diff_ci95']}, SMD {b['smd']:+.4f} "
                     f"CI {b['smd_ci95']}")
        if g == "bot":
            a = r["adjusted_ols"]
            L.append("\n### Covariate-adjusted dose effect (OLS on heavy + pre_O + pre_logvol)\n")
            L.append(f"- dO: raw gap {a['raw_vs_adjusted_dO'][0]:+.4f} -> adjusted "
                     f"{a['heavy_coef_dO']:+.4f} CI {a['heavy_coef_dO_ci95']}")
            L.append(f"- retreat (LPM): raw gap {a['raw_vs_adjusted_retreat'][0]:+.4f} -> adjusted "
                     f"{a['heavy_coef_retreat']:+.4f} CI {a['heavy_coef_retreat_ci95']}")
            m = r["matching"]
            L.append(f"\n### Matching ({m['spec']})\n")
            L.append(f"- heavy events matched: {m['n_matched']}/{m['n_heavy']} "
                     f"({m['pct_discarded']}% discarded outside caliper)")
            L.append(f"- matched heavy-light dO: {m['matched_diff_dO']:+.4f} "
                     f"CI {m['matched_diff_dO_ci95']}")
            L.append(f"- matched heavy-light retreat: {m['matched_diff_retreat']:+.4f} "
                     f"CI {m['matched_diff_retreat_ci95']}")
    v = res["verdict"]
    L.append("\n## Verdict\n")
    L.append(v["text"] + "\n")
    L.append(f"(max |SMD| bots = {v['max_abs_smd_bots']}; selection >=0.10: "
             f"{v['selection_present_smd_ge_0.10']}; >=0.25: {v['selection_strong_smd_ge_0.25']}; "
             f"dO dose effect survives adjustment+matching: "
             f"{v['dose_effect_on_dO_survives_adjustment_and_matching']})\n")
    L.append("Caveats: doses are still observational within arm (no randomization); matching/"
             "adjustment control only for OBSERVED pre-event behavior (pre_O, pre_logvol) — "
             "unobserved-confounder selection cannot be excluded. Light/heavy assignment is "
             "held fixed at the full-sample median inside bootstrap replicates.\n")
    SUMMARY_MD.write_text("\n".join(L), encoding="utf-8")


# --------------------------------------------------------------------------- #
def run(max_agents: int = 0, analyze_only: bool = False) -> dict | None:
    config.ensure_dirs()
    roster = pd.read_csv(ROSTER_CSV)
    if not analyze_only:
        mu_H, sigma_H, _ = _centroid(roster)
        if not collect(roster, mu_H, sigma_H, max_agents=max_agents):
            return None                      # budget hit; rerun to continue
    res = analyze()
    _log("analysis complete -> " + str(SUMMARY_MD))
    print(json.dumps(res, indent=2, default=str))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-agents", type=int, default=0,
                    help="process at most N agents this run (0 = all); resumable")
    ap.add_argument("--analyze-only", action="store_true",
                    help="skip collection (requires a complete DB_dose_events.csv)")
    args = ap.parse_args()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        run(max_agents=args.max_agents, analyze_only=args.analyze_only)
