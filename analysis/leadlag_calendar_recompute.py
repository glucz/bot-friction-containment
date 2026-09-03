"""Lead-lag delays in calendar time, on the arm of record, against a 999-replicate null.

Cross-correlates each agent's experienced friction (daily block rate) with its behavioural
response (404 probing) over lags of +/-14 days, on a CALENDAR index: a day with no requests
is a real zero, not a missing row. An agent is a responder when its peak |ccf| falls on the
positive side, a leader when it falls on the negative side.

The null rotates one series by a random offset and recomputes, 999 times per agent. It is
built over the agents whose series can vary, so the observed share compared against it is
taken over that same null-eligible set: comparing an all-agent share against a null-eligible
distribution tests one population against another's reference.

Both indexings are emitted. Calendar is primary, because that is the one the null is
calibrated on; active-observation indexing is reported beside it so the difference is visible
rather than looking like two analyses disagreeing.

Read-only, cache-only. Usage: python leadlag_calendar_recompute.py [--workers N] [--nnull N]
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

HERE = Path(__file__).resolve().parent
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from _layout import data_root, on_path, require  # noqa: E402

ES = on_path()
CACHE = ES / "cache" / "agents"

from input_contract import admit, load_admitted, merge_per_id  # noqa: E402

# Admission happens once, in the parent, before any pool starts. Workers hold a copy of
# the per-id manifest so they can refuse content that changed after admission; a worker
# cannot enforce anything else, since its own register never reaches the parent.
_ADMITTED: dict = {}
_PER_ID: dict = {}


def _install_admitted(per_id):
    global _ADMITTED
    _ADMITTED = per_id

ROSTER = ES / "outputs" / "roster_v2.csv"

MIN_DAYS_LL = 30
MIN_BLOCK_DAYS = 3
LAG_MAX = 14
SHIFT_MIN = 15



def _ccf(resp, frict):
    r = (resp - np.nanmean(resp)) / (np.nanstd(resp) + 1e-9)
    f = (frict - np.nanmean(frict)) / (np.nanstd(frict) + 1e-9)
    ccf = signal.correlate(r, f, mode="full") / len(r)
    lags = signal.correlation_lags(len(r), len(f), mode="full")
    m = np.abs(lags) <= LAG_MAX
    return lags[m], ccf[m]


def _peak_sides(lags, ccf):
    out = {}
    for side, mask in (("post", lags >= 1), ("pre", lags <= -1)):
        if not mask.any():
            out[f"{side}_lag"], out[f"{side}_ccf"] = np.nan, np.nan
            continue
        i = int(np.argmax(np.abs(ccf[mask])))
        out[f"{side}_lag"] = int(lags[mask][i])
        out[f"{side}_ccf"] = float(ccf[mask][i])
    return out


def _calendarize(df):
    """Reindex to the agent's own daily calendar span.

    Missing-versus-zero rule, stated explicitly because the review demands one: a day on
    which the agent sent no request is treated as a real zero, not as missing data. It probed
    nothing (c404 = 0) and experienced no friction (cblock = 0), so both rates are 0.

    That choice is defensible but not neutral, and it is not the only one. The alternative,
    masking the day out and computing each lag only from days observed in both series, is
    implemented in `leadlag_mask_sensitivity.py` and gives a materially different answer:
    pooled tau_A = 6 and tau_D = 6 against the 4 and 5 here, so the loop reads 12 rather
    than 9. Run both. The disagreement is itself the finding - a delay magnitude that moves
    with the imputation rule is not identified by these data.

    """
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    full = pd.date_range(d["date"].min(), d["date"].max(), freq="D")
    d = d.set_index("date").reindex(full)
    for c in ("hits", "c404", "c200", "cblock", "c429", "robots"):
        if c in d:
            d[c] = d[c].fillna(0.0)
    h = d["hits"].to_numpy(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        d["p404"] = np.where(h > 0, d["c404"].to_numpy(float) / np.where(h > 0, h, 1.0), 0.0)
        d["p_block"] = np.where(h > 0, d["cblock"].to_numpy(float) / np.where(h > 0, h, 1.0), 0.0)
    return d.reset_index(drop=True)


_C: dict = {}


def _init(per_id, nnull):
    _install_admitted(per_id)
    _C["nnull"] = nnull


def _one(args):
    a_id, grp = args
    # Input failure and scientific exclusion are different facts; see input_contract.py.
    df = load_admitted(a_id, CACHE, _ADMITTED)
    # cohort criteria on ACTIVE days, exactly as published
    if len(df) < MIN_DAYS_LL or int((df["cblock"] > 0).sum()) < MIN_BLOCK_DAYS:
        return None
    if np.nanstd(df["p_block"].to_numpy(float)) < 1e-9:
        return None

    rec = {"a_id": a_id, "group": grp, "n_active": len(df)}
    cal = _calendarize(df)
    rec["n_calendar"] = len(cal)
    rec["density"] = len(df) / max(len(cal), 1)

    for tag, d in (("active", df), ("calendar", cal)):
        frict = d["p_block"].to_numpy(float)
        resp = d["p404"].to_numpy(float)
        if np.nanstd(frict) < 1e-9 or np.nanstd(resp) < 1e-9:
            continue
        lags, ccf = _ccf(resp, frict)
        s = _peak_sides(lags, ccf)
        is_resp = bool(abs(s["post_ccf"]) > abs(s["pre_ccf"]))
        rec[f"{tag}_post_lag"] = s["post_lag"]
        rec[f"{tag}_pre_lag"] = s["pre_lag"]
        rec[f"{tag}_post_ccf"] = s["post_ccf"]
        rec[f"{tag}_responder"] = is_resp

    # null: circular shift of friction on the CALENDAR series
    frict = cal["p_block"].to_numpy(float)
    resp = cal["p404"].to_numpy(float)
    n = len(cal)
    if n >= 2 * SHIFT_MIN and np.nanstd(frict) > 1e-9 and np.nanstd(resp) > 1e-9:
        rng = np.random.default_rng([42, int(a_id)])
        nr, nl, nresp, nccf = [], [], [], []
        for _ in range(_C["nnull"]):
            off = int(rng.integers(SHIFT_MIN, n - SHIFT_MIN + 1))
            lags, ccf = _ccf(resp, np.roll(frict, off))
            s = _peak_sides(lags, ccf)
            r = bool(abs(s["post_ccf"]) > abs(s["pre_ccf"]))
            nresp.append(r)
            nr.append(s["post_lag"] if r else np.nan)
            nl.append(abs(s["pre_lag"]) if not r else np.nan)
            # Coupling strength under the null. The peak |ccf| against its null is the
            # discriminator both manuscripts lean on, so it is emitted here rather than left
            # to a separate run.
            nccf.append(abs(s["post_ccf"]))
        rec["_null_responder"] = np.array(nresp)
        rec["_null_resp_lag"] = np.array(nr, float)
        rec["_null_lead_lag"] = np.array(nl, float)
        rec["_null_abs_ccf"] = np.array(nccf, float)
    return rec


def _boot_median(x, n=2000, seed=42):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    m = np.median(rng.choice(x, (n, len(x)), True), axis=1)
    return (float(np.median(x)), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--nnull", type=int, default=999)
    a = ap.parse_args()

    ro = pd.read_csv(require(ROSTER, "the roster (`outputs/roster_v2.csv`)"))
    tasks = [(int(r.a_id), "abusive" if r.role == "abusive_share"
              else "verified" if r.role == "verified_human"
              else "declared" if r.role == "declared" else "pothuman")
             for r in ro.itertuples()]
    # Admit every declared input BEFORE any pool starts. A cohort that shrinks to whatever is on
    # disk is not a cohort, and a worker's own register never reaches the parent.
    global _PER_ID
    _REPORTS = admit({g: [i for i, gg in tasks if gg == g] for g in sorted({gg for _, gg in tasks})}, CACHE, "leadlag_calendar", out_dir=HERE,
                     analysis_min_rows=MIN_DAYS_LL, required_cols=("date", "cblock", "p404", "hits"))
    _PER_ID = merge_per_id(_REPORTS)


    recs = []
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init,
                             initargs=(_PER_ID, a.nnull)) as ex:
        for r in ex.map(_one, tasks, chunksize=50):
            if r is not None:
                recs.append(r)
    print(f"cohort: {len(recs)} agents")

    out = {"n_null": a.nnull, "populations": {}}
    for grp in ("abusive", "verified", "declared", "pothuman"):
        G = [r for r in recs if r["group"] == grp]
        if len(G) < 5:
            continue
        e = {"n": len(G), "median_density": float(np.median([r["density"] for r in G]))}
        for tag in ("active", "calendar"):
            resp = [r for r in G if r.get(f"{tag}_responder") is True]
            lead = [r for r in G if r.get(f"{tag}_responder") is False]
            tauA = _boot_median([r[f"{tag}_post_lag"] for r in resp])
            tauD = _boot_median([abs(r[f"{tag}_pre_lag"]) for r in lead])
            e[tag] = {"responder_share": len(resp) / len(G),
                      "n_responders": len(resp), "n_leaders": len(lead),
                      "tau_A": tauA[0], "tau_A_ci": [tauA[1], tauA[2]],
                      "tau_D": tauD[0], "tau_D_ci": [tauD[1], tauD[2]],
                      "loop": (tauA[0] + tauD[0]) if np.isfinite(tauA[0]) and np.isfinite(tauD[0]) else None,
                      "median_abs_ccf": float(np.nanmedian(
                          [abs(r[f"{tag}_post_ccf"]) for r in G
                           if np.isfinite(r.get(f"{tag}_post_ccf", np.nan))]))
                      if any(np.isfinite(r.get(f"{tag}_post_ccf", np.nan)) for r in G) else None}
        # null distributions over replicates
        H = [r for r in G if "_null_responder" in r]
        if H:
            R = np.vstack([r["_null_responder"] for r in H])
            RL = np.vstack([r["_null_resp_lag"] for r in H])
            CC = np.vstack([r["_null_abs_ccf"] for r in H])
            LL = np.vstack([r["_null_lead_lag"] for r in H])
            share = R.mean(axis=0)
            with np.errstate(invalid="ignore"):
                mlagA = np.nanmedian(RL, axis=0)
                mlagD = np.nanmedian(LL, axis=0)
            obs_tag = "calendar"
            obs = e[obs_tag]
            def pval(observed, null, tail):
                null = null[np.isfinite(null)]
                if not len(null) or not np.isfinite(observed):
                    return None
                k = (null >= observed).sum() if tail == "hi" else (null <= observed).sum()
                return float((k + 1) / (len(null) + 1))
            # Observed share on the NULL-ELIGIBLE denominator. Computing it over all len(G)
            # agents while every null replicate is built over the len(H) agents with a varying
            # series would put different denominators on the two sides of the comparison, which
            # tests one population against another's reference. The agents excluded from H are
            # structural
            # non-responders -- no varying response or friction series to rotate -- so the
            # like-for-like comparison restricts the observed share to H as well. The
            # population-level share stays available as `calendar.responder_share`.
            n_resp_H = sum(1 for r in H if r.get(f"{obs_tag}_responder") is True)
            obs_share_H = n_resp_H / len(H)
            e["null"] = {
                "n_agents_in_null": len(H),
                "responder_share": {"obs": obs_share_H,
                                    "obs_n_responders": n_resp_H,
                                    "obs_denominator": len(H),
                                    "obs_all_agents": obs["responder_share"],
                                    "null_mean": float(share.mean()),
                                    "null_max": float(share.max()),
                                    "p": pval(obs_share_H, share, "hi"),  # matched denominator: the
                                    # observed share and every replicate must be built
                                    # from the same agents. Storing obs_share_H and then
                                    # The full null distribution, not summary statistics: the
                                    # falsification figure plots replicate-level shares, and they
                                    # exist nowhere else.
                                    "null_distribution": [float(x) for x in share]},
                "tau_A": {"obs": obs["tau_A"], "null_median": float(np.nanmedian(mlagA)),
                          "null_lo": float(np.nanpercentile(mlagA, 2.5)),
                          "null_hi": float(np.nanpercentile(mlagA, 97.5)),
                          "p_shorter": pval(obs["tau_A"], mlagA, "lo")},
                "tau_D": {"obs": obs["tau_D"], "null_median": float(np.nanmedian(mlagD)),
                          "null_lo": float(np.nanpercentile(mlagD, 2.5)),
                          "null_hi": float(np.nanpercentile(mlagD, 97.5)),
                          "p_shorter": pval(obs["tau_D"], mlagD, "lo")},
                "median_abs_ccf": {"obs": obs.get("median_abs_ccf"),
                                   "null_mean": float(np.nanmean(np.nanmedian(CC, axis=0))),
                                   "null_max": float(np.nanmax(np.nanmedian(CC, axis=0))),
                                   "p": pval(obs.get("median_abs_ccf"),
                                             np.nanmedian(CC, axis=0), "hi")},
            }
        out["populations"][grp] = e

    print(f"\n{'population':<12}{'n':>6}{'density':>9}"
          f"{'tauA act':>10}{'tauA cal':>10}{'tauD act':>10}{'tauD cal':>10}{'loop cal':>10}")
    for g, e in out["populations"].items():
        print(f"{g:<12}{e['n']:>6}{e['median_density']:>9.2f}"
              f"{e['active']['tau_A']:>10.1f}{e['calendar']['tau_A']:>10.1f}"
              f"{e['active']['tau_D']:>10.1f}{e['calendar']['tau_D']:>10.1f}"
              f"{(e['calendar']['loop'] or float('nan')):>10.1f}")

    print(f"\nNull at {a.nnull} circular-shift replicates (calendar indexing):")
    for g, e in out["populations"].items():
        if "null" not in e:
            continue
        n = e["null"]
        print(f"  [{g}]  responder share obs {n['responder_share']['obs']:.3f} vs null mean "
              f"{n['responder_share']['null_mean']:.3f} (max {n['responder_share']['null_max']:.3f}), "
              f"p={n['responder_share']['p']:.4f}")
        for k, lbl in (("tau_A", "tau_A"), ("tau_D", "tau_D")):
            v = n[k]
            print(f"          {lbl} obs {v['obs']:.1f} vs null median {v['null_median']:.1f} "
                  f"[{v['null_lo']:.1f}, {v['null_hi']:.1f}], p(shorter)={v['p_shorter']:.4f}")

    p = Path(__file__).with_name("leadlag_calendar_recompute.json")
    out["inputs"] = _REPORTS
    json.dump(out, open(p, "w"), indent=2, default=float)
    print(f"\nwrote {p}")

    # Per-agent rows, added 2026-08-25. The companion's response-lag figure was still drawn
    # from the 2026-06-11 per-agent table -- a different cohort (883 vs 890) and a different
    # responder set -- while the text quoted this run. A figure cannot be re-captioned onto
    # data it does not plot, so the per-agent detail has to be persisted somewhere.
    keep = ("a_id", "group", "density",
            "active_responder", "active_post_lag", "active_pre_lag", "active_post_ccf",
            "calendar_responder", "calendar_post_lag", "calendar_pre_lag", "calendar_post_ccf")
    cp = Path(__file__).with_name("leadlag_calendar_per_agent.csv")
    pd.DataFrame([{k: r.get(k) for k in keep} for r in recs]).to_csv(cp, index=False)
    print(f"wrote {cp}  ({len(recs)} agents)")


if __name__ == "__main__":
    main()
