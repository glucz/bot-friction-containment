"""Event-time study with leads: the parallel-trends evidence a window estimator needs.

A window estimator compares one pre-window mean against one post-window mean, which is two
numbers per event and carries no information about whether a trend was already running
inside the baseline.

For every event with full +/-10 coverage, the observability trajectory is
normalized to **t = -6**, one step outside the estimator's own pre-window, so that days
-5..-1 show whether a trend is already running *inside the window the DiD treats as
baseline*. Days +1..+5 show the effect; +6..+10 show persistence.

Parallel-trends is then testable two ways: each pre-period point's bot-minus-control
difference should straddle zero, and its slope over t in [-5,-1] should not differ from zero.

Runs at the corrected event spacing and with the block-insensitive 404 coordinate, which are
the definitions of record.

Read-only, cache-only. Usage: python event_time_study.py [--workers N] [--spacing 11]
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

MIN_DAYS_OBS = 20
EVENT_W = 5
LEAD = 10          # event-time half-width
REF = -6           # normalization period: one step outside the DiD pre-window
_C: dict = {}


def _init(per_id, frames, spacing, coord):
    _install_admitted(per_id)
    _C.update(frames=frames, spacing=spacing, coord=coord)


def _p404(df, coord):
    if coord == "baseline":
        return df["p404"].to_numpy(float)
    h = df["hits"].to_numpy(float)
    den = h - df["cblock"].to_numpy(float) - df["c429"].to_numpy(float)
    num = df["c404"].to_numpy(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def _behavior(df, coord):
    return np.column_stack([_p404(df, coord),
                            df["robots_rate"].to_numpy(float),
                            np.log1p(df["hits"].to_numpy(float))])


def _block_events(df, spacing):
    blk = df["cblock"].to_numpy(float)
    r429 = df["c429"].to_numpy(float)
    idx = set()
    if len(blk) >= 3 and blk.std() > 0:
        idx.update(np.flatnonzero(blk > max(blk.mean() + 1.5 * blk.std(), 0)).tolist())
    idx.update(np.flatnonzero(r429 > 0).tolist())
    kept, last = [], -10 ** 9
    for i in sorted(idx):
        if i - last >= spacing:
            kept.append(i)
            last = i
    return kept


def _load(a_id):
    # Input failure and scientific exclusion are different facts; see input_contract.py.
    df = load_admitted(a_id, CACHE, _ADMITTED)
    return df if len(df) >= MIN_DAYS_OBS else None


def _profile(args):
    a_id, coord = args
    df = _load(a_id)
    if df is None:
        return None
    with np.errstate(invalid="ignore"):
        return np.nanmean(_behavior(df, coord), axis=0)


def _traj(args):
    """Per-agent mean event-time trajectory of O, normalized to t = REF."""
    a_id, grp = args
    df = _load(a_id)
    if df is None:
        return None
    B = _behavior(df, _C["coord"])
    mu, sd = _C["frames"]
    O = np.linalg.norm((B - mu) / sd, axis=1)
    # balanced panel: only events with the full +/-LEAD span available
    ev = [i for i in _block_events(df, _C["spacing"]) if i - LEAD >= 0 and i + LEAD < len(df)]
    if not ev:
        return None
    rows = []
    for i in ev:
        seg = O[i - LEAD: i + LEAD + 1].astype(float)
        ref = seg[LEAD + REF]
        if not np.isfinite(ref):
            continue
        rows.append(seg - ref)
    if not rows:
        return None
    with np.errstate(invalid="ignore"):
        return {"a_id": a_id, "group": grp, "n_events": len(rows),
                "traj": np.nanmean(np.vstack(rows), axis=0)}


def _boot(M, n=2000, seed=42):
    """Bootstrap over agents (rows). Returns mean and 95% band per column."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(M), (n, len(M)))
    with np.errstate(invalid="ignore"):
        draws = np.nanmean(M[idx], axis=1)
        return np.nanmean(M, axis=0), np.nanpercentile(draws, 2.5, axis=0), \
            np.nanpercentile(draws, 97.5, axis=0), draws


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--spacing", type=int, default=11)
    ap.add_argument("--coord", default="block_insensitive",
                    choices=["baseline", "block_insensitive"])
    a = ap.parse_args()

    ro = pd.read_csv(require(ROSTER, "the roster (`outputs/roster_v2.csv`)"))
    pops = {"abusive": sorted(set(ro[ro.role == "abusive_share"].a_id)),
            "verified": sorted(set(ro[ro.role == "verified_human"].a_id))}
    # Admit every declared input BEFORE any pool starts. A cohort that shrinks to whatever is on
    # disk is not a cohort, and a worker's own register never reaches the parent.
    global _PER_ID
    _REPORTS = admit(pops, CACHE, "event_time", out_dir=HERE,
                     analysis_min_rows=MIN_DAYS_OBS, required_cols=("hits", "p404", "robots_rate", "cblock", "c429"))
    _PER_ID = merge_per_id(_REPORTS)


    with ProcessPoolExecutor(max_workers=a.workers, initializer=_install_admitted,
                             initargs=(_PER_ID,)) as ex:
        P = [m for m in ex.map(_profile, [(i, a.coord) for i in pops["verified"]], chunksize=20)
             if m is not None]
    M = np.array(P, float)
    frames = (np.nanmean(M, axis=0), np.nanstd(M, axis=0) + 1e-9)
    print(f"standardization frame: {len(M)} verified-human agents")

    tasks = [(i, g) for g, ids in pops.items() for i in ids]
    recs = []
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init,
                             initargs=(_PER_ID, frames, a.spacing, a.coord)) as ex:
        for r in ex.map(_traj, tasks, chunksize=100):
            if r is not None:
                recs.append(r)

    t = np.arange(-LEAD, LEAD + 1)
    out = {"spacing": a.spacing, "coord": a.coord, "ref_period": REF, "t": t.tolist()}
    curves = {}
    for g in ("abusive", "verified"):
        G = [r for r in recs if r["group"] == g]
        M = np.vstack([r["traj"] for r in G])
        m, lo, hi, draws = _boot(M)
        curves[g] = (M, m, lo, hi, draws)
        out[g] = {"n_agents": len(G), "n_events": int(sum(r["n_events"] for r in G)),
                  "mean": m.tolist(), "lo": lo.tolist(), "hi": hi.tolist()}
        print(f"  {g}: {len(G)} agents, {out[g]['n_events']} balanced events")

    # bot - control difference, bootstrapped jointly
    dm = curves["abusive"][1] - curves["verified"][1]
    dd = curves["abusive"][4] - curves["verified"][4]
    dlo = np.nanpercentile(dd, 2.5, axis=0)
    dhi = np.nanpercentile(dd, 97.5, axis=0)
    out["difference"] = {"mean": dm.tolist(), "lo": dlo.tolist(), "hi": dhi.tolist()}

    pre = np.where((t >= -5) & (t <= -1))[0]
    post = np.where((t >= 1) & (t <= 5))[0]
    print(f"\n{'t':>4}{'bot-ctrl':>11}{'95% CI':>22}   parallel-trends")
    for k in range(len(t)):
        star = ""
        if t[k] in (-5, -4, -3, -2, -1):
            star = "  <-- PRE (should straddle 0)" if not (dlo[k] > 0 or dhi[k] < 0) \
                else "  <-- PRE **VIOLATION**"
        print(f"{t[k]:>4}{dm[k]:>11.4f}   [{dlo[k]:>7.4f},{dhi[k]:>7.4f}]{star}")

    # slope of the difference over the pre-window
    slopes = np.polyfit(t[pre], dd[:, pre].T, 1)[0]
    s_lo, s_hi = np.percentile(slopes, [2.5, 97.5])
    s_obs = float(np.polyfit(t[pre], dm[pre], 1)[0])
    viol = int(sum((dlo[k] > 0) or (dhi[k] < 0) for k in pre))
    out["pretrend"] = {"slope": s_obs, "slope_ci": [float(s_lo), float(s_hi)],
                       "slope_sig": bool(s_lo > 0 or s_hi < 0),
                       "n_pre_points_excluding_zero": viol,
                       "did_check_post_minus_pre": float(dm[post].mean() - dm[pre].mean())}
    print(f"\nPARALLEL TRENDS")
    print(f"  pre-window points whose CI excludes zero: {viol} of 5")
    print(f"  pre-window slope of the difference: {s_obs:+.5f} per day, "
          f"95% CI [{s_lo:+.5f}, {s_hi:+.5f}] -> "
          f"{'VIOLATED' if (s_lo > 0 or s_hi < 0) else 'not rejected'}")
    print(f"  post-mean minus pre-mean of the difference: "
          f"{dm[post].mean() - dm[pre].mean():+.4f}  (DiD analogue)")

    # ---- re-specification: is there a CLEAN pre-window further back?
    farpre = np.where((t >= -LEAD) & (t <= REF))[0]
    fslopes = np.polyfit(t[farpre], dd[:, farpre].T, 1)[0]
    f_lo, f_hi = np.percentile(fslopes, [2.5, 97.5])
    f_obs = float(np.polyfit(t[farpre], dm[farpre], 1)[0])
    print("")
    print(f"  FAR-PRE window t in [{-LEAD}, {REF}]: slope {f_obs:+.5f} per day, "
          f"95% CI [{f_lo:+.5f}, {f_hi:+.5f}] -> "
          f"{'VIOLATED' if (f_lo > 0 or f_hi < 0) else 'not rejected -> usable baseline'}")
    print(f"  far-pre level of the difference: {dm[farpre].mean():+.4f}")

    for lbl, pw in (("published window  pre=[-5,-1]", pre),
                    ("clean window      pre=[-10,-6]", farpre)):
        est = dd[:, post].mean(axis=1) - dd[:, pw].mean(axis=1)
        e_obs = float(dm[post].mean() - dm[pw].mean())
        lo_, hi_ = np.percentile(est, [2.5, 97.5])
        print(f"  DiD, {lbl}, post=[+1,+5]: {e_obs:+.4f}  [{lo_:+.4f}, {hi_:+.4f}]")
    out["respec"] = {
        "farpre_slope": f_obs, "farpre_slope_ci": [float(f_lo), float(f_hi)],
        "farpre_usable": bool(not (f_lo > 0 or f_hi < 0)),
        "did_published_window": float(dm[post].mean() - dm[pre].mean()),
        "did_clean_window": float(dm[post].mean() - dm[farpre].mean()),
        "did_clean_ci": [float(x) for x in np.percentile(
            dd[:, post].mean(axis=1) - dd[:, farpre].mean(axis=1), [2.5, 97.5])]}

    # ---- figure
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for g, c, lbl in (("abusive", "tab:red", "abusive automation"),
                      ("verified", "tab:blue", "verified humans")):
        _, m, lo, hi, _ = curves[g]
        ax[0].plot(t, m, "-o", ms=3, color=c, label=lbl)
        ax[0].fill_between(t, lo, hi, color=c, alpha=.18)
    ax[0].set_title("Observability around block events", fontsize=10)
    ax[0].set_ylabel(f"$O$ relative to $t={REF}$")
    ax[1].plot(t, dm, "-o", ms=3, color="k", label="abusive - human")
    ax[1].fill_between(t, dlo, dhi, color="k", alpha=.15)
    ax[1].set_title("Difference, with the DiD pre-window shaded", fontsize=10)
    ax[1].set_ylabel("difference in $O$")
    for x in ax:
        x.axvline(0, color="grey", lw=.8, ls="--")
        x.axhline(0, color="grey", lw=.6)
        x.axvspan(-5, -1, color="tab:green", alpha=.08)
        x.set_xlabel("event time (active days; 0 = block event)")
        x.legend(fontsize=8)
        x.grid(alpha=.25)
    fig.suptitle(f"Event-time study, spacing {a.spacing}, {a.coord} 404 coordinate "
                 f"(balanced panel, normalized to t={REF})", fontsize=9)
    fig.tight_layout()
    fp = Path(__file__).with_name("event_time_study.png")
    fig.savefig(fp, dpi=150)
    print(f"\nwrote {fp}")

    jp = Path(__file__).with_name("event_time_study.json")
    out["inputs"] = _REPORTS
    json.dump(out, open(jp, "w"), indent=2, default=float)
    print(f"wrote {jp}")


if __name__ == "__main__":
    main()
