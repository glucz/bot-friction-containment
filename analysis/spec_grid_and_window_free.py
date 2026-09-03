"""Is the headline observability DiD a design-invariant magnitude? Specification grid plus window-free estimators.

A 576-cell cache sweep could not flip the sign, but it did
show the magnitude is design-dependent, and that the [-10,-6] baseline was chosen after the
adjacent baseline failed a pre-trend test. That is a researcher degree of freedom, and a
window-free estimator does not require it.

This reproduces both independently rather than citing the reviewer's artifacts:

  * a specification grid over half-width, event spacing, baseline gap and 404 coordinate;
  * two estimators that never choose a baseline window at all:
      FULL_PRE_LINEAR  - fit the pre-period trend over t in [-10,-1], extrapolate it through
                         the post window, and take the post-period deviation from that trend.
                         This is the honest answer to "what if the pre-trend is real?"
      SEGMENTED        - a single event-time regression on t in [-10,-1] u [+1,+10] with a
                         post indicator and a common linear trend; the post coefficient is
                         the level shift net of trend.

Read-only, cache-only. Usage: python spec_grid_and_window_free.py [--workers N]
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

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
LEAD = 10
_C: dict = {}


def _init(per_id, frames, coord):
    _install_admitted(per_id)
    _C.update(frames=frames, coord=coord)


def _p404(df, coord):
    if coord == "baseline":
        return df["p404"].to_numpy(float)
    h = df["hits"].to_numpy(float)
    if coord == "nonblocked":
        den = h - df["cblock"].to_numpy(float) - df["c429"].to_numpy(float)
    else:  # vs200
        den = df["c404"].to_numpy(float) + df["c200"].to_numpy(float)
    num = df["c404"].to_numpy(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def _behavior(df, coord):
    return np.column_stack([_p404(df, coord), df["robots_rate"].to_numpy(float),
                            np.log1p(df["hits"].to_numpy(float))])


def _events(df, spacing):
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


def _agent(args):
    """Per-agent event-time trajectory of O on the balanced +/-LEAD panel."""
    a_id, grp, spacing = args
    df = _load(a_id)
    if df is None:
        return None
    mu, sd = _C["frames"]
    O = np.linalg.norm((_behavior(df, _C["coord"]) - mu) / sd, axis=1)
    ev = [i for i in _events(df, spacing) if i - LEAD >= 0 and i + LEAD < len(df)]
    if not ev:
        return None
    rows = [O[i - LEAD:i + LEAD + 1].astype(float) for i in ev]
    with np.errstate(invalid="ignore"):
        return {"a_id": a_id, "group": grp, "traj": np.nanmean(np.vstack(rows), axis=0)}


def _did_from_traj(M, W, gap):
    """post mean over [+1,+W] minus pre mean over [-(gap+W), -(gap+1)]."""
    t = np.arange(-LEAD, LEAD + 1)
    pre = np.where((t <= -(gap + 1)) & (t >= -(gap + W)))[0]
    post = np.where((t >= 1) & (t <= W))[0]
    if not len(pre) or not len(post):
        return None
    with np.errstate(invalid="ignore"):
        return np.nanmean(M[:, post], axis=1) - np.nanmean(M[:, pre], axis=1)


def _boot_diff(a, b, n=2000, seed=42):
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    rng = np.random.default_rng(seed)
    d = rng.choice(a, (n, len(a)), True).mean(1) - rng.choice(b, (n, len(b)), True).mean(1)
    return float(a.mean() - b.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    ro = pd.read_csv(require(ROSTER, "the roster (`outputs/roster_v2.csv`)"))
    pops = {"abusive": sorted(set(ro[ro.role == "abusive_share"].a_id)),
            "verified": sorted(set(ro[ro.role == "verified_human"].a_id))}
    # Admit every declared input BEFORE any pool starts. A cohort that shrinks to whatever is on
    # disk is not a cohort, and a worker's own register never reaches the parent.
    global _PER_ID
    _REPORTS = admit(pops, CACHE, "spec_grid", out_dir=HERE,
                     analysis_min_rows=MIN_DAYS_OBS, required_cols=("hits", "p404", "robots_rate", "cblock", "c429"))
    _PER_ID = merge_per_id(_REPORTS)


    out = {"grid": [], "window_free": {}}
    traj_cache = {}
    for coord in ("baseline", "nonblocked", "vs200"):
        with ProcessPoolExecutor(max_workers=a.workers, initializer=_install_admitted,
                                 initargs=(_PER_ID,)) as ex:
            P = [m for m in ex.map(_profile, [(i, coord) for i in pops["verified"]],
                                   chunksize=20) if m is not None]
        frames = (np.nanmean(np.array(P, float), axis=0),
                  np.nanstd(np.array(P, float), axis=0) + 1e-9)
        # 16 added 2026-08-25: the preferred specification pairs the [-10,-6] baseline
        # with post [+1,+5], which needs a gap of 16 to be disjoint. Without a 16 cell the
        # grid cannot contain the submitted estimator, and its percentile ranks a spec the
        # paper no longer uses.
        for spacing in (5, 11, 15, 16, 21):
            tasks = [(i, g, spacing) for g, ids in pops.items() for i in ids]
            recs = []
            with ProcessPoolExecutor(max_workers=a.workers, initializer=_init,
                                     initargs=(_PER_ID, frames, coord)) as ex:
                for r in ex.map(_agent, tasks, chunksize=100):
                    if r is not None:
                        recs.append(r)
            B = np.vstack([r["traj"] for r in recs if r["group"] == "abusive"])
            C = np.vstack([r["traj"] for r in recs if r["group"] == "verified"])
            traj_cache[(coord, spacing)] = (B, C)
            for W, gap in product((3, 5, 7), (0, 1, 3, 5)):
                db, dc = _did_from_traj(B, W, gap), _did_from_traj(C, W, gap)
                if db is None:
                    continue
                est, lo, hi = _boot_diff(db, dc)
                out["grid"].append({"coord": coord, "spacing": spacing, "W": W, "gap": gap,
                                    "did": est, "lo": lo, "hi": hi,
                                    "n_bot": int(len(B)), "n_ctrl": int(len(C))})
            print(f"  {coord:<11} spacing {spacing:<3} bots {len(B):>4} ctrl {len(C):>4}")

    g = pd.DataFrame(out["grid"])
    chosen = g[(g.coord == "nonblocked") & (g.spacing == 11) & (g.W == 5) & (g.gap == 5)]
    print(f"\n{len(g)} specifications")
    print(f"  all negative: {bool((g.did < 0).all())}")
    print(f"  min {g.did.min():.4f}  p2.5 {g.did.quantile(.025):.4f}  median {g.did.median():.4f}"
          f"  p97.5 {g.did.quantile(.975):.4f}  max {g.did.max():.4f}")
    print(f"  significant (CI excludes 0): {int((g.hi < 0).sum())}/{len(g)}")
    if len(chosen):
        c = float(chosen.did.iloc[0])
        print(f"  submitted spec: {c:.4f}, at the {100*(g.did <= c).mean():.0f}th percentile "
              f"(fraction at least as negative)")
    print("\n  median by baseline gap: "
          + ", ".join(f"gap {k}: {v:.4f}" for k, v in g.groupby('gap').did.median().items()))
    print("  median by coordinate:   "
          + ", ".join(f"{k}: {v:.4f}" for k, v in g.groupby('coord').did.median().items()))

    # ---- window-free estimators on the submitted event set --------------------
    # The window-free estimators are the alternative to the preferred WINDOW choice, so they
    # must run on the preferred specification's event set: spacing 16, not the superseded 11.
    B, C = traj_cache[("nonblocked", 16)]
    t = np.arange(-LEAD, LEAD + 1)
    pre_m = np.where(t <= -1)[0]
    post_m = np.where(t >= 1)[0]

    def full_pre_linear(M):
        """Extrapolate each agent's own pre-trend and take the mean post deviation."""
        res = []
        for row in M:
            y = row[pre_m]
            ok = np.isfinite(y)
            if ok.sum() < 4:
                res.append(np.nan); continue
            b1, b0 = np.polyfit(t[pre_m][ok], y[ok], 1)
            pred = b0 + b1 * t[post_m]
            with np.errstate(invalid="ignore"):
                res.append(float(np.nanmean(row[post_m] - pred)))
        return np.array(res, float)

    def segmented(M):
        """Per-agent OLS of O on [1, t, post]; the post coefficient nets out the trend."""
        res = []
        idx = np.concatenate([pre_m, post_m])
        post_flag = (t[idx] > 0).astype(float)
        X = np.column_stack([np.ones(len(idx)), t[idx], post_flag])
        for row in M:
            y = row[idx]
            ok = np.isfinite(y)
            if ok.sum() < 6:
                res.append(np.nan); continue
            res.append(float(np.linalg.lstsq(X[ok], y[ok], rcond=None)[0][2]))
        return np.array(res, float)

    for name, fn in (("FULL_PRE_LINEAR", full_pre_linear), ("SEGMENTED", segmented)):
        est, lo, hi = _boot_diff(fn(B), fn(C))
        out["window_free"][name] = {"did": est, "ci": [lo, hi], "sig": bool(hi < 0)}
        print(f"\n  {name:<16}{est:+.4f}  [{lo:+.4f}, {hi:+.4f}]  "
              f"{'significant' if hi < 0 else 'NOT significant'}")

    p = Path(__file__).with_name("spec_grid_and_window_free.json")
    json.dump({"grid": out["grid"], "window_free": out["window_free"],
               "summary": {"n_specs": int(len(g)), "all_negative": bool((g.did < 0).all()),
                           "min": float(g.did.min()), "median": float(g.did.median()),
                           "max": float(g.did.max()),
                           "n_significant": int((g.hi < 0).sum())},
               "inputs": _REPORTS},
              open(p, "w"), indent=2, default=float)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
