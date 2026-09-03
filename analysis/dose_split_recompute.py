"""Finding 3 (dose split) at the corrected event spacing.

Definitions follow `db_analysis_dose_robustness.py`: intensity is the event-day block rate, the
light/heavy split is at the within-group median of intensity, and dO is the post-minus-pre change
in the observability score. This runs them on the specification of record - 16-observation
spacing, block-insensitive 404 coordinate, `[-10,-6]` baseline.

The load-bearing claim is not the raw gap but its reversal under adjustment: the paper reads
the raw pattern as defender targeting rather than a dose response. That is what this checks.

Read-only, cache-only. Usage: python dose_split_recompute.py [--workers N]
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
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
# SPACING is 16, not 11. With the baseline [-10,-6] and post [+1,+5], events
# 11 apart have event-1's post window IDENTICAL to event-2's pre window: both are
# [t+1, t+5]. Eleven is the disjointness condition for ADJACENT +/-5 windows only, and
# moving the baseline recreated the overlap in a new position. 2*W + REF_span + 1 = 16
# is the first gap that makes the two windows disjoint.
MIN_DAYS_OBS, W, SPACING, LEAD, REF = 20, 5, 16, 10, -6
_C: dict = {}


def _init(per_id, frames, spec):
    _install_admitted(per_id)
    _C.update(frames=frames, spec=spec)


def _p404(df, coord):
    if coord == "baseline":
        return df["p404"].to_numpy(float)
    h = df["hits"].to_numpy(float)
    den = h - df["cblock"].to_numpy(float) - df["c429"].to_numpy(float)
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


def _wm(a, lo, hi):
    seg = a[max(0, lo):hi]
    if not len(seg) or np.all(np.isnan(seg)):
        return np.nan
    return float(np.nanmean(seg))


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


def _rows(args):
    a_id, grp = args
    df = _load(a_id)
    if df is None:
        return []
    spacing, coord, base = _C["spec"]
    mu, sd = _C["frames"]
    B = _behavior(df, coord)
    O = np.linalg.norm((B - mu) / sd, axis=1)
    pblk = df["p_block"].to_numpy(float)
    lv = np.log1p(df["hits"].to_numpy(float))
    need = LEAD if base == "clean" else W
    out = []
    for i in _events(df, spacing):
        if i - need < 0 or i + W >= len(df):
            continue
        pre = _wm(O, i - LEAD, i + REF + 1) if base == "clean" else _wm(O, i - W, i)
        post = _wm(O, i + 1, i + 1 + W)
        if not (np.isfinite(pre) and np.isfinite(post)):
            continue
        # Post-volume and the retreat flag. Both need per-event volume on BOTH sides of the
        # event; only the pre side was being carried.
        pre_lv_adj = _wm(lv, i - W, i)                      # adjacent [-5,-1]
        pre_lv_clean = _wm(lv, i - LEAD, i + REF + 1)       # clean [-10,-6], matches pre_O
        pre_lv = pre_lv_clean if base == "clean" else pre_lv_adj
        post_lv = _wm(lv, i + 1, i + 1 + W)
        retreat = (np.isfinite(pre_lv) and np.isfinite(post_lv)
                   and np.expm1(post_lv) < 0.5 * np.expm1(pre_lv))
        out.append({"a_id": a_id, "group": grp, "intensity": float(pblk[i]),
                    "pre_O": pre, "dO": post - pre,
                    "pre_logvol": pre_lv, "pre_logvol_adjacent": pre_lv_adj,
                    "pre_logvol_clean": pre_lv_clean, "post_logvol": post_lv,
                    "dvol": (post_lv - pre_lv) if (np.isfinite(post_lv) and np.isfinite(pre_lv)) else np.nan,
                    "retreat": bool(retreat)})
    return out


def _cluster_boot(d, fn, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    ids = d["a_id"].to_numpy()
    uniq = np.unique(ids)
    groups = {u: d[ids == u] for u in uniq}
    vals = []
    for _ in range(n):
        pick = rng.choice(uniq, len(uniq), True)
        s = pd.concat([groups[u] for u in pick], ignore_index=True)
        v = fn(s)
        if np.isfinite(v):
            vals.append(v)
    return np.array(vals)


def _gap(d):
    med = d["intensity"].median()
    h, l = d[d.intensity > med], d[d.intensity <= med]
    if len(h) < 5 or len(l) < 5:
        return np.nan
    return float(h.dO.mean() - l.dO.mean())


def _adj(d):
    med = d["intensity"].median()
    heavy = (d["intensity"] > med).to_numpy(float)
    X = np.column_stack([np.ones(len(d)), heavy, d.pre_O.to_numpy(float),
                         d.pre_logvol.to_numpy(float)])
    y = d.dO.to_numpy(float)
    m = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    if m.sum() < 10:
        return np.nan
    return float(np.linalg.lstsq(X[m], y[m], rcond=None)[0][1])


CALIPER = 0.2   # matching caliper in pooled-SD units, as in db_analysis_dose_robustness.py


def _match(G, ycol):
    """1-NN with replacement on (pre_O, pre_logvol), pooled within-dose SD, 0.2 caliper.

    Standardization is by the POOLED WITHIN-DOSE standard deviation, not the overall event SD,
    and a 0.2 caliper applies: matching every heavy event regardless of distance would report a
    matched estimate over pairs that are not comparable. Both choices change the answer, so both
    are stated here rather than left to the reader to infer from the code.
    Returns (heavy-minus-light difference in ycol, n matched, n heavy).
    """
    med = G.intensity.median()
    heavy = (G.intensity > med).to_numpy()
    pre_O = G.pre_O.to_numpy(float)
    pre_lv = G.pre_logvol.to_numpy(float)
    y = G[ycol].to_numpy(float)
    ok = np.isfinite(pre_O) & np.isfinite(pre_lv) & np.isfinite(y)
    heavy, pre_O, pre_lv, y = heavy[ok], pre_O[ok], pre_lv[ok], y[ok]
    if heavy.sum() < 10 or (~heavy).sum() < 10:
        return np.nan, 0, int(heavy.sum())
    zsd = np.array([np.sqrt((np.var(pre_O[heavy], ddof=1) + np.var(pre_O[~heavy], ddof=1)) / 2),
                    np.sqrt((np.var(pre_lv[heavy], ddof=1) + np.var(pre_lv[~heavy], ddof=1)) / 2)])
    zmu = np.array([pre_O.mean(), pre_lv.mean()])
    Z = (np.column_stack([pre_O, pre_lv]) - zmu) / (zsd + 1e-12)
    zh, zl = Z[heavy], Z[~heavy]
    dist, j = cKDTree(zl).query(zh, k=1)
    keep = dist <= CALIPER
    if not keep.any():
        return np.nan, 0, int(heavy.sum())
    diff = float(y[heavy][keep].mean() - y[~heavy][j[keep]].mean())
    return diff, int(keep.sum()), int(heavy.sum())


def _matched_gap(G):
    return _match(G, "dO")[0]


def _matched_retreat(G):
    return _match(G, "retreat")[0]


def _retreat_gap(G):
    """Heavy-minus-light difference in the share of events classified as retreat."""
    med = G.intensity.median()
    h, l = G[G.intensity > med], G[G.intensity <= med]
    if len(h) < 10 or len(l) < 10:
        return np.nan
    return float(h.retreat.mean() - l.retreat.mean())


def _adj_retreat(d):
    """OLS dose coefficient on RETREAT, controlling for pre-event covariates.

    The article's conclusion that "at the margin, dose works in the deterrence direction"
    rests on the adjusted and matched dose effect on retreat, not on the raw gap -- the raw
    gap is negative because defenders aim heavier friction at agents already behaving
    evasively. Both are recomputed here.
    """
    med = d["intensity"].median()
    heavy = (d["intensity"] > med).to_numpy(float)
    X = np.column_stack([np.ones(len(d)), heavy, d.pre_O.to_numpy(float),
                         d.pre_logvol.to_numpy(float)])
    y = d.retreat.to_numpy(float)
    m = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    if m.sum() < 10:
        return np.nan
    return float(np.linalg.lstsq(X[m], y[m], rcond=None)[0][1])


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
    _REPORTS = admit(pops, CACHE, "dose_split", out_dir=HERE,
                     analysis_min_rows=MIN_DAYS_OBS, required_cols=("hits", "p404", "robots_rate", "cblock", "c429"))
    _PER_ID = merge_per_id(_REPORTS)

    out = {}
    for label, spec in (("published (spacing 5, baseline 404, pre [-5,-1])", (5, "baseline", "near")),
                        # Label derived from the constant so it can never drift from the
                        # data again: it read "spacing 11" after SPACING became 16, and
                        # the manuscript cites this key.
                        (f"corrected (spacing {SPACING}, block-insensitive, pre [-10,-6])",
                         (SPACING, "block_insensitive", "clean"))):
        coord = spec[1]
        with ProcessPoolExecutor(max_workers=a.workers, initializer=_install_admitted,
                                 initargs=(_PER_ID,)) as ex:
            P = [m for m in ex.map(_profile, [(i, coord) for i in pops["verified"]],
                                   chunksize=20) if m is not None]
        M = np.array(P, float)
        frames = (np.nanmean(M, axis=0), np.nanstd(M, axis=0) + 1e-9)
        tasks = [(i, g) for g, ids in pops.items() for i in ids]
        rows = []
        with ProcessPoolExecutor(max_workers=a.workers, initializer=_init,
                                 initargs=(_PER_ID, frames, spec)) as ex:
            for r in ex.map(_rows, tasks, chunksize=100):
                rows.extend(r)
        d = pd.DataFrame(rows)
        # Persist the per-event frame for the specification of record. The balance figure used
        # to come from a different script with a different baseline, so its panels could drift
        # from the numbers they illustrate. Rendering the
        # figure from this frame makes the two consistent by construction.
        if label.startswith("corrected"):
            d.to_csv(HERE / "dose_events_corrected.csv", index=False)
            print(f"      per-event frame -> dose_events_corrected.csv ({len(d):,} rows)")
        res = {}
        for g in ("abusive", "verified"):
            G = d[d.group == g].reset_index(drop=True)
            if len(G) < 20:
                continue
            med = G.intensity.median()
            h, l = G[G.intensity > med], G[G.intensity <= med]
            gap = _gap(G)
            gb = _cluster_boot(G, _gap)
            adj = _adj(G)
            ab = _cluster_boot(G, _adj)
            smd = {c: float((h[c].mean() - l[c].mean()) /
                            np.sqrt((h[c].var() + l[c].var()) / 2))
                   for c in ("pre_O", "pre_logvol")}
            # Pre-event levels behind those SMDs. The manuscript states the balance test in
            # levels as well as in standard units, and a standardized difference alone cannot
            # be checked against the data by a reader; emitting both keeps the quoted levels
            # tied to the same spacing as the SMD rather than to a superseded run.
            levels = {c: {"heavy": float(h[c].mean()), "light": float(l[c].mean()),
                          "diff": float(h[c].mean() - l[c].mean())}
                      for c in ("pre_O", "pre_logvol")}
            _pre_O_diff = lambda D: float(D[D.intensity > D.intensity.median()].pre_O.mean()
                                          - D[D.intensity <= D.intensity.median()].pre_O.mean())
            _pb = _cluster_boot(G, _pre_O_diff)
            levels["pre_O"]["diff_ci"] = [float(np.percentile(_pb, 2.5)),
                                          float(np.percentile(_pb, 97.5))]
            res[g] = {"n_events": int(len(G)), "n_agents": int(G.a_id.nunique()),
                      "dO_light": float(l.dO.mean()), "dO_heavy": float(h.dO.mean()),
                      "raw_gap": gap, "raw_gap_ci": [float(np.percentile(gb, 2.5)),
                                                     float(np.percentile(gb, 97.5))],
                      "adjusted_gap": adj,
                      "adjusted_gap_ci": [float(np.percentile(ab, 2.5)),
                                          float(np.percentile(ab, 97.5))],
                      "smd_pre_O": smd["pre_O"], "smd_pre_logvol": smd["pre_logvol"],
                      "pre_levels": levels,
                      "matched_gap": _matched_gap(G),
                      "matched_gap_ci": [float(np.percentile(_cluster_boot(G, _matched_gap), 2.5)),
                                         float(np.percentile(_cluster_boot(G, _matched_gap), 97.5))],
                      "retreat_light": float(l.retreat.mean()),
                      "retreat_heavy": float(h.retreat.mean()),
                      "retreat_gap": _retreat_gap(G),
                      "retreat_gap_ci": [float(np.percentile(_cluster_boot(G, _retreat_gap), 2.5)),
                                         float(np.percentile(_cluster_boot(G, _retreat_gap), 97.5))],
                      "adjusted_retreat": _adj_retreat(G),
                      "adjusted_retreat_ci": [float(np.percentile(_cluster_boot(G, _adj_retreat), 2.5)),
                                              float(np.percentile(_cluster_boot(G, _adj_retreat), 97.5))],
                      "matched_retreat": _matched_retreat(G),
                      "matched_retreat_ci": [float(np.percentile(_cluster_boot(G, _matched_retreat), 2.5)),
                                             float(np.percentile(_cluster_boot(G, _matched_retreat), 97.5))],
                      "n_matched": _match(G, "dO")[1], "n_heavy": _match(G, "dO")[2],
                      "caliper": CALIPER,
                      # Baseline sensitivity: the adjusted retreat coefficient with the
                      # volume covariate on the ADJACENT window instead of the outcome's own.
                      # Reported rather than chosen, because the sign depends on it.
                      "adjusted_retreat_adjacent_vol": _adj_retreat(
                          G.assign(pre_logvol=G.pre_logvol_adjacent)),
                      "dvol_light": float(l.dvol.mean()), "dvol_heavy": float(h.dvol.mean())}
        out[label] = res
        print(f"\n=== {label} ===")
        for g, r in res.items():
            print(f"  [{g}] {r['n_events']:,} events / {r['n_agents']:,} agents")
            print(f"      dO light {r['dO_light']:+.3f}   dO heavy {r['dO_heavy']:+.3f}")
            print(f"      raw gap (heavy-light) {r['raw_gap']:+.4f} "
                  f"[{r['raw_gap_ci'][0]:+.4f}, {r['raw_gap_ci'][1]:+.4f}]")
            print(f"      adjusted gap          {r['adjusted_gap']:+.4f} "
                  f"[{r['adjusted_gap_ci'][0]:+.4f}, {r['adjusted_gap_ci'][1]:+.4f}]")
            print(f"      pre-event balance SMD: pre_O {r['smd_pre_O']:+.2f}, "
                  f"pre_logvol {r['smd_pre_logvol']:+.2f}")
    p = Path(__file__).with_name("dose_split_recompute.json")
    out["inputs"] = _REPORTS
    json.dump(out, open(p, "w"), indent=2, default=float)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
