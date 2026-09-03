"""Post-block response estimands on the non-overlapping event set.

Windows are [i-W, i+W] with W = 5, so two kept events have disjoint windows only when
their centres differ by at least 2W+1 = 11 observations. Eleven is therefore the record
spacing for every estimand here, all of which use adjacent pre/post windows. (The
[-10,-6] baseline estimators elsewhere need 16; that is a different window geometry, not
a different opinion about the same one.)

Four populations are admitted before any estimate is computed - the abusive arm and the
curated declared, verified and potential-human classes, defined in POPULATIONS-KEY.md.

Emits, per population pairing: observability difference-in-differences, the four response
channels (404 probing, signed robots.txt rate, log volume, normalized robots.txt
response), retreat shares, cohort counts, the input manifests, and the event-aligned
figure data behind the robots.txt response figure.

Read-only, cache-only; no database. Usage: python event_spacing_recompute.py [--workers N]
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
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
EVENT_W = 5
SPACINGS = {"corrected_11": 2 * EVENT_W + 1}



def _block_events(df, spacing):
    blk = df["cblock"].to_numpy(dtype=float)
    r429 = df["c429"].to_numpy(dtype=float)
    idx = set()
    if len(blk) >= 3 and blk.std() > 0:
        thr = blk.mean() + 1.5 * blk.std()
        idx.update(np.flatnonzero(blk > max(thr, 0)).tolist())
    idx.update(np.flatnonzero(r429 > 0).tolist())
    kept, last = [], -10 ** 9
    for i in sorted(idx):
        if i - last >= spacing:
            kept.append(i)
            last = i
    return kept


def _winmean(a, c, side, w=EVENT_W):
    lo, hi = (max(0, c - w), c) if side == "pre" else (c + 1, min(len(a), c + 1 + w))
    seg = a[lo:hi]
    if not len(seg) or np.all(np.isnan(seg)):
        return np.nan
    return float(np.nanmean(seg))


def _did(a, b, n=2000):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 5 or len(b) < 5:
        return {"did": None, "ci": [None, None], "sig": False, "n_bot": len(a), "n_ctrl": len(b)}
    rng = np.random.default_rng(42)
    d = rng.choice(a, (n, len(a)), True).mean(1) - rng.choice(b, (n, len(b)), True).mean(1)
    lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    # Components, added 2026-08-25. A DiD is bot-minus-control, and a manuscript sentence
    # about what BOTS do cannot be written from the contrast alone: a positive DiD is equally
    # consistent with bots rising and with controls falling. The Finding 5 volume channel is
    # the live case -- it flips to significantly positive at spacing 11 -- so the components
    # decide whether the honest wording is "volume rises" or "volume rises RELATIVE TO
    # CONTROLS". Emit them rather than re-deriving them by hand later.
    return {"did": float(a.mean() - b.mean()), "ci": [round(lo, 4), round(hi, 4)],
            "sig": bool(lo > 0 or hi < 0), "n_bot": len(a), "n_ctrl": len(b),
            "bot_mean": float(a.mean()), "ctrl_mean": float(b.mean()),
            "bot_median": float(np.median(a)), "ctrl_median": float(np.median(b))}


def _behavior(df):
    return np.column_stack([df["p404"].to_numpy(float),
                            df["robots_rate"].to_numpy(float),
                            np.log1p(df["hits"].to_numpy(float))])


def _load(a_id):
    # Input failure and scientific exclusion are different facts; see input_contract.py.
    df = load_admitted(a_id, CACHE, _ADMITTED)
    return df if len(df) >= MIN_DAYS_OBS else None


def _profile(a_id):
    df = _load(a_id)
    return None if df is None else _behavior(df).mean(axis=0)


_C: dict = {}


def _init(per_id, frames):
    _install_admitted(per_id)
    _C["frames"] = frames


def _delta(a_id):
    df = _load(a_id)
    if df is None:
        return None
    B = _behavior(df)
    rb = df["robots_rate"].to_numpy(float)
    lh = np.log1p(df["hits"].to_numpy(float))
    p4 = df["p404"].to_numpy(float)
    vol = lh
    vol_rb = float(np.std(np.diff(rb))) if len(rb) > 1 else 0.0
    rec = {"a_id": a_id}
    for sname, sp in SPACINGS.items():
        idx = [i for i in _block_events(df, sp) if i >= EVENT_W and i + EVENT_W < len(df)]
        if not idx:
            rec[f"{sname}__n_events"] = 0
            continue
        retreat = [bool(np.expm1(_winmean(vol, i, "post")) < 0.5 * np.expm1(_winmean(vol, i, "pre")))
                   for i in idx]
        stealth = [i for i, r in zip(idx, retreat) if not r]
        rec[f"{sname}__n_events"] = len(idx)
        rec[f"{sname}__n_retreat"] = int(sum(retreat))
        for ch, arr in (("d_robots", rb), ("d_loghits", lh), ("d_p404", p4)):
            rec[f"{sname}__{ch}"] = float(np.nanmean(
                [_winmean(arr, i, "post") - _winmean(arr, i, "pre") for i in idx]))
        rec[f"{sname}__z_robots"] = float(np.nanmean(
            [abs(_winmean(rb, i, "post") - _winmean(rb, i, "pre")) / (vol_rb + 1e-9) for i in idx]))
        # Event-aligned robots.txt profile, baseline-subtracted, from the same events this
        # specification keeps. The figure behind Section 7.7 is drawn from these numbers rather
        # than from a separate pipeline with its own cohort rule.
        prof = np.full(2 * EVENT_W + 1, np.nan)
        for k, off in enumerate(range(-EVENT_W, EVENT_W + 1)):
            vals = [rb[i + off] - _winmean(rb, i, "pre") for i in idx
                    if 0 <= i + off < len(rb)]
            if vals:
                prof[k] = float(np.nanmean(vals))
        rec[f"{sname}__profile"] = prof.tolist()

        for fname, (mu, sd) in _C["frames"].items():
            O = np.linalg.norm((B - mu) / sd, axis=1)
            f = lambda i: _winmean(O, i, "post") - _winmean(O, i, "pre")
            rec[f"{sname}__O_{fname}"] = float(np.nanmean([f(i) for i in idx]))
            rec[f"{sname}__O_{fname}_stealth"] = (
                float(np.nanmean([f(i) for i in stealth])) if stealth else np.nan)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    # Populations exactly as db_analysis_arms_v2 builds them: the abusive arm from the
    # roster of record, the other three from the curated per-agent label files. Using the
    # roster for `declared`/`pothuman` gives slightly different sets and will not replicate.
    import glob, os, sys
    sys.path.insert(0, str(ES))
    import config

    def curated(ext):
        out = set()
        for fp in glob.glob(str(config.DATA_DIR / ("*." + ext))):
            try:
                aid = int(os.path.splitext(os.path.basename(fp))[0])
            except ValueError:
                continue
            if aid <= 2_147_483_647:
                out.add(aid)
        return sorted(out)

    ro = pd.read_csv(require(ROSTER, "the roster (`outputs/roster_v2.csv`)"))
    pops = {"abusive": sorted(set(ro[ro.role == "abusive_share"].a_id)),
            "verified": curated("human"),
            "declared": curated("bot"),
            "pothuman": curated("pothuman")}
    print({k: len(v) for k, v in pops.items()})
    # Admit every declared input BEFORE any pool starts. A cohort that shrinks to whatever is on
    # disk is not a cohort, and a worker's own register never reaches the parent.
    global _PER_ID
    _REPORTS = admit(pops, CACHE, "event_spacing", out_dir=HERE,
                     analysis_min_rows=MIN_DAYS_OBS, required_cols=("hits", "p404", "robots_rate", "cblock", "c429"))
    _PER_ID = merge_per_id(_REPORTS)


    frames = {}
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_install_admitted,
                             initargs=(_PER_ID,)) as ex:
        for fname in ("verified", "pothuman"):
            M = np.array([m for m in ex.map(_profile, pops[fname], chunksize=20) if m is not None])
            frames[fname] = (M.mean(axis=0), M.std(axis=0) + 1e-9)
            print(f"frame {fname}: {len(M)} agents")

    targets = sorted(set().union(*[set(v) for v in pops.values()]))
    recs = []
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init, initargs=(_PER_ID, frames)) as ex:
        for r in ex.map(_delta, targets, chunksize=100):
            if r is not None:
                recs.append(r)
    pa = pd.DataFrame(recs).set_index("a_id")

    out = {}
    _FIGDATA = {"robots_response": {}}
    for sname in SPECS_ORDER:
        col = lambda s: f"{sname}__{s}"
        has = pa[col("n_events")].fillna(0) > 0
        g = {k: pa[pa.index.isin(v) & has] for k, v in pops.items()}
        r = {"n_agents": {k: int(len(v)) for k, v in g.items()},
             "n_events": {k: int(v[col("n_events")].sum()) for k, v in g.items()},
             "retreat_share": {k: (float(v[col("n_retreat")].sum() / v[col("n_events")].sum())
                                   if v[col("n_events")].sum() else None) for k, v in g.items()}}
        r["observability"] = {
            "abusive_vs_verified_full": _did(g["abusive"][col("O_verified")], g["verified"][col("O_verified")]),
            "abusive_vs_verified_stealth": _did(g["abusive"][col("O_verified_stealth")], g["verified"][col("O_verified_stealth")]),
            "abusive_vs_pothuman_full": _did(g["abusive"][col("O_pothuman")], g["pothuman"][col("O_pothuman")]),
            "abusive_vs_pothuman_stealth": _did(g["abusive"][col("O_pothuman_stealth")],
                                                g["pothuman"][col("O_pothuman_stealth")])
            if col("O_pothuman_stealth") in g["abusive"].columns else None,
            # The declared arm is scored against both reference profiles, so a declared contrast
            # quoted in the article comes from this run rather than from a separate one on a
            # different cohort.
            "declared_vs_verified_full": _did(g["declared"][col("O_verified")],
                                              g["verified"][col("O_verified")]),
            "declared_vs_pothuman_full": _did(g["declared"][col("O_pothuman")],
                                              g["pothuman"][col("O_pothuman")]),
        }
        # Every population pairing the article quotes, computed in one run on one cohort. A table
        # whose rows come from separate runs is a mixed-specification table however each row is
        # labelled, so the sensitivity column is emitted here rather than sourced elsewhere.
        CHS = ("d_p404", "d_robots", "d_loghits", "z_robots")
        r["channels"] = {ch: _did(g["abusive"][col(ch)], g["verified"][col(ch)]) for ch in CHS}
        r["channels_declared"] = {ch: _did(g["declared"][col(ch)], g["verified"][col(ch)])
                                  for ch in CHS}
        r["channels_pothuman"] = {ch: _did(g["abusive"][col(ch)], g["pothuman"][col(ch)])
                                  for ch in CHS}
        r["channels_declared_pothuman"] = {ch: _did(g["declared"][col(ch)], g["pothuman"][col(ch)])
                                           for ch in CHS}
        # The raw group means each contrast is built from, so a manuscript quoting a level rather
        # than a difference takes it from the same run.
        r["raw_means"] = {ch: {k: (float(v[col(ch)].mean()) if len(v) else None)
                               for k, v in g.items()} for ch in CHS}
        prof_col = f"{sname}__profile"
        if prof_col in pa.columns:
            fig = {}
            for k, v in g.items():
                mat = np.array([row for row in v[prof_col] if isinstance(row, list)], float)
                if not len(mat):
                    continue
                n = np.sum(~np.isnan(mat), axis=0)
                fig[k] = {
                    "mean": np.nanmean(mat, axis=0).tolist(),
                    "sem": (np.nanstd(mat, axis=0) / np.sqrt(np.maximum(n, 1))).tolist(),
                    "n_agents": int(len(mat)),
                    "n_events": int(v[col("n_events")].sum()),
                }
            _FIGDATA["robots_response"][sname] = {
                "offsets": list(range(-EVENT_W, EVENT_W + 1)),
                "measure": "mean baseline-subtracted robots_rate, baseline = the event's own "
                           "pre-window mean",
                "uncertainty": "standard error across agents",
                "spacing": SPACINGS[sname],
                "populations": fig,
                "input_digests": {k: v.get("content_digest_sha256")
                                  for k, v in _REPORTS.items()},
            }
        out[sname] = r

    # ---------------- report ----------------
    r11 = out["corrected_11"]
    print(f"\n{'quantity':<40}{'spacing 11':>16}")
    print("  " + "-" * 56)
    for k in ("abusive", "verified", "declared", "pothuman"):
        print(f"  {'agents with events [' + k + ']':<40}{r11['n_agents'][k]:>16,}")
    for k in ("abusive", "verified", "declared", "pothuman"):
        print(f"  {'events [' + k + ']':<40}{r11['n_events'][k]:>16,}")
    print("  " + "-" * 56)
    for key, blk in sorted(r11["observability"].items()):
        if blk is None:
            continue
        print(f"  {key:<40}{blk['did']:>16.4f}   CI {blk['ci']}"
              f"  {'sig' if blk['sig'] else 'NOT SIG'}")
    print("  " + "-" * 56)
    for group in ("channels", "channels_declared", "channels_pothuman",
                  "channels_declared_pothuman"):
        for ch in ("d_p404", "d_robots", "d_loghits", "z_robots"):
            blk = r11[group][ch]
            print(f"  {group + '.' + ch:<40}{blk['did']:>16.4f}   CI {blk['ci']}"
                  f"  {'sig' if blk['sig'] else 'NOT SIG'}")

    p = Path(__file__).with_name("event_spacing_recompute.json")
    json.dump({"results": out, "inputs": _REPORTS, "figure_data": _FIGDATA},
              open(p, "w"), indent=2, default=float)
    print(f"\nwrote {p}")


SPECS_ORDER = ["corrected_11"]

if __name__ == "__main__":
    main()
