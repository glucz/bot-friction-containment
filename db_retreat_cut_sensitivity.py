"""Retreat share under alternative volume-drop cuts, on the honeypot-share arm.

Section 7.4 states that the 50% volume-drop convention separating retreat from
stealth adaptation is not load-bearing, and supports that with the retreat
fraction at a permissive 30% cut and a strict 70% cut. The cohort is the arm of
record (7,812 agents, read from `outputs/DB_arm_of_record_ids.csv`: the honeypot-share criterion minus the
hand-verified humans), so the event count here matches the one the findings
report rather than the raw criterion output.

Runs entirely off the per-agent parquet cache; no database access.

Usage: python db_retreat_cut_sensitivity.py --spacing 11 --output DB_retreat_cut_sensitivity_sp11.json
Writes: outputs/DB_retreat_cut_sensitivity_sp11.json (the defaults; both are arguments)
"""
from __future__ import annotations

import json
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import sys as _sys
from pathlib import Path as _Path


def _find_analysis() -> _Path:
    """Locate `input_contract` in the source tree OR in the public bundle.

    A path computed for one layout breaks in the other, so both are searched and a failure names
    what was looked for rather than surfacing as a bare import error.
    """
    here = _Path(__file__).resolve()
    for cand in (here.parents[1] / "papers" / "D-adversarial-friction" / "analysis",  # source tree
                 here.parent / "analysis",                                            # bundle
                 here.parent):
        if (cand / "input_contract.py").exists():
            return cand
    raise SystemExit(
        f"input_contract.py not found from {here}. Looked in the source tree "
        f"(../papers/D-adversarial-friction/analysis) and the public bundle (./analysis).")


_sys.path.insert(0, str(_find_analysis()))
from input_contract import admit, load_admitted, merge_per_id  # noqa: E402

import config
import db_source as dbs
from arm_ids import arm_of_record_ids
from population_of_record import EVENT_W, MIN_DAYS_OBS, curated
import event_study_primitives as esp
from event_study_primitives import _block_events, _winmean

CUTS = (0.30, 0.50, 0.70)
# Spacing and destination are arguments, not environment: a documented command whose result
# depends on an unmentioned variable produces a different artifact for the reader who follows the
# instructions than for the author who set it. The spacing this run used is written into the JSON,
# so a value always carries the geometry it was computed under.
DEFAULT_SPACING = 11
DEFAULT_OUT = "DB_retreat_cut_sensitivity_sp11.json"


_ADMITTED: dict = {}


def _install_admitted(per_id):
    global _ADMITTED
    _ADMITTED = per_id


def _counts(a_id):
    """Per-agent event count and, per cut, how many events count as retreat.

    Every id here was admitted in the parent, so a missing or altered input is a pipeline failure
    and must raise. Returning None for an unreadable file would make it indistinguishable from an
    agent that simply has too few observations - the conflation the input contract exists to stop.
    """
    df = load_admitted(a_id, dbs._cache_path(1).parent, _ADMITTED)
    if len(df) < MIN_DAYS_OBS:
        return None
    idx = [i for i in _block_events(df) if i >= EVENT_W and i + EVENT_W < len(df)]
    if not idx:
        return None
    vol = np.log1p(df["hits"].to_numpy(float))
    ratio = np.array([np.expm1(_winmean(vol, i, "post", EVENT_W))
                      / max(np.expm1(_winmean(vol, i, "pre", EVENT_W)), 1e-9)
                      for i in idx])
    out = {"a_id": a_id, "n_events": len(idx)}
    for c in CUTS:
        # a "c cut" means: post volume fell to below (1 - c) of baseline
        out[f"n_retreat_{int(c * 100)}"] = int((ratio < (1.0 - c)).sum())
    return out


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__ and __doc__.split(chr(10))[0])
    ap.add_argument("--spacing", type=int, default=DEFAULT_SPACING,
                    help=f"minimum active observations between kept events "
                         f"(default {DEFAULT_SPACING}: these windows are adjacent, so 2*W+1)")
    ap.add_argument("--output", default=DEFAULT_OUT,
                    help=f"artifact filename under outputs/ (default {DEFAULT_OUT})")
    ap.add_argument("--check-inputs", action="store_true",
                    help="admit the declared population and stop, writing nothing")
    return ap.parse_args(argv)


def run(spacing: int = DEFAULT_SPACING, output: str = DEFAULT_OUT,
        check_inputs: bool = False) -> dict:
    config.ensure_dirs()
    if spacing < 1:
        raise SystemExit(f"--spacing must be at least 1, not {spacing}")

    # The workers are separate processes and re-import the primitives, so the spacing has to reach
    # them through the environment; setting it here keeps that an implementation detail of process
    # start-up rather than something the documented command depends on the reader to know.
    os.environ["AGWA_EVENT_SPACING"] = str(spacing)
    esp.EVENT_SPACING = spacing
    out_json = config.OUT_DIR / output
    A = {"abusive": set(arm_of_record_ids()),
         "declared": curated("bot"), "verified": curated("human"),
         "pothuman": curated("pothuman")}

    targets = sorted(set().union(*A.values()))
    reports = admit({"all": targets}, dbs._cache_path(1).parent, "retreat_cut",
                    out_dir=config.OUT_DIR, analysis_min_rows=MIN_DAYS_OBS,
                    required_cols=("hits", "cblock"))
    per_id = merge_per_id(reports)
    if check_inputs:
        print(f"  inputs admitted for {len(targets):,} agent(s); nothing written")
        return {"checked_inputs": True, "n_targets": len(targets), "inputs": reports}
    with ProcessPoolExecutor(max_workers=8, initializer=_install_admitted,
                             initargs=(per_id,)) as ex:
        recs = [r for r in ex.map(_counts, targets, chunksize=100) if r is not None]
    pa = pd.DataFrame(recs).set_index("a_id")

    res = {"cuts": list(CUTS), "n_events": {}, "retreat_share": {}, "inputs": reports}
    for k, ids in A.items():
        g = pa[pa.index.isin(ids)]
        n = int(g["n_events"].sum()) if len(g) else 0
        res["n_events"][k] = n
        res["retreat_share"][k] = {
            f"{int(c * 100)}": (round(float(g[f'n_retreat_{int(c * 100)}'].sum() / n), 4)
                                if n else None)
            for c in CUTS}

    res["event_spacing"] = spacing
    res["window_half_width"] = EVENT_W
    res["note"] = ("Windows are adjacent [-5,-1] and [+1,+5], so the disjointness condition is "
                   "2*W+1 = 11, not the 16 the [-10,-6] baseline of Findings 2 and 3 requires.")
    json.dump(res, open(out_json, "w"), indent=2)
    print(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    _a = _parse_args()
    run(spacing=_a.spacing, output=_a.output, check_inputs=_a.check_inputs)
