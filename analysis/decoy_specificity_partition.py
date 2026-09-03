"""Emit the decoy-bridge specificity partition and the one-date deletion as an artifact.

Reads the shipped pair table and writes the four `min_pop` bins, the predefined precision
core, and the effect of deleting the single most-represented date. The bins are asserted to
be a partition of the envelope PER ROW: every pair must fall in exactly one, so a bin cannot
be dropped or an edge moved without the script failing rather than emitting a plausible table
with a hole in it. The source pair file is recorded by sha256 so a stale JSON beside a
regenerated table is caught.

Reads the pair table; writes only its own JSON.
Usage: python decoy_specificity_partition.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from _layout import data_root, on_path, require  # noqa: E402

TAB = data_root() / "outputs" / "tables"
PAIRS = TAB / "DB_rotation_pairs_v5.csv"
OUT = HERE / "decoy_specificity_partition.json"

# CORE_POP = 25 in db_analysis_identity_rotation, so the first two bins are exactly the
# predefined precision core. The 4-5 split is reported separately because it is the sharpest
# stratum in the file and omitting it is the error this artifact exists to prevent.
BINS = [(0, 5), (6, 25), (26, 100), (101, 10**9)]
CORE_POP = 25
BIG_DATE = "2022-02-10"
N_BOOT, SEED = 2000, 42


def _date_cluster_ci(sub: pd.DataFrame, col: str = "dB_v") -> list[float]:
    dates = sub["t_A"].unique()
    groups = {d: sub[sub["t_A"] == d][col].to_numpy(float) for d in dates}
    rng = np.random.default_rng(SEED)
    vals = [np.nanmean(np.concatenate([groups[d] for d in rng.choice(dates, len(dates), replace=True)]))
            for _ in range(N_BOOT)]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main() -> None:
    d = pd.read_csv(PAIRS)
    d["t_A"] = pd.to_datetime(d["t_A"]).dt.date.astype(str)
    out: dict = {"source": PAIRS.name, "n_envelope_pairs": int(len(d)), "bins": []}

    covered = 0
    for lo, hi in BINS:
        g = d[(d.min_pop >= lo) & (d.min_pop <= hi)]
        covered += len(g)
        out["bins"].append({
            "min_pop_lo": lo, "min_pop_hi": None if hi > 10**8 else hi,
            "n_pairs": int(len(g)), "n_deaths": int(g.a_death.nunique()),
            "n_dates": int(g.t_A.nunique()),
            "succ_dvol_mean": float(g.dB_v.mean()) if len(g) else None,
            "succ_dbot_mean": float(g.dB_b.mean()) if len(g) else None,
        })

    # EXHAUSTIVENESS, per row rather than by total. A matching sum can hide a compensating gap
    # and overlap; counting how many bins each row falls into cannot. Every row must land in
    # exactly one.
    hits = np.zeros(len(d), dtype=int)
    for lo, hi in BINS:
        hits += ((d.min_pop >= lo) & (d.min_pop <= hi)).to_numpy(int)
    out["n_pairs_covered"] = int((hits == 1).sum())
    out["n_pairs_uncovered"] = int((hits == 0).sum())
    out["n_pairs_double_counted"] = int((hits > 1).sum())
    out["exhaustive"] = bool((hits == 1).all())
    if not out["exhaustive"]:
        raise SystemExit(
            f"PARTITION NOT A PARTITION: {int((hits == 0).sum())} rows in no bin, "
            f"{int((hits > 1).sum())} rows in more than one, of {len(d)}")
    assert covered == len(d), "bin totals disagree with per-row coverage"

    # Provenance: a stale JSON beside a changed pair table would otherwise pass unnoticed.
    out["source_sha256"] = hashlib.sha256(PAIRS.read_bytes()).hexdigest()

    core = d[d.min_pop <= CORE_POP]
    out["predefined_core"] = {
        "definition": f"min_pop <= {CORE_POP} (CORE_POP), i.e. the first two bins",
        "n_pairs": int(len(core)), "n_deaths": int(core.a_death.nunique()),
        "succ_dvol_mean": float(core.dB_v.mean()),
        "succ_dvol_ci_date_cluster": _date_cluster_ci(core),
    }
    assert len(core) == out["bins"][0]["n_pairs"] + out["bins"][1]["n_pairs"], \
        "the core must be exactly the first two bins"

    big = d[d.t_A == BIG_DATE]
    rest = d[d.t_A != BIG_DATE]
    out["date_concentration"] = {
        "date": BIG_DATE, "n_pairs": int(len(big)),
        "share_of_envelope": round(len(big) / len(d), 4),
        "succ_dvol_mean": float(big.dB_v.mean()),
        "without_that_date": {
            "n_pairs": int(len(rest)), "n_deaths": int(rest.a_death.nunique()),
            "n_dates": int(rest.t_A.nunique()),
            "succ_dvol_mean": float(rest.dB_v.mean()),
            "succ_dvol_ci_date_cluster": _date_cluster_ci(rest),
        },
    }

    print(f"envelope pairs: {len(d):,}   exhaustive: {out['exhaustive']}")
    for b in out["bins"]:
        hi = b["min_pop_hi"] or "inf"
        print(f"  min_pop {b['min_pop_lo']:>3}-{str(hi):>5}: {b['n_pairs']:5,} pairs, "
              f"{b['n_deaths']:3} deaths, {b['n_dates']:3} dates, dvol {b['succ_dvol_mean']:+.4f}")
    c = out["predefined_core"]
    print(f"  predefined core: {c['n_pairs']} pairs / {c['n_deaths']} deaths, "
          f"{c['succ_dvol_mean']:+.4f}, date-CI "
          f"[{c['succ_dvol_ci_date_cluster'][0]:+.3f}, {c['succ_dvol_ci_date_cluster'][1]:+.3f}]")
    w = out["date_concentration"]["without_that_date"]
    print(f"  without {BIG_DATE}: {w['n_pairs']:,} pairs, {w['succ_dvol_mean']:+.4f}, date-CI "
          f"[{w['succ_dvol_ci_date_cluster'][0]:+.3f}, {w['succ_dvol_ci_date_cluster'][1]:+.3f}]")

    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT.name}")


if __name__ == "__main__":
    main()
