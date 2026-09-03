"""Share of each population that ever fetches robots.txt, from the per-agent series.

The estimand is a POPULATION SHARE: the fraction of agents in each population whose series carries
at least one day of nonzero robots.txt retrieval. It depends on no part of the event design - not
spacing, window width or baseline - and is computed directly from the per-agent series.

The denominator is agents meeting the same minimum-observation rule the rest of Section 7.7 uses,
so the share is comparable with the other per-agent statistics rather than with a raw roster count.

Populations are defined in `POPULATIONS-KEY.md` and are not restated here: the abusive arm is the
roster role `abusive_share`; declared, verified and potential-human are the curated classes. Every
declared id is admitted through the shared input contract before any agent is counted, so the
denominator is the population it names rather than whatever parsed.

Read-only apart from the JSON it writes.
Usage: python robots_engagement.py
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _layout import data_root, on_path, require  # noqa: E402

ES = on_path()
CACHE = ES / "cache" / "agents"
ROSTER = ES / "outputs" / "roster_v2.csv"
OUT = HERE / "robots_engagement.json"

sys.path.insert(0, str(ES))
from input_contract import admit, load_admitted, merge_per_id  # noqa: E402

MIN_DAYS_OBS = 20


def _curated(ext: str) -> list[int]:
    import config
    out = set()
    for fp in glob.glob(str(config.DATA_DIR / ("*." + ext))):
        try:
            aid = int(os.path.splitext(os.path.basename(fp))[0])
        except ValueError:
            continue
        if aid <= 2_147_483_647:
            out.add(aid)
    return sorted(out)


def main() -> None:
    ro = pd.read_csv(require(ROSTER, "the roster (`outputs/roster_v2.csv`)"))
    pops = {
        "abusive": sorted({int(i) for i in ro[ro.role == "abusive_share"].a_id}),
        "declared": _curated("bot"),
        "verified": _curated("human"),
        "pothuman": _curated("pothuman"),
    }

    # Admission first, as everywhere else: a share whose denominator is "whatever parsed" is not
    # a share of the population it names. `admit` also pins content across runs through the
    # registry, so a cached file replaced between runs is refused rather than moving the share.
    inputs = admit(pops, CACHE, "robots", out_dir=HERE,
                   analysis_min_rows=MIN_DAYS_OBS, required_cols=("robots_rate",))
    per_id = merge_per_id(inputs)

    out: dict = {"min_days_obs": MIN_DAYS_OBS,
                 "definition": ("share of agents with at least one day of nonzero robots.txt "
                                "retrieval, over agents meeting the minimum-observation rule"),
                 "populations": {}, "inputs": inputs}
    for g, ids in sorted(pops.items()):
        n = ever = 0
        for a_id in ids:
            df = load_admitted(a_id, CACHE, per_id)
            if len(df) < MIN_DAYS_OBS:
                continue
            n += 1
            if (df["robots_rate"].fillna(0) > 0).any():
                ever += 1
        out["populations"][g] = {
            "n_declared": len(ids), "n_qualifying": n, "n_ever_robots": ever,
            "pct_ever_robots": round(100.0 * ever / max(n, 1), 2),
        }
        print(f"  {g:10s} {len(ids):5d} declared, {n:5d} qualifying, {ever:5d} ever "
              f"= {out['populations'][g]['pct_ever_robots']:.2f}%")

    a = out["populations"]["abusive"]["pct_ever_robots"]
    d = out["populations"]["declared"]["pct_ever_robots"]
    out["declared_over_abusive_ratio"] = round(d / a, 1) if a else None
    print(f"\n  declared / abusive ratio: {out['declared_over_abusive_ratio']}x")

    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
