"""Lead-lag estimates split by how densely each agent is observed.

The circular-shift null needs a series long enough to shift. On a sparse agent it degenerates
toward the search boundary, so a pooled delay estimate is dominated by agents for which the
instrument does not discriminate. This splits the abusive arm by observation density and reports
each band separately, which is why the article sweeps the loop delay rather than fixing it.

Two denominators are reported and never merged. `responder_share_all` is over every agent in the
band; `responder_share_test` is over the null-eligible subset, and it is the one compared with the
null, because the null distribution is built from exactly those agents. Comparing an all-agent
share against a null-eligible distribution tests one population against another's reference.

Read-only apart from the JSON it writes, beside this file rather than in the working directory.
Usage: python leadlag_density_split.py [--workers N] [--nnull N]
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import leadlag_calendar_recompute as M  # noqa: E402
from input_contract import admit, merge_per_id  # noqa: E402

OUT = HERE / "leadlag_density_split.json"
BANDS = [("dense  d>=0.50", lambda d: d >= 0.50),
         ("mid    0.25-0.50", lambda d: 0.25 <= d < 0.50),
         ("sparse d<0.25", lambda d: d < 0.25),
         ("ALL", lambda d: True)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--nnull", type=int, default=999)
    a = ap.parse_args()

    ro = pd.read_csv(M.ROSTER)
    ids = sorted({int(r.a_id) for r in ro.itertuples() if r.role == "abusive_share"})

    # Admission before the pool, as everywhere else. Without it this script measures the
    # intersection of the arm with whatever is on disk, which is the defect the split is meant to
    # illuminate rather than reproduce.
    reports = admit({"abusive": ids}, M.CACHE, "leadlag_density", out_dir=HERE,
                    analysis_min_rows=M.MIN_DAYS_LL,
                    required_cols=("date", "cblock", "p404", "hits"))
    per_id = merge_per_id(reports)

    tasks = [(i, "abusive") for i in ids]
    recs = []
    with ProcessPoolExecutor(max_workers=a.workers, initializer=M._init,
                             initargs=(per_id, a.nnull)) as ex:
        for r in ex.map(M._one, tasks, chunksize=50):
            if r is not None:
                recs.append(r)
    print(f"abusive cohort: {len(recs)}")

    print(f"\n{'band':<20}{'n_all':>7}{'n_null':>8}{'resp%_all':>11}{'resp%_test':>12}"
          f"{'tauA':>7}{'tauD':>7}{'nullA med':>11}{'p_share':>9}")
    out: dict = {"n_null_replicates": a.nnull, "bands": {}, "inputs": reports}
    for lbl, f in BANDS:
        G = [r for r in recs if f(r["density"])]
        if len(G) < 10:
            continue
        resp = [r for r in G if r.get("calendar_responder") is True]
        lead = [r for r in G if r.get("calendar_responder") is False]
        tA = M._boot_median([r["calendar_post_lag"] for r in resp])[0]
        tD = M._boot_median([abs(r["calendar_pre_lag"]) for r in lead])[0]

        H = [r for r in G if "_null_responder" in r]
        R = np.vstack([r["_null_responder"] for r in H])
        share = R.mean(axis=0)
        with np.errstate(invalid="ignore"):
            mA = np.nanmedian(np.vstack([r["_null_resp_lag"] for r in H]), axis=0)
            mD = np.nanmedian(np.vstack([r["_null_lead_lag"] for r in H]), axis=0)

        # The share tested against the null is taken over the null-eligible set, not over every
        # agent in the band.
        share_all = len(resp) / len(G)
        share_test = sum(r.get("calendar_responder") is True for r in H) / len(H)
        p = float(((share >= share_test).sum() + 1) / (len(share) + 1))

        print(f"{lbl:<20}{len(G):>7}{len(H):>8}{100 * share_all:>11.1f}{100 * share_test:>12.1f}"
              f"{tA:>7.1f}{tD:>7.1f}{np.nanmedian(mA):>11.1f}{p:>9.4f}")
        out["bands"][lbl] = {
            "n_all": len(G), "n_null_eligible": len(H),
            "responder_share_all": share_all, "responder_share_test": share_test,
            "tau_A": tA, "tau_D": tD,
            "null_tauA_median": float(np.nanmedian(mA)),
            "null_tauA_ci": [float(np.nanpercentile(mA, 2.5)),
                             float(np.nanpercentile(mA, 97.5))],
            "null_tauD_median": float(np.nanmedian(mD)),
            "p_responder_share": p,
        }

    OUT.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
