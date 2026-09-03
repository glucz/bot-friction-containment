"""How robust is the 0.66% disappearance rate?

Three objections the headline number does not address, and each is
checkable from the cache:

  1. **Follow-up.** "Never seen again" is only meaningful relative to how long we could have
     seen the agent. If the rate falls to zero among episodes with long follow-up, the 18
     events are partly a censoring artifact.
  2. **Episode weighting.** An identifier can disappear once but contributes many episodes to
     the denominator, so the episode-weighted rate is not the per-agent risk.
  3. **Short absence is not disappearance.** How many episodes go quiet for the post window
     and then come back?

Read-only, cache-only. Usage: python disappearance_sensitivity.py [--workers N]
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
# SPACING 11 -> 16 to agree with combined_numbers_of_record.py, which is cited in the same
# sentence. This measure has no pre-window, so window overlap does not force 16; the reason
# is that two artifacts defining the same episode risk set differently, both cited, is not
# shippable. The looser rule is the more conservative risk set here.
MIN_DAYS_OBS, W, SPACING = 20, 5, 16
PANEL_END = pd.Timestamp("2023-03-05")


def _events(blk, r429, spacing):
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


def _one(args):
    a_id, grp = args
    # Input failure and scientific exclusion are different facts; see input_contract.py.
    d = load_admitted(a_id, CACHE, _ADMITTED)
    if len(d) < MIN_DAYS_OBS:
        return []
    d = d.copy()
    d["date"] = pd.to_datetime(d["date"])
    last_seen = d["date"].max()
    out = []
    for i in _events(d["cblock"].to_numpy(float), d["c429"].to_numpy(float), SPACING):
        if i < W:
            continue
        ed = d["date"].iloc[i]
        post_end = ed + pd.Timedelta(days=W)
        censored = post_end > PANEL_END
        later = d["date"][d["date"] > ed]
        gone = len(later) == 0
        # did it go quiet for the post window and then return?
        quiet5 = not ((d["date"] > ed) & (d["date"] <= post_end)).any()
        returned = quiet5 and len(later[later > post_end]) > 0
        out.append({"a_id": a_id, "group": grp, "event_date": ed,
                    "censored": bool(censored), "gone": bool(gone),
                    "quiet5": bool(quiet5), "returned_after_quiet": bool(returned),
                    "followup_days": int((PANEL_END - ed).days),
                    "days_to_last_seen": int((last_seen - ed).days)})
    return out


def wilson(k, n, z=1.96):
    if not n:
        return (float("nan"),) * 2
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    # clamp: these are proportions, and at k = 0 the algebra returns -1e-17 for the lower edge
    return max(0.0, 100 * (c - h)), min(100.0, 100 * (c + h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    ro = pd.read_csv(require(ROSTER, "the roster (`outputs/roster_v2.csv`)"))
    tasks = [(int(r.a_id), "abusive" if r.role == "abusive_share" else "verified")
             for r in ro.itertuples() if r.role in ("abusive_share", "verified_human")]
    # Admit every declared input BEFORE any pool starts. A cohort that shrinks to whatever is on
    # disk is not a cohort, and a worker's own register never reaches the parent.
    global _PER_ID
    _REPORTS = admit({g: [i for i, gg in tasks if gg == g] for g in ("abusive", "verified")}, CACHE, "disappearance", out_dir=HERE,
                     analysis_min_rows=MIN_DAYS_OBS, required_cols=("date", "cblock", "c429"))
    _PER_ID = merge_per_id(_REPORTS)

    rows = []
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_install_admitted,
                             initargs=(_PER_ID,)) as ex:
        for r in ex.map(_one, tasks, chunksize=100):
            rows.extend(r)
    d = pd.DataFrame(rows)
    out = {}
    for g in ("abusive", "verified"):
        G = d[(d.group == g) & (~d.censored)]
        n, k = len(G), int(G.gone.sum())
        lo, hi = wilson(k, n)
        out[g] = {"evaluable": n, "gone": k, "pct": 100 * k / n,
                  "wilson": [lo, hi],
                  "agents_with_events": int(G.a_id.nunique()),
                  "agents_that_vanish": int(G[G.gone].a_id.nunique())}
        print(f"[{g}] {k}/{n} episodes = {100*k/n:.2f}%  Wilson [{lo:.2f}, {hi:.2f}]")
        print(f"      agent-weighted: {out[g]['agents_that_vanish']}/"
              f"{out[g]['agents_with_events']} agents = "
              f"{100*out[g]['agents_that_vanish']/max(out[g]['agents_with_events'],1):.2f}%")

    A = d[(d.group == "abusive") & (~d.censored)]
    print(f"\nshort absence is not disappearance:")
    q = A[A.quiet5]
    print(f"  quiet for the whole post window: {len(q)}")
    print(f"  of those, returned later:        {int(q.returned_after_quiet.sum())} "
          f"({100*q.returned_after_quiet.mean():.1f}%)")
    out["quiet5"] = {"n": int(len(q)), "returned": int(q.returned_after_quiet.sum())}

    print(f"\nsensitivity to required follow-up (abusive):")
    out["followup"] = []
    for req in (5, 30, 90, 180, 365, 730):
        S = A[A.followup_days >= req]
        if not len(S):
            continue
        k, n = int(S.gone.sum()), len(S)
        lo, hi = wilson(k, n)
        out["followup"].append({"min_days": req, "n": n, "gone": k, "pct": 100 * k / n,
                                "wilson": [lo, hi]})
        print(f"  >= {req:>4} d follow-up: {k:>3}/{n:<5} = {100*k/n:5.2f}%  [{lo:.2f}, {hi:.2f}]")

    p = Path(__file__).with_name("disappearance_sensitivity.json")
    out["inputs"] = _REPORTS
    json.dump(out, open(p, "w"), indent=2, default=float)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
