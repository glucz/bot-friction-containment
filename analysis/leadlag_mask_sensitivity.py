"""The missing-day sensitivity that `leadlag_calendar_recompute.py` promises and never ran.

That script's docstring states an explicit missing-versus-zero rule and says the alternative,
masking the day out, "is reported as a sensitivity by the caller". No caller ever did. The
Masking moves the pooled result, so the alternative is run rather than asserted.

Zero-fill treats a silent day as a real zero: nothing probed, no friction experienced. That is
behaviourally defensible but it makes ~85% of a sparse agent's series identical zeros, which is
why the circular-shift null degenerates. Masking instead computes each lag from the days where
BOTH series are observed, so the estimate never sees an imputed value.

Neither is obviously right. The point of running both is that if they disagree, the delay is
not identified by the data, which is what the manuscript now claims.

Read-only, cache-only. Usage: python leadlag_mask_sensitivity.py [--workers N]
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
MIN_DAYS_LL, MIN_BLOCK_DAYS, LAG_MAX = 30, 3, 14


def _masked_ccf(resp, frict, lag_max=LAG_MAX):
    """Pairwise-complete cross-correlation: each lag uses only days observed in both."""
    out_l, out_c = [], []
    for k in range(-lag_max, lag_max + 1):
        if k >= 0:
            r, f = resp[k:], frict[:len(frict) - k] if k else frict
        else:
            r, f = resp[:len(resp) + k], frict[-k:]
        ok = np.isfinite(r) & np.isfinite(f)
        if ok.sum() < 10:
            out_l.append(k); out_c.append(np.nan); continue
        rr, ff = r[ok], f[ok]
        sr, sf = rr.std(), ff.std()
        if sr < 1e-9 or sf < 1e-9:
            out_l.append(k); out_c.append(np.nan); continue
        out_l.append(k)
        out_c.append(float(((rr - rr.mean()) * (ff - ff.mean())).mean() / (sr * sf)))
    return np.array(out_l), np.array(out_c, float)


def _peaks(lags, ccf):
    out = {}
    for side, m in (("post", lags >= 1), ("pre", lags <= -1)):
        c = ccf[m]
        if not m.any() or np.all(~np.isfinite(c)):
            out[side + "_lag"], out[side + "_ccf"] = np.nan, np.nan
            continue
        i = int(np.nanargmax(np.abs(c)))
        out[side + "_lag"] = int(lags[m][i])
        out[side + "_ccf"] = float(c[i])
    return out


def _one(args):
    a_id, grp = args
    # Input failure and scientific exclusion are different facts; see input_contract.py.
    df = load_admitted(a_id, CACHE, _ADMITTED)
    if len(df) < MIN_DAYS_LL or int((df["cblock"] > 0).sum()) < MIN_BLOCK_DAYS:
        return None
    if np.nanstd(df["p_block"].to_numpy(float)) < 1e-9:
        return None
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    span = pd.date_range(d["date"].min(), d["date"].max(), freq="D")
    cal = d.set_index("date").reindex(span)          # missing days stay NaN: masked
    resp = cal["p404"].to_numpy(float)
    frict = cal["p_block"].to_numpy(float)
    if np.all(~np.isfinite(resp)) or np.nanstd(frict) < 1e-9:
        return None
    lags, ccf = _masked_ccf(resp, frict)
    s = _peaks(lags, ccf)
    is_resp = bool(abs(s["post_ccf"]) > abs(s["pre_ccf"])) if (
        np.isfinite(s["post_ccf"]) and np.isfinite(s["pre_ccf"])) else None
    return {"a_id": a_id, "group": grp, "density": len(d) / len(span),
            "post_lag": s["post_lag"], "pre_lag": s["pre_lag"], "responder": is_resp}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    ro = pd.read_csv(require(ROSTER, "the roster (`outputs/roster_v2.csv`)"))
    tasks = [(int(r.a_id), "abusive" if r.role == "abusive_share"
              else "verified" if r.role == "verified_human" else "other")
             for r in ro.itertuples() if r.role in ("abusive_share", "verified_human")]
    # Admit every declared input BEFORE any pool starts. A cohort that shrinks to whatever is on
    # disk is not a cohort, and a worker's own register never reaches the parent.
    global _PER_ID
    _REPORTS = admit({g: [i for i, gg in tasks if gg == g] for g in sorted({gg for _, gg in tasks})}, CACHE, "leadlag_mask", out_dir=HERE,
                     analysis_min_rows=MIN_DAYS_LL, required_cols=("date", "cblock", "p404"))
    _PER_ID = merge_per_id(_REPORTS)

    recs = []
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_install_admitted,
                             initargs=(_PER_ID,)) as ex:
        for r in ex.map(_one, tasks, chunksize=50):
            if r is not None:
                recs.append(r)
    d = pd.DataFrame(recs)
    out = {}
    print(f"{'population':<12}{'n':>6}{'resp%':>8}{'tau_A':>8}{'tau_D':>8}{'loop':>7}")
    for g in ("abusive", "verified"):
        G = d[d.group == g]
        resp = G[G.responder == True]                       # noqa: E712
        lead = G[G.responder == False]                      # noqa: E712
        tA = float(np.nanmedian(resp.post_lag)) if len(resp) else np.nan
        tD = float(np.nanmedian(np.abs(lead.pre_lag))) if len(lead) else np.nan
        out[g] = {"n": int(len(G)), "responder_share": float((G.responder == True).mean()),
                  "tau_A": tA, "tau_D": tD, "loop": tA + tD}
        print(f"{g:<12}{len(G):>6}{100*out[g]['responder_share']:>7.1f}%"
              f"{tA:>8.1f}{tD:>8.1f}{tA + tD:>7.1f}")

    A = d[d.group == "abusive"]
    print(f"\nabusive arm by density band (masked):")
    out["bands"] = {}
    for lbl, sel in (("dense d>=0.50", A.density >= .50), ("mid 0.25-0.50",
                     (A.density >= .25) & (A.density < .50)), ("sparse d<0.25", A.density < .25)):
        B = A[sel]
        if len(B) < 10:
            continue
        r = B[B.responder == True]                          # noqa: E712
        l = B[B.responder == False]                         # noqa: E712
        tA = float(np.nanmedian(r.post_lag)); tD = float(np.nanmedian(np.abs(l.pre_lag)))
        out["bands"][lbl] = {"n": int(len(B)), "tau_A": tA, "tau_D": tD, "loop": tA + tD}
        print(f"  {lbl:<16}n={len(B):<5} tau_A {tA:>5.1f}  tau_D {tD:>5.1f}  loop {tA + tD:>5.1f}")

    print("\ncompare zero-filled (leadlag_calendar_recompute.json): abusive tau_A 4, tau_D 5, "
          "loop 9; bands 4 / 15 / 10")
    p = Path(__file__).with_name("leadlag_mask_sensitivity.json")
    out["inputs"] = _REPORTS
    json.dump(out, open(p, "w"), indent=2, default=float)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
