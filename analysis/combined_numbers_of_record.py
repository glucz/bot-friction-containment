"""The numbers of record: the observability difference-in-differences and the retreat estimand.

Two estimands, computed in one run because their specification choices interact.

**Retreat.** The risk set is CALENDAR, not active-day. Every block event enters; the post window
is five calendar days; a day with no requests is a real zero rather than a missing row; an event
whose window runs past the panel edge is reported as censored rather than dropped; and the outcome
is split three ways - disappeared, retreated, persisted. An active-day risk set would condition on
post-block survival, which excludes permanent disappearance by construction from a statistic about
disappearance, and would classify by a geometric mean of hits, which means "volume halved" rather
than "gone".

**Observability DiD.** The block-insensitive 404 coordinate, the event spacing and the
$[-10,-6]$ baseline interact, so the estimate of record comes from a single run with all three in
force rather than from composing separate deltas.

Read-only, cache-only. Usage: python combined_numbers_of_record.py [--workers N]
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from _layout import data_root, on_path, require  # noqa: E402

ES = on_path()
CACHE = ES / "cache" / "agents"
ROSTER = ES / "outputs" / "roster_v2.csv"

MIN_DAYS_OBS = 20
W = 5                     # pre/post half-window
# Minimum separation between two events of the same agent, so that no observation is reused.
# 11 makes ADJACENT +/-5 windows disjoint, but the baseline is [-10,-6] (below), and events 11
# apart have event-1's post window [t+1,t+5] IDENTICAL to event-2's baseline. Partial reuse
# persists to a gap of 15, so 16 is the first genuinely disjoint spacing: g_min = LEAD + W + 1.
SPACING = 16
LEAD, REF = 10, -6        # baseline t in [-10, -6], clear of the event's own pre-window
PANEL_END = pd.Timestamp("2023-03-05")
RETREAT_DROP = 0.5
_C: dict = {}


from input_contract import load_admitted  # noqa: E402

_ADMITTED: dict = {}


def _install_admitted(per_id):
    global _ADMITTED
    _ADMITTED = per_id or {}


def _init(frames, per_id=None):
    if per_id is not None:
        _install_admitted(per_id)
    _C["frames"] = frames


def _p404_bi(df):
    """404 share of responses a block cannot manufacture, so a block cannot move the coordinate."""
    h = df["hits"].to_numpy(float)
    den = h - df["cblock"].to_numpy(float) - df["c429"].to_numpy(float)
    num = df["c404"].to_numpy(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def _behavior(df):
    return np.column_stack([_p404_bi(df), df["robots_rate"].to_numpy(float),
                            np.log1p(df["hits"].to_numpy(float))])


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


class InputUnavailable(Exception):
    """A required input could not be read. Distinct from an agent that is scientifically excluded."""


# Every agent the run declines to use, with the reason. A scientific exclusion is a result; an
# unreadable input is a failure, and the two must not share a code path.
_EXCLUDED: dict[int, str] = {}


def _load(a_id, strict: bool = False):
    """Load an agent's cached series.

    Three outcomes, kept apart. A missing or unreadable file is an *input* failure: under
    `strict` it raises, because a cohort that silently shrinks to whatever is on disk is not a
    cohort. Too few observations is a *scientific* exclusion and is recorded with its reason.
    Otherwise the frame is returned.

    Collapsing the first two into the third defines the analysis by cache contents rather than by
    its roster, so they are kept apart and reported separately.
    """
    if _ADMITTED:
        # Under a frozen cohort every id was admitted in the parent; the worker's remaining job
        # is to prove the bytes it reads are the bytes that were admitted.
        df = load_admitted(a_id, CACHE, _ADMITTED)
    else:
        p = CACHE / f"{a_id}.parquet"
        if not p.exists():
            if strict:
                raise InputUnavailable(f"agent {a_id}: no cached series at {p.name}")
            _EXCLUDED[int(a_id)] = "input-missing"
            return None
        try:
            df = pd.read_parquet(p)
        except Exception as exc:
            if strict:
                raise InputUnavailable(f"agent {a_id}: unreadable ({type(exc).__name__})")
            _EXCLUDED[int(a_id)] = f"input-unreadable:{type(exc).__name__}"
            return None
    if len(df) < MIN_DAYS_OBS:
        _EXCLUDED[int(a_id)] = f"below-min-days:{len(df)}"
        return None
    return df


def _profile(a_id):
    df = _load(a_id)
    if df is None:
        return None
    with np.errstate(invalid="ignore"):
        return np.nanmean(_behavior(df), axis=0)


def _one(args):
    a_id, grp = args
    df = _load(a_id)
    if df is None:
        return None
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    last_seen = d["date"].max()

    # ---- calendar risk set for retreat -------------------------------------
    cal = d.set_index("date").reindex(
        pd.date_range(d["date"].min(), d["date"].max(), freq="D"))
    for c in ("hits", "cblock", "c429"):
        cal[c] = cal[c].fillna(0.0)
    ch = cal["hits"].to_numpy(float)
    cdates = cal.index
    # events located on the ACTIVE index, then mapped to calendar
    act_ev = _events(df["cblock"].to_numpy(float), df["c429"].to_numpy(float), SPACING)
    rec = {"a_id": a_id, "group": grp}
    n_all = n_cens = n_dis = n_ret = n_per = 0
    for i in act_ev:
        if i < W:
            continue
        ed = d["date"].iloc[i]
        n_all += 1
        post_end = ed + pd.Timedelta(days=W)
        if post_end > PANEL_END:
            n_cens += 1                       # right-censored at the panel edge
            continue
        pre_m = np.arange(len(cdates))[(cdates >= ed - pd.Timedelta(days=W)) & (cdates < ed)]
        post_m = np.arange(len(cdates))[(cdates > ed) & (cdates <= post_end)]
        if not len(pre_m):
            continue
        pre_v = ch[pre_m].mean()
        # a calendar day beyond the agent's last observation is a genuine zero
        post_v = ch[post_m].mean() if len(post_m) else 0.0
        if post_end > last_seen and post_v == 0.0:
            n_dis += 1                        # never seen again: true disappearance
        elif post_v < RETREAT_DROP * pre_v:
            n_ret += 1
        else:
            n_per += 1
    rec.update(n_events_all=n_all, n_censored=n_cens, n_disappear=n_dis,
               n_retreat=n_ret, n_persist=n_per)

    # ---- combined DiD on the clean baseline --------------------------------
    B = _behavior(df)
    mu, sd = _C["frames"]
    O = np.linalg.norm((B - mu) / sd, axis=1)
    ev = [i for i in act_ev if i - LEAD >= 0 and i + W < len(df)]
    vals, sv = [], []
    vol = np.log1p(df["hits"].to_numpy(float))
    for i in ev:
        base = O[i - LEAD: i + REF + 1]       # t in [-10, -6]
        post = O[i + 1: i + 1 + W]            # t in [+1, +5]
        if not len(base) or not len(post):
            continue
        with np.errstate(invalid="ignore"):
            v = np.nanmean(post) - np.nanmean(base)
        if not np.isfinite(v):
            continue
        vals.append(v)
        pre_v = np.expm1(np.nanmean(vol[max(0, i - W):i]))
        post_v = np.expm1(np.nanmean(vol[i + 1:i + 1 + W]))
        if not (post_v < RETREAT_DROP * pre_v):
            sv.append(v)
    if vals:
        rec["dO"] = float(np.mean(vals))
        rec["dO_stealth"] = float(np.mean(sv)) if sv else np.nan
        rec["n_dO_events"] = len(vals)
    return rec


def _did(a, b, n=2000):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 5 or len(b) < 5:
        return {"did": None, "ci": [None, None], "n_bot": len(a), "n_ctrl": len(b)}
    rng = np.random.default_rng(42)
    dd = rng.choice(a, (n, len(a)), True).mean(1) - rng.choice(b, (n, len(b)), True).mean(1)
    lo, hi = np.percentile(dd, [2.5, 97.5])
    # The component means are stored, not just their difference. A downstream document that quotes
    # "bot side fell X, control side rose Y" otherwise has no field to cite and its numbers cannot
    # be checked against anything -- which is how a pair that differences to a superseded DiD
    # survives a version bump unnoticed.
    return {"did": float(a.mean() - b.mean()), "ci": [round(float(lo), 4), round(float(hi), 4)],
            "sig": bool(lo > 0 or hi < 0), "n_bot": len(a), "n_ctrl": len(b),
            "mean_bot": round(float(a.mean()), 4), "mean_ctrl": round(float(b.mean()), 4)}


def partition_roster(ids):
    """Split a roster into cached / uncached, so the second group can be reported."""
    cached, uncached = [], []
    for i in ids:
        (cached if (CACHE / f"{i}.parquet").exists() else uncached).append(int(i))
    return cached, uncached


def load_frozen(path):
    """The cohort of record, as an explicit id list per estimate.

    Freezing the ids is what makes the estimate reproducible: without it a re-run
    inherits whatever the cache holds at that moment, which is not a property of the study.
    """
    if path is None:
        return None
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return {k: set(map(int, v)) for k, v in d["cohorts"].items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cohort", default=None,
                    help="freeze to the id lists in this JSON (default: cohort_of_record.json)")
    ap.add_argument("--no-freeze", action="store_true",
                    help="recompute the cohort from whatever the cache currently holds")
    ap.add_argument("--exclude", default=None,
                    help="comma-separated agent ids to drop, for reconstructing a prior cohort")
    ap.add_argument("--out", default=None, help="write the results JSON here instead of the default")
    a = ap.parse_args()
    # Freezing is the default: without it a re-run silently inherits the cache's current
    # contents, which is how this estimate drifted once already.
    default_cohort = Path(__file__).with_name("cohort_of_record.json")
    cohort_path = a.cohort or (None if a.no_freeze
                               else (default_cohort if default_cohort.exists() else None))
    frozen = load_frozen(cohort_path)
    if frozen:
        print(f"cohort frozen to {Path(cohort_path).name}")
        # Admission runs BEFORE any estimate. Existence is not admission: the parquet must
        # parse, carry the columns the analysis indexes, and meet the row minimum. Each
        # admitted id is pinned by content hash so a file swapped between admission and use
        # is detected rather than averaged in.
        from input_contract import preflight, merge_per_id
        # Content pins live in ONE registry, `input_pins.json`, consulted by `preflight`
        # itself. This script supplies no digests of its own; there is no second authority.
        _INPUT_MANIFEST = {
            g: preflight(ids, CACHE, f"combined_{g}", out_dir=Path(__file__).parent,
                         analysis_min_rows=MIN_DAYS_OBS,
                         required_cols=("date", "hits", "p404", "robots_rate"))
            for g, ids in sorted(frozen.items())
        }
        _install_admitted(merge_per_id(_INPUT_MANIFEST))
    else:
        print("cohort NOT frozen: computed from the cache as it currently stands")
    drop = {int(x) for x in a.exclude.split(",")} if a.exclude else set()
    ro = pd.read_csv(require(ROSTER, "the roster (`outputs/roster_v2.csv`)"))
    pops = {"abusive": sorted(set(ro[ro.role == "abusive_share"].a_id)),
            "verified": sorted(set(ro[ro.role == "verified_human"].a_id)),
            "pothuman": sorted(set(ro[ro.role == "pothuman"].a_id))}

    # The roster is the population; the cache decides who of it is reachable. Report that gap
    # rather than letting it disappear into the qualification filter.
    cache_gap = {}
    for g, ids in pops.items():
        cached, uncached = partition_roster(ids)
        cache_gap[g] = {"roster": len(ids), "cached": len(cached),
                        "uncached": len(uncached), "uncached_ids": uncached}
        if drop:
            pops[g] = [i for i in pops[g] if int(i) not in drop]
        if frozen and g in frozen:
            pops[g] = [i for i in pops[g] if int(i) in frozen[g]]

    def _frame(ids):
        with ProcessPoolExecutor(max_workers=a.workers, initializer=_install_admitted,
                                 initargs=(_ADMITTED,)) as ex:
            P = [m for m in ex.map(_profile, ids, chunksize=20) if m is not None]
        M = np.array(P, float)
        return (np.nanmean(M, axis=0), np.nanstd(M, axis=0) + 1e-9)

    frames = _frame(pops["verified"])
    frames_pot = _frame(pops["pothuman"])

    tasks = [(i, g) for g in ("abusive", "verified") for i in pops[g]]
    recs = []
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init,
                             initargs=(frames, _ADMITTED)) as ex:
        for r in ex.map(_one, tasks, chunksize=100):
            if r is not None:
                recs.append(r)
    pa = pd.DataFrame(recs)

    # Sensitivity arm on its own frame, same spacing and baseline as the headline.
    tasks_pot = [(i, g) for g in ("abusive", "pothuman") for i in pops[g]]
    recs_pot = []
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init,
                             initargs=(frames_pot, _ADMITTED)) as ex:
        for r in ex.map(_one, tasks_pot, chunksize=100):
            if r is not None:
                recs_pot.append(r)
    pp = pd.DataFrame(recs_pot)

    out = {}
    print(f"\n{'RETREAT ON A CALENDAR RISK SET':<62}{'abusive':>12}{'verified':>12}")
    for g in ("abusive", "verified"):
        G = pa[pa.group == g]
        out[g] = {k: int(G[k].sum()) for k in
                  ("n_events_all", "n_censored", "n_disappear", "n_retreat", "n_persist")}
        out[g]["n_agents"] = int(len(G))
    rows = [("agents", "n_agents"), ("all block events entering the risk set", "n_events_all"),
            ("  right-censored at the panel edge", "n_censored"),
            ("  DISAPPEARED (never seen again)", "n_disappear"),
            ("  retreated (volume more than halved)", "n_retreat"),
            ("  persisted", "n_persist")]
    for lbl, k in rows:
        print(f"  {lbl:<60}{out['abusive'][k]:>12,}{out['verified'][k]:>12,}")
    for g in ("abusive", "verified"):
        o = out[g]
        den = o["n_events_all"] - o["n_censored"]
        o["evaluable"] = den
        o["disappear_share"] = o["n_disappear"] / den if den else None
        o["retreat_share"] = o["n_retreat"] / den if den else None
        o["exit_share"] = (o["n_disappear"] + o["n_retreat"]) / den if den else None
    print(f"\n  {'evaluable events':<60}{out['abusive']['evaluable']:>12,}"
          f"{out['verified']['evaluable']:>12,}")
    for lbl, k in (("disappearance share", "disappear_share"),
                   ("retreat share (volume halved)", "retreat_share"),
                   ("EXIT share = disappeared + retreated", "exit_share")):
        print(f"  {lbl:<60}{100*out['abusive'][k]:>11.1f}%{100*out['verified'][k]:>11.1f}%")

    bot = pa[pa.group == "abusive"]; ctl = pa[pa.group == "verified"]
    out["did_combined"] = _did(bot["dO"], ctl["dO"])
    out["did_combined_stealth"] = _did(bot["dO_stealth"], ctl["dO_stealth"])
    print(f"\nCOMBINED DiD (spacing {SPACING}, block-insensitive 404, clean [-10,-6] baseline)")
    for lbl, k in (("full", "did_combined"), ("stealth-only", "did_combined_stealth")):
        v = out[k]
        print(f"  {lbl:<16}{v['did']:+.4f}  {v['ci']}  "
              f"(bots {v['n_bot']}, controls {v['n_ctrl']})")

    p = Path(__file__).with_name("combined_numbers_of_record.json")
    # sensitivity-arm contrast, on the same specification as the headline
    try:
        ab_p = pp[pp.group == "abusive"]["dO"].to_numpy(float)
        po_p = pp[pp.group == "pothuman"]["dO"].to_numpy(float)
        res_p = _did(ab_p, po_p)
        res_p["note"] = ("abusive versus the larger model-selected sensitivity arm, same spacing "
                         "and baseline as did_combined, on the sensitivity arm's own frame")
        out["did_combined_pothuman"] = res_p
        print(f"  sensitivity arm  {res_p['did']:+.4f}  {res_p['ci']}  "
              f"(bots {res_p['n_bot']}, controls {res_p['n_ctrl']})")
    except Exception as exc:                       # pragma: no cover - diagnostic only
        print(f"  sensitivity arm: not computed ({exc})")

    out["cache_gap"] = cache_gap
    out["cohorts"] = {g: sorted(int(x) for x in pa[pa.group == g]["a_id"])
                      for g in ("abusive", "verified")}
    out["cohorts"]["pothuman"] = sorted(int(x) for x in pp[pp.group == "pothuman"]["a_id"])
    out["cohort_frozen"] = bool(frozen)
    out["excluded"] = {str(k): v for k, v in sorted(_EXCLUDED.items())}
    out["exclusion_counts"] = {r: sum(1 for v in _EXCLUDED.values() if v.split(":")[0] == r)
                               for r in sorted({v.split(":")[0] for v in _EXCLUDED.values()})}
    if frozen:
        for g, ids in frozen.items():
            emitted = set(out["cohorts"].get(g, []))
            declared = {int(i) for i in ids}
            missing = declared - emitted
            extra = emitted - declared
            if missing or extra:
                raise SystemExit(
                    f"emitted {g} cohort differs from the frozen one: "
                    f"{len(missing)} declared-but-absent, {len(extra)} present-but-undeclared")
        print("  emitted cohorts match the frozen register exactly")
        # The manifest was computed before the analysis ran; publish it rather than
        # recomputing it afterwards, which would not describe the bytes the workers used.
        out["inputs"] = _INPUT_MANIFEST
    if a.out:
        p = Path(a.out)
    json.dump(out, open(p, "w"), indent=2, default=float)
    print(f"\nwrote {p}")
    for g, v in cache_gap.items():
        if v["uncached"]:
            print("  cache gap: " + g + ": " + str(v["uncached"]) + " of "
                  + str(v["roster"]) + " roster agents have no cached series")


if __name__ == "__main__":
    main()
