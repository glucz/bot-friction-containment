"""
Overnight analysis battery on the DB-extracted, calendar-dated enriched series.

Runs a single memory-safe STREAMING pass over the per-agent Parquet cache (so it
scales to ~50k agents without loading everything at once), then synthesizes. It
leverages exactly the signals the local derived files lacked:

  A1  RFC-9309 compliance response: do bots check robots.txt MORE after their
      first real block (401/403)? -- per-agent before/after, DiD bots vs controls.
  A2  Filter-response event study on REAL block events (401/403/429): normalized
      behavioral response (robots / volume / 404 / resistance), DiD vs controls.
  A3  Population co-evolution over CALENDAR time (2019-2023): monthly block rate,
      robots rate, and active-agent counts by role -- the population-dynamics companion's real-time story.
  A4  Non-stationarity of robots_rate / block rate per agent (trend), bot vs ctrl.

Honeypot-confirmed bots are the treatment; controls = labeled humans/chrome +
honeypot-negative + whitelist (per the roster). Resumable: pure function of the
cache + roster, safe to re-run anytime.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import db_source as dbs
import stats_tests as st
from roster import ROSTER_CSV

MIN_DAYS = 20          # agents with fewer active days are skipped for trajectory stats
EVENT_W = 5            # +/- active-day window around a block event
FIRST_BLOCK_W = 30     # active-day window before/after the FIRST block (slow adaptation)
OUT_TAB = config.TAB_DIR
OUT_FIG = config.FIG_DIR


def _block_events(df: pd.DataFrame) -> np.ndarray:
    """Positional indices of real-block-pressure days: cblock spikes above the
    agent's own baseline (z>=1.5), OR any day with >=1 rate-limit (429)."""
    blk = df["cblock"].to_numpy(dtype=float)
    r429 = df["c429"].to_numpy(dtype=float)
    idx = set()
    if len(blk) >= 3 and blk.std() > 0:
        thr = blk.mean() + 1.5 * blk.std()
        idx.update(np.flatnonzero(blk > max(thr, 0)).tolist())
    idx.update(np.flatnonzero(r429 > 0).tolist())
    ev = sorted(idx)
    # enforce spacing so pre/post windows don't overlap
    kept, last = [], -10**9
    for i in ev:
        if i - last >= EVENT_W:
            kept.append(i); last = i
    return np.array(kept, dtype=int)


def _winmean(a: np.ndarray, c: int, side: str, w: int) -> float:
    lo, hi = (max(0, c - w), c) if side == "pre" else (c + 1, min(len(a), c + 1 + w))
    seg = a[lo:hi]
    return float(np.nanmean(seg)) if len(seg) else np.nan


def run() -> dict:
    config.ensure_dirs()
    roster = pd.read_csv(ROSTER_CSV).set_index("a_id")

    per_agent = []
    # event-aligned robots_rate curves (baseline-subtracted), by group
    curves = {"bot": [], "control": []}
    # monthly population accumulators: {(role_group, 'YYYY-MM'): [hits, cblock, robots, c200, n_days, n_agents_marker]}
    monthly = {}

    n_seen = 0
    for a_id, row in roster.iterrows():
        cp = dbs._cache_path(int(a_id))
        if not cp.exists():
            continue
        df = pd.read_parquet(cp)
        if df.empty:
            continue
        n_seen += 1
        is_bot = bool(row["is_bot"])
        grp = "bot" if is_bot else ("control" if bool(row["is_control"]) else "other")

        rec = {"a_id": int(a_id), "role": row["role"], "group": grp, "n_days": len(df),
               "total_hits": int(df["hits"].sum()),
               "mean_block_rate": float(df["p_block"].mean()),
               "mean_robots_rate": float(df["robots_rate"].mean()),
               "ever_robots": int(df["robots"].sum() > 0),
               "n_block_days": int((df["cblock"] > 0).sum()),
               "n_429_days": int((df["c429"] > 0).sum())}

        # --- monthly population co-evolution contribution ---
        mser = pd.to_datetime(df["date"]).dt.to_period("M").astype(str)
        g = df.groupby(mser)
        agg = g[["hits", "cblock", "robots", "c200"]].sum()
        ndays = g.size()
        for m in agg.index:
            key = (grp, m)
            acc = monthly.setdefault(key, [0.0, 0.0, 0.0, 0.0, 0, 0])
            acc[0] += agg.loc[m, "hits"]; acc[1] += agg.loc[m, "cblock"]
            acc[2] += agg.loc[m, "robots"]; acc[3] += agg.loc[m, "c200"]
            acc[4] += int(ndays.loc[m]); acc[5] += 1

        if len(df) >= MIN_DAYS:
            # --- A4 non-stationarity: robots_rate / block_rate trend ---
            rec["robots_trend"] = st.mann_kendall(df["robots_rate"].to_numpy()).direction
            rec["block_trend"] = st.mann_kendall(df["p_block"].to_numpy()).direction

            # --- A2 event study around real block events ---
            events = _block_events(df)
            rb = df["robots_rate"].to_numpy(); lh = np.log1p(df["hits"].to_numpy())
            p4 = df["p404"].to_numpy(); res = df["resistance"].to_numpy()
            d_rb, d_lh, d_p4, d_res = [], [], [], []
            vol_rb = np.std(np.diff(rb)) if len(rb) > 1 else 0.0
            z_rb = []
            for i in events:
                if i < EVENT_W or i + EVENT_W >= len(df):
                    continue
                drb = _winmean(rb, i, "post", EVENT_W) - _winmean(rb, i, "pre", EVENT_W)
                d_rb.append(drb); z_rb.append(abs(drb) / (vol_rb + 1e-9))
                d_lh.append(_winmean(lh, i, "post", EVENT_W) - _winmean(lh, i, "pre", EVENT_W))
                d_p4.append(_winmean(p4, i, "post", EVENT_W) - _winmean(p4, i, "pre", EVENT_W))
                d_res.append(_winmean(res, i, "post", EVENT_W) - _winmean(res, i, "pre", EVENT_W))
                # event-aligned robots curve (baseline-subtracted)
                seg = rb[i - EVENT_W:i + EVENT_W + 1] - rb[i - EVENT_W:i].mean()
                if grp in curves and len(seg) == 2 * EVENT_W + 1:
                    curves[grp].append(seg)
            rec["n_events"] = len(d_rb)
            rec["d_robots"] = float(np.mean(d_rb)) if d_rb else np.nan
            rec["z_robots"] = float(np.mean(z_rb)) if z_rb else np.nan
            rec["d_loghits"] = float(np.mean(d_lh)) if d_lh else np.nan
            rec["d_p404"] = float(np.mean(d_p4)) if d_p4 else np.nan

            # --- A1 first-block compliance response (slow adaptation) ---
            blk_days = np.flatnonzero(df["cblock"].to_numpy() > 0)
            if len(blk_days):
                fb = int(blk_days[0])
                pre = rb[max(0, fb - FIRST_BLOCK_W):fb]
                post = rb[fb + 1:fb + 1 + FIRST_BLOCK_W]
                if len(pre) and len(post):
                    rec["firstblock_d_robots"] = float(np.nanmean(post) - np.nanmean(pre))
        per_agent.append(rec)

    pa = pd.DataFrame(per_agent)
    pa.to_csv(OUT_TAB / "DB_per_agent.csv", index=False)

    results = {"n_agents_analyzed": int(n_seen),
               "group_counts": pa["group"].value_counts().to_dict()}
    results.update(_synthesize(pa))
    _write_monthly(monthly)
    _plot_event_curves(curves)
    _plot_co_evolution()

    (config.OUT_DIR / "DB_summary.json").write_text(json.dumps(results, indent=2, default=str))
    _write_report(results, pa)
    return results


def _did(a: np.ndarray, b: np.ndarray, n=2000) -> dict:
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 5 or len(b) < 5:
        return {"did": None, "ci": [None, None], "sig": False, "bot": None, "ctrl": None}
    rng = np.random.default_rng(42)
    ba = rng.choice(a, (n, len(a)), True).mean(1)
    bb = rng.choice(b, (n, len(b)), True).mean(1)
    d = ba - bb
    lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    return {"did": float(a.mean() - b.mean()), "ci": [round(lo, 4), round(hi, 4)],
            "sig": bool(lo > 0 or hi < 0), "bot": round(float(a.mean()), 4), "ctrl": round(float(b.mean()), 4)}


def _synthesize(pa: pd.DataFrame) -> dict:
    bot = pa[pa.group == "bot"]; ctl = pa[pa.group == "control"]
    out = {}
    # A1 robots.txt compliance
    out["robots_compliance"] = {
        "pct_bots_ever_robots": round(100 * bot["ever_robots"].mean(), 1),
        "pct_controls_ever_robots": round(100 * ctl["ever_robots"].mean(), 1),
        "firstblock_d_robots_DiD": _did(bot.get("firstblock_d_robots", pd.Series(dtype=float)).to_numpy(),
                                        ctl.get("firstblock_d_robots", pd.Series(dtype=float)).to_numpy()),
    }
    # A2 filter-response DiD (normalized robots response + signed volume/404)
    out["filter_response"] = {
        "robots_norm_response_DiD": _did(bot.get("z_robots", pd.Series(dtype=float)).to_numpy(),
                                         ctl.get("z_robots", pd.Series(dtype=float)).to_numpy()),
        "loghits_DiD": _did(bot.get("d_loghits", pd.Series(dtype=float)).to_numpy(),
                            ctl.get("d_loghits", pd.Series(dtype=float)).to_numpy()),
        "p404_DiD": _did(bot.get("d_p404", pd.Series(dtype=float)).to_numpy(),
                         ctl.get("d_p404", pd.Series(dtype=float)).to_numpy()),
    }
    # A4 non-stationarity
    out["nonstationarity"] = {
        "pct_bots_robots_trend": round(100 * (bot.get("robots_trend", pd.Series(dtype=float)) != 0).mean(), 1),
        "pct_controls_robots_trend": round(100 * (ctl.get("robots_trend", pd.Series(dtype=float)) != 0).mean(), 1),
    }
    return out


def _write_monthly(monthly: dict) -> None:
    rows = []
    for (grp, m), v in monthly.items():
        hits, cblock, robots, c200, ndays, nag = v
        rows.append({"group": grp, "month": m, "agent_months": nag, "active_days": ndays,
                     "hits": hits, "block_rate": cblock / max(1, hits),
                     "robots_rate": robots / max(1, hits), "success_rate": c200 / max(1, hits)})
    pd.DataFrame(rows).sort_values(["group", "month"]).to_csv(OUT_TAB / "DB_monthly_coevolution.csv", index=False)


def _plot_event_curves(curves: dict) -> None:
    off = np.arange(-EVENT_W, EVENT_W + 1)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    for grp, color in [("bot", "tab:red"), ("control", "tab:blue")]:
        if not curves.get(grp):
            continue
        M = np.vstack(curves[grp]); m = M.mean(0); se = M.std(0) / np.sqrt(len(M))
        ax.plot(off, m, color=color, marker="o", ms=3, label=f"{grp} (n_events={len(M)})")
        ax.fill_between(off, m - 1.96 * se, m + 1.96 * se, color=color, alpha=0.15)
    ax.axvline(0, color="k", ls="--", lw=1, label="real block (401/403/429)")
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_xlabel("active days relative to block event")
    ax.set_ylabel("robots.txt rate (baseline-subtracted)")
    ax.set_title("RFC-9309 compliance response to a real block: bots vs controls")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(OUT_FIG / "DB_robots_response_to_block.png", dpi=150); plt.close(fig)


def _plot_co_evolution() -> None:
    fp = OUT_TAB / "DB_monthly_coevolution.csv"
    if not fp.exists():
        return
    d = pd.read_csv(fp)
    d = d[d.agent_months >= 20]  # drop noisy thin months
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), dpi=150, sharex=True)
    for grp, color in [("bot", "tab:red"), ("control", "tab:blue")]:
        s = d[d.group == grp].sort_values("month")
        if s.empty:
            continue
        axes[0].plot(s.month, s.block_rate, color=color, label=grp)
        axes[1].plot(s.month, s.robots_rate, color=color, label=grp)
    axes[0].set_ylabel("block rate (401/403)/hit"); axes[0].set_title("Population co-evolution over calendar time")
    axes[1].set_ylabel("robots.txt rate"); axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.3)
    step = max(1, len(d.month.unique()) // 12)
    axes[1].set_xticks(axes[1].get_xticks()[::step]); axes[1].tick_params(axis="x", rotation=90)
    fig.tight_layout(); fig.savefig(OUT_FIG / "DB_co_evolution.png", dpi=150); plt.close(fig)


def _write_report(res: dict, pa: pd.DataFrame) -> None:
    L = ["# DB battery results — calendar-dated AGWA evidence (Papers F & D)\n"]
    L.append(f"Agents analyzed: {res['n_agents_analyzed']}  |  groups: {res['group_counts']}\n")
    rc = res["robots_compliance"]
    L.append("## A1 — RFC-9309 (robots.txt) compliance\n")
    L.append(f"- bots that ever access robots.txt: **{rc['pct_bots_ever_robots']}%** vs controls {rc['pct_controls_ever_robots']}%")
    fb = rc["firstblock_d_robots_DiD"]
    L.append(f"- change in robots.txt rate after the FIRST real block (bot−control DiD): "
             f"{fb['did']} CI {fb['ci']} {'(sig)' if fb['sig'] else '(ns)'}\n")
    fr = res["filter_response"]
    L.append("## A2 — filter-response to REAL block events (401/403/429)\n")
    for k, lab in [("robots_norm_response_DiD", "normalized robots response"),
                   ("loghits_DiD", "log-hits (retreat)"), ("p404_DiD", "404 probing")]:
        v = fr[k]
        L.append(f"- {lab}: bot={v['bot']} ctrl={v['ctrl']} DiD={v['did']} CI {v['ci']} {'(sig)' if v['sig'] else '(ns)'}")
    L.append("")
    ns = res["nonstationarity"]
    L.append("## A4 — non-stationarity\n")
    L.append(f"- bots with a significant robots_rate trend: {ns['pct_bots_robots_trend']}% vs controls {ns['pct_controls_robots_trend']}%\n")
    L.append("## A3 — population co-evolution\n")
    L.append("See `DB_monthly_coevolution.csv` and `figures/DB_co_evolution.png` (monthly block/robots rates 2019-2023).\n")
    (config.OUT_DIR / "DB_SUMMARY.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, default=str))
