"""
db_analysis_identity_rotation.py — Identity rotation / respawn as a third escape
channel (this paper, Finding 6). See PLAN_identity_rotation.md.

Question: when a honeypot-confirmed bot is blocked and its volume collapses
(currently labeled `retreat`), does the *operation* reappear under a different
a_id (= a different user-agent), time-locked to the block and linked by shared
infrastructure? If so, "retreat" overcounts true suppression and rotation is a
second observability-loss channel.

BRIDGE — what survives the UA change. Two bridges are run; both are reported.
(1) HONEYPOT-URL bridge: url2agent_honeypot (194k rows, agent-ids + url-ids, NO
PII) is pulled ONCE and all linkage is computed in memory. A fresh UA hitting the
SAME decoy paths is a strong handoff signal.
(2) IP bridge: initially blocked because the `ip` table (52M rows) could not be
looked up by i_agent (non-leftmost in its only index -> TokuDB error 1152).
Adding idx_ip_agent_time made it feasible, and it now runs as a server-side
self-join on non-proxy IPs returning a_ids + shared-IP COUNTS only — no address
ever leaves the server (see bridge_candidates_ip).

SCOPE (state plainly in this paper §7.8): successors found via the honeypot bridge
are themselves honeypot-confirmed bots, so that bridge measures rotation WITHIN the
confirmed-bot fleet (both endpoints are ground-truth bots -> no label-contamination
ambiguity) and its rate is a LOWER BOUND. The IP bridge reaches ANY successor,
including fresh / non-honeypot identities, and so closes exactly that gap; operations
that rotate UA and infrastructure together still escape, leaving a residual lower
bound.

RARITY: popular honeypots are hit by thousands of unrelated bots, so shared URLs
are weighted by inverse popularity. Precision core = shares a RARE decoy;
recall envelope = shares any decoy.

CLUSTER UNIT: death-cluster (a death A + its bridged successors = one bootstrap
resample unit) — privacy-safe and removes the bridge/cluster circularity.

Stages (resumable): deaths -> bridge graph -> pairs (event-study around t_A) ->
inference (conservation + nulls + per-pair placebo).
"""
from __future__ import annotations

import datetime as _dt
import json
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import db
import db_source as dbs
from roster import ROSTER_CSV

# ---- knobs ----
WC = 14               # +/- CALENDAR-day window around the death date t_A (= 2*tau_A)
POP_CAP = 1000        # ignore honeypot URLs hit by more agents than this for candidate generation
CORE_POP = 25         # "rare" decoy: a shared URL hit by <= this many agents = precision core
TOP_K = 40            # max candidate successors examined per death (by rarity score)
PLACEBO_OFFSET = 60   # days before t_A for the per-pair placebo event
MIN_DAYS = 15
VOL_FLOOR = 1.0
RANDOM_STATE = 42
N_BOOT = 2000

# ---- IP bridge (requires idx_ip_agent_time on the ip table) ----
TOPK_IP = 12               # candidate successors per death (by shared-IP count)
IP_BRIDGE_TIMEOUT_MS = 30000   # per-query cap; pathological popular-IP joins skip
IP_OVERLAP_CORE = 0.5      # precision core: successor shares >=50% of A's IP footprint
IP_SHARED_MIN = 3          # ...and >=3 shared IPs (kills single-popular-IP coincidences)

TAB = config.TAB_DIR
FIG = config.FIG_DIR
DEATHS_CSV = TAB / "DB_rotation_deaths.csv"
PAIRS_CSV = TAB / "DB_rotation_pairs.csv"
IP_PAIRS_CSV = TAB / "DB_rotation_ip_pairs.csv"


# ===================== botness axis =====================

def _behavior_matrix(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack([
        df["p404"].to_numpy(float),
        df["robots_rate"].to_numpy(float),
        np.log1p(df["hits"].to_numpy(float)),
    ])


AXIS_NPZ = TAB / "_botness_axis.npz"


def build_botness_axis(roster: pd.DataFrame, min_days: int = 20, cache: bool = False) -> dict:
    if cache and AXIS_NPZ.exists():
        d = np.load(AXIS_NPZ)
        return {"mu_H": d["mu_H"], "sigma_H": d["sigma_H"], "axis": d["axis"],
                "bot_centroid_b": float(d["bot_centroid_b"])}
    ctl, bot = [], []
    for _, r in roster.iterrows():
        cp = dbs._cache_path(int(r["a_id"]))
        if not cp.exists():
            continue
        df = pd.read_parquet(cp)
        if len(df) < min_days:
            continue
        v = _behavior_matrix(df).mean(axis=0)
        if bool(r["is_bot"]):
            bot.append(v)
        elif bool(r["is_control"]):
            ctl.append(v)
    ctl, bot = np.array(ctl), np.array(bot)
    mu_H, sigma_H, mu_B = ctl.mean(0), ctl.std(0) + 1e-9, bot.mean(0)
    muB_z = (mu_B - mu_H) / sigma_H
    axis = muB_z / (np.linalg.norm(muB_z) + 1e-9)
    out = {"mu_H": mu_H, "sigma_H": sigma_H, "axis": axis,
           "bot_centroid_b": float(np.linalg.norm(muB_z))}
    if cache:
        config.ensure_dirs()
        np.savez(AXIS_NPZ, **out)
    return out


def _b_series(df: pd.DataFrame, ax: dict) -> np.ndarray:
    return ((_behavior_matrix(df) - ax["mu_H"]) / ax["sigma_H"]) @ ax["axis"]


# ===================== honeypot bridge (in-memory, no PII) =====================

def load_honeypot_graph() -> dict:
    """Pull url2agent_honeypot once; build agent->urls, url->popularity. a_ids and
    url-ids only (no PII)."""
    print("[bridge] pulling url2agent_honeypot (once)...")
    t = db.read_sql("SELECT uh_a_id AS a, uh_u_id AS u FROM url2agent_honeypot WHERE uh_total > 0")
    t = t[t["a"] <= dbs.MAX_A_ID]
    agent_urls: dict[int, set] = {}
    for a, u in zip(t["a"].astype(int), t["u"].astype(int)):
        agent_urls.setdefault(a, set()).add(u)
    url_pop = t.groupby("u")["a"].nunique().to_dict()
    url_agents: dict[int, list] = {}
    for a, u in zip(t["a"].astype(int), t["u"].astype(int)):
        if url_pop[u] <= POP_CAP:                        # only non-mega-popular decoys link agents
            url_agents.setdefault(u, []).append(a)
    print(f"  {len(agent_urls)} honeypot agents, {len(url_pop)} decoy URLs "
          f"(median pop {int(np.median(list(url_pop.values())))})")
    return {"agent_urls": agent_urls, "url_pop": url_pop, "url_agents": url_agents}


def bridge_candidates(a_id: int, g: dict, k: int = TOP_K) -> pd.DataFrame:
    urls = g["agent_urls"].get(a_id, set())
    if not urls:
        return pd.DataFrame(columns=["a_id", "shared_hp", "rare_score", "min_pop", "jaccard"])
    cand: dict[int, list] = {}
    for u in urls:
        if g["url_pop"].get(u, 10**9) > POP_CAP:
            continue
        for b in g["url_agents"].get(u, []):
            if b != a_id:
                cand.setdefault(b, []).append(u)
    cols = ["a_id", "shared_hp", "rare_score", "min_pop", "jaccard"]
    if not cand:
        return pd.DataFrame(columns=cols)
    rows = []
    for b, shared in cand.items():
        pops = [g["url_pop"][u] for u in shared]
        union = len(urls | g["agent_urls"].get(b, set()))
        rows.append({"a_id": b, "shared_hp": len(shared),
                     "rare_score": float(sum(1.0 / p for p in pops)),
                     "min_pop": int(min(pops)),
                     "jaccard": len(shared) / union if union else 0.0})
    return pd.DataFrame(rows).sort_values("rare_score", ascending=False).head(k)


# ===================== IP bridge (server-side self-join; a_id + counts only, no PII) =====================

_IP_NA = "SELECT COUNT(DISTINCT i_name) c FROM ip WHERE i_agent = %s AND i_proxy = 0"
_IP_BRIDGE = """
SELECT b.i_agent AS a_id, COUNT(DISTINCT b.i_name) AS shared_ips
FROM ip a JOIN ip b ON a.i_name = b.i_name
WHERE a.i_agent = %s AND b.i_agent <> a.i_agent AND a.i_proxy = 0
GROUP BY b.i_agent ORDER BY shared_ips DESC LIMIT %s
"""


def bridge_candidates_ip(a_id: int, conn, k: int = TOPK_IP) -> pd.DataFrame:
    """Successors sharing non-proxy IPs with the dead bot. Reaches ANY agent (incl.
    fresh / non-honeypot identities) -> closes the channel the honeypot bridge missed.
    Strength = overlap fraction (shared / A's IP footprint). Never returns IPs."""
    cols = ["a_id", "shared_ips", "overlap_frac", "in_core"]
    try:
        nA = int(db.read_sql(_IP_NA, (a_id,), conn=conn).c.iloc[0])
        if nA == 0:
            return pd.DataFrame(columns=cols)
        df = db.read_sql(_IP_BRIDGE, (a_id, k * 4), conn=conn)
    except Exception as e:
        print(f"  [ip-bridge] skip {a_id}: {str(e)[:70]}")
        return pd.DataFrame(columns=cols)
    if df.empty:
        return pd.DataFrame(columns=cols)
    df["overlap_frac"] = df["shared_ips"] / nA
    df["in_core"] = ((df["overlap_frac"] >= IP_OVERLAP_CORE) & (df["shared_ips"] >= IP_SHARED_MIN)).astype(int)
    df["rank"] = df["in_core"] * 10**6 + df["shared_ips"]
    return df.sort_values("rank", ascending=False).head(k).drop(columns="rank")


# ===================== pairs: event-study around the shared death date t_A =====================

def _series(a_id: int, conn) -> pd.DataFrame:
    cp = dbs._cache_path(int(a_id))
    if cp.exists():
        return pd.read_parquet(cp)
    try:
        return dbs.extract_series(int(a_id), conn=conn, use_cache=True)
    except Exception:
        return pd.DataFrame()


def _cal_stats(df: pd.DataFrame, t_center, ax: dict, days: int = WC):
    """Calendar-aligned windows around t_center (missing days = 0 volume, so a
    silent-then-active successor registers a real ramp). Returns
    (log-vol pre, log-vol post, botness pre, botness post, n_pre_days, n_post_days)."""
    dates = pd.to_datetime(df["date"]).dt.date.to_numpy()
    hits = df["hits"].to_numpy(float)
    b = _b_series(df, ax)
    lo, hi = t_center - _dt.timedelta(days=days), t_center + _dt.timedelta(days=days)
    pre = (dates >= lo) & (dates < t_center)
    post = (dates >= t_center) & (dates < hi)
    return (float(np.log1p(hits[pre].sum())), float(np.log1p(hits[post].sum())),
            float(np.nanmean(b[pre])) if pre.any() else np.nan,
            float(np.nanmean(b[post])) if post.any() else np.nan,
            int(pre.sum()), int(post.sum()))


def stage_pairs(deaths: pd.DataFrame, ax: dict, g: dict, out_csv=PAIRS_CSV) -> pd.DataFrame:
    partial = out_csv.with_suffix(".partial.csv")
    progress = out_csv.with_suffix(".progress.txt")
    rows, start = [], 0
    if partial.exists() and progress.exists():
        rows = pd.read_csv(partial).to_dict("records")
        start = int(progress.read_text().strip())
        print(f"[pairs] resuming from death {start}/{len(deaths)} ({len(rows)} pairs so far)")
    conn = db.get_connection()
    for n, (_, drow) in enumerate(deaths.iloc[start:].iterrows(), start + 1):
        a_id = int(drow["a_id"])
        t_A = pd.to_datetime(drow["t_death"]).date()
        dfa = pd.read_parquet(dbs._cache_path(a_id))
        vA_pre, vA_post, bA_pre, bA_post, _, _ = _cal_stats(dfa, t_A, ax)
        for _, c in bridge_candidates(a_id, g).iterrows():
            b_id = int(c["a_id"])
            dfb = _series(b_id, conn)
            if len(dfb) < MIN_DAYS:
                continue
            vB_pre, vB_post, bB_pre, bB_post, _, npost = _cal_stats(dfb, t_A, ax)
            if npost == 0:                                # B not observed after t_A at all
                continue
            # per-pair placebo: B's change around a date PLACEBO_OFFSET days earlier
            t_P = t_A - _dt.timedelta(days=PLACEBO_OFFSET)
            vBp_pre, vBp_post, _, _, _, _ = _cal_stats(dfb, t_P, ax)
            rows.append({
                "a_death": a_id, "a_succ": b_id, "t_A": t_A,
                "shared_hp": int(c["shared_hp"]), "rare_score": float(c["rare_score"]),
                "min_pop": int(c["min_pop"]), "jaccard": float(c["jaccard"]),
                "in_core": int(c["min_pop"] <= CORE_POP),
                "bA_pre": bA_pre, "dA_b": bA_post - bA_pre, "dA_v": vA_post - vA_pre,
                "bB_post": bB_post, "dB_b": bB_post - bB_pre, "dB_v": vB_post - vB_pre,
                "dB_v_placebo": (vBp_post - vBp_pre),
            })
        if n % 50 == 0:
            print(f"  [pairs] {n}/{len(deaths)} deaths, {len(rows)} pairs")
            pd.DataFrame(rows).to_csv(partial, index=False)
            progress.write_text(str(n))
    conn.close()
    pairs = pd.DataFrame(rows).drop_duplicates(subset=["a_death", "a_succ"], keep="first")
    pairs.to_csv(out_csv, index=False)
    partial.unlink(missing_ok=True)
    progress.unlink(missing_ok=True)
    print(f"[pairs] {len(pairs)} candidate pairs -> {out_csv.name}")
    return pairs


def stage_pairs_ip(deaths: pd.DataFrame, ax: dict, out_csv=IP_PAIRS_CSV) -> pd.DataFrame:
    """Same event-study as stage_pairs but using the IP bridge. Resumable. Successors
    may be ANY agent (extracted on demand), so this tests handoffs into fresh /
    non-honeypot identities that the honeypot bridge cannot reach."""
    partial = out_csv.with_suffix(".partial.csv")
    progress = out_csv.with_suffix(".progress.txt")
    rows, start = [], 0
    if partial.exists() and progress.exists():
        rows = pd.read_csv(partial).to_dict("records")
        start = int(progress.read_text().strip())
        print(f"[ip-pairs] resuming from death {start}/{len(deaths)} ({len(rows)} pairs so far)")
    bconn = db.get_connection(timeout_ms=IP_BRIDGE_TIMEOUT_MS)
    sconn = db.get_connection()
    for n, (_, drow) in enumerate(deaths.iloc[start:].iterrows(), start + 1):
        a_id = int(drow["a_id"])
        t_A = pd.to_datetime(drow["t_death"]).date()
        dfa = pd.read_parquet(dbs._cache_path(a_id))
        vA_pre, vA_post, bA_pre, bA_post, _, _ = _cal_stats(dfa, t_A, ax)
        for _, c in bridge_candidates_ip(a_id, bconn).iterrows():
            b_id = int(c["a_id"])
            dfb = _series(b_id, sconn)
            if len(dfb) < MIN_DAYS:
                continue
            vB_pre, vB_post, bB_pre, bB_post, _, npost = _cal_stats(dfb, t_A, ax)
            if npost == 0:
                continue
            t_P = t_A - _dt.timedelta(days=PLACEBO_OFFSET)
            vBp_pre, vBp_post, _, _, _, _ = _cal_stats(dfb, t_P, ax)
            rows.append({
                "a_death": a_id, "a_succ": b_id, "t_A": t_A,
                "shared_ips": int(c["shared_ips"]), "overlap_frac": float(c["overlap_frac"]),
                "in_core": int(c["in_core"]),
                "bA_pre": bA_pre, "dA_b": bA_post - bA_pre, "dA_v": vA_post - vA_pre,
                "bB_post": bB_post, "dB_b": bB_post - bB_pre, "dB_v": vB_post - vB_pre,
                "dB_v_placebo": (vBp_post - vBp_pre),
            })
        if n % 25 == 0:
            print(f"  [ip-pairs] {n}/{len(deaths)} deaths, {len(rows)} pairs")
            pd.DataFrame(rows).to_csv(partial, index=False)
            progress.write_text(str(n))
    bconn.close(); sconn.close()
    pairs = pd.DataFrame(rows).drop_duplicates(subset=["a_death", "a_succ"], keep="first")
    pairs.to_csv(out_csv, index=False)
    partial.unlink(missing_ok=True)
    progress.unlink(missing_ok=True)
    print(f"[ip-pairs] {len(pairs)} candidate pairs -> {out_csv.name}")
    return pairs


# ===================== inference =====================

def _boot(x: np.ndarray, stat, n: int = N_BOOT, seed: int = RANDOM_STATE):
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return (np.nan, np.nan, np.nan, len(x))
    rng = np.random.default_rng(seed)
    bs = [stat(x[rng.integers(0, len(x), len(x))]) for _ in range(n)]
    return (float(stat(x)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), len(x))


def _transfer(sub: pd.DataFrame, label: str) -> dict:
    """Does the successor's volume RAMP (dB_v>0) and its botness rise, as A dies?
    Headline = mean successor volume change around t_A; CI by death-cluster bootstrap."""
    if sub.empty:
        return {"label": label, "n": 0}
    # death-cluster bootstrap: resample deaths, keep their candidate rows together
    deaths = sub["a_death"].unique()
    rng = np.random.default_rng(RANDOM_STATE)
    groups = {d: sub[sub.a_death == d] for d in deaths}

    def cluster_mean(col):
        vals = []
        for _ in range(N_BOOT):
            chosen = rng.choice(deaths, len(deaths), replace=True)
            vals.append(np.nanmean(np.concatenate([groups[d][col].to_numpy() for d in chosen])))
        point = float(np.nanmean(sub[col]))
        return point, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

    dBv, dBv_lo, dBv_hi = cluster_mean("dB_v")
    dBb, dBb_lo, dBb_hi = cluster_mean("dB_b")
    plac = float(np.nanmean(sub["dB_v_placebo"]))
    return {"label": label, "n_pairs": int(len(sub)), "n_deaths": int(len(deaths)),
            "succ_dvol_mean": dBv, "succ_dvol_ci": [dBv_lo, dBv_hi],
            "succ_dbot_mean": dBb, "succ_dbot_ci": [dBb_lo, dBb_hi],
            "succ_dvol_placebo": plac,
            "ramp_minus_placebo": dBv - plac}


def _link_permutation_null(pairs: pd.DataFrame, deaths: pd.DataFrame, ax: dict, seed: int = RANDOM_STATE) -> dict:
    """Valid falsification: recompute each successor's volume change around a RANDOM
    OTHER death's date (break the specific A<->B temporal link, keep B fixed). If the
    real ramp is link-specific it exceeds this; if equal, the ramp is generic growth."""
    if pairs.empty:
        return {"label": "link_permutation_null", "succ_dvol_mean": None, "n_eval": 0}
    rng = np.random.default_rng(seed)
    pool = pd.to_datetime(deaths["t_death"]).dt.date.to_numpy()
    cache: dict[int, pd.DataFrame | None] = {}

    def get(b):
        if b not in cache:
            cp = dbs._cache_path(int(b))
            cache[b] = pd.read_parquet(cp) if cp.exists() else None
        return cache[b]

    vals = []
    for _, r in pairs.iterrows():
        dfb = get(int(r["a_succ"]))
        if dfb is None:
            vals.append(np.nan); continue
        vpre, vpost, _, _, _, npost = _cal_stats(dfb, rng.choice(pool), ax)
        vals.append((vpost - vpre) if npost > 0 else np.nan)
    vals = np.array(vals, float)
    return {"label": "link_permutation_null", "succ_dvol_mean": float(np.nanmean(vals)),
            "n_eval": int(np.isfinite(vals).sum())}


def stage_inference(pairs: pd.DataFrame, deaths: pd.DataFrame, ax: dict, tag: str = "") -> dict:
    core = pairs[pairs["in_core"] == 1]
    env = pairs
    res = {
        "bridge": "ip" if tag == "_ip" else "honeypot",
        "n_deaths": int(len(deaths)),
        "n_deaths_with_candidate": int(pairs["a_death"].nunique()),
        "n_candidate_pairs": int(len(pairs)),
        "transfer_core": _transfer(core, "precision_core"),
        "transfer_envelope": _transfer(env, "recall_envelope"),
        "null_link_perm_core": _link_permutation_null(core, deaths, ax),
        "null_link_perm_envelope": _link_permutation_null(env, deaths, ax),
        "transfer_rate_core": round(core["a_death"].nunique() / max(1, len(deaths)), 4),
        "transfer_rate_envelope": round(env["a_death"].nunique() / max(1, len(deaths)), 4),
    }
    _plot_conservation(env, core, tag)
    (config.OUT_DIR / f"DB_rotation_summary{tag}.json").write_text(json.dumps(res, indent=2, default=str))
    _write_summary(res, tag)
    return res


def _plot_conservation(env: pd.DataFrame, core: pd.DataFrame, tag: str = "") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    ax = axes[0]
    if len(env):
        ax.scatter(-env["dA_v"], env["dB_v"], s=10, alpha=0.25, color="gray", label="recall envelope")
    if len(core):
        ax.scatter(-core["dA_v"], core["dB_v"], s=18, alpha=0.7, color="tab:red", label="precision core")
    ax.axhline(0, color="k", lw=0.6); ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("dying agent's volume LOSS  (-dA_v)")
    ax.set_ylabel("successor's volume change  (dB_v)")
    ax.set_title("Volume transfer around the block"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax2 = axes[1]
    if len(env):
        ax2.scatter(env["bA_pre"], env["bB_post"], s=10, alpha=0.25, color="gray")
    if len(core):
        ax2.scatter(core["bA_pre"], core["bB_post"], s=18, alpha=0.7, color="tab:red")
    ax2.set_xlabel("b_A_pre (dying agent botness)"); ax2.set_ylabel("b_B_post (successor botness)")
    ax2.set_title("Botness level at handoff"); ax2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / f"DB_rotation_conservation{tag}.png", dpi=150); plt.close(fig)


def _write_summary(res: dict, tag: str = "") -> None:
    c, e = res["transfer_core"], res["transfer_envelope"]
    nc, ne = res["null_link_perm_core"], res["null_link_perm_envelope"]
    bridge = res.get("bridge", "honeypot")
    core_desc = "rare shared IP (>=50% overlap, >=3 IPs)" if bridge == "ip" else "rare shared decoy"

    def fmt(t):
        if not t.get("n_pairs"):
            return "no pairs"
        return (f"successor dvol {t['succ_dvol_mean']:+.3f} CI {[round(x,3) for x in t['succ_dvol_ci']]}, "
                f"dbot {t['succ_dbot_mean']:+.3f} CI {[round(x,3) for x in t['succ_dbot_ci']]}, "
                f"placebo dvol {t['succ_dvol_placebo']:+.3f} (n={t['n_pairs']} pairs / {t['n_deaths']} deaths)")

    def fmtnull(t):
        return ("n/a" if t.get("succ_dvol_mean") is None
                else f"successor dvol around a RANDOM other death = {t['succ_dvol_mean']:+.3f} (n={t['n_eval']})")
    if bridge == "ip":
        bridge_line = ("Bridge = shared non-proxy IPs (overlap-weighted); successors may be ANY agent "
                       "incl. fresh / non-honeypot identities -> this CLOSES the channel the honeypot "
                       "bridge missed. UA+IP co-rotation still escapes -> residual lower bound.")
    else:
        bridge_line = ("Bridge = shared honeypot decoy URLs (rarity-weighted); successors are themselves "
                       "confirmed bots (intra-fleet rotation) -> LOWER BOUND. The IP-bridge variant "
                       "(DB_rotation_SUMMARY_ip.md) reaches non-honeypot successors and closes that gap.")
    L = [f"# Identity rotation / respawn — Finding 6 ({bridge} bridge)\n",
         f"Deaths (persistent post-block collapses among honeypot bots): **{res['n_deaths']}**; "
         f"with >=1 bridged successor: {res['n_deaths_with_candidate']}; candidate pairs: {res['n_candidate_pairs']}.\n",
         bridge_line + "\n",
         "## Transfer rate (deaths with a bridged successor)\n",
         f"- precision core ({core_desc}): **{res['transfer_rate_core']}**",
         f"- recall envelope (any shared): {res['transfer_rate_envelope']}\n",
         "## Does the successor ramp up as the bot dies? (event-study around t_A)\n",
         f"- precision core: {fmt(c)}",
         f"- recall envelope: {fmt(e)}\n",
         "## Falsification\n",
         f"- link-permutation null (core): {fmtnull(nc)}",
         f"- link-permutation null (envelope): {fmtnull(ne)}",
         "  (real ramp must EXCEED the successor's ramp around a random unrelated death; "
         "if equal, the ramp is generic successor growth, not a handoff)",
         "- per-pair placebo: `placebo dvol` above is the successor's volume change around a date "
         f"{PLACEBO_OFFSET} d before the block "
         f"(core ramp-minus-placebo = {c.get('ramp_minus_placebo', float('nan')):+.3f}, "
         f"envelope = {e.get('ramp_minus_placebo', float('nan')):+.3f}).\n",
         "_Planted-handoff positive control: run with `--selftest` (separate pass)._\n"]
    (config.OUT_DIR / f"DB_rotation_SUMMARY{tag}.md").write_text("\n".join(L), encoding="utf-8")


# ===================== positive control (planted handoff) =====================

def selftest_planted(ax: dict, n: int = 300, seed: int = RANDOM_STATE) -> dict:
    """Split real confirmed-bot series at their midpoint: first half = a dying A,
    second half = a born successor B (same operator => a genuine handoff). The
    statistic must show dA_v<0 (collapse), dB_v>0 (ramp), bB_post~bA_pre
    (botness conserved), and a NULL temporal placebo. Validates sensitivity."""
    roster = pd.read_csv(ROSTER_CSV)
    bots = roster[roster["is_bot"] == True]["a_id"].astype(int).to_numpy()   # noqa: E712
    rng = np.random.default_rng(seed)
    rows = []
    for a_id in rng.permutation(bots):
        if len(rows) >= n:
            break
        cp = dbs._cache_path(int(a_id))
        if not cp.exists():
            continue
        df = pd.read_parquet(cp)
        dates = pd.to_datetime(df["date"]).dt.date
        span = (dates.max() - dates.min()).days
        if span < 4 * WC:
            continue
        t_split = dates.min() + _dt.timedelta(days=span // 2)
        A, B = df[dates < t_split], df[dates >= t_split]
        if len(A) < MIN_DAYS or len(B) < MIN_DAYS:
            continue
        vA_pre, vA_post, bA_pre, bA_post, _, _ = _cal_stats(A, t_split, ax)
        vB_pre, vB_post, bB_pre, bB_post, _, npost = _cal_stats(B, t_split, ax)
        if npost == 0:
            continue
        t_P = t_split - _dt.timedelta(days=PLACEBO_OFFSET)
        vBp_pre, vBp_post, _, _, _, _ = _cal_stats(B, t_P, ax)
        rows.append({"dA_v": vA_post - vA_pre, "dB_v": vB_post - vB_pre,
                     "bA_pre": bA_pre, "bB_post": bB_post,
                     "dB_v_placebo": vBp_post - vBp_pre})
    p = pd.DataFrame(rows)
    m = p[["bA_pre", "bB_post"]].dropna()
    corr = float(np.corrcoef(m["bA_pre"], m["bB_post"])[0, 1]) if len(m) > 2 else float("nan")
    res = {"n_planted": int(len(p)),
           "dA_v_mean_want_neg": float(p["dA_v"].mean()),
           "dB_v_mean_want_pos": float(p["dB_v"].mean()),
           "dB_v_placebo_mean_want_0": float(p["dB_v_placebo"].mean()),
           "b_transfer_bias_want_0": float((p["bB_post"] - p["bA_pre"]).mean()),
           "pearson_bApre_bBpost_want_1": corr,
           "frac_ramp_exceeds_placebo": float((p["dB_v"] > p["dB_v_placebo"]).mean())}
    (config.OUT_DIR / "DB_rotation_selftest.json").write_text(json.dumps(res, indent=2, default=str))
    print("[selftest] planted-handoff positive control:")
    print(json.dumps(res, indent=2, default=str))
    return res


# ===================== placebo-death pass (organic, non-block collapses) =====================

def build_placebo_deaths(seed: int = RANDOM_STATE, n_match: int | None = None) -> pd.DataFrame:
    """Deaths defined by a persistent volume collapse NOT coincident with a block
    spike (organic churn). If respawn is friction-driven, the bridged-successor
    ramp here should be WEAKER than for the real (block-driven) deaths."""
    from db_battery import _block_events
    roster = pd.read_csv(ROSTER_CSV)
    bots = roster[roster["is_bot"] == True]["a_id"].astype(int)   # noqa: E712
    rows = []
    for a_id in bots:
        cp = dbs._cache_path(int(a_id))
        if not cp.exists():
            continue
        df = pd.read_parquet(cp)
        if len(df) < 25:
            continue
        hits = df["hits"].to_numpy(float)
        cblock = df["cblock"].to_numpy(float)
        blk = set(_block_events(df).tolist())
        for i in range(5, len(df) - 1):
            if i in blk or cblock[max(0, i - 3):i + 4].sum() > 0:    # require NO block nearby
                continue
            pre = hits[i - 5:i].mean()
            if pre < 5:
                continue
            tail = hits[i + 1:].mean()
            if tail < 0.5 * pre and hits[i + 1:i + 21].max() < pre:
                rows.append({"a_id": int(a_id),
                             "t_death": pd.to_datetime(df["date"].iloc[i]).date(),
                             "pre_vol": float(pre), "tail_vol": float(tail)})
                break
    placebo = pd.DataFrame(rows)
    if n_match and len(placebo) > n_match:
        placebo = placebo.sample(n_match, random_state=seed)
    placebo.to_csv(TAB / "DB_rotation_placebo_deaths.csv", index=False)
    print(f"[placebo] {len(placebo)} organic (non-block) deaths")
    return placebo


def run_placebo() -> dict:
    config.ensure_dirs()
    roster = pd.read_csv(ROSTER_CSV)
    ax = build_botness_axis(roster, cache=True)
    real = pd.read_csv(DEATHS_CSV)
    placebo = build_placebo_deaths(n_match=len(real))
    if placebo.empty:
        print("[placebo] none found"); return {}
    placebo["t_death"] = pd.to_datetime(placebo["t_death"])
    g = load_honeypot_graph()
    pairs = stage_pairs(placebo, ax, g, out_csv=TAB / "DB_rotation_placebo_pairs.csv")
    t = _transfer(pairs[pairs["in_core"] == 1], "placebo_core")
    t["transfer_rate_core"] = round(pairs[pairs.in_core == 1]["a_death"].nunique() / max(1, len(placebo)), 4)
    (config.OUT_DIR / "DB_rotation_placebo.json").write_text(json.dumps(t, indent=2, default=str))
    print("[placebo] core transfer:", json.dumps(t, indent=2, default=str))
    return t


# ===================== main =====================

def run() -> dict:
    config.ensure_dirs()
    roster = pd.read_csv(ROSTER_CSV)
    print("[axis] building botness axis...")
    ax = build_botness_axis(roster, cache=True)
    print(f"  bot centroid b = {ax['bot_centroid_b']:.3f}")
    deaths = pd.read_csv(DEATHS_CSV, parse_dates=["t_death"])
    print(f"[deaths] {len(deaths)} loaded")
    if len(deaths) < 20:
        print(f"[GATE] only {len(deaths)} deaths — stopping.")
        return {"n_deaths": int(len(deaths)), "gated": True}
    partial = PAIRS_CSV.with_suffix(".partial.csv")
    if PAIRS_CSV.exists() and not partial.exists():
        pairs = pd.read_csv(PAIRS_CSV, parse_dates=["t_A"])
        print(f"[pairs] reused complete {len(pairs)} from {PAIRS_CSV.name}")
    else:
        g = load_honeypot_graph()                 # only needed when (re)computing pairs
        pairs = stage_pairs(deaths, ax, g)
    if pairs.empty:
        print("[done] no candidate pairs — rotation undetectable via honeypot bridge.")
        return {"n_deaths": int(len(deaths)), "n_pairs": 0}
    res = stage_inference(pairs, deaths, ax)
    print(json.dumps(res, indent=2, default=str))
    return res


def run_ipbridge() -> dict:
    """IP-bridge variant: tests handoffs into ANY agent (incl. fresh/non-honeypot
    identities). Requires idx_ip_agent_time on the ip table. Writes _ip outputs."""
    config.ensure_dirs()
    roster = pd.read_csv(ROSTER_CSV)
    print("[axis] building botness axis...")
    ax = build_botness_axis(roster, cache=True)
    print(f"  bot centroid b = {ax['bot_centroid_b']:.3f}")
    deaths = pd.read_csv(DEATHS_CSV, parse_dates=["t_death"])
    print(f"[deaths] {len(deaths)} loaded")
    partial = IP_PAIRS_CSV.with_suffix(".partial.csv")
    if IP_PAIRS_CSV.exists() and not partial.exists():
        pairs = pd.read_csv(IP_PAIRS_CSV, parse_dates=["t_A"])
        print(f"[ip-pairs] reused complete {len(pairs)} from {IP_PAIRS_CSV.name}")
    else:
        pairs = stage_pairs_ip(deaths, ax)
    if pairs.empty:
        print("[done] no IP-bridge candidate pairs.")
        return {"n_deaths": int(len(deaths)), "n_pairs": 0}
    res = stage_inference(pairs, deaths, ax, tag="_ip")
    print(json.dumps(res, indent=2, default=str))
    return res


if __name__ == "__main__":
    import sys
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        arg = sys.argv[1] if len(sys.argv) > 1 else ""
        if arg == "--selftest":
            ax = build_botness_axis(pd.read_csv(ROSTER_CSV), cache=True)
            selftest_planted(ax)
        elif arg == "--placebo":
            run_placebo()
        elif arg == "--ipbridge":
            run_ipbridge()
        else:
            run()
