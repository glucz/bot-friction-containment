"""Build 14-feature classifier-input TSVs from the AGWA DB.

The original generator is not in the research tree; the column conventions were
recovered by validating against known files and are documented in
TSV_RECONSTRUCTION.md. This script applies them to agents that have no TSV -- in
particular the ever-blocked subset of Paper D's 7,812-agent treatment arm, which
the transformer classifier has never scored and which was never in its training
set (training globs only *.bot and *.human).

Emits, per agent:
  arm_tsv/<a_id>.<class>       15-column TSV **with a header row**, so the
                               published parse_user_agent_file()'s skiprows=1
                               skips the header instead of silently dropping day 0
  arm_tsv/parquet/<a_id>.parquet  same series PLUS the 401/403/429 block channel,
                               which the classifier TSV format cannot carry

Grid: every calendar day from the agent's first to last hit, minus the 234
confirmed panel-zero days (_panel_zero_days_full.pkl). Applied identically to
treatment and control agents. No trailing 30-day-window truncation, so series run
~29 days longer than the published files; do not pool generated b with published b.

Queries per agent: 1 aggregate + 4 separate DISTINCT channels + 1 extension
histogram. Separate DISTINCT queries (rather than one combined) keep each inside
the server's statement budget. On error 1030/1152/1969 the agent is retried
year-by-year, then recorded as failed.

Usage:
  python db_build_classifier_tsv.py --set arm            # 1,787 ever-blocked arm
  python db_build_classifier_tsv.py --set control        # pothuman minus verified
  python db_build_classifier_tsv.py --set verified       # the 200 hand-verified
  python db_build_classifier_tsv.py --validate 172877 human
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys
import threading
import time
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import config
import db

ROBOTS_U_ID = 64
OUT = config.HERE / "arm_tsv"
PQ = OUT / "parquet"
MASK = config.HERE / "_panel_zero_days_full.pkl"
HTML_EXT = ("html", "htm")
COLS = ["day", "hits", "p404", "p200", "robots", "img", "html", "js", "php",
        "folder", "ip_ent", "dom_ent", "ctry_ent", "url_ent", "husource"]
RETRYABLE = (1030, 1152, 1969)          # oversized query -> chunk by year
TRANSIENT = (2003, 2006, 2013, 1040, 1203)  # connection/load -> plain retry
_lock = threading.Lock()
_zero: set = set()


def _chunks(lo, hi):
    """Year boundaries covering [lo, hi], for retrying oversized agents."""
    out, a = [], pd.Timestamp(lo)
    hi = pd.Timestamp(hi)
    while a <= hi:
        b = min(pd.Timestamp(year=a.year, month=12, day=31), hi)
        out.append((a.date(), b.date()))
        a = b + pd.Timedelta(days=1)
    return out


def _q(sql, params, span=None):
    """read_sql, retried on transient connection errors, chunked per-year on
    server-limit errors."""
    last = None
    for attempt in range(5):
        try:
            return db.read_sql(sql, params)
        except Exception as e:
            code = e.args[0] if e.args else None
            last = e
            if code in TRANSIENT:
                time.sleep(2 * (attempt + 1))
                continue
            break
    e = last
    if True:
        code = e.args[0] if e.args else None
        if code not in RETRYABLE or span is None:
            raise
        frames = []
        for lo, hi in _chunks(*span):
            sub = sql.replace("WHERE h_a_id=%s", "WHERE h_a_id=%s AND h_ts>=%s AND h_ts<%s") \
                     .replace("WHERE h.h_a_id=%s", "WHERE h.h_a_id=%s AND h.h_ts>=%s AND h.h_ts<%s")
            p = tuple(params) + (str(lo), str(pd.Timestamp(hi) + pd.Timedelta(days=1)))
            frames.append(db.read_sql(sub, p))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build(a_id: int) -> pd.DataFrame:
    sp = db.read_sql("SELECT MIN(DATE(h_ts)) lo, MAX(DATE(h_ts)) hi FROM hits WHERE h_a_id=%s",
                     (a_id,))
    if sp.empty or pd.isna(sp.lo[0]):
        raise ValueError("no hits")
    span = (sp.lo[0], sp.hi[0])

    agg = _q("""SELECT DATE(h_ts) d, COUNT(*) hits, SUM(h_status=200) c200,
                SUM(h_status=404) c404, SUM(h_status IN (401,403)) cblock,
                SUM(h_status=429) c429, SUM(h_u_id=%s) robots, SUM(h_ccode='HU') hu
                FROM hits WHERE h_a_id=%s GROUP BY DATE(h_ts)""",
             (ROBOTS_U_ID, a_id), span)
    # one DISTINCT per channel; COALESCE because the generator counted NULL as a value
    dist = {}
    for name, expr in [("n_ip", "COALESCE(h_i_id,-1)"), ("n_dom", "COALESCE(h_d_id,-1)"),
                       ("n_ctry", "h_ccode"), ("n_url", "h_u_id")]:
        dist[name] = _q(f"""SELECT DATE(h_ts) d, COUNT(DISTINCT {expr}) v
                            FROM hits WHERE h_a_id=%s GROUP BY DATE(h_ts)""", (a_id,), span)
    ext = _q("""SELECT DATE(h.h_ts) d, u.u_extension x, COUNT(*) c
                FROM hits h JOIN url u ON u.u_id=h.h_u_id
                WHERE h.h_a_id=%s GROUP BY DATE(h.h_ts), u.u_extension""", (a_id,), span)

    agg["d"] = pd.to_datetime(agg["d"])
    agg = agg.groupby("d", as_index=False).sum(numeric_only=True)
    days = [d for d in pd.date_range(span[0], span[1]) if d.date() not in _zero]
    g = pd.DataFrame({"d": days}).merge(agg, on="d", how="left").fillna(0)
    h = g["hits"].replace(0, np.nan)

    out = pd.DataFrame({"d": g["d"], "day": np.arange(len(g)), "hits": g["hits"]})
    out["p404"] = (g["c404"] / h).fillna(0)
    out["p200"] = (g["c200"] / h).fillna(0)
    out["robots"] = (g["robots"] / h).fillna(0)
    out["img"] = 0.0                      # identically zero in all 377 training files
    out["js"] = 0.0                       # ditto
    for name, key in [("n_ip", "ip_ent"), ("n_dom", "dom_ent"),
                      ("n_ctry", "ctry_ent"), ("n_url", "url_ent")]:
        df = dist[name]
        if df.empty:
            out[key] = 0.0
            continue
        df["d"] = pd.to_datetime(df["d"])
        df = df.groupby("d", as_index=False)["v"].max()
        out[key] = (g["d"].map(df.set_index("d")["v"]).fillna(0).to_numpy() / h).fillna(0)
    out["husource"] = (g["hu"] / h).fillna(0)

    if ext.empty:
        out["html"] = out["php"] = out["folder"] = 0.0
    else:
        ext["d"] = pd.to_datetime(ext["d"])
        ext["x"] = ext["x"].fillna("(null)").str.lower()
        piv = (ext.pivot_table(index="d", columns="x", values="c", aggfunc="sum")
               .reindex(g["d"]).fillna(0))
        def share(cols):
            use = [c for c in cols if c in piv.columns]
            v = piv[use].sum(axis=1).to_numpy() if use else np.zeros(len(g))
            return pd.Series(v / h.to_numpy()).fillna(0).to_numpy()
        out["html"] = share(HTML_EXT)
        out["php"] = share(("php",))
        out["folder"] = share(("(null)",))

    out["cblock"] = g["cblock"]
    out["c429"] = g["c429"]
    return out


def write(a_id: int, klass: str, out: pd.DataFrame) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PQ.mkdir(parents=True, exist_ok=True)
    tsv = out[COLS].copy()
    with open(OUT / f"{a_id}.{klass}", "w", encoding="utf-8", newline="") as fh:
        fh.write("\t".join(COLS) + "\n")       # real header: skiprows=1 now skips THIS
        tsv.to_csv(fh, sep="\t", header=False, index=False, float_format="%.6g")
    out.to_parquet(PQ / f"{a_id}.parquet", index=False)


def one(job):
    a_id, klass = job
    if (OUT / f"{a_id}.{klass}").exists():
        return ("skip", a_id, "")
    try:
        df = build(int(a_id))
        if len(df) < 30:
            return ("short", a_id, f"{len(df)} rows < 30-day window")
        write(int(a_id), klass, df)
        return ("ok", a_id, f"{len(df)} rows")
    except Exception as e:
        return ("fail", a_id, f"{type(e).__name__}: {str(e)[:120]}")


def ids_from(pattern):
    return sorted(int(os.path.basename(f).split(".")[0])
                  for f in glob.glob(str(config.DATA_DIR / pattern)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["arm", "control", "verified"])
    ap.add_argument("--validate", nargs=2, metavar=("A_ID", "CLASS"))
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()

    global _zero
    _zero = set(pickle.load(open(MASK, "rb")))
    print(f"panel-zero mask: {len(_zero)} days")

    if a.validate:
        aid, kl = int(a.validate[0]), a.validate[1]
        got = build(aid)
        pub = pd.read_csv(config.DATA_DIR / f"{aid}.{kl}", sep="\t", header=None, names=COLS)
        n = min(len(got), len(pub))
        print(f"generated {len(got)} rows vs published {len(pub)}; comparing {n}")
        print(f"{'column':10} {'maxdiff':>13}")
        for c in COLS[1:]:
            d = float(np.nanmax(np.abs(got[c].to_numpy()[:n] - pub[c].to_numpy()[:n])))
            print(f"{c:10} {d:13.8f} {'EXACT' if d < 2e-6 else 'DIFF'}")
        return

    verified = set(ids_from("*.human"))
    if a.set == "arm":
        jobs = [(i, "arm") for i in
                pd.read_csv(config.HERE / "outputs" / "_arm_blocked_agents.csv").a_id.tolist()]
    elif a.set == "verified":
        jobs = [(i, "human") for i in sorted(verified)]
    else:
        jobs = [(i, "pothuman") for i in ids_from("*.pothuman") if i not in verified]

    print(f"set={a.set}: {len(jobs)} agents, {a.threads} threads")
    tally, fails = {}, []
    with ThreadPoolExecutor(max_workers=a.threads) as ex:
        for k, (st, aid, msg) in enumerate(ex.map(one, jobs), 1):
            tally[st] = tally.get(st, 0) + 1
            if st in ("fail", "short"):
                fails.append({"a_id": aid, "status": st, "msg": msg})
            if k % 100 == 0 or k == len(jobs):
                print(f"  {k}/{len(jobs)} {tally}", flush=True)
    if fails:
        pd.DataFrame(fails).to_csv(config.HERE / "outputs" / f"_tsv_build_fails_{a.set}.csv",
                                   index=False)
        print(f"failures written: outputs/_tsv_build_fails_{a.set}.csv")
    print("done:", tally)


if __name__ == "__main__":
    main()
