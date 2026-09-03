"""Census of the three analysis populations, as an output of record.

Section 7.1 quotes several structural facts about how the arms were formed.
They were derived by hand during the 2026-08-02 revision and had no output
backing them, which the paper's own "summaries of record" standard forbids.
This script produces them.

Reported:
  - honeypot join composition: agents that ever requested a trap path, split by
    whether the fillout script recorded a positive total (the treatment arm),
    an explicit zero, or nothing at all
  - size and mean request volume of the in-window and out-of-window groups,
    which is the selection the treatment-arm definition induces
  - roster versus extraction-cohort counts for each population
  - overlap between the curated declared set and the honeypot-positive set,
    which the roster's dedup would otherwise hide

Usage: python db_arm_census.py
Writes: outputs/DB_arm_census.json
"""
from __future__ import annotations

import glob
import json
import os

import pandas as pd

import config
import db
import db_source as dbs
from roster import ROSTER_CSV


def _curated_ids(ext: str) -> set[int]:
    out = set()
    for fp in glob.glob(str(config.DATA_DIR / ("*." + ext))):
        try:
            aid = int(os.path.splitext(os.path.basename(fp))[0])
        except ValueError:
            continue
        if aid <= dbs.MAX_A_ID:
            out.add(aid)
    return out


def run() -> dict:
    config.ensure_dirs()

    comp = db.read_sql("""
        SELECT COUNT(*) AS distinct_agents,
               SUM(tot IS NULL) AS null_total,
               SUM(tot = 0)     AS explicit_zero,
               SUM(tot > 0)     AS positive
        FROM (SELECT uh_a_id, SUM(uh_total) AS tot
              FROM url2agent_honeypot GROUP BY uh_a_id) x
    """).to_dict("records")[0]

    vol = db.read_sql("""
        SELECT CASE WHEN x.tot > 0 THEN 'in_window' ELSE 'out_of_window' END AS grp,
               COUNT(*) AS agents, ROUND(AVG(a.a_totalhit), 0) AS mean_requests,
               ROUND(AVG(DATEDIFF(a.a_last_5p, a.a_start_5p)), 1) AS mean_window_days
        FROM (SELECT uh_a_id, SUM(uh_total) AS tot FROM url2agent_honeypot
              GROUP BY uh_a_id HAVING SUM(uh_total) IS NOT NULL) x
        JOIN agent a ON a.a_id = x.uh_a_id
        GROUP BY grp
    """).to_dict("records")

    roster = pd.read_csv(ROSTER_CSV)
    declared = _curated_ids("bot")
    verified = _curated_ids("human")
    pothuman = _curated_ids("pothuman")
    hp_pos = set(db.read_sql(
        "SELECT uh_a_id AS a_id FROM url2agent_honeypot GROUP BY uh_a_id "
        "HAVING SUM(uh_total) > 0")["a_id"].astype(int))

    per_agent = pd.read_csv(config.OUT_DIR / "tables" / "DB_per_agent.csv")
    cohort = set(per_agent["a_id"].astype(int))

    out = {
        "honeypot_join": comp,
        "window_selection": vol,
        "roster_counts": roster["role"].value_counts().to_dict(),
        "curated_sizes": {"declared": len(declared), "verified": len(verified),
                          "pothuman": len(pothuman)},
        "extraction_cohort": {
            "abusive": len(set(roster.loc[roster.role == "bot_hp", "a_id"].astype(int)) & cohort),
            "declared": len(declared & cohort),
            "verified": len(verified & cohort),
            "pothuman": len(pothuman & cohort),
        },
        "sampling": {
            "whitelist_note": "ctrl_white is capped at WHITE_SAMPLE=5000 drawn from the qualifying set",
            "hpneg_note": "ctrl_hpneg is sampled at HPNEG_SAMPLE=6000 from the explicit-zero pool",
        },
        "declared_intersect_honeypot_positive": len(declared & hp_pos),
        "verified_intersect_honeypot_positive": len(verified & hp_pos),
        "pothuman_intersect_honeypot_positive": len(pothuman & hp_pos),
    }
    json.dump(out, open(config.OUT_DIR / "DB_arm_census.json", "w"), indent=2, default=float)
    print(json.dumps(out, indent=2, default=float))
    return out


if __name__ == "__main__":
    run()
