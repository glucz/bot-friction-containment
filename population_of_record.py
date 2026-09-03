"""Current population definitions and the constants every event study here shares.

The populations are DEFINED in `POPULATIONS-KEY.md`; this module is the runnable form of those
definitions and nothing else. It exists so that a current script does not have to import a whole
historical analysis to learn what the arm of record is.

  * `curated(ext)`      - a curated label class, by file extension
  * `abusive_arm()`     - the raw honeypot-share criterion output
  * `arm_of_record()`   - that output minus the hand-verified humans: the ONLY analysable arm
  * `MIN_DAYS_OBS`      - the minimum observed days an agent needs to enter an estimate
  * `EVENT_W`           - the half-width of the pre/post windows

`abusive_arm()` is a raw intermediate. Every estimate uses `arm_of_record()`.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import pandas as pd

import config
import db
import db_source as dbs

MIN_DAYS_OBS = 20
EVENT_W = 5
SHARE_THRESHOLD = 0.25

OUT_IDS = config.OUT_DIR / "DB_arm_abusive_ids.csv"          # raw criterion output, provenance only
OUT_IDS_RECORD = config.OUT_DIR / "DB_arm_of_record_ids.csv"  # the arm of record


def _assert_data_dir() -> None:
    """A curated class read from a directory that does not exist is an EMPTY SET, not an error.

    That silent-empty failure is the one this package exists to prevent, so the directory is
    asserted rather than assumed: in the public bundle `config.DATA_DIR` may point somewhere that
    was never shipped, and a caller would otherwise receive a well-formed empty population.
    """
    if not Path(config.DATA_DIR).is_dir():
        raise SystemExit(
            f"config.DATA_DIR does not exist: {config.DATA_DIR}. The curated label files are not "
            f"distributed with the public bundle; a curated population cannot be constructed here, "
            f"and returning an empty one would be worse than failing.")


def curated(ext: str) -> set[int]:
    _assert_data_dir()
    out = set()
    for fp in glob.glob(str(config.DATA_DIR / ("*." + ext))):
        try:
            aid = int(os.path.splitext(os.path.basename(fp))[0])
        except ValueError:
            continue
        if aid <= dbs.MAX_A_ID:
            out.add(aid)
    return out


def abusive_arm() -> pd.DataFrame:
    d = db.read_sql("""
        SELECT x.uh_a_id AS a_id, x.tot AS hp_hits, a.a_totalhit AS total_hits
        FROM (SELECT uh_a_id, SUM(uh_total) AS tot FROM url2agent_honeypot
              GROUP BY uh_a_id HAVING SUM(uh_total) > 0) x
        JOIN agent a ON a.a_id = x.uh_a_id
        WHERE a.a_totalhit > 0
    """)
    d["share"] = (d["hp_hits"] / d["total_hits"]).clip(0, 1)
    arm = d[d["share"] > SHARE_THRESHOLD].copy()
    arm.to_csv(OUT_IDS, index=False)
    return arm


def arm_of_record() -> set[int]:
    """The treatment arm of record: 7,812 agents.

    `abusive_arm()` applies the criterion, honeypot share > 0.25, and returns 7,819 ids. Seven of
    those are hand-verified humans; hand verification outranks an automated criterion, so the arm
    excludes them and the control keeps them. An agent in both the treatment and the control vector
    of the same difference-in-differences is an estimator defect, not a rounding convention.

    Current overlaps against this arm: 0 verified humans, 7 declared crawlers, 60 potential humans.

    This is the same set as `roster_v2.csv`'s `abusive_share` role, and the
    equality is asserted whenever that roster exists. Seven declared crawlers
    and 67 sensitivity-pool agents also clear the threshold and DO stay in the
    arm: neither label is hand-verified, so neither outranks the criterion.
    See longtail/POPULATIONS-KEY.md for the full overlap matrix.
    """
    ids = set(abusive_arm()["a_id"].astype(int)) - curated("human")
    roster_v2 = config.OUT_DIR / "roster_v2.csv"
    if roster_v2.exists():
        r = pd.read_csv(roster_v2)
        role = set(r.loc[r.role == "abusive_share", "a_id"].astype(int))
        if role != ids:
            raise SystemExit(f"arm_of_record ({len(ids)}) disagrees with roster_v2 "
                             f"abusive_share ({len(role)}); reconcile before running")
    pd.DataFrame({"a_id": sorted(ids)}).to_csv(OUT_IDS_RECORD, index=False)
    return ids
