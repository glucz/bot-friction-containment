"""Build the second-generation roster: the four current population roles.

Roles, in precedence order, each agent taking the first that applies:

  * `verified_human`  - the curated `.human` class, hand verified
  * `declared`        - the curated `.bot` class, minus any agent in the arm
  * `abusive_share`   - the arm of record: honeypot share above threshold, minus the seven
                        agents that are also hand-verified humans
  * `pothuman`        - the curated `.pothuman` class

Precedence matters: hand verification outranks the automated criterion, which is why the arm
of record is 7,812 and not the raw criterion output. Definitions live in POPULATIONS-KEY.md.

Writes: outputs/roster_v2.csv
"""
from __future__ import annotations

import pandas as pd

import config
from population_of_record import abusive_arm, curated

ROSTER_V2_CSV = config.OUT_DIR / "roster_v2.csv"


def build() -> pd.DataFrame:
    config.ensure_dirs()
    # Verified humans are claimed first: 7 of the 200 clear the share threshold
    # (the 3.5% false-positive rate of Section 7.1), and hand verification
    # outranks an automated criterion. Moving them costs the headline DiD
    # little; the arm of record excludes them (POPULATIONS-KEY.md).
    groups = [
        ("verified_human", sorted(curated("human")), False, True),
        ("abusive_share", sorted(set(abusive_arm()["a_id"].astype(int))), True, False),
        ("declared", sorted(curated("bot")), False, False),
        ("pothuman", sorted(curated("pothuman")), False, False),
    ]
    seen: set[int] = set()
    rows = []
    for role, ids, is_bot, is_control in groups:
        for a in ids:
            if a in seen:                      # first listed group wins; no double counting
                continue
            seen.add(a)
            rows.append({"a_id": a, "role": role,
                         "is_bot": is_bot, "is_control": is_control})
    d = pd.DataFrame(rows)
    d.to_csv(ROSTER_V2_CSV, index=False)
    print(d.groupby(["role", "is_bot", "is_control"]).size())
    print("written", ROSTER_V2_CSV)
    return d


if __name__ == "__main__":
    build()
