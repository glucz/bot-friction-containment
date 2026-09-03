"""The arm of record, read from the released ID file. One reader, no database.

`population_of_record.arm_of_record()` DERIVES the arm: it queries the database and rewrites
`DB_arm_abusive_ids.csv` and `DB_arm_of_record_ids.csv`. That is correct when the population is
being established and wrong in an analysis, which must be cache-only and read-only.

Analyses read the published authority through this module. It exists once so the validation is
defined in one place rather than copied into each caller.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import config

ARM_IDS_CSV = config.OUT_DIR / "DB_arm_of_record_ids.csv"
N_ARM = 7812


def arm_of_record_ids(path: Path | None = None, expect: int = N_ARM) -> list[int]:
    """Every id in the released arm file, validated, sorted, and free of duplicates.

    Refuses rather than returning a population of unknown size or shape: a weight, share or
    contrast reported for "the arm" must be computed over the arm the manuscripts name.
    """
    src = Path(path or ARM_IDS_CSV)
    if not src.exists():
        raise SystemExit(
            f"{src} is missing. It is the released arm-of-record authority; regenerate it "
            f"deliberately with population_of_record.py rather than letting an analysis derive a "
            f"population of its own.")

    try:
        frame = pd.read_csv(src)
    except Exception as exc:                                   # noqa: BLE001
        raise SystemExit(f"{src.name} could not be read as CSV ({type(exc).__name__}: {exc})")
    if "a_id" not in frame.columns:
        raise SystemExit(
            f"{src.name} has no `a_id` column; it carries {', '.join(map(str, frame.columns))}")
    col = frame["a_id"]

    # Every rejection below must be a CONTRACT refusal. A column of mixed types reaches pandas as
    # object dtype, so arithmetic on it raises a bare TypeError from inside the check rather than
    # a message naming the file and the problem, and a raw traceback is indistinguishable from a
    # broken script. Decide the shape first, then do arithmetic.
    if col.dtype == bool:
        raise SystemExit(f"{src.name}: a_id is Boolean, not an agent identifier")
    bad = [repr(v) for v in col
           if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating))]
    if bad:
        raise SystemExit(
            f"{src.name}: a_id contains {len(bad)} non-numeric value(s), first {bad[0]}. "
            f"Agent identifiers are integers.")
    if col.isna().any():
        raise SystemExit(f"{src.name}: a_id contains missing values")
    if not (col % 1 == 0).all():
        raise SystemExit(f"{src.name}: a_id contains non-integer values")

    # A zero-padded id is accepted and normalised: `07` and `7` name the same agent, and the ids
    # are compared as integers everywhere downstream. Recorded as a decision, not an accident.
    

    ids = col.astype(int).tolist()
    if any(i <= 0 for i in ids):
        raise SystemExit(f"{src.name}: a_id contains non-positive ids")

    uniq = sorted(set(ids))
    if len(uniq) != len(ids):
        raise SystemExit(
            f"{src.name}: {len(ids) - len(uniq)} duplicate id(s); a population counted twice is "
            f"not the population it names")
    if expect and len(uniq) != expect:
        raise SystemExit(
            f"{src.name} holds {len(uniq):,} unique ids, not the {expect:,} of the arm of record. "
            f"Refusing rather than reporting a result for a population of unknown size.")
    return uniq
