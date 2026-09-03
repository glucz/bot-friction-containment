"""Emit the dose split at the specification of record, without the superseded branch.

`dose_split_recompute.py` computes more than one specification. This lifts the branch of record -
spacing 16, block-insensitive 404 coordinate, `[-10,-6]` baseline - into a file of its own, records
which branch it came from and the sha256 of the file it came from, and is what the manifest and the
manuscripts read. A single file holding several specifications is a file a reader can quote the
wrong key from. The
two-branch recomputation stays internal.

Read-only apart from the JSON it writes.
Usage: python dose_split_of_record.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "dose_split_recompute.json"
OUT = HERE / "dose_split_of_record.json"

DESCRIPTION = (
    "Dose split at the specification of record: event spacing 16, the block-insensitive 404 "
    "coordinate, and the [-10,-6] baseline. The raw dose gradient is defender-targeting "
    "selection and is never read as a causal gradient; only the adjusted and matched estimates "
    "carry weight, and the verified-human control arm shows a larger adjusted retreat gradient, "
    "so no measured retreat contrast identifies a bot-specific response."
)


def main() -> None:
    # Stage 1's intermediate is deliberately not distributed: it carries both the of-record
    # branch and a superseded one, and only the current branch is published. Say so rather than
    # raising a bare FileNotFoundError, which reads as a broken script.
    if not SRC.exists():
        raise SystemExit(
            f"{SRC.name} is not distributed with this bundle, so this stage cannot run.\n"
            f"  expected at {SRC}\n"
            f"It is produced by `python analysis/dose_split_recompute.py`, which needs the "
            f"restricted per-agent cache. The published result of this stage is "
            f"`analysis/dose_split_of_record.json`.")
    d = json.loads(SRC.read_text(encoding="utf-8"))
    keys = [k for k in d if k.startswith("corrected")]
    if len(keys) != 1:
        raise SystemExit(f"expected exactly one 'corrected...' branch, found {keys}")
    out = {
        "specification": keys[0],
        "description": DESCRIPTION,
        "populations": d[keys[0]],
        "source": SRC.name,
        "source_sha256": hashlib.sha256(SRC.read_bytes()).hexdigest(),
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    ab = out["populations"]["abusive"]
    print(f"wrote {OUT.name} from {keys[0]}")
    print(f"  retreat light {ab['retreat_light']:.4f}  heavy {ab['retreat_heavy']:.4f}  "
          f"n_events {ab['n_events']}  n_agents {ab['n_agents']}")


if __name__ == "__main__":
    main()
