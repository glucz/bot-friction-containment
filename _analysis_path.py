"""Where the `analysis` package sits, from a root-level pipeline script, in either layout.

In the source tree the analyses live at `papers/D-adversarial-friction/analysis/`, one level up
and sideways from here. In the published bundle they sit at `analysis/`, directly beside the
script. The directory is resolved, never assumed, and resolved in ONE place: a helper copied into each
script is a second implementation that drifts from the one the next correction reaches.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def analysis_dir() -> Path:
    """The directory containing `input_contract.py`."""
    for cand in (HERE.parent / "papers" / "D-adversarial-friction" / "analysis",  # source tree
                 HERE / "analysis"):                                             # bundle
        if (cand / "input_contract.py").exists():
            return cand
    raise SystemExit(
        "input_contract.py was not found beside this script or in the source tree. This analysis "
        "will not run without the input contract, and running it without one would produce a "
        "cohort of whatever happens to be readable rather than the declared population.")


def on_path() -> Path:
    """Put that directory on `sys.path` and return it."""
    d = analysis_dir()
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
    return d
