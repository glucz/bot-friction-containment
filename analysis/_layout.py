"""Where the pipeline's data root sits, in whichever layout this copy is running from.

The analyses live in two places. In the source tree they sit at
`papers/D-adversarial-friction/analysis/` with the data root at `longtail/empirical-support/`, three
levels up and sideways. In the published bundle the same files sit at `artifacts/analysis/` with the
data root one level up, at `artifacts/` itself.

The root is resolved, never assumed. A hard-coded layout imports cleanly in the other one and then
fails on the first read, which is indistinguishable from a broken script and tells a reader nothing
about the restricted inputs the bundle deliberately withholds.

The root is identified by the `outputs/` directory every layout has, not by its name, so neither
directory has to be called anything in particular.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def data_root() -> Path:
    """The directory holding `outputs/`, and `cache/` when the cache is distributed."""
    # The source tree is tried FIRST. Both candidates are identified by an `outputs/` directory,
    # and the paper directory does not have one today - but if it ever gained one, preferring the
    # nearer candidate would silently switch every analysis onto a different root while all of
    # them still ran. Order by which layout is the authoritative one, not by proximity.
    for cand in (HERE.parents[2] / "empirical-support",            # source tree
                 HERE.parents[3] / "empirical-support",
                 HERE.parent):                                     # bundle: artifacts/
        try:
            if (cand / "outputs").is_dir():
                return cand
        except IndexError:
            continue
    raise SystemExit(
        "no data root was found: neither this package's parent nor the source tree's "
        "empirical-support directory contains an `outputs/` directory. Nothing can be read or "
        "verified from here.")


def on_path() -> Path:
    """Put the data root on `sys.path` so its modules (`config`, `db_source`) import.

    Analyses that import `config` inside `main()` need the root on the path before they run.
    """
    root = data_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def require(path: Path, what: str) -> Path:
    """Return `path`, or refuse with the message the bundle's documentation promises.

    A bare `FileNotFoundError` on a restricted input reads as a broken script. The distinction the
    package rests on is between an input that is absent because it is not distributed and one that
    is absent because something is wrong, and only the first is a refusal.
    """
    if path.exists():
        return path
    raise SystemExit(
        f"{what} is not distributed with this bundle, so this analysis refuses rather than "
        f"running on a smaller population.\n"
        f"  expected at {path}\n"
        f"The per-agent series derive from request-level logs and the curated label files identify "
        f"hand-verified people; see the README for what can be verified without them.")
