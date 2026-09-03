# -*- coding: utf-8 -*-
"""One-off: remove dissertation draft letter references ("Paper D", "Paper F",
"Paper B") from the pipeline sources and generated summaries, so the public
artifact repository reads as the code companion of a titled paper rather than
of a lettered draft. Also renames sim_paper_d.py -> sim_friction_policy.py.

Run once per tree:
    python tools_dedraft.py <tree-root>
"""
from __future__ import annotations

import io
import os
import re
import sys

# Ordered: specific phrasings first, generic fallbacks last.
RULES = [
    # --- headings / labels in generated summaries -------------------------
    ("## Paper F calibration -- adaptation dynamics",
     "## Population-dynamics calibration -- adaptation dynamics"),
    ("## Paper F calibration -- three-strategy population mix",
     "## Population-dynamics calibration -- three-strategy population mix"),
    ("- **Paper D**: empirical botness density",
     "- **Friction containment**: empirical botness density"),
    ("- **Paper B**: bot traffic as an SLA perturbation",
     "- **SLA perturbation**: bot traffic as an SLA perturbation"),
    ("| botness density (Paper D map) + SLA perturbation (Paper B) |",
     "| botness density (containment map) + SLA perturbation |"),
    ("- **Paper D** — *Computers & Security* — adversarial friction menus.",
     "- **Friction containment** — *Computers & Security* — adversarial "
     "friction schedules."),

    # --- cross-reference phrasings ---------------------------------------
    ("bot adaptation delay tau in Paper B T4 / Paper D T5",
     "bot adaptation delay tau in the SLA-market companion's Theorem 4 / "
     "this paper's Theorem 5"),
    ("for Paper F (Chaos, Solitons & Fractals) and Paper D (Computers & Security).",
     "for the population-dynamics companion (Chaos, Solitons & Fractals) and\n"
     "this paper (Computers & Security)."),
    ("E1 (Paper D): empirical botness density. Paper D's central containment-map figure",
     "E1: empirical botness density. The central containment-map figure"),
    ("E2 (Paper B): bot traffic as a time-varying perturbation. Paper B must stay",
     "E2: bot traffic as a time-varying perturbation. The SLA-market companion "
     "must stay"),
    ("# --- E1: empirical botness density (Paper D) ---",
     "# --- E1: empirical botness density -----------"),
    ("# --- E2: bot-perturbation calibration (Paper B) ---",
     "# --- E2: bot-perturbation calibration ------------"),
    ("Supports: Paper F (adaptation is real and common -> replicator dynamics is",
     "Supports: the population-dynamics companion (adaptation is real and common"),
    ("warranted) and Paper D (observed tactic drift, not just synthetic strategies).",
     "-> replicator dynamics is warranted) and this paper (observed tactic\ndrift, not just synthetic strategies)."),
    ("Supports: Paper D (empirical f_switch: visible->evasive strategy switch triggered",
     "Supports: this paper (empirical f_switch: visible->evasive strategy switch triggered"),
    ("by friction; Theorem 5 attacker adaptation) and Paper F (the feedback that drives",
     "by friction; Theorem 5 attacker adaptation) and the population-dynamics\ncompanion (the feedback that drives"),
    ("Analysis C -- Trajectory dynamics  (supports Paper F directly)",
     "Analysis C -- Trajectory dynamics  (population-dynamics calibration)"),
    ("(supports Paper F's three-strategy replicator model)",
     "(three-strategy replicator calibration)"),

    # --- section / finding references ------------------------------------
    ("Paper D, Section ", "this paper, Section "),
    ("Paper D Section ", "this paper, Section "),
    ("Paper D Finding ", "Finding "),
    ("Paper D §", "this paper §"),
    ("Paper D f_switch", "f_switch"),
    ("(Paper D)", "(this paper)"),
    ("Paper D's", "this paper's"),
    ("Paper D", "this paper"),

    # --- remaining letters ------------------------------------------------
    ("Paper F's", "the population-dynamics companion's"),
    ("Paper F", "the population-dynamics companion"),
    ("Paper B's", "the SLA-market companion's"),
    ("Paper B", "the SLA-market companion"),

    # --- module rename ----------------------------------------------------
    ("sim_paper_d", "sim_friction_policy"),
]

SKIP_DIRS = {".git", "__pycache__", ".cache", "cache", "figures"}
EXTS = {".py", ".md", ".cff", ".txt", ".ini", ".template"}


def sweep(root: str) -> None:
    changed, hits = [], 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if os.path.splitext(fn)[1] not in EXTS and not fn.endswith(".template"):
                continue
            path = os.path.join(dirpath, fn)
            if os.path.basename(path) == os.path.basename(__file__):
                continue
            with io.open(path, "r", encoding="utf-8", errors="surrogateescape") as f:
                orig = f.read()
            text = orig
            for old, new in RULES:
                text = text.replace(old, new)
            if text != orig:
                n = sum(1 for _ in re.finditer(r"Paper [A-G]\b|sim_paper_d", orig))
                hits += n
                with io.open(path, "w", encoding="utf-8", errors="surrogateescape",
                             newline="") as f:
                    f.write(text)
                changed.append((os.path.relpath(path, root), n))

    # rename the simulation module
    old_p = os.path.join(root, "sim_paper_d.py")
    new_p = os.path.join(root, "sim_friction_policy.py")
    if os.path.exists(old_p) and not os.path.exists(new_p):
        os.rename(old_p, new_p)
        changed.append(("sim_paper_d.py -> sim_friction_policy.py", 1))

    for name, n in changed:
        print("  %-52s %d" % (name, n))
    print("%s: %d files touched, %d letter/module references removed"
          % (root, len(changed), hits))

    leftover = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if os.path.splitext(fn)[1] not in EXTS:
                continue
            path = os.path.join(dirpath, fn)
            with io.open(path, "r", encoding="utf-8", errors="surrogateescape") as f:
                for i, line in enumerate(f, 1):
                    if re.search(r"Paper [A-G]\b|sim_paper_d", line):
                        leftover.append("%s:%d %s" % (
                            os.path.relpath(path, root), i, line.strip()[:100]))
    if leftover:
        print("LEFTOVERS:")
        for l in leftover:
            print("   " + l)
    else:
        print("no leftovers")


if __name__ == "__main__":
    sweep(sys.argv[1])
