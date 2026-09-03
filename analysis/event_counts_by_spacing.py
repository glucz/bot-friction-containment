"""Event counts under each spacing rule, emitted rather than asserted.

The article states what enforcing the spacing rule costs in events, and a quoted number is a
field in a JSON, so this emits it.

The comparison must hold the cache fixed. Counting events at one spacing on one day's cache
against another spacing on a later cache confounds the rule with cache growth, and only the
fixed-cache contrast isolates what the rule costs.

Read-only. Reads the per-agent tables produced by `db_battery.py`; writes
`event_counts_by_spacing.json`.

Usage: python event_counts_by_spacing.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from _layout import data_root, on_path, require  # noqa: E402

TAB = data_root() / "outputs" / "tables"

# (label, filename, cache vintage, spacing) -- the cache vintage is what makes a comparison
# valid or confounded, so it is recorded rather than assumed.
TABLES = [
    ("current_cache_spacing5", "DB_per_agent_cachenow_sp5.csv", "current", 5),
    ("current_cache_spacing11", "DB_per_agent_spacing11.csv", "current", 11),
]


def main() -> None:
    out = {"tables": {}, "comparisons": {}}
    for key, fn, vintage, spacing in TABLES:
        p = TAB / fn
        if not p.exists():
            out["tables"][key] = {"error": f"missing: {fn}"}
            print(f"  {key:26s} MISSING ({fn})")
            continue
        d = pd.read_csv(p)
        rec = {"file": fn, "cache_vintage": vintage, "spacing": spacing,
               "n_events": int(d["n_events"].sum()), "n_agents": int(len(d)),
               "n_agents_with_event": int((d["n_events"] > 0).sum())}
        out["tables"][key] = rec
        print(f"  {key:26s} events {rec['n_events']:>7,}  agents {rec['n_agents']:>6,}")

    # Only fixed-cache comparisons are emitted. A cross-vintage contrast mixes the rule with
    # cache growth, and an artifact that carries one invites it to be quoted.
    def cmp(a, b, label, valid, note):
        ta, tb = out["tables"].get(a), out["tables"].get(b)
        if not ta or not tb or "error" in ta or "error" in tb:
            return
        lost = ta["n_events"] - tb["n_events"]
        out["comparisons"][label] = {
            "from": a, "to": b, "events_from": ta["n_events"], "events_to": tb["n_events"],
            "events_dropped": lost, "pct_dropped": round(100 * lost / ta["n_events"], 2),
            "isolates_spacing": valid, "note": note}
        print(f"  {label:34s} {ta['n_events']:>7,} -> {tb['n_events']:>7,}  "
              f"({100*lost/ta['n_events']:.1f}% dropped)  {'VALID' if valid else 'CONFOUNDED'}")

    cmp("current_cache_spacing5", "current_cache_spacing11", "spacing_only_5_to_11", True,
        "cache held fixed, so the difference is the spacing rule and nothing else. "
        "This is the figure the article quotes.")

    p = HERE / "event_counts_by_spacing.json"
    json.dump(out, open(p, "w"), indent=2)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
