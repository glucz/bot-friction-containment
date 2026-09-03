"""Theorem 5's safe-gain budget, evaluated at the panel's measured loop delay.

The theorem bounds the normalized loop gain of a sampled delayed controller:

    0 < k_phi * s < 2 sin( pi / (2(2*tau + 1)) )

where `tau` is the loop delay expressed in update steps. A defender updating every `P` days
against a loop delay of `T` days has tau = ceil(T / P) steps, so the budget is a function of how
often the defender acts, not only of how slow the ecosystem is.

This emits the cadence table the manuscript prints, and the per-day budget it restates in prose,
so both are fields rather than arithmetic performed in a sentence. The numbers are closed-form
consequences of the theorem; nothing here is measured. `T` is the measured quantity, and its
identification limits are the manuscript's (Section 7.6).

Read-only apart from the JSON it writes.
Usage: python theorem5_cadence.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

HERE = Path(__file__).resolve().parent
OUT = HERE / "theorem5_cadence.json"

T_DAYS = 9          # the pooled loop delay reported in Section 7.6
PERIODS = (1, 2, 3, 5, 9)


def budget(tau: int) -> float:
    """The theorem's bound at a delay of `tau` update steps."""
    return 2.0 * math.sin(math.pi / (2 * (2 * tau + 1)))


def main() -> None:
    rows = []
    for p in PERIODS:
        # CEILING, not rounding: a defender updating every 2 days against a 9-day loop sees
        # 4.5 steps of delay and must budget for 5. Rounding down would grant a gain the
        # loop cannot absorb, which is the wrong direction for a stability bound.
        tau = max(1, math.ceil(T_DAYS / p))
        rows.append({
            "update_period_days": p,
            "delay_steps": tau,
            "denominator": 2 * (2 * tau + 1),
            "safe_gain_budget": round(budget(tau), 4),
            "safe_gain_budget_pct": round(100.0 * budget(tau), 1),
        })

    daily = rows[0]
    out = {
        "loop_delay_days": T_DAYS,
        "bound": "0 < k_phi * s < 2 sin(pi / (2(2*tau + 1)))",
        "note": ("closed-form consequences of Theorem 5 at the measured loop delay; the delay is "
                 "the measured quantity and carries the identification limits of Section 7.6"),
        "cadence": rows,
        "daily_budget": daily["safe_gain_budget"],
        "daily_budget_pct": daily["safe_gain_budget_pct"],
        "ratio_period_over_daily": round(
            rows[-1]["safe_gain_budget"] / daily["safe_gain_budget"], 2),
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for r in rows:
        print(f"  P={r['update_period_days']:>2}d  tau={r['delay_steps']:>2}  "
              f"budget={r['safe_gain_budget']:.4f} ({r['safe_gain_budget_pct']:.1f}%)")
    print(f"\n  daily budget {out['daily_budget_pct']:.1f}%, "
          f"{out['ratio_period_over_daily']:.2f}x smaller than at the loop period")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
