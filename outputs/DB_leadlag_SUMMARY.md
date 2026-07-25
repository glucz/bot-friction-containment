# Lead-lag: friction <-> behavior (sign convention verified by self-test)

Agents: 8761 (bots 6254, controls 2507); filters: >=30 days, >=3 block days, lag 0 excluded (compositional same-day coupling).

## bot
- responders (behavior follows friction): 41.6%
- tau_response median 7.0 d, CI [7.0, 7.0]
- leaders' implied defender reaction time: 5.0 d
- median post-peak CCF sign: 0.275 (negative = probing is SUPPRESSED after friction)

## control
- responders (behavior follows friction): 50.0%
- tau_response median 7.0 d, CI [7.0, 8.0]
- leaders' implied defender reaction time: 7.0 d
- median post-peak CCF sign: 0.145 (negative = probing is SUPPRESSED after friction)


## Paper mapping
- tau_response = bot adaptation delay tau in the SLA-market companion's Theorem 4 / this paper's Theorem 5.
- The pre-side (probing precedes blocks) measures the DEFENDER's reaction time — the other delay in the same loop. Both delays enter the Nyquist denominator; the loop period is bounded below by their sum.

## Addendum: lag-distribution diagnostics (verification pass)

- bot: mode = 1 day; 29.3% of responders within 3 days; 11.2% piled at the 14-day search boundary (true lag may exceed window).
- control: mode = 1 day; 24.8% of responders within 3 days; 8.6% piled at the 14-day search boundary (true lag may exceed window).
- No spike at lag 7/14 -> no day-of-week seasonality artifact (checked).
- The robust bot-vs-control discriminator is coupling STRENGTH: median post-peak CCF 0.275 (bots) vs 0.145 (controls), ~1.9x.
