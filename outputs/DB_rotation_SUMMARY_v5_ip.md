# Identity rotation / respawn — Finding 6 (ip bridge)

Deaths (persistent post-block collapses among honeypot bots): **271**; with >=1 bridged successor: 271; candidate pairs: 3100.

Bridge = shared non-proxy IPs (overlap-weighted); successors may be ANY agent incl. fresh / non-honeypot identities -> this CLOSES the channel the honeypot bridge missed. UA+IP co-rotation still escapes -> residual lower bound.

## Transfer rate (deaths with a bridged successor)

- precision core (rare shared IP (>=50% overlap, >=3 IPs)): **0.7491**
- recall envelope (any shared): 1.0

## Does the successor ramp up as the bot dies? (event-study around t_A)

- precision core: successor dvol -0.936 CI [-1.049, -0.819], dbot +0.970 CI [0.817, 1.131], placebo dvol -0.039 (n=2211 pairs / 203 deaths)
- recall envelope: successor dvol -0.743 CI [-0.852, -0.639], dbot +0.752 CI [0.632, 0.881], placebo dvol +0.034 (n=3100 pairs / 271 deaths)

## Falsification

- link-permutation null (core): successor dvol around dates it is NOT linked to = +0.582, 2.5-97.5% [-0.045, +1.283] over 200 draws, on 1944 of 2211 pairs (88%; successors without a full-history cache are excluded)
- link-permutation null (envelope): successor dvol around dates it is NOT linked to = +0.418, 2.5-97.5% [-0.047, +0.952] over 200 draws, on 2526 of 3100 pairs (81%; successors without a full-history cache are excluded)
  (real ramp must EXCEED the successor's ramp around a random unrelated death; if equal, the ramp is generic successor growth, not a handoff)
- per-pair placebo: `placebo dvol` above is the successor's volume change around a date 60 d before the block (core ramp-minus-placebo = -0.897, envelope = -0.777).

_Planted-handoff positive control: run with `--selftest` (separate pass)._
