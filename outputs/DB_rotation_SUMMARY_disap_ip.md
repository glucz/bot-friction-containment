# Identity rotation / respawn — Finding 6 (ip bridge)

Deaths (persistent post-block collapses among honeypot bots): **18**; with >=1 bridged successor: 17; candidate pairs: 94.

Bridge = shared non-proxy IPs (overlap-weighted); successors may be ANY agent incl. fresh / non-honeypot identities -> this CLOSES the channel the honeypot bridge missed. UA+IP co-rotation still escapes -> residual lower bound.

## Transfer rate (deaths with a bridged successor)

- precision core (rare shared IP (>=50% overlap, >=3 IPs)): **0.3889**
- recall envelope (any shared): 0.9444

## Does the successor ramp up as the bot dies? (event-study around t_A)

- precision core: successor dvol +0.360 CI [0.045, 0.932], dbot -0.284 CI [-0.755, -0.037], placebo dvol +0.481 (n=13 pairs / 7 deaths)
- recall envelope: successor dvol -0.849 CI [-1.221, -0.378], dbot +0.166 CI [0.018, 0.277], placebo dvol +0.281 (n=94 pairs / 17 deaths)

## Falsification

- link-permutation null (core): successor dvol around dates it is NOT linked to = +0.448, 2.5-97.5% [-1.083, +1.084] over 200 draws, on 9 of 13 pairs (69%; successors without a full-history cache are excluded)
- link-permutation null (envelope): successor dvol around dates it is NOT linked to = +0.541, 2.5-97.5% [+0.055, +1.062] over 200 draws, on 59 of 94 pairs (63%; successors without a full-history cache are excluded)
  (real ramp must EXCEED the successor's ramp around a random unrelated death; if equal, the ramp is generic successor growth, not a handoff)
- per-pair placebo: `placebo dvol` above is the successor's volume change around a date 60 d before the block (core ramp-minus-placebo = -0.121, envelope = -1.131).

_Planted-handoff positive control: run with `--selftest` (separate pass)._
