# Identity rotation / respawn — Finding 6 (ip bridge)

Deaths (persistent post-block collapses among honeypot bots): **1059**; with >=1 bridged successor: 885; candidate pairs: 5014.

Bridge = shared non-proxy IPs (overlap-weighted); successors may be ANY agent incl. fresh / non-honeypot identities -> this CLOSES the channel the honeypot bridge missed. UA+IP co-rotation still escapes -> residual lower bound.

## Transfer rate (deaths with a bridged successor)

- precision core (rare shared IP (>=50% overlap, >=3 IPs)): **0.3654**
- recall envelope (any shared): 0.8357

## Does the successor ramp up as the bot dies? (event-study around t_A)

- precision core: successor dvol -0.326 CI [-0.437, -0.216], dbot +0.585 CI [0.457, 0.719], placebo dvol +0.312 (n=2625 pairs / 387 deaths)
- recall envelope: successor dvol -0.108 CI [-0.19, -0.026], dbot +0.359 CI [0.285, 0.441], placebo dvol +0.156 (n=5014 pairs / 885 deaths)

## Falsification

- link-permutation null (core): successor dvol around a RANDOM other death = +0.223 (n=1413)
- link-permutation null (envelope): successor dvol around a RANDOM other death = +0.255 (n=3247)
  (real ramp must EXCEED the successor's ramp around a random unrelated death; if equal, the ramp is generic successor growth, not a handoff)
- per-pair placebo: `placebo dvol` above is the successor's volume change around a date 60 d before the block (core ramp-minus-placebo = -0.638, envelope = -0.264).

_Planted-handoff positive control: run with `--selftest` (separate pass)._
