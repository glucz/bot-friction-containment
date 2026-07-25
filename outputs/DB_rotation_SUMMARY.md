# Identity rotation / respawn — Finding 6 (honeypot-fleet rotation)

Deaths (persistent post-block collapses among honeypot bots): **1059**; with >=1 bridged successor: 430; candidate pairs: 9170.

Bridge = shared honeypot decoy URLs (rarity-weighted); successors are themselves confirmed bots (intra-fleet rotation) -> LOWER BOUND. The IP-bridge variant (DB_rotation_SUMMARY_ip.md) reaches non-honeypot successors and closes that gap.

## Transfer rate (deaths with a bridged successor)

- precision core (rare decoy, pop<= 25): **0.017**
- recall envelope (any decoy): 0.406

## Does the successor ramp up as the bot dies? (event-study around t_A)

- precision core: successor dvol +0.160 CI [-0.106, 0.418], dbot +0.066 CI [-0.078, 0.218], placebo dvol +0.016 (n=183 pairs / 18 deaths)
- recall envelope: successor dvol +0.299 CI [0.203, 0.409], dbot +0.049 CI [0.019, 0.079], placebo dvol -0.038 (n=9170 pairs / 430 deaths)

## Falsification

- link-permutation null (core): successor dvol around a RANDOM other death = +0.130 (n=143)
- link-permutation null (envelope): successor dvol around a RANDOM other death = +0.175 (n=7009)
  (real ramp must EXCEED the successor's ramp around a random unrelated death; if equal, the ramp is generic successor growth, not a handoff)
- per-pair placebo: `placebo dvol` above is the successor's volume change around a date 60 d before the block (core ramp-minus-placebo = +0.144, envelope = +0.337).

_Planted-handoff positive control: run with `--selftest` (separate pass)._
