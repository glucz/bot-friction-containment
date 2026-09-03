# Identity rotation / respawn — Finding 6 (honeypot bridge)

Deaths (persistent post-block collapses among honeypot bots): **271**; with >=1 bridged successor: 132; candidate pairs: 4207.

Bridge = shared honeypot decoy URLs (rarity-weighted); successors are themselves confirmed bots (intra-fleet rotation); IP bridge infeasible on this schema -> LOWER BOUND.

## Transfer rate (deaths with a bridged successor)

- precision core (rare shared decoy): **0.0406**
- recall envelope (any shared): 0.4871

## Does the successor ramp up as the bot dies? (event-study around t_A)

- precision core: successor dvol -0.094 CI [-0.423, 0.335], dbot +0.106 CI [0.003, 0.188], placebo dvol -0.069 (n=191 pairs / 11 deaths)
- recall envelope: successor dvol +0.495 CI [0.364, 0.636], dbot -0.036 CI [-0.066, -0.003], placebo dvol -0.177 (n=4207 pairs / 132 deaths)

## Falsification

- link-permutation null (core): successor dvol around dates it is NOT linked to = -0.086, 2.5-97.5% [-0.559, +0.418] over 200 draws, on 160 of 191 pairs (84%; successors without a full-history cache are excluded)
- link-permutation null (envelope): successor dvol around dates it is NOT linked to = +0.118, 2.5-97.5% [-0.113, +0.405] over 200 draws, on 3625 of 4207 pairs (86%; successors without a full-history cache are excluded)
  (real ramp must EXCEED the successor's ramp around a random unrelated death; if equal, the ramp is generic successor growth, not a handoff)
- per-pair placebo: `placebo dvol` above is the successor's volume change around a date 60 d before the block (core ramp-minus-placebo = -0.026, envelope = +0.672).

_Planted-handoff positive control: run with `--selftest` (separate pass)._
