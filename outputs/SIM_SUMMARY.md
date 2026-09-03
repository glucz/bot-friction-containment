# SIM_SUMMARY — Calibrated Policy Simulation for this paper (Section 8)

*This summary reports the seed-42 simulation and the empirical anchors used by Paper D
v37. Empirical anchors are provenance-only unless the parameter table says otherwise.
This file is maintained by hand; `sim_friction_policy.py` does not write it.*

Script: `sim_friction_policy.py` (self-contained, all RNGs seeded 42).
Outputs: `outputs/figures/SIM_containment_map.png`, `outputs/figures/SIM_stability_timedomain.png`,
`outputs/tables/SIM_policy_comparison.csv`, `outputs/SIM_headline.json`, this file.

Notation follows the paper: **beta = 0.234** is the classifier's operating threshold on the
botness score, and tau_A / tau_D are the loop delays.

## Scope and honesty statement

**This simulation demonstrates the model's internal logic under measured delays
and densities. It is not an independent empirical finding.** The measured AGWA
quantities (loop delays, botness density, retreat fractions, volume CV, operating
threshold) discipline the inputs; the initial strategy mix is **not** among them and
is an assumed parameter, swept for sensitivity; the attacker cost parameters and
defender loss scales are assumed modeling choices, flagged below, and the outputs
are model outputs. Where the model's behavior diverges from the measurements
(retreat under heavy friction — see "Consistency checks") we report the divergence
rather than tune it away.

## Parameter table

| Parameter | Value | Status | Source / rationale |
|---|---|---|---|
| Attacker adaptation delay tau_A | 4 days | **measured** | Section 7.6 lead-lag study, abusive arm median (9 d verified humans, 2 d declared crawlers) |
| Defender reaction delay tau_D | 5 days | **measured** | Section 7.6 (leaders' implied reaction time) |
| Minimum loop period tau_A + tau_D | ≈9 days | **measured** | Section 7.6 |
| Botness density (bots) | empirical histogram | **measured** | `outputs/tables/E_botness_density.csv` (n=1,191 browser-labelled agents, mean b=0.454, 62% of mass above beta=0.234). A *contamination* density over a mixed stream, which is the population a triage rule faces — see the caveat below the model line |
| Operating threshold beta | 0.234 | **measured** | classifier `best_threshold.txt` |
| Initial strategy mix visible/evasive | 0.22 / 0.78 | **ASSUMED** | The network-spread mixture on declared automation does not identify the initial sunk-evasion share of abusive automation. 0.22 / 0.78 is the assumed baseline setting only; swept over 0.20-0.90 in `sim_sweep_mix_evasive.py` |
| Bot volume CV | 1.22 | **measured** | Section 7.9; per-agent volume caps X_i ~ lognormal, mean 1, CV 1.22 |
| Retreat fraction light/heavy | 0.098 / 0.099 | **measured** | Section 7.5 dose-response table at event spacing 16 (`analysis/dose_split_of_record.json`, `populations.abusive`). Used as a consistency check, never fitted, and selection-confounded; the raw gradient is null, gap +0.000 with an interval spanning zero |
| Observability collapse | DiD −0.364 (−0.491 stealth-only) | **measured** | Section 7.4, on the 7,812-agent arm at event spacing 16. Evidence for the mechanism L_E represents; it does not enter the simulation and does not set L_E's size |
| HUMAN score density | Beta(1.5, 8) | **ASSUMED** | no table in the pipeline provides a human score density; low-concentration Beta, mean 0.158, 22% of humans above beta = 0.234 (overlap region exists by construction) |
| v_A distribution | lognormal, median 0.60, sigma=0.60 | **ASSUMED** | places the band in the interior of [0,1] for mid-range v_A; alternative sigma=1.00 used for the robustness check |
| c0 (visible baseline unit cost) | 0.10 | **ASSUMED** | scraping is cheap absent friction |
| c1, r (visible friction cost c1·f^r) | 1.20, 1.5 | **ASSUMED** | convex friction cost; f_min(v_A)=((v_A−c0)/c1)^(1/r) spans (0,1) for v_A in (0.10, 1.30) |
| c_E (evasion variable unit cost) | 0.35 | **ASSUMED** | evasion is operationally costlier per unit than visible scraping |
| rho (residual friction exposure) | 0.30 | **ASSUMED** | evasion avoids 70% of the friction cost curve; residual cost c2·rho·f^r with c2 = 1.0, **carrying the same exponent r as the visible branch** |
| K_E (evasion fixed setup cost) | 0.50 | **ASSUMED** | sensitivity: halved/doubled below |
| p_C (compliant channel price) | 1.25 | **ASSUMED** | priced above evasion's effective cost for all feasible f; consistent with measured non-adoption of compliance (Section 7.7: 2.23% of the abusive arm ever fetch robots.txt, DiD ≈ 0 after first block) |
| X (volume cap) | 1.0 (mean) | **ASSUMED** | normalization; per-agent caps carry the measured CV 1.22 |
| gamma_D (infra cost / unit friction-volume) | 0.05 | **ASSUMED** | defense delivery is cheap relative to extraction loss |
| h (human-harm scale, L_H = h·f^2) | 1.00 | **ASSUMED** | convex harm per Section 5; fully blocking one human costs as much as one unit of extraction |
| e_D (loss / unit successful extraction) | 1.00 | **ASSUMED** | normalization of L_D |
| L_E (evasion-adaptation loss / agent·period) | 0.25 | **ASSUMED** | a declared loss scale: an evading agent costs the defender ≈25% of a unit-extraction loss per period in degraded detection and second-line defense. No measured quantity enters it; the observability collapse is evidence that the mechanism exists, not a calibration of its magnitude |
| H_budget (per-user harm budget) | 0.25 | **ASSUMED** | human-harm boundary f_H = sqrt(H_budget/h) = **0.50** |
| Population | 12,000 bots + 12,000 humans | **ASSUMED** | equal masses; n=24,000; results are policy-comparative, not level-calibrated |

Model: attacker picks argmax over U_V = v_A·x − (c0+c1·f^r)·x, **U_E = v_A·x − K_E − (c_E+c2·rho·f^r)·x**,
U_C = (v_A−p_C)·x, U_R = 0, with x at the cap when the unit margin is positive (zero-utility maxima = retreat).
Defender loss L_D = e_D·x_success + gamma_D·(friction-volume served) + h·sum_humans phi(b)^2 + L_E·#{s_E}.

**Caveat on the botness density.** Bot scores are drawn from the browser-labelled density, which
is over a *mixed* stream, and an assumed human population is then added separately, so low-score
mass is represented twice. The 38.3% sub-beta figure below is therefore a property of the mixed
high-risk stream a triage rule faces, not of confirmed automation. Replacing it means scoring the
treatment arm itself, whose agents have no per-agent files in the classifier's corpus. No
Section 7 finding depends on the density.

## Analytic band (Theorem 2 quantities)

- **f_min(v_A) = ((v_A − 0.10)/1.20)^(1/1.5)**; at the reference value v_ref = 0.60 (population median): **f_min = 0.558**.
- **f_switch = 0.886** (K_E = 0.50, unpaid), from (c1 − c2·rho)·f^r = K_E/X + c_E − c0. Under the linear-in-volume specification with a common cap this indifference friction is **independent of v_A** (both utilities share the v_A·x term), so it becomes operative only for types above v*; the v_A-dependence of the band's effective upper edge comes from the evasion-profitability boundary U_E = 0, visible in the map.
- For bots with **K_E sunk**, the switch point falls to **f_switch(sunk) = 0.426**.
- **Band at v_ref: [0.558, 0.886], width 0.328.** Band-collapse boundary (f_min = f_switch): **v\* = 1.100**; 15.5% of simulated bots (**35.2% of bot value mass**) lie beyond v\* and cannot be contained at any feasible friction — they evade.
- **kappa at v_ref = 0.1250, for every r.** With g(f) = rho·f^r the f_min^r term cancels, so the containment threshold does not depend on the cost-curve exponent: r = 1.2, 1.5, 2.0 and 3.0 all give 0.1250. This is result R1, and it is exact only under this functional form.

## Sensitivity: K_E halved / doubled (band-collapse boundary)

| K_E | f_switch | band at v_ref = 0.60 | width | band-collapse v\* |
|---|---|---|---|---|
| 0.25 (halved) | 0.676 | [0.558, 0.676] | **0.118** (−64%) | 0.767 |
| 0.50 (base) | 0.886 | [0.558, 0.886] | **0.328** | 1.100 |
| 1.00 (doubled) | none in (0,1] | [0.558, 1.00] | **0.442** (capped at f=1) | 1.300 |

Halving K_E narrows the band by 64% and pulls the collapse boundary down to v\* = 0.767; doubling
K_E removes the evasion switch from the feasible friction range entirely, leaving the band bounded
only by f = 1. **Cheap evasion is the band-collapse direction**, as Theorem 2's sufficient condition
predicts.

## Five-policy comparison (n = 24,000; indices: no-defense = 100)

Static graded schedule tuned by L_D grid search: **phi(b) = 1.00·sigmoid((b−0.234)/0.08)**
(ceiling swept 0.20-1.00 in steps of 0.025, width in {0.05, 0.08, 0.12}).
Adaptive graded: same schedule scaled by m_t, gain k_phi = 0.60 = 1/2·k_crit (k_crit = 1.20),
time-averaged after burn-in.

| Policy | Extraction (idx) | Infra cost | Human harm | Evasion prob (value mass) | Observed bot vol (idx) | L_D (idx) | SLA distortion | Retreat share |
|---|---|---|---|---|---|---|---|---|
| No defense (f=0) | 11,981 (100) | 0 | 0 | 0.000 | 11,981 (100) | 11,981 (100) | 0 | 0.00 |
| Binary block, beta=0.234 | 7,442 (62.1) | 177 | 2,677 | 0.364 | 4,591 (38.3) | 10,962 (**91.5**) | 2,677 | 0.39 |
| Aggressive throttle f=min(1,3b) | 6,999 (58.4) | 346 | 3,483 | **0.422** | 3,592 (**30.0**) | 11,630 (**97.1**) | 5,448 | 0.43 |
| Static graded (tuned) | 7,403 (61.8) | 263 | 1,948 | 0.386 | 4,294 (35.8) | 10,342 (**86.3**) | 3,790 | 0.40 |
| Adaptive graded (k=0.60) | 7,486 (62.5) | 261 | 1,911 | 0.394 | 4,295 (35.8) | 10,400 (**86.8**) | 3,753 | 0.39 |

Headlines:

1. **Ranking by total defender loss: static graded < adaptive graded < binary block < aggressive throttle < no defense.**
2. **The overdeterrence trap is visible in the table:** aggressive throttling produces the *best dashboard* (observed bot volume 30.0) and the *worst loss among active defenses* (97.1) — it buys the largest evasion share (0.422) and barely beats doing nothing. A defender choosing by observed bot volume would choose the worst active policy.
3. Static graded vs binary block at comparable dashboard levels (35.8 against 38.3 observed): **−27% human harm** (1,948 vs 2,677) and −5 points of L_D — the Theorem 4 chord effect under the assumed convex harm. The *sign* is robust across the cost grid; the magnitude is not (see below).
4. Across these five policies successful extraction never falls below ~58 of 100. Two floors are why: 38.3% of the simulated stream's volume scores b < beta = 0.234 and receives ≈no friction under any schedule that respects the classifier's informativeness, and 35.2% of bot value mass lies beyond v\*. **Friction cannot zero out extraction; it can only price the middle of the value distribution out of visibility or out of the market.** This is a statement about the five policies compared, not a theorem: a flat f = 1 schedule drives extraction to 39, at the cost of full friction on every human.
5. **The tuned schedule leaves the band, and that is a result rather than a tuning defect.** It places high-botness mass at **f = 1.00**, above the fresh switch (0.886) as well as the sunk-cost switch (0.426), so it pushes even entrants whose evasion cost is unpaid past the point where evasion becomes their best response. No lower ceiling on the tuning grid reaches its total loss, but the surface near the top is shallow and the loss is **not monotone in the ceiling** (0.500 gives 11,000.2 against 0.400's 10,695.9). The runner-up, 0.950, is **1.7 units away, 0.017%**. Ceilings within 0.1% of the minimum are exactly {0.950, 0.975, 1.000}, all above the fresh switch; widening to 0.5% admits [0.850, 1.000], which reaches inside the band. **Read the result as the direction the optimizer pushes, not as the exact ceiling it picks.** Full per-ceiling grid: `outputs/tables/SIM_ceiling_grid.csv` and `outputs/SIM_ceiling_grid.json` (`sim_ceiling_grid.py`). Relative to binary blocking, the same schedule buys its human-harm advantage with a slightly larger evasive share (0.386 against 0.364); under the declared costs the loss-minimizing schedule declines to contain and instead spends the collateral efficiently.
6. Under the tuned schedule **22.3% of humans exceed the harm boundary f_H = 0.50** — at a ceiling of 1.00 this is exactly the human mass scoring above beta. That is the convex-harm cost the L_D optimizer accepts; a stricter H_budget would force the ceiling down.

## Adaptive loop and critical gain (SIM_stability_timedomain.png)

Daily loop: f_i(t) = min(1, m_t·phi(b_i)); defender updates m_{t+1} = m_t + k_phi·(V_obs(t−tau_D) − V\*)/V0
with tau_D = 5 d on the dashboard (visible-only) signal; attackers best-respond to friction lagged
tau_A = 4 d; evasion setup is a ratchet (K_E sunk once paid). The dashboard target V\* is the
operating point the tuned schedule actually holds, 4,294 = 35.8% of no-defense volume.
The main simulation and `sim_oscillation_cost.py` use the same 4,294 target.

- **Empirical critical gain k_crit = 1.20** (scan step 0.05; instability = peak-to-peak dashboard swing > 0.15·V0 in the last 150 days).
- Below k_crit (k = 0.60): friction converges to the tuned operating point and the dashboard tracks the target (residual swing 0.0013·V0). Note the operating point is f = 1.00 on high-b mass, which is *outside* the band [0.558, 0.886]; the loop is stable there, it is simply not containing.
- Above k_crit (k = 2.40): sustained oscillation between under- and over-friction with **period ≈ 26.7 days ≈ 2.96× the measured minimum loop period of 9 days** (tau_A + tau_D) — the loop alternates between visible-extraction episodes and induced-evasion episodes, the threshold-probing dynamic of Theorem 5.
- Cost of instability: the oscillating loop averages **L_D 12,386/day vs 10,400 stable (+19.1%)** and human harm 2,862 vs 1,911/day (**+49.8%**). Its time-averaged dashboard is **49% worse** (observed volume ratio 1.486), so the instability is visible. What the dashboard gets wrong is the size and the composition of the damage: a 49% rise in visible volume stands for a 19% rise in total loss, and about half of that extra loss is human harm the dashboard never shows. Worse, the dashboard moves in the direction that invites tightening the gain further, which makes matters worse.

## Robustness: alternative v_A spread (sigma = 1.00 instead of 0.60, same median)

| Policy | Extraction | Human harm | Evasion prob | L_D |
|---|---|---|---|---|
| Static graded (re-tuned: f_target = 0.425) | 9,038 | 362 | 0.000 | **9,572** |
| Binary block beta=0.234 | 7,490 | 2,677 | 0.445 | 11,056 |
| Aggressive throttle | 6,875 | 3,483 | 0.505 | 11,534 |
| No defense | 11,560 | 0 | 0.000 | 11,560 |

**The policy ranking does not flip** (graded < binary < throttle < no defense). The interior optimum
moves: with a fatter value tail the re-tuned schedule drops its ceiling to f_target = 0.425 — just
*below* the sunk-cost switch point 0.426 — and induces **zero** evasion, conceding extraction by the
high-value tail instead of paying L_E and harm to fight it. The qualitative conclusion (graded
friction dominates; maximal friction is self-defeating) is robust to the value-distribution
assumption; the optimal operating point is not.

## Cost-parameter sweep (`sim_sweep_costs.py`, 13 settings, graded re-tuned at each)

Two conclusions are grid-robust: the best-dashboard policy is never the best-loss policy, and the
tuned graded schedule is the best active policy whenever human harm is strictly convex. The one
assumption that moves the ranking is the harm exponent, and it moves it where Theorem 4 says it
must — at linear harm (q = 1) binary blocking becomes the best active policy (indexed L_D 91.5 vs
95.8). The **magnitude** of the graded-versus-binary harm advantage is not robust: it swings from
−90% to −25% across the grid, so the −27% above should be quoted as a sign, not a number. Full
table in `outputs/SIM_cost_sweep_SUMMARY.md`.

## Sensitivity to the initial sunk-evasion share (`sim_sweep_mix_evasive.py`)

The tuned ceiling holds at f_target = 1.00 for sunk shares of 0.20 through 0.78 and then drops to
0.425, the sunk-cost switch point, at 0.90 — where human harm falls from 1,948 to 362, more than
fivefold, and evasion goes to zero. Past a threshold the optimizer stops trying to contain a
population that is already committed and simply stops paying for the attempt. Full table in
`outputs/SIM_mix_evasive_sweep.md`.

## Consistency checks against measurements (reported, not fitted)

- Simulated retreat share under a light schedule (f_target 0.45): **0.181** vs measured light-dose retreat **0.098** — same order.
- Simulated retreat under the heavy schedule (f_target 0.95) is **0.381**; the descriptive heavy-dose retreat anchor is **0.099**, a ratio of **3.9**. This is not evidence about the direction of a bot-specific treatment response: dose assignment is endogenous, and the covariate-adjusted retreat gradient is larger for the verified-human controls (+17.9 pp) than for abusive automation (+10.3 pp). The simulated retreat direction is therefore unvalidated by these data.
- Compliance share under 1% across policies with p_C = 1.25 — consistent with the measured non-adoption of compliant channels (Section 7.7: 2.23% of the abusive arm ever fetch robots.txt). Making s_C the off-ramp requires pricing it below evasion's effective cost (p_C < c_E + c2·rho·f^r + K_E/X ≈ 1.0-1.15 at band-edge friction), a pricing decision rather than a signaling one (Design Rule 8).

## Files

- `sim_friction_policy.py` — simulation (seed 42, single file)
- `sim_oscillation_cost.py` — instability cost, shares the tuned target
- `sim_sweep_costs.py`, `sim_sweep_mix_evasive.py` — the two sweeps
- `sim_ceiling_grid.py` — loss against schedule ceiling, the artifact behind headline 5
- `figures/SIM_containment_map.png` — central figure
- `figures/SIM_stability_timedomain.png` — Theorem 5 time-domain
- `tables/SIM_policy_comparison.csv` — full metric table (raw + indexed)
- `SIM_headline.json` — machine-readable headline numbers
