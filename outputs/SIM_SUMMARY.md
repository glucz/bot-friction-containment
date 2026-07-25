# SIM_SUMMARY — Calibrated Policy Simulation for this paper (Section 8)

Script: `empirical-support/sim_friction_policy.py` (self-contained, all RNGs seeded 42).
Outputs: `outputs/figures/SIM_containment_map.png`, `outputs/figures/SIM_stability_timedomain.png`,
`outputs/tables/SIM_policy_comparison.csv`, `outputs/SIM_headline.json`, this file.

## Scope and honesty statement

**This simulation demonstrates the model's internal logic under measured delays
and densities. It is not an independent empirical finding.** The measured AGWA
quantities (loop delays, botness density, initial strategy mix, retreat
fractions, volume CV, operating threshold) discipline the inputs; the attacker
cost parameters and defender loss scales are assumed modeling choices, flagged
below, and the outputs are model outputs. Where the model's behavior diverges
from the measurements (retreat under heavy friction — see "Consistency checks")
we report the divergence rather than tune it away.

## Parameter table

| Parameter | Value | Status | Source / rationale |
|---|---|---|---|
| Attacker adaptation delay τ_A | 7 days | **measured** | Section 7.6 lead-lag study (median; mode 1 d, heavy tail) |
| Defender reaction delay τ_D | 5 days | **measured** | Section 7.6 (leaders' implied reaction time) |
| Minimum loop period τ_A + τ_D | ≈12 days | **measured** | Section 7.6 |
| Botness density (bots) | empirical histogram | **measured** | `outputs/tables/E_botness_density.csv` (n=1,191 scored agents, mean b=0.454, 62% of mass above b=0.234) |
| Operating threshold τ | 0.234 | **measured** | classifier `best_threshold.txt` (Section 7.9) |
| Initial strategy mix visible/evasive | 0.22 / 0.78 | **measured** | `D_class_mix.csv` bot row; used as the share of bots with evasion setup cost K_E already **sunk** |
| Bot volume CV | 1.22 | **measured** | Section 7.9; per-agent volume caps X_i ~ lognormal, mean 1, CV 1.22 |
| Retreat fraction light/heavy | 0.141 / 0.073 | **measured** | Section 7.5 dose-response table; used as a consistency check, not fitted |
| Observability collapse | DiD −0.23 (−0.31 stealth-only) | **measured** | Section 7.4; qualitative anchor for L_E (below) |
| HUMAN score density | Beta(1.5, 8) | **ASSUMED** | no table in the pipeline provides a human score density; low-concentration Beta, mean 0.158, 22% of humans above τ=0.234 (overlap region exists by construction) |
| v_A distribution | lognormal, median 0.60, σ=0.60 | **ASSUMED** | places the band in the interior of [0,1] for mid-range v_A; alternative σ=1.00 used for the robustness check |
| c₀ (visible baseline unit cost) | 0.10 | **ASSUMED** | scraping is cheap absent friction |
| c₁, r (visible friction cost c₁·f^r) | 1.20, 1.5 | **ASSUMED** | convex friction cost; f_min(v_A)=((v_A−c₀)/c₁)^{1/r} spans (0,1) for v_A∈(0.10, 1.30) |
| c_E (evasion variable unit cost) | 0.35 | **ASSUMED** | evasion is operationally costlier per unit than visible scraping |
| ρ (residual friction exposure) | 0.30 | **ASSUMED** | evasion avoids 70% of friction; c_residual(f)=c₂·ρ·f with c₂=1.0 |
| K_E (evasion fixed setup cost) | 0.50 | **ASSUMED** | sensitivity: halved/doubled below |
| p_C (compliant channel price) | 1.25 | **ASSUMED** | priced above evasion's effective cost for all feasible f; consistent with measured non-adoption of compliance (Section 7.7: 6.1% robots.txt, DiD ≈ 0 after first block) |
| X (volume cap) | 1.0 (mean) | **ASSUMED** | normalization; per-agent caps carry the measured CV 1.22 |
| γ_D (infra cost / unit friction-volume) | 0.05 | **ASSUMED** | defense delivery is cheap relative to extraction loss |
| h (human-harm scale, L_H = h·f²) | 1.00 | **ASSUMED** | convex harm per Section 5; fully blocking one human costs as much as one unit of extraction |
| e_D (loss / unit successful extraction) | 1.00 | **ASSUMED** | normalization of L_D |
| L_E (evasion-adaptation loss / agent·period) | 0.25 | **ASSUMED, anchored** | magnitude set qualitatively to the measured observability collapse (DiD −0.23 to −0.31): an evading agent costs the defender ≈25% of a unit-extraction loss per period in degraded detection and second-line defense. The anchor is qualitative, not an estimate. |
| H_budget (per-user harm budget) | 0.25 | **ASSUMED** | human-harm boundary f_H = √(H_budget/h) = **0.50** |
| Population | 12,000 bots + 12,000 humans | **ASSUMED** | equal masses; n=24,000 ≥ 20,000; results are policy-comparative, not level-calibrated |

Model: attacker picks argmax over U_V = v_A·x − (c₀+c₁f^r)·x, U_E = v_A·x − K_E − (c_E+c₂ρf)·x,
U_C = (v_A−p_C)·x, U_R = 0, with x at the cap when the unit margin is positive (zero-utility maxima = retreat).
Defender loss L_D = e_D·x_success + γ_D·(friction-volume served) + h·Σ_humans φ(b)² + L_E·#{s_E}.

## Analytic band (Theorem 2 quantities)

- **f_min(v_A) = ((v_A − 0.10)/1.20)^{1/1.5}**; at the reference value v_ref = 0.60 (population median): **f_min = 0.558**.
- **f_switch = 0.897** (K_E = 0.50, unpaid), from c₁f^r − c₂ρf = K_E/X + c_E − c₀. Under the linear-in-volume specification with a common cap, this indifference friction is **independent of v_A** (both utilities share the v_A·x term); the v_A-dependence of the band's effective upper edge comes from the evasion-profitability boundary U_E = 0 (v_A = c_E + c₂ρf + K_E/X), visible in the map.
- For the 78% of bots with **K_E sunk**, the switch point falls to **f_switch(sunk) = 0.475**.
- **Band at v_ref: [0.558, 0.897], width 0.339.** Band-collapse boundary (f_min = f_switch): **v* = 1.119**; 14.8% of simulated bots (34.2% of bot *value* mass) lie beyond v* and cannot be contained at any feasible friction — they evade.

## Sensitivity: K_E halved / doubled (band-collapse boundary)

| K_E | f_switch | band at v_ref = 0.60 | width | band-collapse v* |
|---|---|---|---|---|
| 0.25 (halved) | 0.706 | [0.558, 0.706] | **0.148** (−56%) | 0.812 |
| 0.50 (base) | 0.897 | [0.558, 0.897] | **0.339** | 1.119 |
| 1.00 (doubled) | none in (0,1] | [0.558, 1.00] | **0.442** (capped at f=1) | 1.30 (where f_min reaches 1) |

Halving K_E narrows the band by 56% and pulls the collapse boundary down to v* = 0.812 (only ~66% of bots below it); doubling K_E removes the evasion switch from the feasible friction range entirely — the band is then bounded only by f = 1. The band width is roughly linear in K_E in this regime; **cheap evasion is the band-collapse direction**, as Theorem 2's sufficient condition predicts.

## Five-policy comparison (n = 24,000; indices: no-defense = 100)

Static graded menu tuned by L_D grid search: φ(b) = 0.80·sigmoid((b−0.234)/0.05).
Adaptive graded: same menu scaled by m_t, gain k_φ = 0.50 = ½·k_crit, time-averaged after 60-day burn-in.

| Policy | Extraction (idx) | Infra cost | Human harm | Evasion prob (value mass) | Observed bot vol (idx) | L_D (idx) | SLA distortion | Retreat share |
|---|---|---|---|---|---|---|---|---|
| No defense (f=0) | 11,981 (100) | 0 | 0 | 0.000 | 11,981 (100) | 11,981 (100) | 0 | 0.00 |
| Binary block, τ=0.234 | 7,442 (62.1) | 177 | 2,677 | 0.364 | 4,591 (38.3) | 10,962 (**91.5**) | 2,677 | 0.39 |
| Aggressive throttle f=min(1,3b) | 6,950 (58.0) | 347 | 3,483 | **0.410** | 3,683 (**30.7**) | 11,547 (**96.4**) | 5,448 | 0.44 |
| Static graded (tuned) | 7,949 (66.3) | 191 | 1,283 | 0.376 | 4,803 (40.1) | 10,166 (**84.8**) | 2,600 | 0.35 |
| Adaptive graded (k=0.50) | 8,077 (67.4) | 184 | 1,178 | 0.392 | 4,792 (40.0) | 10,204 (**85.2**) | 2,490 | 0.34 |

Headlines:

1. **Ranking by total defender loss: static graded < adaptive graded < binary block < aggressive throttle < no defense.**
2. **The overdeterrence trap is visible in the table:** aggressive throttling produces the *best dashboard* (observed bot volume 30.7) and the *worst loss among active defenses* (96.4) — it buys the largest evasion share (0.410), 2.7× the human harm of the graded menu, and barely beats doing nothing.
3. Static graded vs binary block at matched dashboard levels (≈38–40 observed): **−52% human harm** (1,283 vs 2,677) and −7 points of L_D — the Theorem 4 (Jensen) effect under the assumed convex harm.
4. Successful extraction never falls below ~58 of 100 for any policy: 38.3% of bot volume scores b < τ = 0.234 and receives ≈no friction under any monotone menu (classifier informativeness floor), and 34% of bot value mass lies beyond the band-collapse value v*. **Friction cannot zero out extraction; it can only price the middle of the value distribution out of visibility or out of the market.**
5. The tuned static menu places the high-botness mass at f ≈ 0.79, just **inside** the band's upper edge (f_switch = 0.897, fresh) but **above** the sunk-cost switch point (0.475) — so the 78% already-evasive majority with sufficient value still evades (evasion probability 0.376). With the assumed costs, fully avoiding evasion by the sunk majority is worse in L_D than accepting it (but see the robustness run, where a fatter value tail flips that local choice while preserving the policy ranking).
6. Under the tuned menu, 17.5% of humans exceed the harm boundary f_H = 0.50 (Beta-tail humans with bot-like scores) — the convex-harm cost the L_D optimizer accepts; a stricter H_budget would force f_target down.

## Adaptive loop and critical gain (SIM_stability_timedomain.png)

Daily loop: f_i(t) = min(1, m_t·φ(b_i)); defender updates m_{t+1} = m_t + k_φ·(V_obs(t−τ_D) − V*)/V₀ with τ_D = 5 d on the dashboard (visible-only) signal; attackers best-respond to friction lagged τ_A = 7 d; evasion setup is a ratchet (K_E sunk once paid). Dashboard target V* = 0.40·V₀ (targets below the 27.7% visibility floor are unreachable — see headline 4).

- **Empirical critical gain k_crit = 1.00** (scan step 0.05; instability = peak-to-peak dashboard swing > 0.15·V₀ in the last 150 days).
- Below k_crit (k = 0.50): friction converges into the band and the dashboard tracks the target (residual swing 0.001·V₀).
- Above k_crit (k = 2.00): sustained oscillation between under- and over-friction with **period ≈ 40 days ≈ 3.3× the measured minimum loop period of 12 days** (τ_A + τ_D) — the loop alternates between visible-extraction episodes and induced-evasion episodes, the threshold-probing dynamic of Theorem 5. (Classic delayed negative feedback oscillates with period ≈ 4× total delay at onset; 40 d vs 4×12 = 48 d is consistent given the relay-like attacker response.)
- Cost of instability: the oscillating loop averages **L_D = 11,531/day vs 10,204/day stable (+13%)** and human harm 2,020 vs 1,178/day (**+71%**), while its *time-averaged* dashboard looks no worse — oscillation is expensive precisely where the dashboard doesn't show it.

## Robustness: alternative v_A spread (σ = 1.00 instead of 0.60, same median)

| Policy | Extraction | Human harm | Evasion prob | L_D |
|---|---|---|---|---|
| Static graded (re-tuned: f_target = 0.47) | 8,774 | 454 | 0.000 | **9,451** |
| Binary block τ=0.234 | 7,490 | 2,677 | 0.445 | 11,056 |
| Aggressive throttle | 6,839 | 3,483 | 0.493 | 11,470 |
| No defense | 11,560 | 0 | 0.000 | 11,560 |

**The policy ranking does not flip** (graded < binary < throttle < no defense). The interior optimum moves: with a fatter value tail, the re-tuned menu drops its ceiling to f_target = 0.47 — just *below* the sunk-cost switch point 0.475 — and induces **zero** evasion, conceding extraction by the high-value tail instead of paying L_E and harm to fight it. The qualitative conclusion (graded friction tuned to the band dominates; maximal friction is self-defeating) is robust to the value-distribution assumption; the optimal operating point inside the band is not.

## Consistency checks against measurements (reported, not fitted)

- Simulated retreat share under a light menu (f_target 0.45): **0.199** vs measured light-dose retreat **0.141** — same order.
- Simulated retreat under a heavy menu (f_target 0.95): **0.387** vs measured heavy-dose retreat **0.073** — the model **overpredicts retreat under heavy friction by ~5×**. The pure best-response model sends low-value bots to retreat when evasion margins are also destroyed; the measured population instead went stealth (retreat *halved* under heavy doses, Section 7.5). Plausible reasons: real evasion at the margin is cheaper than the assumed c_E/ρ for the surviving (selected/trained) population, and the dose split is endogenous. This divergence is stated in the paper as a limitation of the static cost specification, and it strengthens, not weakens, the overdeterrence warning: reality produces *more* stealth per unit of heavy friction than this model does.
- Compliance share ≈ 0–2% across policies with p_C = 1.25 — consistent with the measured non-adoption of compliant channels (Section 7.7); making s_C the off-ramp requires pricing it below evasion's effective cost (p_C < c_E + c₂ρf + K_E/X ≈ 1.0–1.15 at band-edge friction), a pricing decision, not a signaling one (operational rule 7).

## Files

- `empirical-support/sim_friction_policy.py` — simulation (seed 42, single file)
- `empirical-support/outputs/figures/SIM_containment_map.png` — central figure
- `empirical-support/outputs/figures/SIM_stability_timedomain.png` — Theorem 5 time-domain
- `empirical-support/outputs/tables/SIM_policy_comparison.csv` — full metric table (raw + indexed)
- `empirical-support/outputs/SIM_headline.json` — machine-readable headline numbers
