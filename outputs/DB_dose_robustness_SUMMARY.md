# Dose-response frontier robustness — selection vs dose (this paper, Section 7.5)

Events: 55249 (0 non-finite dropped). Same event definition, windows (+/-5 active days), MIN_DAYS_OBS=20, retreat threshold 0.5, and within-group median intensity split as the original frontier. All CIs are 95% cluster bootstraps resampling AGENTS (n=2000, seed 42).


## bots — 28367 events from 10955 agents (median split at event-day p_block = 0.3077)

### Raw frontier (reproduction)

- light (n=14197): dO -0.1087, dvol +0.0400, retreat 0.141
- heavy (n=14170): dO -0.2648, dvol +0.0242, retreat 0.073
- raw heavy-light gap: dO -0.1561 CI [-0.1884, -0.1235], retreat -0.0681 CI [-0.0757, -0.0604]

### Pre-event balance (light vs heavy)

- pre_O: light 2.0926 vs heavy 2.5563, diff +0.4638 CI [0.3752, 0.5475], SMD +0.2570 CI [0.1951, 0.3235]
- pre_logvol: light 2.4340 vs heavy 1.2162, diff -1.2178 CI [-1.2604, -1.1755], SMD -1.1633 CI [-1.1953, -1.1314]

### Covariate-adjusted dose effect (OLS on heavy + pre_O + pre_logvol)

- dO: raw gap -0.1561 -> adjusted +0.1325 CI [0.0919, 0.1737]
- retreat (LPM): raw gap -0.0681 -> adjusted +0.0687 CI [0.0608, 0.0767]

### Matching (1-NN with replacement on standardized (pre_O, pre_logvol), caliper 0.2 pooled SD)

- heavy events matched: 14148/14170 (0.16% discarded outside caliper)
- matched heavy-light dO: +0.1692 CI [-0.2438, 0.2791]
- matched heavy-light retreat: +0.0399 CI [0.033, 0.0459]

## controls — 26882 events from 3665 agents (median split at event-day p_block = 0.0345)

### Raw frontier (reproduction)

- light (n=13477): dO +0.0050, dvol +0.0556, retreat 0.114
- heavy (n=13405): dO -0.0100, dvol +0.0420, retreat 0.153
- raw heavy-light gap: dO -0.0150 CI [-0.0398, 0.0093], retreat +0.0391 CI [0.0283, 0.05]

### Pre-event balance (light vs heavy)

- pre_O: light 2.4549 vs heavy 2.0692, diff -0.3857 CI [-0.4907, -0.2837], SMD -0.2418 CI [-0.3126, -0.1754]
- pre_logvol: light 5.2658 vs heavy 3.1668, diff -2.0991 CI [-2.1787, -2.0142], SMD -1.3241 CI [-1.3863, -1.2596]

## Verdict

Pre-event covariates are STRONGLY imbalanced across doses (max |SMD| = 1.163 >= 0.25): selection into heavy friction is real, and the matched/adjusted estimates are the paper numbers. The heavy-dose gap does NOT clearly survive adjustment: raw -0.1561, adjusted +0.1325 (95% CI [0.0919, 0.1737]), matched +0.1692 (95% CI [-0.2438, 0.2791]) — treat the frontier as descriptive, not causal. Retreat dose gap: raw -0.0681, adjusted +0.0687 (95% CI [0.0608, 0.0767]), matched +0.0399 (95% CI [0.033, 0.0459]).

(max |SMD| bots = 1.1633; selection >=0.10: True; >=0.25: True; dO dose effect survives adjustment+matching: False)

Caveats: doses are still observational within arm (no randomization); matching/adjustment control only for OBSERVED pre-event behavior (pre_O, pre_logvol) — unobserved-confounder selection cannot be excluded. Light/heavy assignment is held fixed at the full-sample median inside bootstrap replicates.
