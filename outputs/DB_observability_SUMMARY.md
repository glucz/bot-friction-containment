# Observability (b_obs) around real friction — behavior-only features

Agents: 28491 (bots 23098, controls 5393). Features: p404, robots_rate, log-volume; defender-response channels EXCLUDED by design (no circularity).


## dO after block events
- all events: bot -0.2576 vs ctrl -0.0305, DiD -0.2271 CI [-0.2607, -0.1922] SIGNIFICANT
- stealth-only (volume kept within 50%): bot -0.2985 vs ctrl +0.0065, DiD -0.3050 CI [-0.3387, -0.2712] SIGNIFICANT
- bot events classified retreat: 0.107


## Calendar trend
- bot slope -0.00647/mo, control slope +0.00800/mo -> bots converge toward controls relative to the control drift


## Light vs heavy friction (f_switch frontier)

- bot (n=28359 events): light dO -0.1087, dvol +0.0399, retreat 0.141  |  heavy dO -0.2640, dvol +0.0242, retreat 0.073
- control (n=24969 events): light dO +0.0034, dvol +0.0511, retreat 0.121  |  heavy dO -0.0067, dvol +0.0399, retreat 0.155