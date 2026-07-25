# DB battery results — calendar-dated AGWA evidence (Papers F & D)

Agents analyzed: 49080  |  groups: {'bot': 35788, 'control': 13292}

## A1 — RFC-9309 (robots.txt) compliance

- bots that ever access robots.txt: **6.1%** vs controls 10.1%
- change in robots.txt rate after the FIRST real block (bot−control DiD): 7.321473959674281e-05 CI [-0.0008, 0.0009] (ns)

## A2 — filter-response to REAL block events (401/403/429)

- normalized robots response: bot=0.0255 ctrl=0.0644 DiD=-0.03887010175549842 CI [-0.0499, -0.0282] (sig)
- log-hits (retreat): bot=0.0157 ctrl=0.0062 DiD=0.009531679740069495 CI [-0.0205, 0.0384] (ns)
- 404 probing: bot=-0.0275 ctrl=-0.0014 DiD=-0.026152998125221007 CI [-0.0299, -0.0225] (sig)

## A4 — non-stationarity

- bots with a significant robots_rate trend: 37.3% vs controls 62.5%

## A3 — population co-evolution

See `DB_monthly_coevolution.csv` and `figures/DB_co_evolution.png` (monthly block/robots rates 2019-2023).
