# Observability-collapse robustness battery (metric / threshold / control arm)

- agents with usable events: 10959 bots, 3755 controls; 28384 bot events

## 1. Distance-metric variants (DiD of post-pre ΔO, bots vs all controls)
- base: -0.230 CI [-0.263, -0.198] (bot -0.257 vs ctrl -0.028)
- maha: -0.226 CI [-0.259, -0.194] (bot -0.253 vs ctrl -0.028)
- novol: -0.237 CI [-0.272, -0.204] (bot -0.251 vs ctrl -0.014)
- strict_centroid: -0.216 CI [-0.250, -0.183] (bot -0.250 vs ctrl -0.034)

## 2. Retreat-threshold sensitivity
- drop >30%: retreat fraction 0.210; stealth-only DiD -0.365 CI [-0.402, -0.330] (bot -0.347 vs ctrl +0.018)
- drop >50%: retreat fraction 0.107; stealth-only DiD -0.306 CI [-0.340, -0.272] (bot -0.302 vs ctrl +0.005)
- drop >70%: retreat fraction 0.046; stealth-only DiD -0.275 CI [-0.310, -0.242] (bot -0.273 vs ctrl +0.002)

## 3. Control-arm composition (base metric)
- all (n=3755): -0.230 CI [-0.263, -0.198] (bot -0.257 vs ctrl -0.028)
- whitelist_only (n=255): -0.256 CI [-0.309, -0.200] (bot -0.257 vs ctrl -0.002)
- labeled_only (n=2302): -0.227 CI [-0.267, -0.187] (bot -0.257 vs ctrl -0.030)
- strict_arm_and_centroid (n=2557): -0.214 CI [-0.252, -0.176] (bot -0.250 vs ctrl -0.035)