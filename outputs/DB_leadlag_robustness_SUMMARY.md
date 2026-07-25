# Lead-lag robustness (this paper, Section 7.6): KM censoring + shuffle null

Conventions mirrored from `db_analysis_leadlag.py`: cohort >=30 active days and >=3 block days; p404 vs p_block channel (the channel used by the published headline numbers); normalized CCF via `scipy.signal.correlate`; lag window +/-14 with lag 0 excluded; responder = post-side (+1..+14) |peak| > pre-side |peak|; tau = post-side peak lag.

## Test 1 — Kaplan-Meier median response lag (tau = 14 right-censored)

### bot
- responders n=2573, censored at +14: 289 (11.23%)
- naive median tau: 7.0 d
- KM median tau: 7.0 d (95% CI [7, 7], Brookmeyer-Crowley on log-log Greenwood bands)

### control
- responders n=1251, censored at +14: 107 (8.55%)
- naive median tau: 7.0 d
- KM median tau: 7.0 d (95% CI [7, 8], Brookmeyer-Crowley on log-log Greenwood bands)

**Verdict (KM):** see figure `figures/DB_leadlag_KM.png`; curve table `tables/DB_leadlag_KM_curves.csv`.


## Test 2 — Circular-shift shuffle falsification (seed 42; 10 replicates/agent; offset ~ U[15, n-15])

QA: recomputed real post-CCF matches published per-agent CSV for 100.0% of sampled agents; responder flags match 100.0%.

### bot (n=3000)
- responder share: real 41.9% vs null 39.6% (range 37.8-41.1% over 10 replicates); real value sits at the 100th percentile of the null
- median post-peak CCF among responders (signed, headline convention): real 0.271 vs null 0.185 (range 0.172 to 0.193)
- median |post-peak CCF|, all sampled agents: real 0.204 vs null 0.100

### control (n=1500)
- responder share: real 50.4% vs null 50.6% (range 49.2-51.9% over 10 replicates); real value sits at the 50th percentile of the null
- median post-peak CCF among responders (signed, headline convention): real 0.155 vs null 0.067 (range 0.061 to 0.073)
- median |post-peak CCF|, all sampled agents: real 0.116 vs null 0.062

**Verdict (shuffle):** see figure `figures/DB_leadlag_shuffle_null.png`; per-agent table `tables/DB_leadlag_shuffle_per_agent.csv`.
