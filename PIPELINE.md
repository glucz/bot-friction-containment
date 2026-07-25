# Pipeline: how the AGWA panel is turned into friction-response evidence

This pipeline mines **AGWA-derived per-agent daily time series** for evidence
that bots change tactics over time and that the change is a response to
friction. It turns "calibrated with AGWA data" from a promissory note into
numbers, tables, and figures.

Two batteries run over the same corpus:

- the **file battery** (`analysis_a`–`analysis_e`), whose time base is *agent
  age*, over the derived per-agent files;
- the **calendar-dated DB battery** (`db_battery.py`, `db_analysis_*.py`), which
  reads the live AGWA MariaDB (6.05B hits, 2019-03→2023-03, 49,080 agents) and
  can be re-run in minutes from the Parquet cache (`cache/agents/`) with no
  database connection.

## The claim, decomposed

| Sub-claim | Test | Analysis |
|---|---|---|
| **C1** tactics change over time | per-agent trend + change-point on 14 features | **A** |
| **C2** the change responds to friction | event study around filter pressure, bot-vs-human DiD | **B** |
| **C3** the population co-evolves | 3-strategy mixture + lifecycle mix shift | **D** |
| adaptation *timescale* | autocorrelation time of the evasion signal | **C** |
| botness density (containment-map input) + demand perturbation | real model outputs + volume volatility | **E** |

## What each analysis supports

- **A — non-stationarity.** Establishes C1: behavioral series are non-stationary
  in ~99% of agents. Bot-specific fingerprint vs. the human control: request-mix
  shift (html/php ↑), probing (404 ↑), robots.txt checking ↑, network-origin
  *concentration* (country/url/domain entropy ↓).
- **B — filter-response event study (the causal core).** Difference-in-differences
  vs. verified humans around filter-pressure events. Bots respond **significantly
  more than humans** (robots.txt, IP rotation, evasion spread, volume), in
  heterogeneous directions — the empirical basis for the strategy-switch and
  adaptation-delay results of the paper.
- **C — trajectory dynamics.** Adaptation timescale (bots ~5 d vs humans ~2 d)
  and oscillatory share (~3× humans) — an individual-level arms-race signature.
- **D — strategy mix.** Empirical (x_V, x_E, x_M) for the bot population, plus
  lifecycle escalation toward evasion.
- **E — calibration.** Empirical botness density + operating threshold (the
  containment-map contours); bot-traffic volatility, timescale, and
  type-distribution distortion.

## The calendar-dated DB battery

The DB battery produces the paper's headline measurements, each with its own
robustness script:

| Result | Source of record | Robustness |
|---|---|---|
| robots.txt ever-read 6.1% bots vs 10.1% controls; no rise after first block (DiD ≈ 0, ns) | `outputs/DB_SUMMARY.md`, `outputs/DB_summary.json` | — |
| 404-probing suppressed after blocks (DiD −0.026 sig), volume holds (ns) | `outputs/DB_SUMMARY.md` | — |
| Observability collapse after blocks (DiD −0.227; stealth-only −0.305; n = 28,491) | `outputs/DB_observability_SUMMARY.md` | `db_analysis_obs_sensitivity.py` (metric + control-arm variants) |
| Dose split: heavy friction ≈2.4× the stealth shift raw, reversing after adjustment/matching | `outputs/DB_dose_robustness_SUMMARY.md` | `db_analysis_dose_robustness.py` (balance, OLS, 1-NN matching) |
| Loop delays: bot responders 41.6%, median 7 d (mode 1 d, 11.2% censored at 14 d); defender ≈5 d; coupling 0.275 vs 0.145 | `outputs/DB_leadlag_SUMMARY.md` | `db_analysis_leadlag_robustness.py` (Kaplan–Meier, circular-shift null) |
| Identity rotation: no handoff; IP-overlap successors co-collapse (Δvol −0.33) | `outputs/DB_rotation_SUMMARY_ip.md` | planted-handoff positive control + link-permutation null (in-script) |
| Strategy mix x_V/x_E/x_M ≈ 0.22/0.78/0.00 (file battery) | `outputs/SUMMARY.md` | — |

**If a battery is re-run, re-sync every number quoted in the manuscripts** —
`outputs/*_SUMMARY.md` and `outputs/*.json` are the sources of record.

## Run it

```bash
python run_all.py            # full file battery (incl. pothuman)
python run_all.py --quick    # bot + human + chrome only (faster)
python db_battery.py         # calendar-dated DB battery
python sim_friction_policy.py    # calibrated policy simulation
python sim_sweep_costs.py        # cost-parameter sweep
```

Outputs:
- `outputs/SUMMARY.md` — human-readable evidence report
- `outputs/summary.json` — machine-readable headline numbers
- `outputs/tables/*.csv`, `outputs/figures/*.png`

Each analysis is also runnable standalone (`python analysis_b_filter_response.py`).
`replot_botness_density.py` re-renders the botness-density figure from the cached
bin table when only its presentation changes (no database needed).

## Architecture & extension points

- `config.py` — paths (override `AGWA_DATA_DIR`, `AGWA_MODEL_PATH`), feature
  schema, hyperparameters.
- `datasource.py` — `DataSource` interface. `LocalFileSource` reads the on-disk
  derived files; `DBSource` reads the full corpus, giving **calendar timestamps**
  (population co-evolution in real time) and **full RFC 7231 status codes**
  (403/429 = explicit block/rate-limit, a sharper friction signal than the
  200/404/robots available in the derived files). Nothing downstream changes.
- `botness_inference.py` — the **only** PyTorch dependency, and it is optional.
  Without torch, Analysis C uses a model-free behavioral trajectory. Install
  torch to also regenerate the botness curve `b(t)` from `botness_model.pt`,
  matching the published classifier exactly.

## Honesty notes (read before citing)

- **Observational, not a controlled experiment.** Agents react to whatever the
  8,780 production sites already did; this is a natural experiment, framed as such.
- **Friction dose assignment is endogenous.** Defenders aim heavier friction at
  agents that already look evasive; the raw dose gradient is a targeting pattern
  and reverses after adjustment. Only the level effect (bot vs control around the
  same events) carries causal weight.
- **The file battery's time base is agent age**, not calendar date; the DB
  battery supplies the dated analyses.
- **Mimicry is under-counted** in the `.bot` class by construction (stealth bots
  are the hardest to self-identify; they sit in `.chrome`/`.pothuman`).
- The naive "bots fan out (rotate IPs) right after a 404" is **not** supported on
  signed means — humans do that too. The real, significant signal is the
  *magnitude* of response normalized to each agent's own volatility.
