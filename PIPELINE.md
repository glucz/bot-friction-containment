# Pipeline: how the AGWA panel is turned into friction-response evidence

This pipeline mines **AGWA-derived per-agent daily time series** for evidence
that bots change tactics over time and that the change is a response to
friction. It turns "calibrated with AGWA data" from a promissory note into
numbers, tables, and figures.

Two batteries run over the same corpus:

- the **file battery** (`analysis_a`–`analysis_e`), whose time base is *agent
  age*, over the derived per-agent files;
- the **calendar-dated analyses** (`analysis/*.py`, `db_rotation_probe.py`), which
  reads the live AGWA MariaDB (6.05B hits, 2019-03→2023-03, 36,029 agents with
  extracted series) and
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

## The calendar-dated analyses

Every measurement the manuscripts quote is a field in an artifact, produced by the script named
beside it. The authoritative list is `analysis/numbers_manifest.py:ART`, which is what the number
checker loads; the table below is that list in prose.

| Artifact | Produced by | Carries |
|---|---|---|
| `analysis/combined_numbers_of_record.json` | `analysis/combined_numbers_of_record.py` | headline observability DiD, stealth-only, sensitivity arm, retreat |
| `analysis/spec_grid_and_window_free.json` | `analysis/spec_grid_and_window_free.py` | the 180-cell grid, both window-free estimators |
| `analysis/derived_ranks_and_monthly.json` | `analysis/derived_ranks_and_monthly.py` | grid percentiles |
| `analysis/dose_split_of_record.json` | `analysis/dose_split_recompute.py` → `dose_split_of_record.py` | raw, adjusted and matched dose contrasts, balance |
| `analysis/event_time_study.json` | `analysis/event_time_study.py` | event-time leads, parallel-trends check |
| `analysis/event_spacing_recompute.json` | `analysis/event_spacing_recompute.py` | observability and four response channels per population pairing; Fig. S7 data |
| `analysis/leadlag_calendar_recompute.json` | `analysis/leadlag_calendar_recompute.py` | response lags, 999-replicate circular-shift null |
| `analysis/leadlag_density_split.json` | `analysis/leadlag_density_split.py` | lags by observation density |
| `analysis/leadlag_mask_sensitivity.json` | `analysis/leadlag_mask_sensitivity.py` | lags under the masked imputation rule |
| `analysis/disappearance_sensitivity.json` | `analysis/disappearance_sensitivity.py` | disappearance rate and follow-up sensitivities |
| `analysis/robots_engagement.json` | `analysis/robots_engagement.py` | share of each population ever fetching robots.txt |
| `analysis/decoy_specificity_partition.json` | `analysis/decoy_specificity_partition.py` | decoy bridge by shared-path popularity |
| `analysis/event_counts_by_spacing.json` | `analysis/event_counts_by_spacing.py` | cost of the spacing rule in events |
| `outputs/tables/DB_rotation_deaths_v5.csv` | `db_rotation_probe.py` | Finding 6 stage 1: the death cohort |
| `outputs/DB_rotation_summary_v5*.json` | `db_analysis_identity_rotation.py` | Finding 6 stage 2: bridges, nulls, self-test |
| `outputs/DB_retreat_cut_sensitivity_sp11.json` | `db_retreat_cut_sensitivity.py` | retreat definition across cut points |
| `outputs/roster_v2.csv` | `roster_v2.py` | the roster the arm of record is asserted against |

**If any analysis is re-run, re-check every number quoted in the manuscripts.**
`python analysis/numbers_manifest.py` does exactly that and names any surface that has drifted.

## Run it

```bash
python run_all.py            # full file battery (incl. pothuman)
python run_all.py --quick    # bot + human + chrome only (faster)
python sim_friction_policy.py    # calibrated policy simulation
python sim_sweep_costs.py        # cost-parameter sweep
```

Outputs:
- `outputs/tables/*.csv`, `outputs/figures/*.png`

The file battery's own summary documents are not released: they report the battery's agent-age
time base rather than the dated analyses the article quotes, and shipping them beside the current
artifacts would put two answers to the same question one keystroke apart.

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
  and reverses after adjustment. Only the level effect (bot versus control around a
  block event) is read as evidence here; the gradient is never cited as a causal
  dose-response.
- **The file battery's time base is agent age**, not calendar date; the DB
  battery supplies the dated analyses.
- **Mimicry is under-counted** in the `.bot` class by construction (stealth bots
  are the hardest to self-identify; they sit in `.chrome`/`.pothuman`).
- The naive "bots fan out (rotate IPs) right after a 404" is **not** supported on
  signed means — humans do that too. The real, significant signal is the
  *magnitude* of response normalized to each agent's own volatility.
