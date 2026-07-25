# Empirical pipeline and simulation artifacts for *Keeping Agentic Traffic Under Control Where Aggressive Blocking Fails*

Géza Lucz, Bertalan Forstner — Department of Automation and Applied Informatics,
Faculty of Electrical Engineering and Informatics, Budapest University of
Technology and Economics, Műegyetem rkp. 3., H-1111 Budapest, Hungary.

This repository contains the complete analysis code, simulation code, and
generated outputs (summaries, tables, figures) behind the empirical sections
and the calibrated simulation of the paper, and behind its methods companion
(the measurement-pipeline article). It is the repository referenced in the
data-availability statement of both articles:
<https://github.com/glucz/bot-friction-containment>. See `CITATION.cff` for how
to cite it.

The underlying AGWA dataset is published at Zenodo:
[10.5281/zenodo.14497695](https://doi.org/10.5281/zenodo.14497695)
(6.05 billion requests, 8,780 sites, 2019-03 → 2023-03, reduced to daily
series for 49,080 agents).

## Layout

| Path | Contents |
|---|---|
| `analysis_[a-e]_*.py` | File-battery analyses A–E (non-stationarity, filter response, trajectory, strategy mix, calibration) |
| `db_battery.py`, `db_analysis_*.py` | Calendar-dated DB battery: observability collapse, lead–lag, dose robustness, identity rotation, sensitivity |
| `db_rotation_probe.py` | Death-set generator for the identity-rotation test (§7.8) |
| `sim_friction_policy.py`, `sim_sweep_costs.py` | Calibrated simulation (§8): containment map, stability, policy comparison, cost sweep |
| `config.py`, `datasource.py`, `db.py`, `db_source.py`, `roster.py`, `signals.py`, `stats_tests.py`, `botness_inference.py` | Shared infrastructure (paths/schema, data access, read-only DB layer, statistics) |
| `run_all.py` | Driver for the file battery (analyses A–E) |
| `tools/make_schematics.py` | Deterministic generator for the conceptual schematics (Figures 1–2) |
| `outputs/` | Summaries of record (`*_SUMMARY.md`, `*.json`), `tables/*.csv`, `figures/*.png` |
| `PIPELINE.md` | Detailed pipeline documentation: claim decomposition, architecture, honesty notes |
| `PLAN_identity_rotation.md` | Design and results of the identity-rotation test (Finding 6) |

## Article ↔ artifact map

The measurement instruments are documented in the **methods companion** (MethodsX);
the findings and the simulation are in the **main article** (Computers & Security).
Section numbers below refer to the main article.


- **§7.3** (lifecycle trends, filter response) — `analysis_a_nonstationarity.py`, `analysis_b_filter_response.py`
- **§7.4–7.5** (observability collapse, dose split) — `db_analysis_observability.py`, `db_analysis_dose_robustness.py`, `db_analysis_obs_sensitivity.py`
- **§7.6** (lead–lag, adaptation timescale) — `db_analysis_leadlag.py`, `db_analysis_leadlag_robustness.py`, `analysis_c_trajectory.py`
- **§7.7** (robots.txt response) — `db_battery.py`
- **§7.8** (identity rotation) — `db_rotation_probe.py`, `db_analysis_identity_rotation.py`
- **§7.9–7.10** (co-evolution, botness density, strategy mix) — `db_battery.py`, `analysis_e_calibration.py`, `analysis_d_strategy_mix.py`
- **§8** (containment map, stability, policy comparison) — `sim_friction_policy.py`, `sim_sweep_costs.py`
- Headline numbers of record: `outputs/DB_summary.json`, `outputs/summary.json`, `outputs/SIM_headline.json`
- Methods companion figures (dose balance, response-lag distribution, Kaplan–Meier curves, circular-shift null, sample cross-correlations, identity rotation) — `db_analysis_dose_robustness.py`, `db_analysis_leadlag*.py`, `db_analysis_identity_rotation.py`
- All figures: `outputs/figures/`

## Reproducing

```bash
pip install -r requirements.txt

# File battery (analyses A–E) — needs the AGWA-derived per-agent files;
# point AGWA_DATA_DIR at the directory holding the .bot/.human/.chrome/.pothuman files
python run_all.py            # full
python run_all.py --quick    # bot + human + chrome only

# DB battery — needs a read-only MySQL connection to the AGWA panel:
# copy db_config.ini.template to db_config.ini and fill it in.
# After the first run, results are re-runnable in minutes from the local
# Parquet cache (cache/agents/) with no DB connection.
python db_battery.py
python db_analysis_observability.py
python db_analysis_leadlag.py
python db_rotation_probe.py && python db_analysis_identity_rotation.py

# Simulation (§8) — no data dependency beyond the calibration constants
python sim_friction_policy.py
python sim_sweep_costs.py

# Conceptual schematics (Figures 1–2)
python tools/make_schematics.py
```

`torch` is optional: without it, Analysis C uses a model-free behavioral
trajectory; with it (plus the classifier repo on the path), the true botness
curve b(t) is regenerated from the Mathematics-2025 classifier.

## Caveats before citing

See the **honesty notes** in `PIPELINE.md`: the panel is observational (a
natural experiment, not a controlled one); the file battery's time base is
agent age, not calendar date (the DB battery is calendar-dated); mimicry is
under-counted in the `.bot` class by construction; and the raw dose gradient
reflects defender targeting — only the within-agent level effect
(observability DiD ≈ −0.23) is load-bearing, as framed in §7.5 of the paper.

## License

Code and generated outputs are released under the MIT License (see `LICENSE`).
The AGWA dataset itself is distributed under its own terms at Zenodo.
