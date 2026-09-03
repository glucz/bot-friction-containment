# Verification and provenance bundle

Analysis code and derived artifacts for *Keeping Agentic Traffic Under Control Where Aggressive
Blocking Fails*. Every number the manuscripts quote is a field in one of the artifacts below,
produced by the script named beside it.

**This bundle lets you verify the artifacts; it cannot regenerate them.** The analyses require
restricted derived inputs that are not distributed — the per-agent series derive from
request-level logs, and the curated label files identify 200 hand-verified *people* in a corpus
where an agent id joins to their traffic. Publishing that roster is not something a reproduction
convenience justifies, and without the human control arm no contrast in the paper can be rebuilt
in any case. What you can do here: read every field the manuscripts cite, re-run the checkers that
bind them, and inspect the code that produced them.

This package describes the **current** analysis only. Superseded runs, earlier population
conventions and prior figure versions are not here; they are archived with the paper under
`_archive/`.

## What you need

- Python 3.11+, with `pandas`, `numpy`, `scipy`, `matplotlib`, `pyarrow`
- the per-agent parquet cache under `cache/agents/`. **Not distributed**: it derives from
  request-level logs. Every analysis refuses to run without the inputs it declares, so an
  incomplete cache produces a refusal rather than a quietly smaller cohort.
- the curated label files. **Not distributed** either; `population_of_record.curated()` refuses
  rather than returning an empty population if their directory is absent.

Commands are run from this directory.

## Populations

Defined once, in `POPULATIONS-KEY.md`, and cited rather than restated. The runnable form is
`population_of_record.py`:

| Function | Returns |
|---|---|
| `arm_of_record()` | the 7,812-agent abusive arm — **the only analysable arm** |
| `abusive_arm()` | the raw honeypot-share criterion output (7,819); a provenance intermediate |
| `curated(ext)` | a curated label class: `bot` 177, `human` 200, `pothuman` 2,159 |

`POPULATIONS-KEY.md` and `NOTATION-KEY.md` are the research family's shared authorities, cited by
six papers. Their internal paths resolve against the research tree rather than against this bundle,
and they are the two files exempted from this bundle's path check for that reason: diverging the
copy shipped here would break the define-once rule those files exist to enforce. Every other
released document has each backticked artifact path checked against what the bundle actually
contains.

`event_study_primitives.py` holds the shared estimator pieces: `_block_events`, `_winmean`, `_did`.

Window geometry matters and is per finding. Estimators using the `[-10,-6]` baseline need **16**
observations between kept events, because events 11 apart reuse the baseline's own observations.
Estimators using adjacent ±5 windows need **11**, the disjointness condition for that geometry.
Both are correct for different geometries; neither is a tuning knob.

## The input contract

Every analysis that reads the per-agent cache admits its whole declared population **before**
computing anything (`analysis/input_contract.py`). Three outcomes are kept apart:

- **input failure** — absent, unreadable, unparseable, missing a required column, or structurally
  unusable (wrong dtype, unordered or duplicated dates). Aborts the run.
- **scientific exclusion** — parses and carries the right schema but does not qualify, *including a
  zero-row extraction*: finding nothing for an agent is a fact about the corpus, not a defect.
  Recorded with its reason.
- **admitted** — returned, and pinned by content hash.

Workers read through `load_admitted()`, which verifies the sha256 of the bytes it parses against
the manifest taken at admission. `analysis/input_pins.json` holds a cross-run pin per
`(analysis, population)`: a missing or mismatched pin is a refusal, and enrolling one requires the
explicit command below, so no run can become its own authority. It also pins the cached botness
axes, under `axis<suffix>` keys, recording the digest of the `.npz` and of the roster that trained
it. Those records are validated as axes rather than as input pins, and an analysis label may not
begin with `axis`, so the two kinds cannot collide.

What that pin is and is not proof against. A provenance sidecar sitting beside its payload can be
rewritten with it: poison the axis, regenerate the sidecar, and the two agree. The registry catches
that, because it is a third file the forger of the pair did not touch. It is **not** a defence
against someone who rewrites the whole bundle, in which the registry ships alongside the axes it
pins; nothing inside a self-contained archive can authenticate that archive against its own author.
The guarantee is internal consistency across the files a reader receives, not provenance against a
hostile publisher.

Every enrolment report shipped here is pinned -- all 29 of them, under `analysis/` and `outputs/`
alike. `sync_artifacts.py --check` walks the whole bundle and fails on any `*_inputs.json` whose
label the registry does not carry, on one it cannot parse, and on one with no label, because a
report that documents which bytes an analysis read, with no pin behind it, describes a constraint
that is not in force.

Enrolment is two steps, and both ship:

```
python analysis/freeze_input_pins.py --show
python analysis/freeze_input_pins.py --from-artifact <artifact>.json --prefix <prefix>
python analysis/freeze_input_pins.py --from-artifact <artifact>.json --prefix <prefix>     --apply --confirm-artifact-sha256 <the hash the dry run printed>
python analysis/freeze_input_pins.py --from-artifact outputs/<label>_inputs.json
```

The first step validates the artifact's `inputs` schema, recomputes each aggregate digest from its
own per-id records, and prints full digests plus the artifact's hash. **The enrolment authority is
the artifact you review, not the cache**: the tool checks that an artifact is internally consistent
and shows you what it would trust, and it does not re-hash the files on disk. An artifact whose
per-id digests were fabricated consistently would enrol; that is why the hash of the artifact must
be repeated back by a person who read the plan. The second writes, and only
if that hash is repeated back. Add `--only <group>` to enrol one population rather than all of
them.

## Analyses and the artifacts they produce

**Requires restricted derived inputs; artifacts can be verified but not regenerated from this
bundle.** Each command is listed so the path from input to field is legible, and each refuses
rather than producing a smaller result when its inputs are absent.

| Command | Artifact | What it carries |
|---|---|---|
| `python analysis/combined_numbers_of_record.py` | `analysis/combined_numbers_of_record.json` | headline observability DiD, stealth-only and sensitivity arms, retreat |
| `python analysis/spec_grid_and_window_free.py` | `analysis/spec_grid_and_window_free.json` | the 180-cell specification grid, both window-free estimators |
| `python analysis/derived_ranks_and_monthly.py` | `analysis/derived_ranks_and_monthly.json` | grid percentiles for the headline and its balanced analogue |
| `python analysis/dose_split_recompute.py`, then `python analysis/dose_split_of_record.py` | `analysis/dose_split_of_record.json` | raw, adjusted and matched dose contrasts, balance diagnostics |
| `python analysis/event_time_study.py` | `analysis/event_time_study.json` | event-time leads and the parallel-trends check |
| `python analysis/event_spacing_recompute.py` | `analysis/event_spacing_recompute.json` | observability and four response channels for every population pairing, plus the Fig. S7 figure data |
| `python analysis/leadlag_calendar_recompute.py` | `analysis/leadlag_calendar_recompute.json` | response lags and the 999-replicate circular-shift null |
| `python analysis/leadlag_density_split.py` | `analysis/leadlag_density_split.json` | those lags split by observation density |
| `python analysis/leadlag_mask_sensitivity.py` | `analysis/leadlag_mask_sensitivity.json` | the same under the masked imputation rule |
| `python analysis/disappearance_sensitivity.py` | `analysis/disappearance_sensitivity.json` | the disappearance rate and its follow-up sensitivities |
| `python analysis/robots_engagement.py` | `analysis/robots_engagement.json` | share of each population that ever fetches robots.txt |
| `python analysis/decoy_specificity_partition.py` | `analysis/decoy_specificity_partition.json` | the decoy bridge split by shared-path popularity |
| `python analysis/event_counts_by_spacing.py` | `analysis/event_counts_by_spacing.json` | what enforcing the spacing rule costs in events |
| `python db_rotation_probe.py` | `outputs/tables/DB_rotation_deaths_v5.csv` | Finding 6, stage 1: the death cohort on the arm of record |
| `python db_analysis_identity_rotation.py` | `outputs/DB_rotation_summary_v5*.json` | Finding 6, stage 2: the identity-rotation bridges and their nulls |
| `python db_event_branches_v2.py` | `outputs/DB_event_branch_v2.json` | branch weights and block-channel reach on the arm of record |
| `python db_retreat_cut_sensitivity.py --spacing 11 --output DB_retreat_cut_sensitivity_sp11.json` | `outputs/DB_retreat_cut_sensitivity_sp11.json` | the retreat definition swept across cut points |
| `python roster_v2.py` | `outputs/roster_v2.csv` | the roster the arm of record is asserted against |

Simulation artifacts (`outputs/SIM_*.json`, `outputs/tables/SIM_policy_comparison.csv`) come from
the FrictionLab package shipped with the paper.

## Checking the manuscripts against the artifacts

`python analysis/numbers_manifest.py` binds every quoted number to its artifact field and reports
any surface that has drifted. **The manuscripts themselves are not distributed here**, so run from
this bundle the command refuses and says so rather than reporting their absence as drift; it is
listed because it is the checker whose results the paper cites, and because its rules are readable
even where its inputs are not. It also checks that each cited figure is no older than its data, that
every artifact path cited in prose resolves, and that every released figure is either cited or
inventoried.

## Figures

Seventeen figures are cited by the manuscripts, each bound to the artifact its generator consumes.
The other 16 in `outputs/figures/` are listed in `analysis/figure_inventory.json` with a status and
a reason. **No superseded render ships beside a current one** — earlier versions are archived with
the paper, not placed a keystroke away from the file the article cites.
