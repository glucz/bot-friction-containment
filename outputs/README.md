# What is in this directory

Outputs of the AGWA analysis pipeline behind Sections 7 and 8 of the co-submitted article. Every
file here is current: it reports the arm of record, event spacing 16 where the `[-10,-6]` baseline
applies and spacing 11 where adjacent `±5` windows apply, and the v5 rotation cohort.

## Where a number's provenance is settled

The authority is the manifest, not a summary file:

- `analysis/numbers_manifest.py` names every number the main article and the MethodsX companion
  quote, the artifact field it comes from, and every surface that must carry it. The supplement
  is not yet under the percentage census. It parses the Section 8 policy table cell by cell and the policy ranking as an
  ordered list, and it exits nonzero on any disagreement.
- `analysis/combined_numbers_of_record.json` — the observability estimates.
- `analysis/dose_split_of_record.json` — the dose and retreat quantities.
- `SIM_headline.json` — the calibrated simulation's headline values and policy ranking.

## Human-readable summaries

| File | Scope |
|---|---|
| `SIM_SUMMARY.md` | the seed-42 policy simulation: parameter table with each input's evidence tier, the policy comparison, the adaptive loop, and the consistency checks |
| `SIM_cost_sweep_SUMMARY.md` | one-at-a-time sweep of the assumed cost parameters, with the graded schedule re-tuned at each setting |
| `SIM_mix_evasive_sweep.md` | sensitivity to the assumed initial sunk-evasion share, swept 0.20-0.90 |
| `DB_rotation_SUMMARY_v5_ip.md` | Finding 6, IP bridge: 271 deaths, the strong-overlap core and the recall envelope, with both placebos and both date weightings |
| `DB_rotation_SUMMARY_v5.md` | Finding 6, decoy bridge: the precision core, the recall envelope and the specificity partition |
| `DB_rotation_SUMMARY_disap_ip.md` | the same bridges run on the disappearance episodes, where the test is uninformative in either direction |
| `MAPPING_NOTES.md`, `SCHEMA_MAP.md` | the source database: server, schema, date span, and how the derived per-agent series are built from it |

Machine-readable outputs sit beside these as `*.json`; generated tables are in `tables/` and figure
renders in `figures/`.

## Two files whose contents are inputs, not claims

`DB_arms_v2.json` and `DB_dose_robustness_v2.json` are the **inputs** that the of-record
recomputations read and correct: `analysis/leadlag_calendar_recompute.py` reads the first,
`analysis/dose_split_recompute.py` reads the second, and each writes a current artifact from it.
Both therefore contain figures produced under specifications this pipeline no longer uses, and both
must stay here or the correction cannot be reproduced from the repository. Values read out of them
directly are not results of this article; the results are in the of-record artifacts named above,
and the manifest checks the manuscripts only against those.

The same holds for `DB_arms_v2_spacing11_regen.json`, which *is* an of-record artifact: it carries
Finding 5's numbers at the spacing that finding requires. The methods companion's circular-shift figure is drawn
from `analysis/leadlag_calendar_recompute.json`, whose caption reports the cohort and replicate
count that figure actually uses.

## Run records

Superseded run records are not kept in this directory. They are retained outside the pipeline's
output tree, under `_archive/superseded_outputs/`, and are not published with the
artifact repository. Their numbers were produced under specifications this pipeline no longer uses,
so they should not be read alongside the files above; Git history is the record of how the current
specification was reached.
