# Family-Wide Population Key

**Every agent group used anywhere in this research line is defined here, once.** If a paper, a
supplement, a script or a figure caption names a population, this file says what it is, where it
comes from, how many agents it holds, and how to verify the count in one command.

**Standing rule.** Do not define a population inside a manuscript paragraph, a table cell or a
script docstring and leave it there. Add it here and cite it.

Counts verified 2026-08-30 against the files and artifacts named in each row. Companion to
`NOTATION-KEY.md`, which does the same job for symbols.

---

## 1. Layer 1: the curated local corpus

Per-agent behavioral files, one per agent, in
`research/Transformer-classifier-modelling/data/`. These predate the AGWA database work and are
the labelled ground truth the transformer classifier of the research line was trained and
evaluated on. `empirical-support/config.py` exposes them as `CLASSES`.

| Class | Glob | Files | What the label actually means |
|---|---|---:|---|
| Declared automation | `*.bot` | **177** | Agents that advertise themselves as automated and follow the published-crawler pattern. The compliant end of the strategy set. Hand-curated. |
| Verified humans | `*.human` | **200** | Hand-curated and individually verified. The control group of record. |
| Browser-labelled | `*.chrome` | **1,191** | **A user-agent claim, not a verified browser.** `config.py` glosses the class as "real browsers, legit but unverified". Automation presenting a Chrome user-agent lands here by construction, which is exactly why the set is a *mixed stream* and is disqualified as a control. |
| Potential humans | `*.pothuman` | **2,159** | The model-selected candidate pool from which the 200 verified humans were curated. Not verified. Used only as a larger sensitivity arm. |

**Distinct agents across all four classes: 3,527, not 3,727.** The four file counts sum to
3,727 because **all 200 verified humans also own a `<id>.pothuman` file** — they were curated
out of that pool and the twin file was never removed. No other pair of classes overlaps.

Verify:
```
cd research/Transformer-classifier-modelling/data
python -c "import glob,os;f=lambda p:{int(os.path.splitext(os.path.basename(x))[0]) for x in glob.glob(p)};h,pt=f('*.human'),f('*.pothuman');print(len(h),len(pt),len(h&pt))"
```

### 1.1 Reading the verified arm

Verified-human analyses read the 200 `*.human` ids directly, or use `roster_v2.csv` role
`verified_human`.

---

## 2. Layer 2: database-derived populations

### 2.1 The treatment arm — the criterion and the arm of record

**The sole analysable arm is `outputs/DB_arm_of_record_ids.csv`, 7,812 rows.**
`DB_arm_abusive_ids.csv` is the raw criterion output and a provenance intermediate; it is not an
arm and nothing quotes it.

| File | Rows | What it is |
|---|---:|---|
| `outputs/DB_arm_abusive_ids.csv` | 7,819 | the raw criterion output: honeypot share > 0.25, written by `abusive_arm()`. Provenance only |
| `outputs/DB_arm_of_record_ids.csv` | **7,812** | **the arm of record**, written by `arm_of_record()`: the criterion minus the 7 hand-verified humans. Every analysis and every figure reads this |

Median share 0.667. `arm_of_record()` asserts equality with `roster_v2.csv`'s `abusive_share`
role on every run and refuses to proceed if they diverge, so the two cannot drift apart again.

Why share and not presence: `url.u_honeypot` matches exploit-path **file names**, and the list
contains ordinary ones (`login`, `wp-login.php`, `admin`), so mere presence in the join is not
evidence of scanning — it flags 174 of the 200 hand-verified humans. The share filter cuts the
human false-positive rate to 3.5%.

### 2.2 Second-generation roster — `empirical-support/outputs/roster_v2.csv`, 10,081 agents

Built by `roster_v2.py`, which claims groups in the order verified_human → abusive_share →
declared → pothuman, first listed wins.

| Role | Count | Source set |
|---|---:|---|
| `abusive_share` | **7,812** | The 7,819 arm minus the 7 agents also in `.human` |
| `pothuman` | 1,899 | 2,159 minus the 200 verified humans minus 60 claimed by the arm |
| `verified_human` | **200** | `.human`, read directly, bypassing the §1.1 trap |
| `declared` | 170 | 177 minus the 7 also in the arm |

### 2.3 Derived panels

| Population | Count | Where |
|---|---:|---|
| Agents with an extracted daily series (the panel) | 36,029 | reported in D §7.1 |
| Cached per-agent parquet series on this machine | 49,352 | `empirical-support/cache/agents/*.parquet` |
| Labelled user agents in the corpus | 2.27 million | AGWA data descriptor |
| Requests / sites / years | 6.05 bn / 8,780 / 2019-2023 | AGWA data descriptor |

---

## 3. Overlaps: the three populations are **not** disjoint

Measured against the arm of record, `DB_arm_of_record_ids.csv` (7,812 agents):

| Overlap | Agents | Consequence |
|---|---:|---|
| arm of record ∩ verified humans | **0** | by construction: the seven agents meeting the criterion that are also hand-verified humans are assigned to the control arm |
| arm of record ∩ declared automation | **7** | the §7.7 abusive-versus-declared contrast is drawn across a 7-agent overlap |
| arm of record ∩ potential humans | **60** | sensitivity pool only; `pothuman` is model-selected, so it has no claim to outrank an automated criterion |

Response counts here are dominated by a few very large agents, so a response-weighted statistic is
sensitive to how a handful of agents are assigned in a way an agent-weighted one is not. Report
agent-weighted statistics, or state the placement.

**Pooled per-code ratios are not quotable from a sample.** Because a single agent can carry tens of
thousands of 401s or 403s, a 403:401 ratio computed from a sample is dominated by whichever large
agents the sample caught. Quote agent-level statistics from a census
(`db_status_histogram.py --n 0`), never a pooled ratio from a sample.

---

## 4. Who uses which population

| Population | Used for | Papers |
|---|---|---|
| Arm of record (7,812) | every arm-dependent finding: observability DiD, retreat, lead-lag, robots.txt, rotation | D §7.4-§7.8; imported as premises by A, B, C, E, F |
| Verified humans (200) | the control arm. **The usable-window count is per finding, not one number**: **150** at the 16-observation spacing Findings 2 and 3 use (`analysis/combined_numbers_of_record.json:did_combined.n_ctrl`, and `dose_split_recompute.json:"corrected...".verified.n_agents`); **149** on the balanced ±10 event-time panel (`analysis/event_time_study.json:verified.n_agents`); **151** at the complete-cache spacing-11 response analysis (`analysis/event_spacing_recompute.json:results.corrected_11.n_agents.verified`). Verify with the command below rather than quoting one of these across the board | D §7.4-§7.8 |
| Declared automation (177 curated; 170 in the disjoint roster) | compliant reference point; **87** usable roster-declared lead-lag agents (`analysis/leadlag_calendar_recompute.json:populations.declared.n`); **99** curated declared agents and **592** events at spacing 11 (`analysis/event_spacing_recompute.json:results.corrected_11.n_agents.declared, n_events.declared`). The input manifests admit all 170 and all 177 with zero input failures | D §7.3, §7.7 |
| Potential humans (2,159) | sensitivity arm; the alternative reference profile | D §7.4 robustness, MX §2.5 |
| Browser-labelled (1,191) | **score distribution only**, as the shape of the botness density the simulation triages over | D §7.1 (set aside as a control) → §8 head; supplement §S.5, §S.6.1, Figure S1 |
| Honeypot-positive (38,355) | the agents that requested at least one flagged exploit path: the pool the share criterion of §2.1 filters down to the arm. Provenance only, never a reported population | `outputs/DB_arm_census.json` `honeypot_join.positive`; MX §1.2 |
| `outputs/roster.csv` (35,836 `is_bot`) | **not a population.** A first-generation id listing retained because `db_arm_census.py` enumerates from it to produce the census row above. No current analysis selects on its role columns, and no reported population is defined from it | `outputs/roster.csv`; role definitions archived in `_archive/POPULATIONS-KEY-legacy.md` |

### 4.1 The botness density chain

The most-traced definition in the project, recorded end to end:

```
data/*.chrome (1,191 files)
  -> transformer classifier (botness_model.pt, bot_training_and_inference9.py)
  -> chromes_ranked_by_distance_to_human_centroid.csv   (1,191 rows, column avg_bot_prob)
  -> empirical-support/analysis_e_calibration.py  (E1)
  -> empirical-support/outputs/tables/E_botness_density.csv  (40 bins)
  -> supplement Figure S1, and the simulation's per-agent botness draw
     operating threshold beta = 0.234 from best_threshold.txt
```

Published values reproduce exactly: mean **0.4536** (quoted 0.454), **62.38%** above beta
(quoted 62%). `bot_score_area` is a different-scale diagnostic and
must not be substituted. Nothing outside the `.chrome` class has ever been scored: there is no
classifier output for `.bot`, `.human` or `.pothuman`, and none for the arm, whose agents have
no local files at all.

---

## 5. Traps that have already cost time

1. **"Browser-labelled" sounds verified and is not.** It is a user-agent claim. The set is a
   mixed stream by construction, which is why its density is a *contamination* density.
2. **7,819 is the raw criterion output; 7,812 is the sole arm of record.**
3. **Honeypot presence is not abuse** (§2.1). The presence criterion puts 174 of the 200
   hand-verified humans into the treatment arm, which is why the arm is defined by honeypot
   *share* above 0.25 and not by presence.
4. **The curated classes sum to 3,727 files but 3,527 agents** (§1).
5. **Response-lag values depend on the indexing, and it must be stated:** 4 days abusive,
   **8 days verified humans on calendar indexing / 9 on active observations**, 2 days declared
   (`analysis/leadlag_calendar_recompute.json:populations.*.calendar.tau_A, active.tau_A`). The
   8-versus-9 split is one run under two indexings, not two runs disagreeing. Calendar is primary,
   because that is the indexing the circular-shift null is calibrated on.

6. **A finding aggregated from a pre-built per-agent table inherits that table's date.** Re-running
   an aggregator does not re-specify the table beneath it; the upstream table has to be
   regenerated. **Check a table's mtime before trusting a number aggregated from it.**

7. **Lead-lag run of record: `papers/D-adversarial-friction/analysis/leadlag_calendar_recompute.json`**
   — 999 circular shifts, roster `roster_v2.csv` role `abusive_share`, usable cohorts **913
   abusive, 121 verified, 87 declared, 648 sensitivity**
   (`populations.*.n`).

8. **Event spacing is per finding, not one number.** Sixteen observations for the estimators that
   use the $[-10,-6]$ baseline, because events 11 apart reuse the baseline's own observations;
   eleven for the estimators that use adjacent $\pm 5$ windows, where 11 is the disjointness
   condition. Both are correct, for different window geometries.

---

## 6. Layer 3: the WAF audit corpus (Paper H)

**A separate corpus from the AGWA panel. Nothing in §1 to §4 applies to it, and no population
defined above appears in it.** Its populations, traps and counts are maintained with that paper at
`papers/H-waf-blocklist-deterrence/POPULATIONS-KEY-H.md`, so Paper D's reviewer package carries
only the corpus this paper measures.
