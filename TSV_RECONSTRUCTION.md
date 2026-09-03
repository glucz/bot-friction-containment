# Reconstructing the 14-feature classifier TSVs from the AGWA DB

**Status: validated on 2026-08-16.** The generator that originally wrote
`Transformer-classifier-modelling/data/*.{bot,human,chrome,pothuman}` is **not** in
the research tree. This document records the conventions recovered by recomputing
known TSVs from the DB and comparing, so TSVs can be built for agents that have
none (e.g. the 1,810 ever-blocked agents of Paper D's 7,812 treatment arm, which
the classifier has never scored and which were never in its training set).

Validation: on `172877.human` the grid and **all 15 columns** reproduce to the TSV's
6-digit rounding. The html bucket was attributed on `430513.bot`. **Caveat: the grid
rule does not generalise to long-lived agents - see below.**

## Column recipe

Source table `hits` (`h_a_id, h_ts, h_status, h_u_id, h_i_id, h_d_id, h_ccode`),
joined to `url` (`u_extension`) only for the content-type buckets.

| # | Column | Rule | Verdict |
|---|---|---|---|
| 1 | `day` | calendar-day offset from the agent's first hit day, **omitting panel-wide-zero days** (see below) | EXACT (row count 114 = 114) |
| 2 | `traffic_hits` | `COUNT(*)` per day | **EXACT** (0.0) |
| 3 | `404_pct` | `SUM(h_status=404)/hits` | **EXACT** (0.0) |
| 4 | `200_pct` | `SUM(h_status=200)/hits` | **EXACT** (4.2e-7) |
| 5 | `robots_pct` | `SUM(h_u_id=64)/hits` | **EXACT** (0.0) |
| 6 | `img_pct` | **always 0** | 0 nonzero days in 0/377 training files |
| 7 | `html_pct` | `u_extension` in **{html, htm}** / hits | **EXACT** (1.8e-7) |
| 8 | `js_pct` | **always 0** | 0 nonzero days in 0/377 training files |
| 9 | `php_pct` | `u_extension='php'`/hits | **EXACT** (2e-8) |
| 10 | `folder_pct` | `u_extension IS NULL`/hits | **EXACT** (3.2e-7) |
| 11 | `ip_entropy` | `COUNT(DISTINCT COALESCE(h_i_id,-1))/hits` | **EXACT** (3.3e-7) |
| 12 | `domain_entropy` | `COUNT(DISTINCT COALESCE(h_d_id,-1))/hits` | **EXACT** (3.3e-7) |
| 13 | `country_entropy` | `COUNT(DISTINCT h_ccode)/hits` | **EXACT** (3.3e-7) |
| 14 | `url_entropy` | `COUNT(DISTINCT h_u_id)/hits` | **EXACT** (4.8e-7) |
| 15 | `husource` | `SUM(h_ccode='HU')/hits` | **EXACT** (4.6e-7) |

Residuals of ~5e-7 are the TSV's 6-significant-digit rounding, not disagreement.

### The "*_entropy" columns are NOT entropies

They are **distinct-count / hits ratios**. Proof: `1032111909.human` day 0 has
7 hits and all four columns = 0.142857 = 1/7; day 1 has 8 hits and all four =
0.125 = 1/8. Confirmed exactly for `url_entropy` (4.8e-7 over 55 nonzero days).
This matters: no per-day histogram queries are needed, and `_EXTRACT_FULL` in
`db_source.py` **already computes all four numerators**.

### `husource`

Per-day Hungarian share. The author's account is the `ip` table's `i_ccode`
(each IP geolocated individually); the value is denormalized onto `hits.h_ccode`,
and `SUM(h_ccode='HU')/hits` reproduces the column exactly. **Use `h_ccode`** — it
avoids joining `ip` entirely, and the `ip` join returned 0 (its `i_id` is not a
plain per-IP key; it carries `i_agent`/`i_year`/`i_month`).

### The day grid is NOT fully recovered (corrected 2026-08-16)

The `day` column is a calendar offset from the agent's first hit day, and rows are
omitted for some zero-hit days. An "omit panel-wide-zero days" rule reproduces
`172877.human` **exactly** (114 = 114), and the 234 panel-zero days over
2019-03-10..2023-03-05 are confirmed by reading **all 49,352** cached agents
(`_panel_zero_days_full.pkl`; the 6,000-agent sample gave the identical set, so the
mask itself is right). But the rule does **not** generalise:

| agent | TSV rows | masked grid | DB active days |
|---|---|---|---|
| `172877.human` | 114 | **114** | 55 |
| `81142.bot` | 837 | 845 | 425 |
| `430513.bot` | 374 | 402 | 400 |
| `158923.bot` | 1175 | 1110 | 558 |

It errs in both directions, so it is not merely a mask that is too aggressive.

**The tail truncation is explained (author, 2026-08-16): the series ends at the last
row that can START a complete 30-day classifier window**, so the trailing 29 rows are
dropped. This is why `430513` and `81142` each show exactly **28** fewer active days
than the DB (400 vs 372, 425 vs 397). Consequence worth knowing: **published b(t)
curves stop ~29 days before an agent's last activity**, at both ends in fact, since
`skiprows=1` also removes day 0.

**There is no date-reliability exclusion to apply.** An earlier concern that the last
~400 days carried wrong timestamps (server clock reset) is **withdrawn**: the author
confirms the unreliable section was already deleted from the `hits` table, so the
current DB span is clean and the pipeline's absence of a date cutoff is correct.

**Why this does not block the intended measurement.** The planned use is a
*within-agent* event study: does b fall after a 401/403/429 event, differenced
against controls measured on the same grid. That needs internal consistency, not
identity with the 2025 grid. Absolute comparability of arm b with the *published*
chrome b would require the original grid, so do not pool the two; compare
within-arm and against controls regenerated the same way.

**Recommended rule for generated TSVs:** every calendar day from the agent's first
to last hit day, minus the 234 confirmed panel-zero days. Document it and keep it
identical across arm and control agents.

## Two quirks of the published pipeline, found while validating

1. **`parse_user_agent_file()` passes `skiprows=1`** on files whose first line is
   *data*, not a header. Every agent's **day 0 is silently dropped** before the
   model ever sees it. Harmless for long series, but it means published b(t)
   curves start at day 1.
2. **`img_pct` and `js_pct` are identically zero in all 377 training files.** The
   classifier was trained with two of its 14 inputs constant, so it is effectively
   a 12-feature model and its weights on those two channels never saw a nonzero
   value.

## Both conventions are now resolved

**NULL counts as one distinct value.** `h_i_id` and `h_d_id` are NULL on some days
(`h_ccode` never is, which is why `country_entropy` matched from the start). SQL's
`COUNT(DISTINCT ...)` skips NULLs and returns 0; the generator treated NULL as a
value and returned 1. On `172877` two days had 11 and 9 all-NULL-IP hits and the
TSV reads 0.090909 = 1/11 and 0.111111 = 1/9. Use
`COUNT(DISTINCT COALESCE(h_i_id,-1))`; that makes both columns EXACT.

**The html bucket is {html, htm}.** Attributed on `430513.bot` over 374 aligned
rows (126 nonzero): `html+htm` maxdiff 1.8e-7, versus 0.0069 for `html` alone and
0.18 for `htm` alone.

## Remaining operational items (not conventions)

1. **Build the panel-zero mask from ALL cached agents, not a sample.**
   `_panel_zero_days.pkl` came from 6,000 of 49,352 agents and appears to contain
   false positives: for `158923.bot` the reconstructed grid came out 1,110 rows
   against a 1,175-row TSV, i.e. the mask removed days the generator kept. Rebuild
   over the full cache before generating arm TSVs. (Agents whose grid matched
   exactly are unaffected; this only bites where a day was active for rare agents.)
2. **DB limits on larger agents.** The extension join and the DISTINCT aggregation
   both fail above a certain size: `339875699` hit `max_statement_time` (error 1969)
   and `334458593`/`437868` hit TokuDB error 1030/1152. `db_source.py` documents the
   same wall and falls back to BASIC. For the arm, chunk by date range or issue the
   DISTINCT channels as separate per-agent queries. **This is the part that needs
   the extra hardware / query budget** - not the entropies, which turned out to be
   plain distinct-count ratios.

## Why this is worth doing

The TSVs carry the 14 classifier features but **no 401/403/429 channel**; the
parquet cache carries the block channel but only 4 of the 14 features. That
disjointness is why Finding 1 uses a non-200 spike rule while Findings 2-6 use
401/403/429, and why the classifier's own score has never been run as an event
outcome. Rebuilding for the ever-blocked arm agents produces, for the first time,
one frame holding both — enabling a b(t) DiD on the real event rule, on a
population that is genuinely out of sample for the model.

## Why the arm cannot be scored by this classifier (2026-08-16)

The reconstruction works: 1,781 of 1,787 ever-blocked arm agents were built and the
extractor reproduces a published TSV exactly. **But the classifier cannot be applied
to them**, for a reason that is a property of the data, not of the extraction.

| | training (.bot+.human) | ever-blocked arm |
|---|---|---|
| agents | 377 | 1,673 built & measurable |
| median hits per ACTIVE day | 72 | **2** |
| median `ip_ent` | 0.0325 | **1.0000** |
| median `dom_ent` | 0.0327 | **1.0000** |
| median `url_ent` | 0.4208 | **1.0000** |
| agent-days where `ip_ent` == 1/hits | 57.2% | 79.8% |
| total hits, median per agent | 55,819 | **289** |

1. **The normalized ratios saturate.** `*_ent` = distinct/total. On 57% of *training*
   agent-days the numerator is already 1, so the feature equals 1/hits - pure volume.
   Within training, rho(log median daily hits, `ip_ent`) = **-0.757**, `dom_ent`
   -0.766. Normalizing by total does not remove volume dependence; it converts it
   into a ceiling.
2. **At 2 hits/active day the ceiling is hit.** distinct/total can only be 0.5 or 1.0
   at 2 hits, and 1.0 at 1 hit. 72% of arm agents lie ABOVE the training 5-95% band
   for `ip_ent`. Three of the four ratio features are pinned at 1.0.
3. **No control has common support.** 94% of the arm falls below the training 5th
   percentile of volume; 2 of 181 verified humans and 39 of 1,856 pothumans fall
   inside the arm's 5-95% volume band. This cannot be fixed by choosing a different
   control group.
4. **The model has never been run in this regime.** Of the 1,116 chrome agents with
   published scores, **zero** fall in the arm's volume range. The mild in-range drift
   (rho = +0.107 between log volume and published b, mean b 0.376 -> 0.493 across
   volume quintiles) neither licenses nor rules out extrapolation two orders of
   magnitude below the observed range.

**Conclusion.** A b(t) event study on the ever-blocked arm is not a measurement, it is
an extrapolation of a saturated feature space. The defensible statements are: the
behavioral observability score O and the transformer agree directionally where both
exist (AUC 0.717, `DB_obs_transformer_crosscheck.json`), and a classifier-score event
study is infeasible on the blocked population because that population's traffic is too
sparse for the classifier's feature construction. The 1,781 rebuilt TSVs and the
validated extractor remain useful for any analysis that does not require the model.
