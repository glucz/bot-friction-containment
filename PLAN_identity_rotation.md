# Analysis plan — Identity rotation as a third escape channel

**Feeds:** this paper (Computers & Security), new **Finding 6** (subsection after §7.7), plus an
update to §7.8 (Threats to Validity) and a corrected "true retreat fraction" in §7.4/§7.9.
**Status:** planned. No results yet. Implement as `db_analysis_identity_rotation.py`.

## 1. The gap this closes

this paper §3.3 lists `s_R` as "retreat **or target switching**", but §7.2 operationalizes retreat
purely as a post-block volume drop >50% on the **same `a_id`**. Because `a_id` *is* the user-agent
string (`agent.a_name`), an operator who abandons agent A and respawns under a fresh UA → fresh
`a_id` produces:

- **A**: volume → 0 → classified `retreat` (s_R). But the threat did not retreat; it *moved*.
- **B**: fresh, no honeypot history → enters the roster as `ctrl_hpneg` (control) until it trips a
  honeypot.

So the per-`a_id` design conflates **three** distinct outcomes into a binary:

| | identity | behavior | what the defender loses |
|---|---|---|---|
| retreat (s_R) | persists | disappears | nothing (threat gone) |
| stealth (s_E) | persists | reshaped | the **sensor** (O(t) degrades) |
| **respawn** *(unmeasured)* | **new `a_id`** | transferred | the **reputation/history** (sensor *resets*) |

Respawn is currently absorbed into `retreat`. It is a *second observability-loss channel*,
arguably stronger than stealth (the defender's per-identity state is reset, not just degraded),
and it strengthens the overdeterrence thesis: aggressive blocking that "kills" an agent may be
relabeling, not suppressing.

## 2. Hypothesis (the conservation signature)

After A is blocked and its botness decays toward the control centroid, there exists a partner B
whose botness moves **from the control region toward the bot centroid by a comparable magnitude**,
**time-locked** to A's block, and **linked by shared infrastructure**. Formally, for a matched
pair (A, B):

- ΔA_botness < 0 (A drifts toward normal),
- ΔB_botness > 0 (B drifts toward bot),
- |ΔB_botness| ≈ |ΔA_botness| (magnitude-matched transfer),
- B's "birth" lies within Δ days of A's "death" (temporal locking),
- A and B share infrastructure (the bridge, §4).

Systematic, anti-correlated, magnitude-matched, infra-linked pairs ⇒ identity rotation is real and
quantifiable.

## 3. Operational definitions

Reuse the existing daily cache (`date, hits, c200, c404, cblock, c429, robots, n_ip, n_dom,
n_ctry, n_url, p_block, robots_rate, p404, resistance`) and the roster (`is_bot`, `is_control`).

- **Botness coordinate** `b(t)`: signed projection of the behavior vector
  `[p404, robots_rate, log1p(hits)]` onto the axis from the control centroid `μ_H` to the bot
  centroid `μ_B` (both estimated from the roster's confirmed groups). `b > 0` = bot-like.
  *Reuse* `db_analysis_observability._behavior_matrix` and the `μ_H`/`σ_H` pass; add `μ_B`.
- **Death event (A):** an `is_bot` agent with a block event (`_block_events`) after which volume
  drops >50% (the current `retreat` rule) **and stays near-zero** for the following `DEATH_W` (e.g.
  20) active days — i.e. a *persistent* exit, not a transient lull. Record the calendar date `t_A`.
- **Birth event (B):** any `a_id` whose first substantial activity (first day above a volume floor)
  falls in `[t_A − ε, t_A + Δ]` (ε small for overlap handoffs; Δ ≈ 14–30 d).
- **ΔA_botness:** `b` post-window − pre-window around A's terminal block (the existing ±5 d windows).
- **ΔB_botness:** `b` over B's first `BIRTH_W` active days − the control-centroid baseline (B has no
  pre-history, so the baseline is `μ_H`-relative).

## 4. The infrastructure bridge (candidate generation)

The UA (`a_id`) changes; the bridge is whatever persists. **None of this is in the daily cache** —
it needs a new extraction from `ip` / `url2agent_honeypot` / `hackip` (all small and indexed;
the whitelist query in `roster.py` already proves the `ip` table is queryable). For each death A, candidate successors B are agents that:

1. **share ≥1 IP**: `ip.i_name` overlap (`SELECT i_agent FROM ip WHERE i_name IN (A's IPs)`), or
2. **share known-bad IPs**: overlap on `hackip.h_name` (with `h_date` near `t_A`), or
3. **hit the same honeypot URLs**: `url2agent_honeypot.uh_u_id` overlap (the strongest signal — a
   fresh UA hammering the *same decoy paths* A hit), or
4. **target the same rare domains** (`hits.h_d_id` overlap restricted to low-popularity domains).

Rank candidates by bridge strength (IP-overlap count, honeypot-URL overlap, domain rarity).

## 5. Test statistics

Per matched (A, B) pair, and aggregated:

1. **Transfer rate:** fraction of "deaths" with ≥1 qualifying bridged birth in the window.
2. **Conservation:** regression of ΔB_botness on −ΔA_botness; slope ≈ 1 and high R² ⇒ magnitude-matched
   transfer. Report slope CI (cluster bootstrap on operators/IP-clusters).
3. **Temporal locking:** distribution of `t_B − t_A`; concentration near 0 vs uniform.
4. **Corrected retreat fraction:** `true_retreat = retreat − respawn`; recompute §7.4's "only 10.7%
   retreat" net of detected rotation (this becomes a *lower* number → strengthens §7.4).

## 6. Verification / falsification test (finalized)

**Step 0 — feasibility probe (run first, gates everything).** Count the death set (persistent
post-block retreats among honeypot-confirmed bots) with one cheap query. The count decides *both*
whether a result is possible (too small → report the null finding cheaply) and the runtime bound
(per-death = a few indexed lookups on `ip`/`url2agent_honeypot` → `db_battery`-scale overnight
batch, guaranteed to terminate). Build the full pipeline only if the count clears a pre-set floor.

**Locked design decisions** (fixed before looking — avoids garden-of-forking-paths):

- **Bridge = dual reporting.** *Precision core:* IP overlap **AND** honeypot-URL (`uh_u_id`) overlap
  → the claim. *Recall envelope:* any shared IP → the upper bound. Report both; the null is applied
  to each.
- **Bootstrap cluster unit = /24** (truncated `i_name`; no external ASN data → finite-time). Where
  it overlaps the IP-bridge it only *widens* CIs (A and B fall in one cluster) → conservative.
  Optional robustness: honeypot-target-fingerprint clustering.
- **Δ window = 2·τ_A ≈ 14 days** (anchored to the measured response lag of §7.6, not fished).
- Botness axis, bridge rule, and Δ pre-registered before inspection.

**Falsification battery — one null per confound** (same philosophy as the lead-lag shuffle null):

| Confound | Spurious mechanism | Null that kills it |
|---|---|---|
| Coincidence | high churn + ubiquitous IP-sharing (NAT/cloud) | **bridge-permutation:** hold death/birth sets + timing fixed, permute the *link* across agents; real transfer rate & conservation slope must beat this null |
| Common cause | A dies, B ramps because the same site/season drives both | **honeypot-URL bridge** + rare-domain restriction (common cause needn't hit the *same decoy paths*); **placebo-death null** (start from non-block collapses → bridged-birth rate should be lower) |
| Reverse causation | B pre-existed A's death | **reverse-time null** (births *before* the block are not handoffs); require B's birth at/after `t_A` |
| Regression to the mean | matching manufactures slope ≈ 1 from bounded extremes | **pairing-permutation:** match A to a *random* risen B (not bridged); slope ≈ 1 there ⇒ not specific |

**Positive control (sensitivity calibration).** Plant known handoffs: take a confirmed bot's series,
split at a date, relabel the second half as a new `a_id`, and verify the detector recovers the
planted pair (mirrors the lead-lag `_selftest` synthetic +3-lag check; an agent carrying a Chrome
user-agent while repeatedly hitting honeypot decoys is a real disguised-bot template). Negative nulls (reject)
+ positive control (recover) ⇒ calibrated sensitivity/specificity, not just a p-value.

## 7. Known limit (state plainly in §7.8)

If the operator rotates UA **and** IP simultaneously, no bridge survives → the handoff is invisible
to this data. Detected rotation is therefore a **lower bound** on the true respawn rate. This bias is
conservative for the overdeterrence thesis (true suppression is even weaker than measured).

## 8. Outputs

- `db_analysis_identity_rotation.py` (streaming over the cache + targeted `ip`/honeypot queries).
- `tables/DB_rotation_pairs.csv`, `tables/DB_rotation_summary.json`,
  `figures/DB_rotation_conservation.png` (ΔB vs −ΔA scatter + slope), `DB_rotation_SUMMARY.md`.
- this paper edits: new **Finding 6** subsection; corrected retreat fraction in §7.4/§7.9; §7.8
  threat upgraded from static (label contamination) to dynamic (identity rotation) with the
  lower-bound caveat.

## 9. RESULTS (run 2026-06-27) — BOUNDED NULL

Ran on the live DB (49,080-agent roster; **1,059** persistent post-block deaths). The **IP
bridge proved infeasible** — the `ip` table (52M rows) has no usable index on `i_agent`
(non-leftmost in its only index → TokuDB error 1152), so only the **honeypot-URL bridge** was
used. Successors are therefore themselves confirmed bots (intra-fleet rotation); rotation into
fresh / never-honeypotted identities is untestable in *this* run → every number below is a
**LOWER BOUND**. (Superseded by §10: the index was added the next day and the IP bridge ran.)

**Detector validated** (positive control, n=300 planted handoffs): ramp dB_v **+3.09**, collapse
dA_v **−2.82**, botness-transfer corr **0.77** (bias +0.006), temporal placebo **0.0**, 100% of
ramps exceed placebo. The statistic reliably catches a true handoff.

**Real data — no robust rotation signal:**
- **Precision core** (rare shared decoy — the only credible same-operator link): only
  **18/1,059 deaths (1.7%)** have a successor; ramp **+0.16, 95% CI [−0.11, 0.42] (n.s.)**;
  link-permutation null **+0.13** → real ≈ null (no link-specificity).
- **Recall envelope** (any shared decoy): ramp +0.30 [0.20, 0.41] *looks* significant, but the
  link-permutation null is **+0.175** → ~60% is generic successor growth; the link-specific
  excess (+0.12) is small and a shared *popular* decoy is not credible operator evidence.
- **Placebo-death** (organic non-block collapses): core ramp **+0.05 (n.s.)**, 0.85% rate → the
  (already null) core signal is not friction-specific.

**Conclusion.** Within the detectable scope, **no robust evidence of intra-fleet identity
rotation**: credible-bridge rate ~1–2% and not significant; broader correlation is mostly generic
growth. Because the detector demonstrably works, this is a genuine null, not a power failure.
UA+IP co-rotation and fresh-identity respawn remain invisible on this schema.

**Implication for this paper — strengthens, not overturns, §7.4.** The "only ~11% retreat" figure is
**not materially inflated by detectable fleet-rotation**, so reading retreat as (mostly) genuine
withdrawal holds. Reframe Finding 6 as a *falsification that survived*: "we tested whether retreat
is masked identity rotation and found no detectable evidence (≤1.7%, n.s.); fresh-identity respawn
is an acknowledged, untestable residual." Demote respawn from a headline Finding to a **§7.8
threats-to-validity** paragraph (bounded). Artifacts: `DB_rotation_summary.json`,
`DB_rotation_selftest.json`, `DB_rotation_placebo.json`, `figures/DB_rotation_conservation.png`.

## 10. RESULTS — IP BRIDGE (run 2026-06-28, after `idx_ip_agent_time` added)

Adding `idx_ip_agent_time (i_agent, i_year, i_month, i_name)` to the `ip` table made the IP bridge feasible
(A's IP lookup 0.07–0.2 s; self-join 0.1–25 s/death). This bridge reaches **any** agent sharing
the dead bot's non-proxy IPs — incl. fresh / non-honeypot identities — so it **CLOSES the channel
the honeypot bridge could not see**. (The ~17 deaths initially skipped on the 30 s
cap — the most IP-rich/evasive bots — were re-run at a 5-min timeout: recovered, and they
**co-collapse identically** (recovered successor Δvol −0.34, core −0.34), so the result is not an
artifact of excluding the most evasive operators. Merged totals: 885/1059 deaths with a successor,
core Δvol −0.326 [95% CI −0.44, −0.22].)

**Reach:** 862/1059 deaths (81%) have an IP-bridged successor; core (≥50% IP overlap, ≥3 IPs)
covers 371 deaths (35%) — far higher than the honeypot bridge.

**Result: the respawn hypothesis is REFUTED — the opposite holds.** IP-co-located successors do
NOT ramp up as the bot dies; they **COLLAPSE WITH IT**:
- core: successor Δvol **−0.33 [95% CI −0.44, −0.21]** (significant, NEGATIVE); link-permutation
  null **+0.22**; per-pair placebo **+0.25** → real is far below both (ramp−placebo = −0.58). Around
  *unrelated* deaths these agents are growing (+0.22); around *their linked bot's* death they
  collapse (−0.33) — strongly time-locked and against-trend.
- envelope: Δvol −0.10 [−0.19, −0.02], same direction.
- successor botness rises +0.61 as volume falls — secondary/ambiguous (likely residual-composition
  in the b-axis); the volume co-death is the primary, robust signal.
- positive control already reports planted ramps as **+3.09**, so −0.33 is a faithful co-collapse,
  not detector bias.

**Interpretation.** Friction on shared infrastructure (IP ranges) produces **correlated
multi-identity collapse, not displacement/respawn**. Block a honeypot bot's IPs and the other
agents on those IPs go down with it. No whack-a-mole handoff to a fresh same-infrastructure identity.

**Implication for this paper — strengthens §7.4 twice.** (1) Retreat is not masked respawn (neither
bridge finds a handoff). (2) NEW positive result: IP-level friction is *effective* against
co-located fleets — they die together — which is containment-positive, with a collateral-risk
caveat (co-tenants on that infrastructure are taken down too). A stronger Finding 6 than a handoff
would have been: **friction contains rather than displaces.** Residual blind spot: operators who
rotate UA *and* migrate to entirely new IPs stay invisible.

Artifacts: `DB_rotation_summary_ip.json`, `DB_rotation_SUMMARY_ip.md`,
`figures/DB_rotation_conservation_ip.png`, `tables/DB_rotation_ip_pairs.csv`.
