"""Field-to-surface manifest: every quoted number, its artifact field, and every surface.

Each row names an artifact field, the string it must render to, and the surfaces that must
carry it. A surface that does not contain the current rendering is reported, whatever the
reason. Four further contracts sit alongside the row check:

  * **superseded renderings** are rejected wherever they survive. Presence of a current value
    and absence of the value it replaced are different questions, and only the first is asked
    by a row. `build_obsolete_register.py` generates the forbidden set from the archived
    artifacts; matching is digit-boundary aware, and any form that is still current for a
    different field is dropped rather than banned.
  * **the Section 8.1 table** is parsed by row label and column header and compared cell by
    cell against its CSV, because a page-wide token count cannot see two cells swapped.
  * **the prose policy ranking** is read as an ordered list and compared with ranking_main.
  * **located values** must appear in the paragraph that interprets them, and an anchor that
    matches twice is an error rather than a silent choice of the first match.

Figures are hashed against the artifact that produced them and checked for citation, so a
stale render is caught even when its caption is correct.

What a green run does NOT prove: a cell may be coarsened and still round correctly; a located
value can be sheltered by a correct twin in the same paragraph; a compensated occurrence-count
edit passes; and a contradictory restatement outside the parsed sentence is not covered.

Read-only. Usage: python numbers_manifest.py [--json out.json] [--relearn]
"""
from __future__ import annotations

import sys

# The diagnostics carry U+2212, which the Windows console codepage cannot encode: a
# release gate that dies on its own output is not a gate. Reconfigure rather than
# relying on the caller to set PYTHONIOENCODING.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):   # not a real tty, or already wrapped
        pass

import argparse
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
ROOT = PAPER.parents[1]

import sys as _sys  # noqa: E402
_sys.path.insert(0, str(HERE))
from _layout import data_root  # noqa: E402

# The artifacts sit under `empirical-support/` in the source tree and beside this package in the
# published bundle. Assuming the source layout made every rotation, simulation and branch field
# read as "field not found" in the bundle, so the documented checker reported a clean release as
# two hundred defects.
SUP = data_root()

M = "−"


def _load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _load_csv(p: Path):
    """A row-keyed CSV becomes {first column value: {column: number}} so _dig can walk it.

    Section 8.1's table is generated as CSV, not JSON, which is why its cells sat outside the
    manifest for six rounds while every JSON-backed number was covered.
    """
    if not p.exists():
        return None
    import csv
    with p.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return None
    key = next(iter(rows[0]))
    out = {}
    for r in rows:
        out[r[key]] = {k: (float(v) if v not in ("", None) and
                           re.fullmatch(r"-?\d+(\.\d+)?", v) else v)
                       for k, v in r.items()}
    return out


def _dig(obj, path):
    """Walk a dotted path. A segment starting with '~' matches a key by prefix.

    A path may also be given as a list of segments, which is the only way to reach a key
    that itself contains a dot - the Section 8 policy rows (`static_graded_ft1.00`) and the
    ceiling-grid tolerance keys (`0.001`) both do.
    """
    cur = obj
    for part in (path.split(".") if isinstance(path, str) else path):
        if cur is None:
            return None
        if part.startswith("~"):
            key = next((k for k in cur if k.startswith(part[1:])), None)
            cur = cur.get(key) if key else None
        elif isinstance(cur, list):
            cur = cur[int(part)]
        else:
            cur = cur.get(part)
    return cur


# --- renderers: how a field becomes the string a manuscript actually contains ----------
def sig3(x):     return f"{M}{abs(x):.3f}" if x < 0 else f"+{x:.3f}"
def bare3(x):    return f"{M}{abs(x):.3f}" if x < 0 else f"{x:.3f}"
def bare4(x):    return f"{M}{abs(x):.4f}" if x < 0 else f"{x:.4f}"
def sig4(x):     return f"{M}{abs(x):.4f}" if x < 0 else f"+{x:.4f}"
def bare2(x):    return f"{M}{abs(x):.2f}" if x < 0 else f"{x:.2f}"
# The highlights quote the grid range as magnitudes ("0.27 to 0.76"), sign carried by
# the sentence rather than the number.
def absbare2(x): return f"{abs(x):.2f}"
def boundary_phrase_abusive(pops):
    p = pops["abusive"]
    return f"{p['n_boundary_responders']} of {p['n_responders']}"


def boundary_phrase_verified(pops):
    p = pops["verified"]
    return f"{p['n_boundary_responders']} of {p['n_responders']}"


def km_median_phrase(pops):
    """Both medians in the order the sentence states them, so neither can move alone."""
    return (f"**{pops['abusive']['median_km']:.0f} days** for abusive automation and "
            f"**{pops['verified']['median_km']:.0f} days** for verified humans")


def cohort_phrase(blk):
    """The headline cohort as the sentence states it: treated against control, in that order."""
    return f"{blk['n_bot']:,} against {blk['n_ctrl']:,}"


def percentile_phrase(blk):
    """The balanced analogue's percentile, which must not trade places with the headline's."""
    return f"{round(blk['percentile_in_grid'])}th percentile"


def grid_range_phrase(summary):
    """The range as the manuscript must state it: smallest magnitude first."""
    lo, hi = abs(summary["max"]), abs(summary["min"])
    return f"{lo:.2f} to {hi:.2f}"
# The abstract writes an ASCII hyphen while the body writes U+2212. That difference is why a
# superseded headline sat in the abstract through two rounds: every row rendered the minus
# the body uses, so nothing a row produced could ever match the abstract's text.
def ascii2(x):   return f"-{abs(x):.2f}" if x < 0 else f"{x:.2f}"
def pct2(x):     return f"{x:.2f}%"
# Already in percent units, quoted to one decimal in prose and two in a table.
def pctp1(x):    return f"{x:.1f}%"
def pct1(x):     return f"{x*100:.1f}%"
def dec1(x):     return f"{x:.1f}"
def dec2(x):     return f"{x:.2f}"
def countr(x):   return f"{round(x):,}"
def pctup1(x):   return f"+{x:.1f}%"
def count(x):    return f"{int(x):,}" if x >= 1000 else str(int(x))
def plain(x):    return str(int(x)) if float(x).is_integer() else str(x)
def pp1(x):      return f"{M}{abs(x)*100:.1f}" if x < 0 else f"+{x*100:.1f}"

def any_of(*renderers):
    """A field may legitimately render more than one way on different surfaces.

    The dose gradients are quoted as a proportion in the tables and as percentage points
    in the prose. Demanding one form made the manifest report a correct sentence as stale,
    which is how a checker trains its owner to ignore it.
    """
    def render(x):
        return [r(x) for r in renderers]
    return render



ART = {
    "combined":   _load(HERE / "combined_numbers_of_record.json"),
    "grid":       _load(HERE / "spec_grid_and_window_free.json"),
    "dose":       _load(HERE / "dose_split_of_record.json"),
    "disap":      _load(HERE / "disappearance_sensitivity.json"),
    "etime":      _load(HERE / "event_time_study.json"),
    "ranks":      _load(HERE / "derived_ranks_and_monthly.json"),
    "counts":     _load(HERE / "event_counts_by_spacing.json"),
    "retreatcut": _load(SUP / "outputs" / "DB_retreat_cut_sensitivity_sp11.json"),
    "rot_ip":     _load(SUP / "outputs" / "DB_rotation_summary_v5_ip.json"),
    "rot_decoy":  _load(SUP / "outputs" / "DB_rotation_summary_v5.json"),
    "rot_self":   _load(SUP / "outputs" / "DB_rotation_selftest_v5.json"),
    "rot_disap":  _load(SUP / "outputs" / "DB_rotation_summary_disap_ip.json"),
    "decoy_part": _load(HERE / "decoy_specificity_partition.json"),
    "leadlag":    _load(HERE / "leadlag_calendar_recompute.json"),
    "sim_head":   _load(SUP / "outputs" / "SIM_headline.json"),
    "sim_grid":   _load(SUP / "outputs" / "SIM_ceiling_grid.json"),
    "sim_osc":    _load(SUP / "outputs" / "SIM_oscillation_cost.json"),
    "sim_policy": _load_csv(SUP / "outputs" / "tables" / "SIM_policy_comparison.csv"),
    "selftest":   _load(SUP / "outputs" / "DB_rotation_selftest_v5.json"),
    "robots":     _load(HERE / "robots_engagement.json"),
    "censor":     _load(HERE / "leadlag_censoring.json"),
    "espace":     _load(HERE / "event_spacing_recompute.json"),
    "contain":    _load(HERE / "containment_shares.json"),
    "summary":    _load(SUP / "outputs" / "summary.json"),
    "cadence":    _load(HERE / "theorem5_cadence.json"),
    "branch":     _load(SUP / "outputs" / "DB_event_branch_v2.json"),
}

D   = "draft/D-v53.md"
S   = "draft/D-v53-supplement.md"
MX  = "draft/MX-v40.md"
H   = "draft/D-v53-highlights.md"
TOML = "frictionlab/frictionlab/calibrations/agwa_d.toml"
PARM = "frictionlab/docs/parameters.md"

# Renderings that are known to be superseded. A field is only clean when its current form is
# present AND none of these survives on the same surface. This is what makes the check
# location-independent without needing line anchors.
OBSOLETE = {
    "rotation deaths":      ("164 agents", "164 deaths", "260 agents", "260 deaths"),
    "successor dvol":       ("−0.754", "-0.754"),
    "successor dbotness":   ("+1.79,", "+1.79 "),
    "core pairs":           ("1,189",),
    "core deaths":          ("114 deaths",),
    "headline DiD":         ("−0.442", "−0.316"),
    "disappearance rate":   ("0.66%",),
    "grid size":            ("144 spec", "144 window"),
}

# Superseded renderings, generated by build_obsolete_register.py from the archived artifacts.
# Presence of a current value and absence of the value it replaced are different questions, and
# only the first was ever asked: that is how a pass reported 0 problems while the previous
# version's numbers were still scattered through the documents.
_OBSOLETE_REGISTER = HERE / "numbers_manifest_obsolete.json"
OBSOLETE_BY_SURFACE: dict[str, dict[str, tuple]] = {}
if _OBSOLETE_REGISTER.exists():
    for _note, _entry in json.loads(_OBSOLETE_REGISTER.read_text(encoding="utf-8")).items():
        OBSOLETE[_note] = tuple(sorted(set(OBSOLETE.get(_note, ())) | set(_entry["forbidden"])))
        if _entry.get("forbidden_by_surface"):
            OBSOLETE_BY_SURFACE[_note] = {k: tuple(v)
                                          for k, v in _entry["forbidden_by_surface"].items()}


def _forbidden_for(note: str, rel: str) -> tuple:
    """Forbidden renderings for one field ON ONE SURFACE.

    The exemption for a form that is current elsewhere is per surface. Applied globally it let a
    correct occurrence of a value shelter an obsolete occurrence of the same digits on a surface
    the current field never reaches.
    """
    per = OBSOLETE_BY_SURFACE.get(note)
    hand = tuple(f for f in OBSOLETE.get(note, ()) if per is None or f not in
                 {x for v in per.values() for x in v})
    return tuple(per.get(rel, ())) + hand if per is not None else OBSOLETE.get(note, ())


def _obsolete_survives(bad: str, txt: str) -> bool:
    """Does a superseded rendering still stand alone in this text?

    Boundary-aware: a bare "0.222" is a substring of "[-0.517, -0.222]" and of any longer
    decimal, so plain containment reports correct prose as stale and trains the reader to ignore
    the checker. Two rules, and the second is where a stale value escaped:

      * a form carrying an explicit sign must match that sign, with the ASCII hyphen and U+2212
        treated as one character, since a manuscript mixes them freely;
      * an UNSIGNED form must not be preceded by a minus, because that would be a different
        value - but a leading "+" is the same value, and refusing to match it is how
        "[+0.222, +1.492]" passed while "0.222" sat in the register.
    """
    HYPH = "[-" + M + "]"
    stem = bad.lstrip("+-" + M)
    lead = bad[0] if bad[0] in ("+", "-", M) else ""
    if lead in ("-", M):
        pat = HYPH + re.escape(stem)
    elif lead == "+":
        pat = r"\+" + re.escape(stem)
    else:
        pat = r"(?<![-" + M + r"])" + re.escape(stem)
    return bool(re.search(r"(?<![0-9.])" + pat + r"(?![0-9])", txt))


# (artifact, field path, renderer, [surfaces that MUST carry it], note)
ROWS = [
    # The supplement's containment and oscillation shares. Previously outside the census entirely:
    # seventeen quoted percentages that no row and no template read.
    ("contain", "shares.per_agent_X|fresh|f_le_1.agents_pct",     pctp1, [S], "sup fresh union agents"),
    ("contain", "shares.per_agent_X|fresh|f_le_1.value_pct",      pctp1, [S], "sup fresh union value"),
    ("contain", "shares.per_agent_X|fresh|f_le_fH.value_pct",     pctp1, [S], "sup fresh fH value"),
    ("contain", "shares.X_eq_1|fresh|f_le_fH.agents_pct",         pctp1, [S], "sup fresh fH agents"),
    ("contain", "shares.X_eq_1|as_drawn|f_le_fH.agents_pct",      pctp1, [S], "sup as-drawn fH agents"),
    ("contain", "shares.per_agent_X|as_drawn|f_le_fH.value_pct",  pctp1, [S], "sup as-drawn fH value"),
    ("contain", "shares.per_agent_X|as_drawn|f_le_1.agents_pct",  pctp1, [S], "sup uncontainable agents"),
    ("contain", "shares.per_agent_X|as_drawn|f_le_1.value_pct",   pctp1, [S], "sup uncontainable value"),
    ("contain", "shares.branch|fresh|f_le_fH|visible_only.agents_pct", pctp1, [S], "sup visible-only"),
    ("contain", "shares.branch|fresh|f_le_1|visible_only.agents_pct",  pctp1, [S], "sup visible branch"),
    ("contain", "shares.branch|fresh|f_le_1|evasion_only.agents_pct",  pctp1, [S], "sup evasion branch"),
    ("contain", "v_collapse_share.agents_pct",                    pctp1, [S], "sup beyond v* agents"),
    ("contain", "v_collapse_share.value_pct",                     pctp1, [S], "sup beyond v* value"),
    ("contain", "bot_volume_below_beta_pct",                      pctp1, [S], "sup volume below beta"),
    ("contain", "sunk_share",                                     pct1,  [S], "sup sunk share"),
    ("sim_osc", "loss_penalty_pct",                               pctp1, [S], "sup oscillation loss"),
    ("sim_osc", "harm_penalty_pct",                               pctp1, [S], "sup oscillation harm"),
    # Containment shares. The artifact was not loaded at all, so four quoted values rested on
    # nothing a checker read.
    ("contain", "v_collapse_share.agents_pct",  pctp1, [D], "beyond v* agents"),
    ("contain", "v_collapse_share.value_pct",   pctp1, [D], "beyond v* value mass"),
    ("contain", "shares.per_agent_X|fresh|f_le_1.value_pct", pctp1, [D], "fresh escaping value"),
    ("contain", "sunk_share",                   pct1,  [D], "assumed sunk share"),
    # The disappearance rate a year out, which the prose summarises as holding near this
    # value across the follow-up windows.
    ("disap", "followup.4.pct", pctp1, [D], "disappearance rate at one year"),
    # Theorem 5 evaluated at the measured loop delay. The prose restates the daily row as a
    # percentage, so both the cell and the restatement are bound to the same field.
    ("cadence", "cadence.0.safe_gain_budget",  bare3, [D], "daily safe-gain budget"),
    ("cadence", "cadence.1.safe_gain_budget",  bare3, [D], "2-day safe-gain budget"),
    ("cadence", "cadence.2.safe_gain_budget",  bare3, [D], "3-day safe-gain budget"),
    ("cadence", "cadence.3.safe_gain_budget",  bare3, [D], "5-day safe-gain budget"),
    ("cadence", "daily_budget_pct",            pctp1, [D], "daily budget as a percentage"),
    ("contain", "human_mass_above_beta_pct_analytic", pctp1, [D], "analytic human mass above beta"),
    ("contain", "shares.per_agent_X|as_drawn|f_le_1.value_pct", pctp1, [D], "as-drawn escaping value"),
    ("contain", "bot_volume_below_beta_pct", pctp1, [D], "bot volume below beta"),
    ("disap", "abusive.agents_with_events", count, [D], "disappearance agent denominator"),
    # The remaining decimal percentages the census showed unbound.
    ("disap", "quiet5.returned", count, [D], "quiet-five returners"),
    ("disap", "quiet5.n",        count, [D], "quiet-five risk set"),
    ("robots", "populations.abusive.pct_ever_robots", pctp1, [MX], "robots abusive share (MX)"),
    # The 1-NN match rate on heavy events, quoted on both surfaces and previously unbound.
    ("dose", "populations.abusive.n_matched", count, [D], "heavy events matched"),
    ("dose", "populations.abusive.n_heavy",   count, [D], "heavy events"),
    # Lead-lag responder shares and the rotation date concentration: supported, never bound.
    ("leadlag", "populations.abusive.active.responder_share",  pct1, [D], "active responder share"),
    ("leadlag", "populations.abusive.calendar.responder_share", pct1, [D], "calendar responder share"),
    ("leadlag", "populations.pothuman.null.responder_share.obs", pct1, [MX], "pothuman responder share"),
    ("rot_ip",  "transfer_core.largest_date_share_of_pairs",   pct1, [D], "largest date share"),
    # The retreat cut-sensitivity artifact was LOADED and never read: a dead load binds nothing,
    # and the two surfaces quoting this sweep went unchecked while the manifest reported a clean
    # run. A loaded artifact with no row is indistinguishable from one that is being verified.
    ("retreatcut", "n_events.abusive",         count, [],     "retreat sweep events"),
    # Branch weights and block-channel reach, on the arm of record with all 7,812 admitted.
    ("branch", "arm_of_record.responses_429",              count,  [MX],     "429 responses"),
    ("branch", "arm_of_record.responses_401_403",          countr, [MX],     "401/403 responses"),
    ("branch", "arm_of_record.events_429_branch_only",     count,  [MX],     "429-branch events"),
    ("branch", "arm_of_record.events_total",               count,  [MX],     "branch event days"),
    ("branch", "arm_of_record.events_both_same_day",       count,  [MX],     "both-branch days"),
    ("branch", "arm_of_record.pct_events_from_429_branch_only", pct2, [MX],  "429-branch weight"),
    # The main article quotes this field one decimal coarser and gets its own row: a renderer
    # that accepts either form on either surface stops distinguishing them.
    ("branch", "arm_of_record.pct_events_from_429_branch_only", pctp1, [D],  "429-branch weight (D)"),
    ("branch", "arm_of_record.pct_events_touching_429_branch",  pct2, [MX],  "429-touching weight"),
    ("branch", "arm_of_record.agents_ever_blocked",        count,  [D, MX],  "agents ever blocked"),
    ("branch", "arm_of_record.pct_agents_ever_blocked",    pct2,   [MX],     "blocked share"),
    ("branch", "arm_of_record.largest_agent_block_share_pct", pctp1, [D, MX], "largest-agent share"),
    ("combined", "did_combined.did",            bare3, [D, MX],          "headline DiD"),
    ("combined", "did_combined.n_bot",          count, [D],              "headline bots"),
    ("combined", "did_combined.n_ctrl",         count, [D, MX],          "headline controls"),
    ("combined", "did_combined_stealth.did",    bare3, [D],              "stealth-only"),
    ("combined", "did_combined_pothuman.did",   bare3, [D, MX],          "sensitivity arm"),
    ("grid",     "summary.n_specs",             count, [D, H],           "grid size"),
    ("grid",     "window_free.SEGMENTED.did",   bare3, [D],              "segmented estimator"),
    ("grid",     "window_free.SEGMENTED.ci.0",  bare3, [D],              "segmented CI lo"),
    ("grid",     "window_free.SEGMENTED.ci.1",  bare3, [D],              "segmented CI hi"),
    ("grid",     "window_free.FULL_PRE_LINEAR.did", sig3, [D],           "pretrend estimator"),
    ("grid",     "window_free.FULL_PRE_LINEAR.ci.0", bare3, [D],         "pretrend CI lo"),
    ("grid",     "window_free.FULL_PRE_LINEAR.ci.1", bare3, [D],         "pretrend CI hi"),
    ("ranks",    "balanced_analogue.did",       bare3, [D],              "balanced analogue"),
    ("disap",    "abusive.pct",                 pct2,  [D, H],           "disappearance rate"),
    ("disap",    "abusive.gone",                count, [D],              "disappearances"),
    ("etime",    "abusive.n_agents",            count, [D],              "event-time bots"),
    ("etime",    "abusive.n_events",            count, [D],              "event-time events"),
    ("etime",    "verified.n_agents",           count, [D],              "event-time controls"),
    ("dose",     "populations.abusive.adjusted_gap",     sig3, [D, MX],   "dose adjusted gap"),
    ("dose",     "populations.abusive.adjusted_retreat", any_of(sig3, pp1), [D, MX],   "dose adjusted retreat"),
    ("dose",     "populations.verified.adjusted_retreat", any_of(sig3, pp1), [D, MX],  "control adjusted retreat"),
    ("dose",     "populations.abusive.n_events",  count, [D, MX],         "dose events"),
    # The raw and matched dose gaps were quoted on three surfaces without a row, which is also
    # why the register could not tell that a current +0.179 was not a superseded one.
    ("dose",     "populations.abusive.raw_gap",     sig3, [D, MX],        "dose raw gap"),
    ("dose",     "populations.abusive.matched_gap", sig3, [D, MX],        "dose matched gap"),
    ("dose",     "populations.abusive.matched_retreat", any_of(sig3, pp1), [D, MX], "dose matched retreat"),
    ("dose",     "populations.verified.matched_retreat", any_of(sig3, pp1), [D],    "control matched retreat"),
    ("dose",     "populations.abusive.retreat_light",  bare3, [D, MX, S], "dose retreat light"),
    ("dose",     "populations.abusive.retreat_heavy",  bare3, [D, MX, S], "dose retreat heavy"),
    ("dose",     "populations.verified.n_events",  count, [MX, S],        "dose control events"),
    ("dose",     "populations.verified.raw_gap",  sig3,  [MX, D],         "control raw gap"),
    ("counts",   "comparisons.spacing_only_5_to_11.events_from", count, [D], "events before rule"),
    ("counts",   "comparisons.spacing_only_5_to_11.events_to",   count, [D], "events after rule"),
    # Finding 6, on the arm-of-record death cohort
    ("rot_ip",    "n_deaths",                      count, [D, MX, S], "rotation deaths"),
    ("rot_ip",    "transfer_core.succ_dvol_mean",  bare3, [D, MX, S], "successor dvol"),
    ("rot_ip",    "transfer_core.succ_dbot_mean",  any_of(sig3, bare3), [D], "successor dbotness"),
    ("rot_ip",    "transfer_core.n_pairs",         count, [D],        "core pairs"),
    ("rot_ip",    "transfer_core.n_deaths",        count, [D],        "core deaths"),
    ("rot_self",  "n_planted",                     count, [D, MX],    "planted controls"),
    # Date counts and cohort sizes for both rotation strata: quoted in the text, so covered.
    ("rot_ip",    "transfer_core.n_dates",             count, [D],        "core death dates"),
    ("rot_ip",    "transfer_envelope.n_dates",         count, [D],        "all death dates"),
    ("rot_ip",    "null_link_perm_core.succ_dvol_mean", sig3, [D, MX],    "permutation placebo"),
    ("rot_ip",    "transfer_core.succ_dvol_ci_date_cluster.0", bare3, [D], "date-clustered CI lo"),
    ("rot_decoy", "transfer_envelope.succ_dvol_mean",  any_of(sig3, bare3), [D, MX], "decoy envelope ramp"),
    ("rot_disap", "transfer_envelope.succ_dvol_mean",  bare3, [D],        "disappearance envelope"),
    ("rot_disap", "transfer_core.succ_dvol_mean",      any_of(sig3, bare3), [D],  "disappearance core"),
    ("disap",     "abusive.evaluable",                 count, [D],        "disappearance denominator"),
    ("disap",     "abusive.agents_with_events",        count, [D],        "disappearance agents"),
    # The decoy partition and both nulls' endpoints. Bin means and counts are covered
    # individually so that a single bin cannot be altered or omitted without a row failing.
    ("rot_decoy", "transfer_core.succ_dvol_mean",       any_of(sig3, bare3), [D], "decoy core"),
    ("rot_decoy", "transfer_core.n_pairs",              count, [D],        "decoy core pairs"),
    ("rot_decoy", "transfer_envelope.n_pairs",          count, [D],        "decoy envelope pairs"),
    ("rot_decoy", "transfer_envelope.succ_dvol_ci_date_cluster.1", bare3, [D], "decoy date-CI hi"),
    # A quantity a REVIEW corrected is exactly the kind that reverts silently: nothing rendered it,
    # so no row missed it and no obsolete form banned it. The overlap share was corrected from
    # 76.6% to 74.9% and could have been reverted on both surfaces without a single check firing.
    ("rot_ip",    "transfer_rate_core",                 pct1,  [D, MX],    "IP overlap share"),
    # The positive control was quoting the v4 self-test while v5 existed; nothing rendered it.
    # A population share that was being read out of an event-spacing artifact, on a June cache.
    # The highlight range and the two abstract population counts had no row, so the coarse check
    # was the only thing standing between them and any plausible-looking number.
    ("grid",      "summary.max",                        absbare2, [H],        "highlight range hi"),
    ("grid",      "summary.min",                        absbare2, [H],        "highlight range lo"),
    # Presence and occurrence counts are ORDER-BLIND: "0.27 to 0.76" and "0.76 to 0.27" contain the
    # same tokens the same number of times, and both values are legitimately in the rendered pool,
    # so a reversed range - a false statement about the grid - passed every check. The ordered
    # phrase is pinned as a whole.
    ("grid",      "summary",                            grid_range_phrase, [H], "highlight range order"),
    # Cohort sizes and percentiles are order-bearing claims that presence rows cannot see: both
    # numbers are covered and both survive a swap. Each is pinned as the ordered phrase it is.
    ("combined",  "did_combined",                       cohort_phrase, [D], "headline cohort order"),
    ("ranks",     "balanced_analogue",                  percentile_phrase, [D], "percentile order"),
    # The boundary shares had no row, which is why 6.5%/6.5% passed 372 checks.
    ("censor", "populations.abusive.n_boundary_responders",  count, [D, MX], "boundary n abusive"),
    ("censor", "populations.verified.n_boundary_responders", count, [D, MX], "boundary n verified"),
    ("censor", "populations.abusive.boundary_share_pct",  pctp1, [D, MX], "boundary share abusive"),
    ("censor", "populations.verified.boundary_share_pct", pctp1, [D, MX], "boundary share verified"),
    ("censor", "populations.abusive.n_responders",        count, [D, MX], "censoring n abusive"),
    ("censor", "populations.verified.n_responders",       count, [D, MX], "censoring n verified"),
    ("robots",    "populations.abusive.n_declared",     count,  [D],          "abstract arm size"),
    ("robots",    "populations.verified.n_declared",    count,  [D],          "abstract verified n"),
    ("robots",    "populations.abusive.pct_ever_robots",  any_of(pct2, pctp1), [D],        "robots abusive share"),
    ("robots",    "populations.verified.pct_ever_robots", any_of(pct2, pctp1), [D],        "robots verified share"),
    ("robots",    "populations.declared.pct_ever_robots", any_of(pct2, pctp1), [D],        "robots declared share"),
    ("selftest",  "n_planted",                          count, [MX],       "planted handoffs"),
    ("selftest",  "dB_v_mean_want_pos",                 dec2,  [MX],       "planted ramp"),
    ("selftest",  "pearson_bApre_bBpost_want_1",        dec2,  [MX],       "planted botness r"),
    ("rot_decoy", "n_deaths_with_candidate",            count, [D, MX],    "decoy deaths with candidate"),
    ("rot_ip",    "null_link_perm_core.ci.0",           bare3, [D, MX],    "IP null band lo"),
    ("rot_ip",    "null_link_perm_core.ci.1",           bare3, [D, MX],    "IP null band hi"),
    ("rot_ip",    "null_link_perm_core.n_successors",   count, [D],        "IP null successors"),
    ("rot_disap", "transfer_envelope.succ_dvol_ci_date_cluster.0", bare3, [D], "disap date-CI lo"),
    # The decoy specificity partition. Every bin, so a truncated table fails; the artifact itself
    # asserts exhaustiveness, so a bin cannot silently vanish upstream either.
    ("decoy_part", "bins.0.succ_dvol_mean",  any_of(sig3, bare3), [D, MX], "decoy bin 4-5"),
    ("decoy_part", "bins.1.succ_dvol_mean",  any_of(sig3, bare3), [D, MX], "decoy bin 6-25"),
    ("decoy_part", "bins.2.succ_dvol_mean",  any_of(sig3, bare3), [D, MX], "decoy bin 26-100"),
    ("decoy_part", "bins.3.succ_dvol_mean",  any_of(sig3, bare3), [D, MX], "decoy bin 101+"),
    ("decoy_part", "bins.0.n_pairs",         count, [D, MX],  "decoy bin 4-5 pairs"),
    ("decoy_part", "bins.3.n_pairs",         count, [D],      "decoy bin 101+ pairs"),
    ("decoy_part", "n_envelope_pairs",       count, [D, MX],  "decoy envelope total"),
    ("decoy_part", "date_concentration.n_pairs", count, [D, MX], "concentrated-date pairs"),
    ("decoy_part", "date_concentration.without_that_date.succ_dvol_mean", any_of(sig3, bare3), [D, MX], "without that date"),
    ("decoy_part", "date_concentration.without_that_date.n_pairs", count, [D, MX], "without-date pairs"),
    # Every cell the manuscript displays, not a sample of them. A page-wide token count cannot see two
    # cells swapped, because it corrupted the
    # 93 and 509 counts, the dormancy share and a deletion endpoint and all four passed.
    ("decoy_part", "bins.1.n_pairs",              count, [D, MX], "decoy bin 6-25 pairs"),
    ("decoy_part", "bins.2.n_pairs",              count, [D, MX], "decoy bin 26-100 pairs"),
    ("decoy_part", "bins.0.n_deaths",             count, [D],     "decoy bin 4-5 deaths"),
    ("decoy_part", "bins.1.n_deaths",             count, [D],     "decoy bin 6-25 deaths"),
    ("decoy_part", "bins.2.n_deaths",             count, [D],     "decoy bin 26-100 deaths"),
    ("decoy_part", "bins.3.n_deaths",             count, [D],     "decoy bin 101+ deaths"),
    ("decoy_part", "date_concentration.succ_dvol_mean", any_of(sig3, bare3), [D, MX], "concentrated-date mean"),
    ("decoy_part", "date_concentration.without_that_date.succ_dvol_ci_date_cluster.0", bare3, [D, MX], "deletion CI lo"),
    ("decoy_part", "date_concentration.without_that_date.succ_dvol_ci_date_cluster.1", bare3, [D, MX], "deletion CI hi"),
    ("decoy_part", "predefined_core.succ_dvol_ci_date_cluster.1", bare3, [D],  "decoy core CI hi"),
    # null support and the correctly weighted dormancy
    ("rot_ip",    "null_link_perm_core.n_pairs_fixed_support",      count, [D], "null fixed pairs"),
    ("rot_ip",    "null_link_perm_core.n_successors_fixed_support", count, [D], "null fixed successors"),
    ("rot_ip",    "null_link_perm_core.dormant_share_pair_weighted", pct1, [D, MX], "dormancy pair-weighted"),
    ("rot_ip",    "null_link_perm_core.dormant_share_of_candidate_dates", pct1, [D], "dormancy unweighted"),
    ("rot_ip",    "null_link_perm_core_unique_dates.succ_dvol_mean", any_of(sig3, bare3), [D], "unique-date null"),

    # ---- Section 8. The policy conclusion rested on a table that no check could see: it is
    # generated as CSV, so six rounds of manifest work covered every JSON field around it and
    # none of its cells. These rows close that, using the ordering (headline
    # table, containment-map parameters, policy ranking) is the order they appear in.
    ("sim_policy", ["binary_block_tau0.234", "extraction_idx"],   dec1,   [D], "8.1 binary extraction"),
    ("sim_policy", ["binary_block_tau0.234", "human_harm"],       countr, [D], "8.1 binary harm"),
    ("sim_policy", ["binary_block_tau0.234", "evasion_prob"],     bare3,  [D], "8.1 binary evasion share"),
    ("sim_policy", ["binary_block_tau0.234", "observed_idx"],     dec1,   [D], "8.1 binary dashboard"),
    ("sim_policy", ["binary_block_tau0.234", "L_D_idx"],          dec1,   [D], "8.1 binary L_D"),
    ("sim_policy", "aggressive_throttle_3b.extraction_idx",  dec1,   [D], "8.1 throttle extraction"),
    ("sim_policy", "aggressive_throttle_3b.human_harm",      countr, [D], "8.1 throttle harm"),
    ("sim_policy", "aggressive_throttle_3b.evasion_prob",    bare3,  [D], "8.1 throttle evasion share"),
    ("sim_policy", "aggressive_throttle_3b.observed_idx",    dec1,   [D], "8.1 throttle dashboard"),
    ("sim_policy", "aggressive_throttle_3b.L_D_idx",         dec1,   [D], "8.1 throttle L_D"),
    ("sim_policy", ["static_graded_ft1.00", "extraction_idx"],    dec1,   [D], "8.1 graded extraction"),
    ("sim_policy", ["static_graded_ft1.00", "human_harm"],        countr, [D], "8.1 graded harm"),
    ("sim_policy", ["static_graded_ft1.00", "evasion_prob"],      bare3,  [D], "8.1 graded evasion share"),
    ("sim_policy", ["static_graded_ft1.00", "L_D_idx"],           dec1,   [D], "8.1 graded L_D"),
    ("sim_policy", ["adaptive_graded_k0.60", "extraction_idx"],   dec1,   [D], "8.1 adaptive extraction"),
    ("sim_policy", ["adaptive_graded_k0.60", "human_harm"],       countr, [D], "8.1 adaptive harm"),
    ("sim_policy", ["adaptive_graded_k0.60", "evasion_prob"],     bare3,  [D], "8.1 adaptive evasion share"),
    ("sim_policy", ["adaptive_graded_k0.60", "L_D_idx"],          dec1,   [D], "8.1 adaptive L_D"),
    # containment-map parameters: the numbers Fig. 3's caption and Section 8.2 both assert
    ("sim_head",  "f_min_at_vref",                           bare3,  [D], "8.2 deterrence level"),
    ("sim_head",  "f_switch",                                bare3,  [D], "8.2 fresh switch"),
    ("sim_head",  "sens.K_E_base.v_collapse",                bare3,  [D], "8.2 critical value v*"),
    ("sim_head",  "static_f_target",                         dec2,   [D], "8.2 tuned ceiling"),
    ("sim_head",  "static_width",                            dec2,   [D], "8.2 sigmoid width"),
    ("sim_grid",  ["tolerance_sets", "0.001", "ceilings", "0"],         dec2,   [D], "8.2 0.1% band floor"),
    ("sim_grid",  ["tolerance_sets", "0.005", "ceilings", "0"],         dec2,   [D], "8.2 0.5% band floor"),
    # the oscillating loop, which Section 8.3 quotes as its cost of instability
    ("sim_osc",   "loss_penalty_pct",                        pctup1, [D], "8.3 oscillation loss"),
    ("sim_osc",   "harm_penalty_pct",                        pctup1, [D], "8.3 oscillation harm"),
    ("sim_osc",   "period_over_loop_delay",                  bare2,  [D], "8.3 period ratio"),
]

# Figures: (render path, artifact whose data it must reflect, surface citing it)
FIGURES = [
    (SUP / "outputs/figures/DB_observability_eventtime_v3.png",
     HERE / "event_time_study.png", D, "Fig. 2 event-time"),
    # The lead-lag renders were caught printing the previous cohort's shares and responder
    # counts while their captions carried current values. A figure that nothing hashes is a
    # surface the checker cannot see.
    (SUP / "outputs/figures/DB_leadlag_shuffle_null_v2.png",
     HERE / "reference_renders/DB_leadlag_shuffle_null_v2.png", MX, "MX Fig. 4 shuffle null"),
    (SUP / "outputs/figures/DB_leadlag_tau_distribution_v2_arm.png",
     HERE / "reference_renders/DB_leadlag_tau_distribution_v2_arm.png", MX, "MX Fig. 2 response lag"),
    (SUP / "outputs/figures/DB_dose_balance_v3.png",
     HERE / "dose_balance_corrected.png", MX, "MX Fig. 1 dose balance"),
    # Compared against a reference copy taken at verification time, not against itself:
    # self-comparison is true by construction and detects nothing.
    (SUP / "outputs/figures/DB_rotation_conservation_v5_ip.png",
     HERE / "reference_renders/DB_rotation_conservation_v5_ip.png", MX, "MX Fig. 6 rotation"),
    (SUP / "outputs/figures/DB_leadlag_ccf_sample_v2_arm.png",
     HERE / "reference_renders/DB_leadlag_ccf_sample_v2_arm.png", MX, "MX Fig. 5 sample CCF"),
]

# Every PNG a manuscript cites, with the artifact its generator consumes. A hash against a
# reference render proves equality to that snapshot, not that the generator saw current data, so
# each row also requires the render to be no older than the data behind it. A render that
# predates its own source is stale whatever its caption says - which is how a figure carrying a
# June cohort survived under a caption quoting August values.
#
# `None` marks a render with no data source in this repository: a hand-drawn schematic, or a
# panel produced by a sibling paper's pipeline. Those are declared, not silently uncovered.
FIGURE_SOURCES = [
    ("DB_observability_eventtime_v3.png",       HERE / "event_time_study.json"),
    ("DB_leadlag_shuffle_null_v2.png",          HERE / "leadlag_calendar_recompute.json"),
    ("DB_leadlag_tau_distribution_v2_arm.png",  HERE / "leadlag_calendar_per_agent.csv"),
    ("DB_leadlag_ccf_sample_v2_arm.png",        HERE / "leadlag_calendar_recompute.json"),
    ("DB_dose_balance_v3.png",                  HERE / "dose_split_of_record.json"),
    ("DB_rotation_conservation_v5_ip.png",      SUP / "outputs/DB_rotation_summary_v5_ip.json"),
    ("SIM_containment_map.png",                 SUP / "outputs/SIM_ceiling_grid.json"),
    ("SIM_stability_timedomain.png",            SUP / "outputs/SIM_oscillation_cost.json"),
    ("DB_leadlag_KM_v2_arm.png",                HERE / "leadlag_censoring.json"),
    # Corpus panels produced by the family-wide analyses rather than by Paper D's own pipeline.
    # Each is bound to the table its generator writes and reads, so a render older than its data is
    # caught here as it is for D's own figures. "Produced elsewhere" is a reason to record the
    # source, not a reason to exempt the figure.
    ("A_trend_differential_bot_vs_human.png",   SUP / "outputs/tables/A_per_agent_nonstationarity.csv"),
    ("B_difference_in_differences.png",         SUP / "outputs/tables/B_difference_in_differences.csv"),
    ("B_normalized_response.png",               SUP / "outputs/tables/B_normalized_response.csv"),
    ("C_adaptation_time.png",                   SUP / "outputs/tables/C_per_agent_trajectory.csv"),
    ("E_botness_density.png",                   SUP / "outputs/tables/E_botness_density.csv"),
    # Hand-drawn regime schematics. These are the only source-free entries, and each says so in
    # its own caption: they illustrate the model's regions, they do not report a measurement.
    ("SCHEMATIC_evasion_advantage.png",         None),
    ("SCHEMATIC_overdeterrence_regime.png",     None),
    ("DB_robots_response_to_block_v3.png",      HERE / "event_spacing_recompute.json"),
]


# ---------------------------------------------------------------------------
# Structural checks: fields at locations, not tokens on a page.
#
# What these checks do NOT prove, so that nobody reads more into a green run than it earns:
# a cell may be coarsened (62.1 -> 62) and still round correctly; a located value is required in
# its paragraph, so a second correct copy in the same paragraph can shelter a corrupted one; an
# occurrence-count edit compensated elsewhere on the same surface passes; and the ranking parser
# reads one sentence, so a contradictory restatement in the abstract or Section 9 is not covered.
#
# Four mutations defeat a page-wide checker while leaving every token count unchanged:
# swapping two table cells that share a digit budget, changing a cell no row covered, moving a
# prose value by one in its last decimal, and reordering the prose ranking. Occurrence counting
# cannot see any of those, because none of them changes how many times a string appears in the
# file. These checks parse the surface instead, and the occurrence counts stay on as a
# secondary stale-copy guard rather than as the provenance contract.
# ---------------------------------------------------------------------------

# Section 8.1's table: displayed row label -> CSV row key, displayed column -> CSV column.
POLICY_ROWS = [
    ("No defense",          "no_defense"),
    ("Binary block",        "binary_block_tau0.234"),
    ("Aggressive throttle", "aggressive_throttle_3b"),
    ("Static graded",       "static_graded_ft1.00"),
    ("Adaptive graded",     "adaptive_graded_k0.60"),
]
POLICY_COLS = ["extraction_idx", "human_harm", "evasion_prob", "observed_idx", "L_D_idx"]

# The prose ranking, cheapest total loss first: displayed phrase -> ranking_main entry.
RANKING_PHRASES = [
    ("static graded",       "static_graded_ft1.00"),
    ("adaptive graded",     "adaptive_graded_k0.60"),
    ("binary block",        "binary_block_tau0.234"),
    ("aggressive throttle", "aggressive_throttle_3b"),
    ("no defense",          "no_defense"),
]

# (artifact, field, renderer, [(surface, anchor phrase)], note)
# The value must appear in the same sentence as the anchor, so moving it without moving the
# sentence fails even when the page-wide occurrence count is unchanged.
LOCATED = [
    # MethodsX Fig. 4 is rendered by db_leadlag_shuffle_figure_v3.py from the 999-replicate
    # run of record. Its caption is pinned to that artifact so it cannot be re-captioned onto
    # the superseded 10-replicate aggregation, whose generator writes the same .png path.
    ("leadlag", "populations.abusive.null.responder_share.obs", pct1,
     [(MX, "Left: each population's observed responder share")], "Fig. 4 abusive share"),
    ("leadlag", "populations.abusive.null.responder_share.null_mean", pct1,
     [(MX, "Left: each population's observed responder share")], "Fig. 4 abusive null mean"),
    ("leadlag", "populations.abusive.null.responder_share.null_max", pct1,
     [(MX, "Left: each population's observed responder share")], "Fig. 4 abusive null max"),
    ("leadlag", "populations.verified.null.responder_share.obs", pct1,
     [(MX, "Left: each population's observed responder share")], "Fig. 4 human share"),
    ("leadlag", "populations.abusive.null.median_abs_ccf.obs", bare3,
     [(MX, "Left: each population's observed responder share")], "Fig. 4 abusive coupling"),
    ("leadlag", "populations.verified.null.median_abs_ccf.obs", bare3,
     [(MX, "Left: each population's observed responder share")], "Fig. 4 human coupling"),
    # The abstract restates four headline quantities. A page-wide presence check cannot see a
    # SWAP between them - "-0.30" is a correct rendering of the light-dose effect and was accepted
    # in the headline's place - so each is pinned to the sentence that interprets it.
    ("combined", "did_combined.did", ascii2,
     [(D, "our preferred one gives")], "abstract headline DiD"),
    ("grid", "window_free.SEGMENTED.did", ascii2,
     [(D, "the two estimators that choose no baseline window")], "abstract segmented"),
    ("grid", "window_free.FULL_PRE_LINEAR.did", ascii2,
     [(D, "the two estimators that choose no baseline window")], "abstract pre-trend"),
    # Sections 7.3 and 7.7 were quoting an event-spacing sensitivity artifact that this manifest
    # did not load, so every value in them passed by not being looked at. Each is now bound to the
    # field it comes from, in the sentence that interprets it.
    ("espace", "results.corrected_11.observability.declared_vs_verified_full.did", bare3,
     [(D, "on the verified-human coordinate")], "declared obs vs verified"),
    ("espace", "results.corrected_11.observability.declared_vs_pothuman_full.did", bare3,
     [(D, "on the verified-human coordinate")], "declared obs vs pothuman"),
    ("espace", "results.corrected_11.channels.z_robots.did", sig3,
     [(D, "| Abusive automation |")], "z_robots abusive"),
    ("espace", "results.corrected_11.channels_declared.z_robots.did", sig3,
     [(D, "| Declared automation |")], "z_robots declared"),
    ("espace", "results.corrected_11.channels_pothuman.z_robots.did", sig3,
     [(D, "| Abusive automation |")], "z_robots abusive vs pothuman"),
    ("espace", "results.corrected_11.channels_declared_pothuman.z_robots.did", sig3,
     [(D, "| Declared automation |")], "z_robots declared vs pothuman"),
    ("espace", "results.corrected_11.channels_declared.d_robots.did", bare4,
     [(D, "The signed channel adds a caution")], "d_robots declared"),
    ("espace", "results.corrected_11.channels.d_robots.did", bare4,
     [(D, "The signed channel adds a caution")], "d_robots abusive"),
    ("espace", "results.corrected_11.channels.d_p404.did", bare4,
     [(D, "The signed channel adds a caution")], "d_p404 abusive"),
    ("espace", "results.corrected_11.channels.d_loghits.did", sig4,
     [(D, "The signed channel adds a caution")], "d_loghits abusive"),
    ("espace", "results.corrected_11.channels_declared.d_loghits.did", sig4,
     [(D, "declared crawlers move the same channel harder")], "d_loghits declared"),
    ("espace", "results.corrected_11.raw_means.z_robots.declared", bare4,
     [(D, "raw movement")], "raw z_robots declared"),
    ("espace", "results.corrected_11.raw_means.z_robots.abusive", bare4,
     [(D, "raw movement")], "raw z_robots abusive"),
    # A two-character render cannot be occurrence-counted, and plain presence is satisfied by any
    # stray "28" elsewhere on the page. The counts and medians are therefore pinned as the ORDERED
    # PHRASES that state them, in the sentence that interprets each.
    ("censor", "populations", boundary_phrase_abusive,
     [(D, "a small minority on the primary calendar indexing"),
      (MX, "abusive responders sit at the")], "boundary phrase abusive"),
    ("censor", "populations", boundary_phrase_verified,
     [(D, "a small minority on the primary calendar indexing"),
      (MX, "verified-human responders")], "boundary phrase verified"),
    ("censor", "populations", km_median_phrase,
     [(MX, "Treating those")], "censoring medians"),
    ("sim_head", "f_switch_sunk", bare3,
     [(D, "once $K_E$ is sunk"),
      (D, "as well as the sunk-cost point"),
      (S, "against the fresh 0.886")],
     "sunk-cost switch"),
]


def _table_rows(block: str):
    """Split a markdown table into rows of stripped cells, dropping the alignment row."""
    rows = []
    for line in block.strip().splitlines():
        if not line.strip().startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if all(set(c) <= set("-: ") for c in cells):
            continue
        rows.append(cells)
    return rows


_NUM_RE = re.compile(r"-?[0-9]+(?:[.]([0-9]+))?$")


def _shown(cell: str):
    """The number a table cell displays, and how many decimals it shows."""
    t = cell.replace("**", "").replace("*", "").replace(",", "").replace(M, "-").strip()
    m = _NUM_RE.match(t)
    if not m or m.end() != len(t):
        return None, None
    return float(t), len(m.group(1) or "")


def check_policy_table(txt, csv_blob, problems) -> int:
    """Every displayed cell of the Section 8.1 table, against its own CSV field."""
    i = txt.find("| Policy | Extraction (idx)")
    if i < 0:
        problems.append("8.1 table: header not found - the table was renamed or removed")
        return 0
    j = txt.index("\n\n", i)
    rows = _table_rows(txt[i:j])
    body = rows[1:]
    if len(body) != len(POLICY_ROWS):
        problems.append(f"8.1 table: {len(body)} data rows, expected {len(POLICY_ROWS)}")
        return 0
    n = 0
    for (label, key), row in zip(POLICY_ROWS, body):
        if label.lower() not in row[0].lower():
            problems.append(f"8.1 table: row {row[0]!r} is not the expected {label!r}")
            continue
        src = (csv_blob or {}).get(key)
        if src is None:
            problems.append(f"8.1 table: {key} missing from the policy CSV")
            continue
        for col, cell in zip(POLICY_COLS, row[1:]):
            shown, dp = _shown(cell)
            if shown is None:
                problems.append(f"8.1 table: {label} / {col} shows no number ({cell!r})")
                continue
            want = round(float(src[col]), dp)
            _note_bound(want)
            if abs(shown - want) > 10 ** -(dp + 3):
                problems.append(
                    f"8.1 table: {label} / {col} shows {shown} but {key}.{col} is {want}")
            n += 1
    return n


def check_ranking(txt, headline, problems) -> int:
    """The prose ranking, read as an ordered list, against ranking_main."""
    i = txt.find("First, the ranking:")
    if i < 0:
        problems.append("policy ranking: the ranking sentence was not found")
        return 0
    end = txt.find(".", i)
    order = []
    for part in txt[i:end].split("<"):
        hit = [key for phrase, key in RANKING_PHRASES if phrase in part.lower()]
        if len(hit) == 1:
            order.append(hit[0])
    want = list((headline or {}).get("ranking_main", []))
    _note_bound(want)
    if order != want:
        problems.append(f"policy ranking: prose reads {order}, ranking_main is {want}")
    return 1


# Blocks that restate headline numbers and nothing else. Every number in them must be a
# correct-precision rounding of some CURRENT artifact field, which is a stronger contract than
# any token blacklist: a superseded estimate quoted to fewer digits than the register stores has
# no current field it can round to, and a page-wide ban on the coarse form would instead reject
# unrelated correct quantities that happen to share those digits.
COARSE_BLOCKS = [(D, "## Abstract", "**Keywords:**"), (H, None, None)]

# Numbers in these blocks that no local artifact can confirm, each with the reason. Design
# constants and HTTP status codes are defined in the text; the corpus-description quantities are
# cited to the dataset publication and are outside this repository's artifacts, so the check
# reports them as unverifiable-by-design rather than pretending to have checked them.
COARSE_ALLOW = {
    "180": "design constant: the complete crossing of four declared specification axes",
    "0": "design constant", "1": "design constant", "2": "design constant",
    "3": "design constant", "4": "design constant", "5": "design constant",
    "6": "design constant", "10": "design constant", "16": "design constant",
    "100": "design constant",
    "2021": "calendar year", "2022": "calendar year", "2024": "calendar year",
    "2025": "calendar year", "2026": "calendar year",
    "401": "HTTP status code", "403": "HTTP status code", "404": "HTTP status code",
    "429": "HTTP status code",
    "8,780": "AGWA corpus description, cited to the dataset publication",
    "6.05": "AGWA corpus description, cited to the dataset publication",
    "2,274,125": "AGWA corpus description, cited to the dataset publication",
}

_NUM = re.compile(r"(?<![\w.])([+" + M + r"-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?)(%?)")


MAX_SEQ = 12


def _rendered_pool() -> list:
    """Every value a manifest ROW or LOCATED entry actually renders, as floats.

    The previous pool walked every scalar in every loaded artifact. With four channel blocks, raw
    group means and 999-replicate distributions in scope, almost any two-decimal token rounds to
    SOMETHING, and the abstract/highlights check degenerated into "is this a number". Restricting
    it to rendered values asks the question that matters: is this a value the manuscript is
    entitled to print?
    """
    pool = []
    for art, path, render, _want, _note in ROWS + [tuple(r) for r in LOCATED]:
        v = _dig(ART.get(art), path)
        if v is None:
            continue
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            pool.append(float(v))
    return pool


# Decimal percentages that are DEFINITIONAL, not measured, and are therefore excluded from the
# percentage census by name rather than left silently unbound. Each names a convention the text
# chooses, so there is no artifact field that could move underneath it.
# Occurrence-specific exclusions. A GLOBAL token whitelist is not durable: excluding the string
# "0.1%" anywhere excuses it everywhere, so a new unsupported sentence carrying that token passes
# the census. Each entry names the surface, the exact clause the token is allowed to appear in,
# and why no artifact field could move underneath it. The token is accounted for ONLY inside that
# clause; the same token in any other sentence is unaccounted and fails.
CENSUS_EXCLUDED_OCCURRENCES = (
    ("draft/D-v53.md", "every ceiling within 0.1% of the minimum", "0.1%",
     "a tolerance level the sweep chooses; the band floor beneath it is bound"),
    ("draft/D-v53.md", "widening the tolerance to 0.5% admits ceilings", "0.5%",
     "the second tolerance level, likewise chosen"),
    ("draft/D-v53.md", "(2.5-97.5% band", "97.5%",
     "the definition of the interval; both endpoints are bound"),
)


_POOL = None


def _rounds_to(tok: str, pct: bool) -> bool:
    """Does some current artifact field round to this token at the token's own precision?"""
    global _POOL
    if _POOL is None:
        _POOL = _rendered_pool()
    body = tok.replace(",", "").replace(M, "-")
    try:
        val = float(body)
    except ValueError:
        return True
    digits = len(body.split(".")[1]) if "." in body else 0
    # Round the ARTIFACT value at the token's precision and compare to the token, never the
    # reverse: scaling the token first and rounding the scaled copy at the token's own precision
    # collapses it (-0.44/100 rounds to -0.00 at two decimals), so every near-zero field in the
    # corpus matched every two-decimal token and the check silently passed everything.
    target = round(val, digits)
    scales = (1.0, 100.0) if pct else (1.0, 100.0, 0.01)
    for v in _POOL:
        if abs(v) > 1e12:
            continue
        for k in scales:
            if round(v * k, digits) == target:
                return True
    return False


def check_release_inventory(problems) -> int:
    """Every released PNG is either cited-and-checked, or listed with a reason for shipping.

    The cited-figure check inspects the seventeen names the manuscripts reference. The release
    directory holds far more, and an uncited render sitting beside a cited one is exactly what a
    reader picks up by mistake - it is how three superseded robots-response images stayed
    available under near-identical names while the article cited a fourth. Silence about them is
    not coverage, so each must appear in the inventory with a status.
    """
    # Inventory the PUBLIC BUNDLE, which is what a reviewer receives. Checking the source
    # directory instead reported figures as released that the sync deliberately withholds, and
    # counted them among the listed entries - the inventory disagreed with the package it
    # described.
    inv_path = HERE / "figure_inventory.json"
    released = sorted(p.name for p in (PAPER / "artifacts" / "outputs" / "figures").glob("*.png"))
    if not inv_path.exists():
        problems.append(f"release inventory: {inv_path.name} is missing; "
                        f"{len(released)} released PNG(s) have no recorded status")
        return 0
    inv = json.loads(inv_path.read_text(encoding="utf-8"))
    declared = {name for name, _ in FIGURE_SOURCES}
    listed = set(inv.get("entries", {}))
    for name in released:
        if name in declared:
            continue
        if name not in listed:
            problems.append(f"release inventory: {name} is released but neither cited nor listed")
    for name in sorted(listed - set(released)):
        problems.append(f"release inventory: {name} is listed but is not in the release directory")
    # The declared counts are part of the claim, so they are checked rather than trusted: the
    # previous inventory asserted 52 released and 17 + 38 entries, which cannot all be true.
    n_cited = len(declared & set(released))
    for field, want in (("n_released", len(released)),
                        ("n_cited_and_checked", n_cited),
                        ("n_listed", len(listed))):
        got = inv.get(field)
        if got != want:
            problems.append(f"release inventory: {field} says {got}, the bundle holds {want}")
    return len(released)


def check_response_table(surfaces, problems) -> int:
    """Section 7.7's response table, parsed as a matrix: header order AND cell mapping.

    Swapping the two column headers leaves every number and every occurrence count identical, and
    reassigns every contrast to the other comparison population. No row, located, coarse, policy,
    ranking or ordered-claim check can see it: the truth is in the header-to-cell mapping, not in
    the values. `check_ordered_claims` cannot help either - neither column order is monotone.
    """
    txt = surfaces.get(D)
    if txt is None:
        return 0
    head = "| Population | versus verified humans | versus the larger sensitivity arm |"
    if head not in txt:
        problems.append("response table: the Section 7.7 header is not in its expected order "
                        "(verified humans first, sensitivity arm second)")
        return 1
    r = _dig(ART.get("espace"), "results.corrected_11")
    if r is None:
        problems.append("response table: event_spacing_recompute.json is not loaded")
        return 1
    want = {
        "Abusive automation": (r["channels"]["z_robots"]["did"],
                               r["channels_pothuman"]["z_robots"]["did"]),
        "Declared automation": (r["channels_declared"]["z_robots"]["did"],
                                r["channels_declared_pothuman"]["z_robots"]["did"]),
    }
    n = 0
    block_start = txt.index(head)
    rows = txt[block_start:].split(chr(10))[2:4]
    for row in rows:
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        if len(cells) < 3:
            problems.append(f"response table: row {row.strip()[:40]!r} does not have three cells")
            n += 1
            continue
        label = cells[0]
        if label not in want:
            problems.append(f"response table: unexpected row label {label!r}")
            n += 1
            continue
        for cell, expected, col in zip(cells[1:3], want[label],
                                       ("versus verified humans",
                                        "versus the larger sensitivity arm")):
            n += 1
            if sig3(expected) not in cell:
                problems.append(f"response table: {label} / {col} shows {cell!r}, but the field "
                                f"for that pairing is {sig3(expected)}")
    return n


_CENSUS_BOUND: set = set()
_PCT_TOKEN = re.compile(r"(?<![\d.])(\d{1,3}\.\d+)%")


_CENSUS_TEMPLATES: set = set()


def _note_template(text) -> None:
    """Record the full clause a semantic check required, not merely the tokens inside it.

    A token-level bound set says "15.6% is accounted for somewhere", which excuses every other
    sentence on the page that happens to use it. A clause-level one says "15.6% is accounted for
    HERE", so a new sentence reusing the token is unaccounted and reported.
    """
    if isinstance(text, str) and len(text) > 12:
        _CENSUS_TEMPLATES.add(text)


def _note_bound(text) -> None:
    """Record every percentage token a check actually requires.

    Derived from the checks themselves rather than kept as a parallel list, because a
    hand-maintained inventory of what is covered drifts from what is covered.
    """
    for chunk in (text if isinstance(text, (list, tuple)) else [text]):
        if isinstance(chunk, str):
            _CENSUS_BOUND.update(m.group(0) for m in _PCT_TOKEN.finditer(chunk))


def _spans(haystack: str, needle: str):
    """Every start offset of `needle` in `haystack`."""
    out, at = [], haystack.find(needle)
    while at != -1:
        out.append(at)
        at = haystack.find(needle, at + 1)
    return out


def check_obsolete_register_surfaces(problems) -> int:
    """Every surface the obsolete register keys must be a surface under review.

    The register is keyed by surface FILENAME. A version cut renames all four, and a register
    still keyed to the previous names silently matches nothing: `_forbidden_for` returns an empty
    tuple for every field, the absence contract on superseded values stops running, and the run
    stays green because nothing reports a check that did not happen. Renaming the surfaces and
    forgetting the register is the ordinary way to reach that state.
    """
    current = {D, S, MX, H}
    keyed = {rel for per in OBSOLETE_BY_SURFACE.values() for rel in per}
    stale = sorted(keyed - current)
    if stale:
        problems.append(
            f"obsolete register: {len(stale)} surface key(s) name a document not under review "
            f"({', '.join(stale[:3])}). The register is keyed by filename, so these entries match "
            f"nothing and the absence contract on superseded values is not running. Remap the "
            f"keys to the current surfaces.")
    return len(current)


def check_percentage_census(surfaces, problems) -> int:
    """Every decimal percentage on a submission surface is bound, or excluded IN ONE CLAUSE.

    The census is only durable if the code enforces it and the exclusions are occurrence-specific.
    A percentage added later must fail a mutation, or appear inside a clause named here with its
    reason; the same token anywhere else is unaccounted and reported.
    """
    n = 0
    for surface in (D, MX, S):
        txt = surfaces.get(surface)
        if txt is None:
            continue
        n += 1
        allowed = [(clause, tok) for surf, clause, tok, _why in CENSUS_EXCLUDED_OCCURRENCES
                   if surf == surface]
        unaccounted = []
        # Tokens a ROW binds are protected by the occurrence-count contract: an extra copy makes
        # the count disagree. Tokens only a TEMPLATE binds are not counted anywhere, so each
        # occurrence must sit inside a clause some check actually required.
        row_bound = _CENSUS_BOUND - {t for tpl in _CENSUS_TEMPLATES
                                     for t in re.findall(r"(?<![\d.])(\d{1,3}\.\d+%)", tpl)}
        for m in re.finditer(r"(?<![\d.])(\d{1,3}\.\d+)%", txt):
            tok = m.group(0)
            if tok in row_bound:
                continue
            if any(tok in tpl and any(pos <= m.start() and m.end() <= pos + len(tpl)
                                      for pos in _spans(txt, tpl))
                   for tpl in _CENSUS_TEMPLATES):
                continue
            # The occurrence must sit INSIDE an exempting clause, not merely near one. Testing a
            # window lets a new claim be whitewashed by quoting the clause beside it.
            if any(tok == t and any(pos <= m.start() and m.end() <= pos + len(c)
                                    for pos in _spans(txt, c))
                   for c, t in allowed):
                continue
            unaccounted.append(f"{tok} (…{txt[max(0, m.start() - 28):m.end() + 10].strip()}…)")
        if unaccounted:
            problems.append(
                f"percentage census: {Path(surface).name} carries {len(unaccounted)} decimal "
                f"percentage occurrence(s) that are neither bound nor excluded in a named clause: "
                f"{'; '.join(unaccounted[:4])}. Bind each, or add an occurrence entry with the "
                f"reason it cannot move.")

        # An exclusion that excuses nothing is not harmless. It is counted as coverage by anyone
        # reading the registry, and it hides the fact that the occurrence it names either moved,
        # was reworded, or never matched the census pattern in the first place.
        for clause, tok in allowed:
            if not any(m2.group(0) == tok
                       and any(pos <= m2.start() and m2.end() <= pos + len(clause)
                               for pos in _spans(txt, clause))
                       for m2 in re.finditer(r"(?<![\d.])(\d{1,3}\.\d+)%", txt)):
                problems.append(
                    f"percentage census: {Path(surface).name} excludes {tok!r} in the clause "
                    f"{clause[:40]!r}, but no census occurrence there matches it. The entry "
                    f"excuses nothing and reads as coverage; remove it or fix the clause.")
    return n

def check_semantic_claims(surfaces, problems) -> int:
    """Bind conclusion-bearing claims to their POPULATION, not merely to the page.

    A presence check asks whether "28 of 413" appears somewhere in the interpreting paragraph. It
    is therefore satisfied when the counts are attached to the wrong populations, because both
    renders are still on the page and every occurrence count is unchanged. The same holds for the
    two shares taken alone. What distinguishes the true sentence from the swapped one is which
    population each figure is predicated of, so the whole predication is required here as one
    ordered template.

    Natural-language entailment does not belong in `_rounds_to`. This is a short explicit table:
    the claims whose reversal would change a conclusion, and nothing else.
    """
    pops = _dig(ART.get("censor"), "populations")
    espace = _dig(ART.get("espace"), "results.corrected_11")
    if pops is None or espace is None:
        problems.append("semantic claim: leadlag_censoring.json or event_spacing_recompute.json "
                        "is not loaded")
        return 1

    ab, ve = pops["abusive"], pops["verified"]

    def _sh(v):
        return f"{v:.1f}"

    templates = [
        (D, "boundary counts and shares, each bound to its population",
         f"{ab['n_boundary_responders']} of {ab['n_responders']} abusive responders "
         f"({_sh(ab['boundary_share_pct'])}%) and {ve['n_boundary_responders']} of "
         f"{ve['n_responders']} verified-human responders ({_sh(ve['boundary_share_pct'])}%)"),
        (MX, "boundary counts and shares, each bound to its population",
         f"**{ab['n_boundary_responders']} of {ab['n_responders']}** abusive responders sit at "
         f"the boundary (**{_sh(ab['boundary_share_pct'])}%**) and "
         f"**{ve['n_boundary_responders']} of {ve['n_responders']}** verified-human responders "
         f"(**{_sh(ve['boundary_share_pct'])}%**)"),
        (MX, "both Kaplan-Meier medians in the order the sentence states them",
         f"**{ab['median_km']:.0f} days** for abusive automation and "
         f"**{ve['median_km']:.0f} days** for verified humans"),
    ]

    n = 0
    for surface, what, want in templates:
        txt = surfaces.get(surface)
        if txt is None:
            continue
        n += 1
        if want not in txt:
            problems.append(
                f"semantic claim: {Path(surface).name} does not carry {what} as stated by the "
                f"artifact. Required verbatim: {want!r}")

    # The branch-weight paragraph states five quantities about two different things: how much
    # each branch contributes, and how far the block channel reaches and how concentrated it is.
    # Every one is a number bound to a role, and presence alone lets the roles be exchanged, which
    # yields a false sentence built entirely from true tokens. Each is required as one ordered
    # template.
    # The retreat cut sweep. Bare presence cannot carry it: "15.4%" also occurs as a median share
    # of calendar days and "3.9%" inside "33.9%", so both endpoints are satisfied by coincidence.
    # The 1-NN match rate: a count pair plus a percentage, all three of which drifted together
    # while nothing bound any of them.
    # The circular-shift null cluster. Six values on each surface in obs / null-mean / null-max
    # role structure: bare presence lets any two of them exchange places while every token stays
    # on the page, and the sentence then reports a null that beats its own observation.
    # "8.0%" is a substring of "18.0%", which appears in the same document, so presence cannot
    # hold it; and the quiet-five share is derived from its two counts rather than stored.
    osc = _dig(ART.get("summary"), "C_trajectory")
    if osc is not None:
        txt = surfaces.get(MX)
        if txt is not None:
            n += 1
            want = (f"({osc['bot']['pct_oscillatory']:.1f}% against "
                    f"{osc['human']['pct_oscillatory']:.1f}%)")
            _note_bound(want)
            _note_template(want)
            if want not in txt:
                problems.append(
                    f"semantic claim: {Path(MX).name} does not state the oscillatory shares as the "
                    f"artifact does. Required verbatim: {want!r}")

    # The disappearance Wilson intervals. Two-decimal percentages in an ordered pair: the
    # endpoints can trade places while both tokens stay on the page, and an interval that runs
    # downward is a different claim from the one the artifact supports.
    # The agent-weighted disappearance rate, derived from two stored counts rather than stored.
    da = _dig(ART.get("disap"), "abusive")
    if da is not None and da.get("agents_with_events"):
        txt = surfaces.get(D)
        if txt is not None:
            n += 1
            want = (f"{100.0 * da['agents_that_vanish'] / da['agents_with_events']:.2f}% "
                    f"({da['agents_that_vanish']} of {da['agents_with_events']:,})")
            _note_bound(want)
            _note_template(want)
            if want not in txt:
                problems.append(
                    f"semantic claim: {Path(D).name} does not state the agent-weighted "
                    f"disappearance rate as the artifact does. Required verbatim: {want!r}")

    dsp = ART.get("disap")
    if dsp is not None:
        for label, lo, hi in (
            ("abusive", *(_dig(dsp, "abusive.wilson") or (None, None))),
            ("verified", *(_dig(dsp, "verified.wilson") or (None, None))),
            ("followup", *(_dig(dsp, "followup.5.wilson") or (None, None))),
        ):
            if lo is None:
                continue
            txt = surfaces.get(D)
            if txt is None:
                continue
            n += 1
            want = f"[{lo:.2f}%, {hi:.2f}%]"
            _note_bound(want)
            _note_template(want)
            if want not in txt:
                problems.append(
                    f"semantic claim: {Path(D).name} does not carry the {label} disappearance "
                    f"interval as the artifact states it. Required verbatim: {want!r}")

    q5 = _dig(ART.get("disap"), "quiet5")
    if q5 is not None and q5.get("n"):
        txt = surfaces.get(D)
        if txt is not None:
            n += 1
            want = (f"**{q5['returned']} of them, "
                    f"{100.0 * q5['returned'] / q5['n']:.1f}%, reappear afterwards**")
            _note_bound(want)
            _note_template(want)
            if want not in txt:
                problems.append(
                    f"semantic claim: {Path(D).name} does not state the quiet-five return share as "
                    f"the artifact does. Required verbatim: {want!r}")

    # The abusive arm's median observation density. Bound as a template because the bare token
    # is shared with the retreat sweep elsewhere on the page, and a shared token is accounted for
    # by the census while the sentence carrying it goes unchecked.
    # The control-arm disappearance rate. A percentage whose only guard is the census's
    # unaccounted-token report can be mutated INTO an excluded token and pass everything; the
    # verbatim binding closes that route for the one sentence where it was demonstrated.
    # The FrictionLab calibration note and its generated parameter page. Both were loaded as
    # surfaces and read by nothing, so the anchors could drift from the artifacts they cite while
    # the manifest reported clean. Each estimate carries its own cohort, and the pair is bound
    # together so a regenerated page cannot lag the calibration it is generated from.
    cb = _dig(ART.get("combined"), "did_combined")
    cs = _dig(ART.get("combined"), "did_combined_stealth")
    gr = _dig(ART.get("grid"), "summary")
    wf = _dig(ART.get("grid"), "window_free")
    if cb and cs and gr and wf:
        want = (f"-{abs(cb['did']):.3f} on {cb['n_bot']:,} abusive agents against "
                f"{cb['n_ctrl']:,} verified humans, and -{abs(cs['did']):.3f} stealth-only on "
                f"{cs['n_bot']:,} against {cs['n_ctrl']:,}")
        span = (f"span -{abs(gr['max']):.3f} to -{abs(gr['min']):.3f}")
        _sg = wf['SEGMENTED']
        _pr = wf['FULL_PRE_LINEAR']
        _sg = _sg['did'] if isinstance(_sg, dict) else _sg
        _pr = _pr['did'] if isinstance(_pr, dict) else _pr
        seg = f"segmented window-free estimator gives -{abs(_sg):.3f}"
        pre = f"pretrend extrapolation gives -{abs(_pr):.3f}"
        for surface in (TOML, PARM):
            txt = surfaces.get(surface)
            if txt is None:
                continue
            for clause in (want, span, seg, pre):
                n += 1
                _note_bound(clause)
                if clause not in txt:
                    problems.append(
                        f"semantic claim: {Path(surface).name} does not carry the observability "
                        f"anchor as the artifacts state it. Required verbatim: {clause!r}")

    # The supplement's branch split. "3.9%" is also the retreat sweep's token on another surface,
    # so a row cannot distinguish them; the clause carries both branch shares in order.
    cshr = _dig(ART.get("contain"), "shares")
    if cshr is not None:
        txt = surfaces.get(S)
        if txt is not None:
            n += 1
            want = (f"{cshr['branch|fresh|f_le_1|visible_only']['agents_pct']:.1f}% of agents "
                    f"escape on the visible branch alone and "
                    f"{cshr['branch|fresh|f_le_1|evasion_only']['agents_pct']:.1f}%")
            _note_bound(want)
            _note_template(want)
            if want not in txt:
                problems.append(
                    f"semantic claim: {Path(S).name} does not state the branch split as the "
                    f"artifact does. Required verbatim: {want!r}")

    dv = _dig(ART.get("disap"), "verified")
    if dv is not None and dv.get("evaluable"):
        txt = surfaces.get(D)
        if txt is not None:
            n += 1
            want = f"{dv['pct']:.2f}%** of control episodes ({dv['gone']} of {dv['evaluable']:,}"
            _note_bound(want)
            _note_template(want)
            if want not in txt:
                problems.append(
                    f"semantic claim: {Path(D).name} does not state the control-arm disappearance "
                    f"rate as the artifact does. Required verbatim: {want!r}")

    md = _dig(ART.get("leadlag"), "populations.abusive.median_density")
    if md is not None:
        txt = surfaces.get(D)
        if txt is not None:
            n += 1
            want = f"active on a median {md * 100:.1f}% of calendar days"
            _note_bound(want)
            _note_template(want)
            if want not in txt:
                problems.append(
                    f"semantic claim: {Path(D).name} does not state the median observation "
                    f"density as the artifact does. Required verbatim: {want!r}")

    ll = _dig(ART.get("leadlag"), "populations")
    if ll is not None:
        def _sh(pop, key):
            return 100.0 * ll[pop]["null"]["responder_share"][key]

        for surface, want in (
            (D,  f"share is **{_sh('abusive','obs'):.1f}%** against a null mean of "
                 f"{_sh('abusive','null_mean'):.1f}% reaching {_sh('abusive','null_max'):.1f}%"),
            (D,  f"({_sh('verified','obs'):.1f}% against a mean of "
                 f"{_sh('verified','null_mean'):.1f}%"),
            (MX, f"share of {_sh('abusive','obs'):.1f}% stands against a null mean of "
                 f"{_sh('abusive','null_mean'):.1f}% reaching {_sh('abusive','null_max'):.1f}%"),
            (MX, f"({_sh('verified','obs'):.1f}% against a mean of "
                 f"{_sh('verified','null_mean'):.1f}%"),
            (MX, f"({_sh('pothuman','obs'):.1f}% against {_sh('pothuman','null_mean'):.1f}%"),
            # The companion restates the same figures in a summary sentence. A restatement
            # is a second claim: unbound, it can drift away from the sentence it summarises.
            (MX, f"The abusive share, {_sh('abusive','obs'):.1f}% on the rotatable cohort, sits above a null mean of {_sh('abusive','null_mean'):.1f}% (maximum {_sh('abusive','null_max'):.1f}%"),
            (MX, f"the verified-human share, {_sh('verified','obs'):.1f}%, clears its own null"),
        ):
            txt = surfaces.get(surface)
            if txt is None:
                continue
            n += 1
            _note_bound(want)
            _note_template(want)
            if want not in txt:
                problems.append(
                    f"semantic claim: {Path(surface).name} does not state the circular-shift null "
                    f"as the artifact does. Required verbatim: {want!r}")

    dm = _dig(ART.get("dose"), "populations.abusive")
    if dm is not None and dm.get("n_heavy"):
        txt = surfaces.get(D)
        if txt is not None:
            n += 1
            pct = 100.0 * dm["n_matched"] / dm["n_heavy"]
            want = (f"{pct:.1f}% of heavy events matched "
                    f"({dm['n_matched']:,}/{dm['n_heavy']:,})")
            _note_bound(want)
            _note_template(want)
            if want not in txt:
                problems.append(
                    f"semantic claim: {Path(D).name} does not state the heavy-event match rate as "
                    f"the artifact does. Required verbatim: {want!r}")
        txt = surfaces.get(MX)
        if txt is not None:
            n += 1
            want = f"{100.0 * dm['n_matched'] / dm['n_heavy']:.1f}% matched"
            _note_bound(want)
            _note_template(want)
            if want not in txt:
                problems.append(
                    f"semantic claim: {Path(MX).name} does not state the match rate as the "
                    f"artifact does. Required verbatim: {want!r}")

    rc = _dig(ART.get("retreatcut"), "retreat_share.abusive")
    if rc is not None:
        for surface, verb, cut in ((D, "moves", "30% cut"), (MX, "runs", "30% volume-drop cut")):
            txt = surfaces.get(surface)
            if txt is None:
                continue
            n += 1
            want = (f"{verb} from {rc['30'] * 100:.1f}% at a permissive {cut} to "
                    f"{rc['70'] * 100:.1f}% at a strict 70% cut")
            _note_bound(want)
            _note_template(want)
            if want not in txt:
                problems.append(
                    f"semantic claim: {Path(surface).name} does not state the retreat cut sweep as "
                    f"the artifact does. Required verbatim: {want!r}")

    br = _dig(ART.get("branch"), "arm_of_record")
    if br is not None:
        for surface, want in (
            (MX, f"the arm receives {br['responses_429']:,.0f} responses with status 429 against "
                 f"{br['responses_401_403']:,.0f} 401/403 responses"),
            (MX, f"only {br['pct_events_from_429_branch_only']:.2f}% of event days, "
                 f"{br['events_429_branch_only']:,} of {br['events_total']:,}, rising to "
                 f"{br['pct_events_touching_429_branch']:.2f}% if the "
                 f"{br['events_both_same_day']} days that trigger both branches are counted "
                 f"with it"),
            (MX, f"({br['agents_ever_blocked']:,} of 7,812, "
                 f"{br['pct_agents_ever_blocked']:.2f}%) ever receiving a block"),
            (MX, f"a single agent carrying {br['largest_agent_block_share_pct']:.1f}% of the "
                 f"arm's block responses"),
            (D,  f"the 429 branch contributes "
                 f"{br['pct_events_from_429_branch_only']:.1f}% of event days"),
            (D,  f"only {br['agents_ever_blocked']:,} agents of 7,812 ever receive one"),
            (D,  f"one of which absorbs {br['largest_agent_block_share_pct']:.1f}% of every "
                 f"block response the arm receives"),
        ):
            txt = surfaces.get(surface)
            if txt is None:
                continue
            n += 1
            _note_bound(want)
            _note_template(want)
            if want not in txt:
                problems.append(
                    f"semantic claim: {Path(surface).name} does not state the branch-weight claim "
                    f"as the artifact does. Required verbatim: {want!r}")

        # Two quantifiers in that paragraph are claims about which side of a half or a quarter a
        # share falls on. They carry no digits of their own, so nothing else can check them.
        for surface, share, small, large, where in (
            (D, br["pct_agents_ever_blocked"], "blocks reach a minority of the arm",
             "blocks reach a majority of the arm", "reach of the block channel"),
            (MX, br["pct_agents_ever_blocked"], "fewer than a quarter of arm agents",
             "more than a quarter of arm agents", "reach of the block channel"),
        ):
            txt = surfaces.get(surface)
            if txt is None:
                continue
            n += 1
            threshold = 50.0 if "minority" in small else 25.0
            want_s, wrong_s = ((small, large) if share < threshold else (large, small))
            if wrong_s in txt:
                problems.append(
                    f"semantic claim: {Path(surface).name} says {wrong_s!r}, but the {where} is "
                    f"{share:.2f}%, which is {'below' if share < threshold else 'above'} "
                    f"{threshold:.0f}%")
            elif want_s not in txt:
                problems.append(
                    f"semantic claim: {Path(surface).name} does not state the {where} "
                    f"quantifier. At {share:.2f}% the artifact requires {want_s!r}")

    # A comparative is a claim about a sign, and flipping the word flips the finding while leaving
    # every number on the page untouched. The two signed differences that license it are read from
    # the artifact, so the word is checked against the data rather than against itself.
    d_ver = espace["channels"]["z_robots"]["did"]
    d_pot = espace["channels_pothuman"]["z_robots"]["did"]
    txt = surfaces.get(D)
    if txt is not None:
        n += 1
        if d_ver < 0 and d_pot < 0:
            want_word, wrong_word = "smaller", "larger"
        elif d_ver > 0 and d_pot > 0:
            want_word, wrong_word = "larger", "smaller"
        else:
            want_word = wrong_word = None
            problems.append(
                f"semantic claim: the abusive robots.txt differences disagree in sign "
                f"({d_ver:+.4f} against verified, {d_pot:+.4f} against the sensitivity arm), so "
                f"the sentence cannot claim a single direction")
        if want_word:
            claim = f"whose own movement is significantly {want_word} than the humans'"
            wrong = f"whose own movement is significantly {wrong_word} than the humans'"
            if wrong in txt:
                problems.append(
                    f"semantic claim: the text reads {wrong_word!r}, but both signed differences "
                    f"are {'negative' if want_word == 'smaller' else 'positive'} "
                    f"({d_ver:+.4f}, {d_pot:+.4f}), which licenses {want_word!r}")
            elif claim not in txt:
                problems.append(
                    f"semantic claim: the direction-bearing clause was not found. The signed "
                    f"differences ({d_ver:+.4f}, {d_pot:+.4f}) require {claim!r}")
    return n


def check_ordered_claims(surfaces, problems) -> int:
    """Claims whose truth is the ORDER of correct numbers, not the numbers themselves.

    Presence rows and occurrence counts are order-blind by construction: transposing an interval or
    swapping two cohort sizes leaves every token and every count identical, so a false statement
    assembled entirely from correct values passes the whole numeric gate. Pinning one phrase per
    site does not scale and misses the next one, so the structural properties are checked instead:

      * a confidence interval must ascend;
      * a range stated as "X to Y" must run from the smaller magnitude to the larger;
      * a median quoted with a range must lie inside it.

    Each is a property of the sentence, independent of which field it came from, so a new
    transposition anywhere on any surface fails without anyone remembering to add a row.
    """
    n = 0
    NUM = r"[+-−]?\d+(?:\.\d+)?"
    for rel, txt in sorted(surfaces.items()):
        if txt is None or not rel.endswith(".md"):
            continue
        for m in re.finditer(r"\[\s*(" + NUM + r")\s*,\s*(" + NUM + r")\s*\]", txt):
            a, b = (float(x.replace("−", "-")) for x in m.groups())
            n += 1
            if a > b:
                line = txt[:m.start()].count("\n") + 1
                problems.append(f"ordered claim: {rel}:{line} interval [{m.group(1)}, "
                                f"{m.group(2)}] descends; an interval must ascend")
        # A RANGE states bounds; a MOVEMENT states a transition, and a decreasing movement is
        # perfectly true ("pulls v* from 1.100 to 0.767"). Only range cues are checked: an
        # explicit "ranges/runs from", a "by X to Y" span, or a pair quoted with its median.
        RANGE_CUE = (r"(?:(?:ranges?|runs?|varies|spans?)\s+from|by)\s+(" + NUM +
                     r")\s+(?:to|and)\s+(" + NUM + r")(?:\s*,?\s*with a median of\s+("
                     + NUM + r"))?")
        # "from X to Y with a median of Z" is unambiguously a range whatever verb precedes it:
        # a movement does not have a median.
        MEDIAN_CUE = (r"from\s+(" + NUM + r")\s+to\s+(" + NUM +
                      r")\s*,?\s*with a median of\s+(" + NUM + r")")
        for m in re.finditer("(?:" + RANGE_CUE + ")|(?:" + MEDIAN_CUE + ")", txt):
            g = [x for x in m.groups() if x is not None]
            m = type("M", (), {"groups": lambda self, g=g: tuple(g + [None] * (3 - len(g))),
                               "group": lambda self, i, g=g: g[i - 1] if i <= len(g) else None,
                               "start": lambda self, st=m.start(): st})()
            a, b = (abs(float(x.replace("−", "-"))) for x in m.groups()[:2])
            n += 1
            if a > b:
                line = txt[:m.start()].count("\n") + 1
                problems.append(f"ordered claim: {rel}:{line} range runs from {m.group(1)} to "
                                f"{m.group(2)}, larger magnitude first")
            med = m.group(3)
            if med is not None:
                mv = abs(float(med.replace("−", "-")))
                n += 1
                if not (a <= mv <= b):
                    line = txt[:m.start()].count("\n") + 1
                    problems.append(f"ordered claim: {rel}:{line} median {med} lies outside the "
                                    f"range it is quoted with")
    return n


def check_cited_paths(surfaces, problems) -> int:
    """Every artifact path a manuscript tells a reader to open must exist.

    The rows verify what an artifact CONTAINS. Nothing was verifying that a path cited in the prose
    resolves at all, so a manuscript could point a reviewer at a file that has moved - which one
    had: the supplement cited `frictionlab/attacker.py` for a module that lives one directory
    deeper. A reviewer opens that; a value checker never does.
    """
    pat = re.compile(r"`((?:analysis|outputs|empirical-support|frictionlab|artifacts)/"
                     r"[A-Za-z0-9_./-]+\.(?:json|csv|py|png|md|toml))`")
    seen: set[str] = set()
    for rel, txt in sorted(surfaces.items()):
        if txt is None or not rel.endswith(".md"):
            continue
        # A path resolves from the root its own reader works in. The manuscripts are read from the
        # paper root, where the simulator sits at `frictionlab/frictionlab/`; the simulator's own
        # docs ship inside the package and are read from ITS root, where the same module is
        # `frictionlab/`. Resolving both against one root reports one of them as broken and invites
        # a "fix" that breaks the other.
        roots = ((PAPER / "frictionlab", PAPER, SUP, ROOT) if rel.startswith("frictionlab/")
                 else (PAPER, SUP, ROOT))
        for cited in sorted(set(pat.findall(txt))):
            if cited in seen:
                continue
            seen.add(cited)
            if not any((base / cited).exists() for base in roots):
                problems.append(f"cited path: {rel} points at {cited}, which does not exist")
    return len(seen)


def check_figure_coverage(surfaces, problems) -> int:
    """Every cited PNG is declared, exists, and is no older than the data behind it.

    Two failures this catches that a render-vs-reference hash cannot: a figure cited by a
    manuscript that nothing in the manifest knows about, and a figure whose pixels are older than
    the artifact its generator reads. The second is the current one.
    """
    declared = {name: src for name, src in FIGURE_SOURCES}
    cited: dict[str, list[str]] = {}
    for rel, txt in surfaces.items():
        if txt is None or not rel.endswith(".md"):
            continue
        for name in set(re.findall(r"[A-Za-z0-9_]+\.png", txt)):
            cited.setdefault(name, []).append(rel)

    for name, rels in sorted(cited.items()):
        if name not in declared:
            problems.append(f"figure coverage: {name} is cited by {', '.join(rels)} but is not "
                            f"declared in FIGURE_SOURCES")
            continue
        render = SUP / "outputs" / "figures" / name
        if not render.exists():
            problems.append(f"figure coverage: {name} is cited but the render is missing")
            continue
        src = declared[name]
        if src is None:
            continue
        if not src.exists():
            problems.append(f"figure coverage: {name}'s declared source {src.name} is missing")
            continue
        # A generator that writes its table and its figure in the same run stamps them within the
        # same second, and sub-second ordering is not staleness. A render that is genuinely behind
        # its data is behind it by hours or months, so a few seconds of slack costs nothing.
        if render.stat().st_mtime < src.stat().st_mtime - 5.0:
            problems.append(f"figure coverage: {name} is OLDER than its source {src.name} "
                            f"- regenerate it rather than re-captioning it")
    for name in sorted(set(declared) - set(cited)):
        problems.append(f"figure coverage: {name} is declared but no manuscript cites it")
    return len(cited)


def check_coarse_blocks(surfaces, problems) -> int:
    """The abstract and the highlights, where a coarse restatement of a dead number hides."""
    n = 0
    for rel, start, end in COARSE_BLOCKS:
        txt = surfaces.get(rel)
        if txt is None:
            problems.append(f"coarse block: {rel} not readable")
            continue
        if start is not None:
            i = txt.find(start)
            j = txt.find(end, i + 1) if i >= 0 else -1
            if i < 0 or j < 0:
                problems.append(f"coarse block: {start!r}..{end!r} not found in {rel}")
                continue
            txt = txt[i:j]
        bad = []
        for m in _NUM.finditer(txt):
            tok, pct = m.group(1), m.group(2) == "%"
            bare = tok.lstrip("+" + M + "-")
            if bare in COARSE_ALLOW or bare.replace(",", "") in COARSE_ALLOW:
                continue
            if not _rounds_to(tok, pct):
                bad.append((tok + m.group(2), txt[max(0, m.start() - 45):m.end() + 20]))
        if bad:
            for tok, ctx in bad:
                problems.append(f"{rel}: {tok} matches no current artifact field "
                                f"at its own precision - ...{ctx.strip()}...")
        n += 1
    return n


def check_located(surfaces, problems) -> int:
    """A value required inside the sentence that interprets it, not merely on the page."""
    n = 0
    for art, path, render, wants, note in LOCATED:
        val = _dig(ART.get(art), path)
        if val is None:
            problems.append(f"{note}: field not found ({art}:{path})")
            continue
        forms = render(val)
        forms = forms if isinstance(forms, list) else [forms]
        for rel, anchor in wants:
            txt = surfaces.get(rel)
            if txt is None or anchor not in txt:
                problems.append(f"{note}: anchor {anchor!r} not found in {rel}")
                continue
            # An anchor that matches twice silently protects whichever paragraph comes first,
            # which is how a figure caption went unchecked while an earlier mention passed.
            if txt.count(anchor) > 1:
                problems.append(f"{note}: anchor {anchor!r} is not unique in {rel} "
                                f"({txt.count(anchor)} matches) - choose a distinctive phrase")
                continue
            # Scope is the paragraph holding the anchor. Sentence scope is not usable here:
            # splitting on "." lands inside every decimal the check exists to protect.
            k = txt.index(anchor)
            lo = txt.rfind("\n", 0, k) + 1
            hi = txt.find("\n", k + len(anchor))
            window = txt[lo:hi if hi > 0 else len(txt)]
            if not any(f in window for f in forms):
                problems.append(f"{note}: {'/'.join(forms)} is not in the paragraph anchored "
                                f"by {anchor!r} in {rel}")
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--relearn", action="store_true",
                    help="rewrite the expected occurrence counts from the current "
                         "tree. Do this ONLY when the tree has just been verified.")
    a = ap.parse_args()

    COUNTS = HERE / "numbers_manifest_counts.json"
    expected = json.loads(COUNTS.read_text()) if COUNTS.exists() else {}
    learned: dict[str, int] = {}
    unlearned: list[str] = []

    surfaces = {}
    for rel in {r for row in ROWS for r in row[3]} | {TOML, PARM}:
        p = PAPER / rel
        surfaces[rel] = p.read_text(encoding="utf-8") if p.exists() else None

    # This checker compares manuscripts against artifacts, and the manuscripts are not distributed
    # with the verification bundle. Reporting their absence as a defect per row turns "there is
    # nothing here to check" into hundreds of failures and tells a reader the package is broken.
    manuscripts = [r for r in surfaces if r.startswith("draft/")]
    if manuscripts and not any(surfaces[r] for r in manuscripts):
        raise SystemExit(
            "the manuscript surfaces are not distributed with this bundle, so there is "
            "nothing here for the manifest to check against.\n"
            f"  expected under {PAPER}/draft\n"
            "What this bundle supports is reading the artifact fields the manuscripts cite "
            "and re-running the analyses that produce them; see the README.")

    out, missing = [], 0
    for art, path, render, want, note in ROWS:
        blob = ART.get(art)
        val = _dig(blob, path) if blob else None
        if val is None:
            print(f"  ??  {note:26s} field not found: {art}:{path}")
            out.append({"note": note, "field": f"{art}:{path if isinstance(path, str) else '.'.join(map(str, path))}", "status": "field-missing"})
            missing += 1
            continue
        s = render(val)
        forms = s if isinstance(s, list) else [s]
        _note_bound(forms)
        gone = []
        for rel in want:
            txt = surfaces.get(rel)
            if txt is None or not any(f in txt for f in forms):
                gone.append(rel)
                continue
            # HOW MANY times, not merely whether. Presence alone let a corrupted occurrence hide
            # behind a correct twin: -0.947 appears four times in the article, and changing one of
            # them to -0.999 passed a presence check. Counting catches that without needing a
            # hand-maintained anchor for every row.
            n_now = sum(txt.count(f) for f in forms)
            key = f"{note}|{rel}"
            if max(len(f) for f in forms) < 3:
                # A one- or two-character rendering counts digits, not values: an unrelated
                # edit moves the total and an in-place corruption does not. Presence and the
                # structural checks carry these rows.
                continue
            if a.relearn:
                learned[key] = n_now
            else:
                n_want = expected.get(key)
                if n_want is None:
                    unlearned.append(key)
                elif n_now != n_want:
                    gone.append(f"{rel} (expected {n_want} occurrence(s), found {n_now} — "
                                f"if the change is intended, re-run with --relearn)")
        # Presence is not enough. Changing ONE restatement of
        # "260 agents" to "261" still passed, because another 260 survived elsewhere in the file.
        # So also reject any OBSOLETE rendering of the same field that is still present.
        for rel in want:
            txt = surfaces.get(rel)
            if txt is None:
                continue
            for bad in _forbidden_for(note, rel):
                if _obsolete_survives(bad, txt):
                    gone.append(f"{rel} (obsolete form {bad!r} still present)")
        s = " | ".join(forms)
        status = "ok" if not gone else "STALE"
        if gone:
            missing += 1
            print(f"  !!  {note:26s} {s:>10s}  absent from: {', '.join(gone)}")
        out.append({"note": note, "field": f"{art}:{path if isinstance(path, str) else '.'.join(map(str, path))}", "rendered": s,
                    "surfaces": want, "absent_from": gone, "status": status})

    print()
    struct: list[str] = []
    n_cells = check_policy_table(surfaces.get(D) or "", ART.get("sim_policy"), struct)
    n_rank = check_ranking(surfaces.get(D) or "", ART.get("sim_head"), struct)
    n_loc = check_located(surfaces, struct)
    n_coarse = check_coarse_blocks(surfaces, struct)
    n_figcov = check_figure_coverage(surfaces, struct)
    n_paths = check_cited_paths(surfaces, struct)
    n_ord = check_ordered_claims(surfaces, struct)
    n_tab = check_response_table(surfaces, struct)
    n_sem = check_semantic_claims(surfaces, struct)
    n_census = check_percentage_census(surfaces, struct)
    n_obs = check_obsolete_register_surfaces(struct)
    n_inv = check_release_inventory(struct)
    n_struct = (n_cells + n_rank + n_loc + n_coarse + n_figcov + n_paths + n_ord
                + n_tab + n_sem + n_census + n_obs + n_inv)
    for msg in struct:
        print(f"  !!  {'structural':26s} {msg}")
    missing += len(struct)
    bad_tab = any(m.startswith("8.1 table") for m in struct)
    bad_rank = any(m.startswith("policy ranking") for m in struct)
    bad_tab = sum(m.startswith("response table") for m in struct)
    bad_ord = sum(m.startswith("ordered claim") for m in struct)
    bad_sem = sum(m.startswith("semantic claim") for m in struct)
    bad_census = sum(m.startswith("percentage census") for m in struct)
    bad_inv = sum(m.startswith("release inventory") for m in struct)
    bad_paths = sum(m.startswith("cited path") for m in struct)
    bad_figcov = sum(m.startswith("figure coverage") for m in struct)
    bad_coarse = sum(m.startswith((D + ":", H + ":", "coarse block")) for m in struct)
    bad_loc = (len(struct) - bad_coarse - bad_figcov - bad_paths - bad_inv - bad_ord - bad_tab
               - bad_sem - bad_census
               - sum(m.startswith(("8.1 table", "policy ranking")) for m in struct))
    print(f"  {'!!' if bad_tab else 'ok'}  {'Section 8.1 table':26s} {n_cells} cells against "
          f"SIM_policy_comparison.csv")
    print(f"  {'!!' if bad_rank else 'ok'}  {'policy ranking':26s} prose order against "
          f"SIM_headline.json:ranking_main")
    print(f"  {'!!' if bad_loc else 'ok'}  {'located values':26s} {n_loc} value(s) required in "
          f"the paragraph that interprets them")
    print(f"  {'!!' if bad_coarse else 'ok'}  {'abstract / highlights':26s} {n_coarse} block(s): "
          f"every number rounds to a current artifact field")
    print(f"  {'!!' if bad_figcov else 'ok'}  {'figure coverage':26s} {n_figcov} cited PNG(s) "
          f"declared, present, and newer than their sources")
    print(f"  {'!!' if bad_paths else 'ok'}  {'cited artifact paths':26s} {n_paths} path(s) cited "
          f"in prose resolve on disk")
    print(f"  {'!!' if bad_tab else 'ok'}  {'7.7 response table':26s} {n_tab} header/cell "
          f"mapping(s) bound to their population pairing")
    print(f"  {'!!' if bad_ord else 'ok'}  {'ordered claims':26s} {n_ord} interval(s), range(s) "
          f"and median(s) are correctly ordered")
    print(f"  {'!!' if bad_sem else 'ok'}  {'semantic claims':26s} {n_sem} claim(s) bound to population "
          f"and direction, not merely to the page")
    print(f"  {'!!' if bad_inv else 'ok'}  {'release inventory':26s} {n_inv} released PNG(s) are "
          f"cited-and-checked or listed with a status")

    print()
    for render_p, source_p, surface, note in FIGURES:
        if not render_p.exists() or not source_p.exists():
            print(f"  ??  {note:26s} render or source missing")
            missing += 1
            continue
        h1 = hashlib.sha256(render_p.read_bytes()).hexdigest()
        h2 = hashlib.sha256(source_p.read_bytes()).hexdigest()
        ok = h1 == h2
        if not ok:
            missing += 1
        # The hash alone passed while MX-v19 cited the STALE v2 rotation image: the `surface`
        # field was collected and never used. A render that nothing cites proves nothing.
        cited = surfaces.get(surface)
        if cited is not None and render_p.name not in cited:
            missing += 1
            ok = False
            print(f"  !!  {note:26s} {surface} does not cite {render_p.name}")
        print(f"  {'ok' if ok else '!!'}  {note:26s} {'render matches source' if ok else 'RENDER DIFFERS FROM SOURCE'}")
        out.append({"note": note, "render": str(render_p), "source": str(source_p),
                    "status": "ok" if ok else "STALE-FIGURE"})

    # Control characters: a single-backslash LaTeX command written through a shell heredoc
    # becomes a real TAB, vertical tab or CR inside the manuscript, invisible in most viewers.
    # Stale-artifact guard: the partition JSON records the hash of the pair table it was built
    # from, so a regenerated pair file with an un-regenerated JSON is caught rather than trusted.
    part = ART.get("decoy_part")
    if part and part.get("source_sha256"):
        src = SUP / "outputs" / "tables" / part.get("source", "")
        if src.exists():
            import hashlib as _h
            live = _h.sha256(src.read_bytes()).hexdigest()
            ok = live == part["source_sha256"]
            if not ok:
                missing += 1
            print(f"  {'ok' if ok else '!!'}  {'partition source hash':26s} "
                  f"{'matches ' + src.name if ok else 'STALE: regenerate decoy_specificity_partition.py'}")
            out.append({"note": "partition source hash", "status": "ok" if ok else "STALE"})

    print()
    for rel in (D, S, MX, H):
        p = PAPER / rel
        if not p.exists():
            continue
        b = p.read_bytes()
        ctrl = [m.start() for m in re.finditer(rb"[\x00-\x09\x0b\x0c\x0e-\x1f\x7f]", b)]
        lone = [m.start() for m in re.finditer(rb"\r(?!\n)", b)]
        # Unbalanced math delimiters. A "$...$" left open renders the rest of the line as
        # mathematics. Cheap to check, invisible by eye.
        _txt = b.decode("utf-8", "replace")
        odd = [k for k, l in enumerate(_txt.splitlines(), 1) if l.count("$") % 2]
        if odd:
            missing += 1
            print(f"  !!  {rel}: unbalanced $ on line(s) {odd[:5]}")
            out.append({"note": f"unbalanced math delimiter in {rel}", "status": "STALE",
                        "lines": odd[:10]})
        if ctrl or lone:
            missing += 1
            print(f"  !!  {rel}: {len(ctrl)} control chars, {len(lone)} lone CR")
            out.append({"note": f"control chars in {rel}", "status": "STALE",
                        "offsets": ctrl[:10] + lone[:10]})
        else:
            print(f"  ok  {rel}: no control characters")

    if a.relearn:
        COUNTS.write_text(json.dumps(learned, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nRELEARNED {len(learned)} occurrence counts -> {COUNTS.name}")
    elif unlearned:
        print(f"\n  ??  {len(unlearned)} row(s) have no learned count yet")
        print("      run once with --relearn on a verified tree")

    print(f"\n{len(out) + n_struct} checks, {missing} problem(s)")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"wrote {a.json}")
    # Exit nonzero so this can gate a build. A checker that always exits 0 is a report,
    # not a check.
    raise SystemExit(1 if missing else 0)


if __name__ == "__main__":
    main()
