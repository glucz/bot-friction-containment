"""
Central configuration for the empirical-support pipeline.

This pipeline extracts evidence from the AGWA-derived per-agent time series to
support the claim "bots change their tactics over time and respond to filters"
for the population-dynamics companion (Chaos, Solitons & Fractals) and
this paper (Computers & Security).

All paths are resolved relative to this file so the pipeline is location-stable.
The data currently lives in the sibling `Transformer-classifier-modelling` repo;
override DATA_DIR via the AGWA_DATA_DIR env var if it moves.
"""
from __future__ import annotations

import os
from pathlib import Path

# --- Paths -----------------------------------------------------------------
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent                      # ...\research\longtail
RESEARCH_ROOT = REPO_ROOT.parent             # ...\research

# The per-agent .bot/.human/.chrome/.pothuman files (AGWA-derived series).
DATA_DIR = Path(
    os.environ.get(
        "AGWA_DATA_DIR",
        RESEARCH_ROOT / "Transformer-classifier-modelling" / "data",
    )
)

# Trained transformer classifier (used only by the torch-gated botness hook).
MODEL_PATH = Path(
    os.environ.get(
        "AGWA_MODEL_PATH",
        RESEARCH_ROOT / "Transformer-classifier-modelling" / "botness_model.pt",
    )
)

# Pre-computed real model outputs that already exist on disk (no torch needed).
CLASSIFIER_REPO = RESEARCH_ROOT / "Transformer-classifier-modelling"
CHROME_RANK_CSV = CLASSIFIER_REPO / "chromes_ranked_by_distance_to_human_centroid.csv"
BOT_SCORES_JSON = CLASSIFIER_REPO / "bot_scores_summary.json"
BEST_THRESHOLD_TXT = CLASSIFIER_REPO / "best_threshold.txt"

OUT_DIR = HERE / "outputs"
FIG_DIR = OUT_DIR / "figures"
TAB_DIR = OUT_DIR / "tables"

# --- Feature schema --------------------------------------------------------
# Column order of the per-agent TSV files (no header row in the data files).
# Matches parse_user_agent_file() in bot_training_and_inference9.py.
COLUMNS = [
    "day", "hits", "p404", "p200", "robots",
    "img", "html", "js", "php", "folder",
    "ip_ent", "dom_ent", "ctry_ent", "url_ent", "husource",
]
# `husource` = per-day Hungarian share of the agent's source IPs, from the AGWA
# `ip` table's `i_ccode` field ('HU'); each IP was geolocated individually when
# the panel was built. Recorded here 2026-08-16 from the study author, because
# the generator that wrote the per-agent TSVs is NOT in this research tree and
# the column is otherwise undefined anywhere in the code. Any re-derivation of
# the 14-feature input for agents that lack a TSV must reproduce it exactly, or
# the resulting botness scores are not comparable with the published ones.

# Behavioral features (everything except the time index and raw volume).
PROPORTION_FEATURES = ["p404", "p200", "robots", "img", "html", "js", "php", "folder"]
ENTROPY_FEATURES = ["ip_ent", "dom_ent", "ctry_ent", "url_ent"]
BEHAVIOR_FEATURES = PROPORTION_FEATURES + ENTROPY_FEATURES

# Distribution-spread features whose increase signals evasion (IP rotation,
# multi-domain / multi-country fan-out, request diversification).
EVASION_FEATURES = ["ip_ent", "dom_ent", "ctry_ent", "url_ent"]

# File-extension -> class label.
CLASSES = {
    "bot": "*.bot",            # self-identified (RFC 9309) bots
    "human": "*.human",        # verified humans (control group)
    "chrome": "*.chrome",      # real browsers, legit but unverified
    "pothuman": "*.pothuman",  # potential humans
}

# Treatment vs control for differential designs.
TREATMENT_CLASSES = ["bot"]
CONTROL_CLASSES = ["human"]

# --- Analysis hyperparameters ---------------------------------------------
MIN_ACTIVE_DAYS = 20        # agents with fewer active days are excluded from trajectory stats
EVENT_WINDOW = 5            # +/- days around a filter-pressure event (Analysis B)
FILTER_PRESSURE_Z = 1.5     # z-score of failure-rate that flags a filter-pressure event
N_STRATEGY_CLUSTERS = 3     # visible / evasive / mimicry (the population-dynamics companion three-strategy model)
RANDOM_STATE = 42
BOOTSTRAP_N = 2000          # bootstrap resamples for CIs


def ensure_dirs() -> None:
    for d in (OUT_DIR, FIG_DIR, TAB_DIR):
        d.mkdir(parents=True, exist_ok=True)
