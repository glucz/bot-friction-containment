"""
Torch-gated hook to the trained transformer classifier.

The model-free analyses (A, B, D, E and the behavioral part of C) need no model.
This module is the ONLY place that touches PyTorch. If torch and the trained
weights (botness_model.pt) are present, it regenerates the true continuous
botness curve b(t) per agent by reusing the classifier code from the sibling
repo (bot_training_and_inference9.py) -- so the curve matches the model in the
Mathematics-2025 paper exactly. If torch is absent, `available()` returns False
and Analysis C falls back to its model-free behavioral trajectory.

Install torch (`pip install torch`) to unlock the b(t) curves; nothing else in
the pipeline changes.
"""
from __future__ import annotations

import sys
from functools import lru_cache

import numpy as np

import config

_SEQ_LEN = 30  # must match the trained model (bot_training_and_inference9.py)


@lru_cache(maxsize=1)
def available() -> bool:
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return config.MODEL_PATH.exists()


@lru_cache(maxsize=1)
def _load():
    """Import the classifier module + load weights. Cached across calls."""
    import torch

    repo = str(config.CLASSIFIER_REPO)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    import bot_training_and_inference9 as clf  # noqa: WPS433

    model = clf.TransformerClassifier().to(clf.DEVICE)
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=clf.DEVICE))
    model.eval()
    return clf, model


def botness_curve(agent_path: str) -> np.ndarray | None:
    """Per-window P(bot) trajectory for one agent file, or None if unavailable."""
    if not available():
        return None
    try:
        import torch

        clf, model = _load()
        df = clf.parse_user_agent_file(agent_path)
        feats = clf.preprocess_features(df)
        if len(feats) < _SEQ_LEN:
            return None
        windows = np.stack([feats[i : i + _SEQ_LEN] for i in range(len(feats) - _SEQ_LEN + 1)])
        x = torch.tensor(windows, dtype=torch.float32).to(clf.DEVICE)
        with torch.no_grad():
            probs = model(x).cpu().numpy()[:, 1]
        return probs
    except Exception:
        return None
