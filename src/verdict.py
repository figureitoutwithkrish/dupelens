"""Three-way verdict rules (same / variant / different), shared by src/variant.py and src/service.py.

Kept free of faiss and torch imports so the live app can use it.
"""
import re

import numpy as np

COMBO = re.compile(r"\bcombo\b|\bkit\b|\bset of \d+|\bpack of \d+|\+|\b\d+\s*in\s*1\b", re.IGNORECASE)


def combo_mismatch(title_a, title_b):
    """True when exactly one title describes a combo, kit, set/pack or N-in-1 product."""
    return bool(COMBO.search(title_a)) != bool(COMBO.search(title_b))


def classify(prob, conflict, colour_d, combo, cfg):
    """Vectorised verdict: 'same', 'variant' or 'different' for each pair."""
    same_model = prob >= cfg["india_same_model_threshold"]
    differs = conflict.astype(bool) | combo.astype(bool) | (colour_d > cfg["colour_cut"])
    return np.where(same_model & ~differs, "same", np.where(same_model, "variant", "different"))
