"""
guardrail/config.py
--------------------
Centralised configuration for the Dual-Gate Medical Guardrail.
All values can be overridden at ``GuardrailPipeline`` construction time;
these serve as sensible, tested defaults.
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Gate 1 – Text Intent Classifier
# ---------------------------------------------------------------------------

# HuggingFace zero-shot pipeline model.
# Using facebook/bart-large-mnli: strong NLI model, works offline after
# first download, ~400 MB. Switch to 'cross-encoder/nli-MiniLM2-L6-H768'
# for a much faster, lighter (~80 MB) alternative.
INTENT_MODEL_NAME: str = "facebook/bart-large-mnli"

# Candidate labels used for the "medical" hypothesis.
MEDICAL_CANDIDATE_LABELS: list[str] = [
    "medical question",
    "radiology question",
    "clinical diagnosis",
    "patient symptom",
    "anatomy question",
    "pathology question",
]

# Candidate labels used for the "non-medical" hypothesis.
NON_MEDICAL_CANDIDATE_LABELS: list[str] = [
    "general knowledge",
    "entertainment",
    "cooking",
    "politics",
    "sports",
    "technology",
]

# Minimum aggregated score the *medical* hypothesis must achieve.
# Below this value Gate 1 fires and the request is refused.
# Tuning guide:
#   0.40 → permissive (almost no false rejections on ambiguous queries)
#   0.55 → default   (good balance, rejects clearly non-medical)
#   0.70 → strict    (may reject borderline clinical terminology)
INTENT_THRESHOLD: float = 0.30

# ---------------------------------------------------------------------------
# Gate 2 – Softmax Confidence Gate
# ---------------------------------------------------------------------------

# Top-1 Softmax probability at the first decoder step must exceed this
# threshold for the image to be considered a valid medical scan.
# Tuning guide:
#   0.20 → very permissive (almost no images refused)
#   0.30 → default         (catches natural photos, cartoons, blank images)
#   0.45 → strict          (may refuse low-quality or unusual scans)
#   0.60 → very strict     (only high-confidence medical images pass)
CONFIDENCE_THRESHOLD: float = 0.30

# ---------------------------------------------------------------------------
# VQA Model paths (mirrors inference.py — kept here so guardrail/ is
# self-contained and does not hard-code paths in multiple places)
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHECKPOINT_PATH: str = os.path.join(_REPO_ROOT, "med_pretrain_29_rad_34.pth")
CONFIG_PATH: str = os.path.join(_REPO_ROOT, "configs", "VQA.yaml")
TEXT_ENCODER: str = "bert-base-uncased"
TEXT_DECODER: str = "bert-base-uncased"
