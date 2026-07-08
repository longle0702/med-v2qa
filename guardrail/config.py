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
# Gate 1 – Text Intent Classifier (Zero-Shot)
# ---------------------------------------------------------------------------
INTENT_MODEL_PATH: str = "typeform/distilbert-base-uncased-mnli"

# Minimum zero-shot probability for the label 'medical question'
INTENT_THRESHOLD: float = 0.30

# ---------------------------------------------------------------------------
# Gate 2 – CLIP Medical Image Classifier
# ---------------------------------------------------------------------------

# Use the HuggingFace Hub directly instead of a local directory.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIP_MODEL_PATH: str = "openai/clip-vit-base-patch32"

# Minimum summed softmax score across all medical prompts for Gate 2 to pass.
# CLIP distributes probability across 10 medical + 10 non-medical prompts;
# a pure random image scores ~0.50, real medical images typically score >0.65.
# Tuning guide:
#   0.45 → permissive  (flags only obviously non-medical images)
#   0.55 → default     (good balance; rejects pets, nature photos, selfies)
#   0.70 → strict      (may reject low-quality or atypical scans)
CLIP_THRESHOLD: float = 0.60

# Kept for backward-compat with any code that imports these names.
CONFIDENCE_THRESHOLD: float = CLIP_THRESHOLD
TRIAGE_YES_THRESHOLD: float = CLIP_THRESHOLD

# ---------------------------------------------------------------------------
# VQA Model paths (mirrors inference.py — kept here so guardrail/ is
# self-contained and does not hard-code paths in multiple places)
# ---------------------------------------------------------------------------

CHECKPOINT_PATH: str = os.path.join(_REPO_ROOT, "med_pretrain_29_rad_34.pth")
CONFIG_PATH: str = os.path.join(_REPO_ROOT, "configs", "VQA.yaml")
TEXT_ENCODER: str = "bert-base-uncased"
TEXT_DECODER: str = "bert-base-uncased"
