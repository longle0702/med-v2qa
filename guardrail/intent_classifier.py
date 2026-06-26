"""
guardrail/intent_classifier.py
--------------------------------
Gate 1 – Text Intent Classifier.

Uses a HuggingFace zero-shot-classification pipeline to determine whether
a user's question is medical/clinical in nature.  The classifier is loaded
lazily on first use so that importing this module has no startup cost when
only Gate 2 is used in isolation.

Design notes
------------
- The pipeline uses an NLI model (BART-large-MNLI by default) with two
  hypothesis templates:
      * "This text is a medical question about {label}."
      * compared against both MEDICAL and NON_MEDICAL candidate labels.
- We aggregate the scores for all medical labels and compare the *maximum
  medical score* against INTENT_THRESHOLD.  Using the maximum (rather than
  the sum) avoids label dilution when many candidates are provided.
- The classifier runs entirely on CPU by default so it does not compete
  with the GPU-resident VQA model.
"""

from __future__ import annotations

import logging
from typing import Optional

# Module-level import so @patch("guardrail.intent_classifier.pipeline") works in tests.
# The actual model weights are only downloaded on first _load() call.
from transformers import pipeline

from guardrail.config import (
    INTENT_MODEL_NAME,
    INTENT_THRESHOLD,
    MEDICAL_CANDIDATE_LABELS,
    NON_MEDICAL_CANDIDATE_LABELS,
)
from guardrail.result_types import IntentResult

logger = logging.getLogger(__name__)

# All candidate labels merged for a single classifier call (more efficient
# than two separate calls).
_ALL_LABELS = MEDICAL_CANDIDATE_LABELS + NON_MEDICAL_CANDIDATE_LABELS


class MedicalIntentClassifier:
    """
    Lightweight zero-shot text classifier for medical intent detection.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier for the zero-shot pipeline.
        Defaults to ``config.INTENT_MODEL_NAME``.
    threshold:
        Minimum score the best *medical* label must achieve to pass.
        Defaults to ``config.INTENT_THRESHOLD``.
    device:
        Device string passed to the HuggingFace pipeline (e.g. ``"cpu"``,
        ``"cuda"``, ``"mps"``).  Defaults to ``"cpu"`` to keep Gate 1
        off the GPU.
    """

    def __init__(
        self,
        model_name: str = INTENT_MODEL_NAME,
        threshold: float = INTENT_THRESHOLD,
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.threshold = threshold
        self.device = device
        self._pipeline = None  # lazy-loaded

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Lazily load the zero-shot classification pipeline."""
        if self._pipeline is not None:
            return
        logger.info(
            "Loading intent classifier '%s' on device '%s' …",
            self.model_name,
            self.device,
        )
        self._pipeline = pipeline(
            "zero-shot-classification",
            model=self.model_name,
            device=self.device,
        )
        logger.info("Intent classifier ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, question: str) -> IntentResult:
        """
        Classify *question* as medical or non-medical.

        Parameters
        ----------
        question:
            The raw question string submitted by the user.

        Returns
        -------
        IntentResult
            ``passed=True`` when the best medical-label score ≥ threshold.
        """
        self._load()

        # Handle edge cases before hitting the model
        cleaned = question.strip()
        if not cleaned:
            logger.debug("Empty question string → automatic refusal.")
            return IntentResult(
                passed=False,
                label="<empty>",
                score=0.0,
                raw_scores={},
            )

        result = self._pipeline(
            cleaned,
            candidate_labels=_ALL_LABELS,
            hypothesis_template="This text is a {}.",
        )

        # Build label → score mapping
        raw_scores: dict[str, float] = dict(
            zip(result["labels"], result["scores"])
        )

        # Find the highest score among all medical labels
        best_medical_label = ""
        best_medical_score = 0.0
        for label in MEDICAL_CANDIDATE_LABELS:
            s = raw_scores.get(label, 0.0)
            if s > best_medical_score:
                best_medical_score = s
                best_medical_label = label

        passed = best_medical_score >= self.threshold

        logger.debug(
            "Intent classify | best_medical_label='%s' score=%.3f threshold=%.3f passed=%s",
            best_medical_label,
            best_medical_score,
            self.threshold,
            passed,
        )

        return IntentResult(
            passed=passed,
            label=best_medical_label,
            score=best_medical_score,
            raw_scores=raw_scores,
        )

    def is_medical(self, question: str) -> bool:
        """
        Convenience wrapper that returns only the boolean gate decision.

        Parameters
        ----------
        question:
            The raw question string.

        Returns
        -------
        bool
            ``True`` when the question is classified as medical.
        """
        return self.classify(question).passed
