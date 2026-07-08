"""
guardrail/intent_classifier.py
--------------------------------
Gate 1 – Text Intent Classifier (Zero-Shot).

Uses distilbert-base-uncased-mnli to classify if the question sits in medical territory.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from transformers import pipeline

from guardrail.config import INTENT_MODEL_PATH, INTENT_THRESHOLD
from guardrail.result_types import IntentResult

logger = logging.getLogger(__name__)


class MedicalIntentClassifier:
    """
    Text intent classifier powered by Zero-Shot Classification.

    Parameters
    ----------
    model_path:
        Path or Hub identifier for the zero-shot model.
    device:
        Torch device the model lives on.
    threshold:
        Threshold for the 'medical question' class probability.
    """

    def __init__(
        self,
        model_path: str = INTENT_MODEL_PATH,
        device: Optional[torch.device] = None,
        threshold: float = INTENT_THRESHOLD,
        # Legacy VQA-model args kept for API compatibility — unused
        model=None,
        tokenizer=None,
    ) -> None:
        self.device = device or torch.device("cpu")
        self.model_path = model_path
        self.threshold = threshold
        self.candidate_labels = ["medical question", "non-medical question"]
        
        self._classifier = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Lazily load the zero-shot pipeline."""
        if self._classifier is not None:
            return
            
        logger.info(
            "Loading zero-shot intent classifier from '%s' on device '%s' …",
            self.model_path,
            self.device,
        )
        
        # Convert device to index for pipeline if cuda
        device_arg = -1
        if self.device.type == "cuda":
            device_arg = self.device.index if self.device.index is not None else 0
            
        self._classifier = pipeline(
            "zero-shot-classification",
            model=self.model_path,
            device=device_arg
        )
        logger.info("Zero-shot intent model ready.")

    def _score_question(self, question: str) -> tuple[float, str]:
        """
        Run the classifier and parse the response.

        Returns
        -------
        (score, label)
            ``score`` — the confidence for 'medical question'.
            ``label`` — 'medical question' or 'non-medical question'.
        """
        res = self._classifier(question, self.candidate_labels)
        scores = dict(zip(res["labels"], res["scores"]))
        
        med_score = scores["medical question"]
        label = "medical question" if med_score >= self.threshold else "non-medical question"
        
        return med_score, label

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_model(self, model=None, tokenizer=None, device: Optional[torch.device] = None) -> None:
        """
        Legacy method kept for pipeline compatibility.
        """
        if device is not None:
            self.device = device

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
            ``passed=True`` when the 'medical question' score >= threshold.
        """
        # Edge case: empty question
        cleaned = question.strip()
        if not cleaned:
            logger.debug("Empty question string → automatic refusal.")
            return IntentResult(
                passed=False,
                label="<empty>",
                score=0.0,
                raw_scores={},
            )

        self._load()

        score, label = self._score_question(cleaned)
        passed = score >= self.threshold

        logger.debug(
            "Intent classify | label='%s' score=%.4f threshold=%.4f passed=%s",
            label,
            score,
            self.threshold,
            passed,
        )

        return IntentResult(
            passed=passed,
            label=label,
            score=score,
            raw_scores={},
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
