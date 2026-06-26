"""
guardrail/tests/test_intent_classifier.py
------------------------------------------
Unit tests for Gate 1: MedicalIntentClassifier.

Mocks out the HuggingFace pipeline so no model weights are required during
CI / fast testing.  The mock simulates the zero-shot classifier output dict:
    {
        "labels": ["label_a", "label_b", ...],
        "scores": [0.9, 0.05, ...],
    }
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from guardrail.intent_classifier import MedicalIntentClassifier
from guardrail.result_types import IntentResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_clf(threshold: float = 0.55) -> MedicalIntentClassifier:
    """Return a classifier instance (pipeline NOT loaded yet)."""
    return MedicalIntentClassifier(threshold=threshold, device="cpu")


def _mock_pipeline_output(top_label: str, top_score: float):
    """
    Build a fake pipeline callable that returns a single high-scoring label
    plus a spread of near-zero scores for all others.
    """
    from guardrail.intent_classifier import _ALL_LABELS  # noqa: PLC0415

    labels = [top_label] + [l for l in _ALL_LABELS if l != top_label]
    remaining = (1.0 - top_score) / max(len(labels) - 1, 1)
    scores = [top_score] + [remaining] * (len(labels) - 1)

    return MagicMock(return_value={"labels": labels, "scores": scores})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMedicalIntentClassifier:

    # ── Construction ────────────────────────────────────────────────────

    def test_default_threshold(self):
        clf = _make_clf()
        assert clf.threshold == 0.55

    def test_custom_threshold(self):
        clf = _make_clf(threshold=0.70)
        assert clf.threshold == 0.70

    def test_pipeline_not_loaded_at_init(self):
        clf = _make_clf()
        assert clf._pipeline is None

    # ── classify() – medical questions ──────────────────────────────────

    @patch("guardrail.intent_classifier.pipeline")
    def test_medical_question_passes(self, mock_pipeline_cls):
        """A clearly medical question should pass Gate 1."""
        clf = _make_clf(threshold=0.55)
        mock_pipeline_cls.return_value = _mock_pipeline_output(
            top_label="medical question", top_score=0.90
        )

        result = clf.classify("What are the findings in this chest X-ray?")

        assert isinstance(result, IntentResult)
        assert result.passed is True
        assert result.score >= 0.55

    @patch("guardrail.intent_classifier.pipeline")
    def test_radiology_question_passes(self, mock_pipeline_cls):
        clf = _make_clf(threshold=0.55)
        mock_pipeline_cls.return_value = _mock_pipeline_output(
            top_label="radiology question", top_score=0.80
        )

        result = clf.classify("Is there evidence of pneumothorax?")
        assert result.passed is True

    @patch("guardrail.intent_classifier.pipeline")
    def test_anatomy_question_passes(self, mock_pipeline_cls):
        clf = _make_clf(threshold=0.55)
        mock_pipeline_cls.return_value = _mock_pipeline_output(
            top_label="anatomy question", top_score=0.75
        )

        result = clf.classify("Where is the aortic arch located?")
        assert result.passed is True

    # ── classify() – non-medical questions ──────────────────────────────

    @patch("guardrail.intent_classifier.pipeline")
    def test_cooking_question_fails(self, mock_pipeline_cls):
        """A cooking question should fail Gate 1."""
        clf = _make_clf(threshold=0.55)
        mock_pipeline_cls.return_value = _mock_pipeline_output(
            top_label="cooking", top_score=0.88
        )

        result = clf.classify("How do I make pasta carbonara?")
        assert result.passed is False

    @patch("guardrail.intent_classifier.pipeline")
    def test_politics_question_fails(self, mock_pipeline_cls):
        clf = _make_clf(threshold=0.55)
        mock_pipeline_cls.return_value = _mock_pipeline_output(
            top_label="politics", top_score=0.92
        )

        result = clf.classify("What is the capital of France?")
        assert result.passed is False

    @patch("guardrail.intent_classifier.pipeline")
    def test_sports_question_fails(self, mock_pipeline_cls):
        clf = _make_clf(threshold=0.55)
        mock_pipeline_cls.return_value = _mock_pipeline_output(
            top_label="sports", top_score=0.77
        )

        result = clf.classify("Who won the 2022 World Cup?")
        assert result.passed is False

    # ── Edge cases ───────────────────────────────────────────────────────

    @patch("guardrail.intent_classifier.pipeline")
    def test_empty_question_fails_without_loading_model(self, mock_pipeline_cls):
        """Empty string must be rejected BEFORE the pipeline callable is invoked."""
        clf = _make_clf()
        # Inject a fresh None pipeline so _load() would call mock_pipeline_cls
        clf._pipeline = None

        result = clf.classify("")

        assert result.passed is False
        assert result.score == 0.0
        # The zero-shot pipeline callable must never have been invoked
        mock_pipeline_cls.return_value.assert_not_called()

    def test_whitespace_only_question_fails(self):
        clf = _make_clf()
        result = clf.classify("   ")
        assert result.passed is False

    # ── Threshold boundary ───────────────────────────────────────────────

    @patch("guardrail.intent_classifier.pipeline")
    def test_score_exactly_at_threshold_passes(self, mock_pipeline_cls):
        """Score == threshold should be accepted (>=)."""
        clf = _make_clf(threshold=0.55)
        mock_pipeline_cls.return_value = _mock_pipeline_output(
            top_label="medical question", top_score=0.55
        )
        result = clf.classify("Any medical question")
        assert result.passed is True

    @patch("guardrail.intent_classifier.pipeline")
    def test_score_just_below_threshold_fails(self, mock_pipeline_cls):
        """Score just below threshold should fail."""
        clf = _make_clf(threshold=0.55)
        mock_pipeline_cls.return_value = _mock_pipeline_output(
            top_label="medical question", top_score=0.549
        )
        result = clf.classify("Any medical question")
        assert result.passed is False

    # ── is_medical() convenience wrapper ────────────────────────────────

    @patch("guardrail.intent_classifier.pipeline")
    def test_is_medical_returns_bool(self, mock_pipeline_cls):
        clf = _make_clf(threshold=0.55)
        mock_pipeline_cls.return_value = _mock_pipeline_output(
            top_label="medical question", top_score=0.80
        )
        assert clf.is_medical("Describe the pulmonary vasculature.") is True

    # ── Return type ──────────────────────────────────────────────────────

    @patch("guardrail.intent_classifier.pipeline")
    def test_raw_scores_populated(self, mock_pipeline_cls):
        """raw_scores dict must contain all candidate labels."""
        from guardrail.intent_classifier import _ALL_LABELS  # noqa: PLC0415

        clf = _make_clf()
        mock_pipeline_cls.return_value = _mock_pipeline_output(
            top_label="medical question", top_score=0.80
        )
        result = clf.classify("Test question")
        assert set(result.raw_scores.keys()) == set(_ALL_LABELS)
