"""
guardrail/tests/test_pipeline.py
----------------------------------
Integration tests for GuardrailPipeline.

All external dependencies (HuggingFace pipeline, VQA model, PIL) are mocked
so the tests run fast with zero GPU and no model weights.

Test matrix
-----------
✓  Medical question  + medical image   → PASSED, answer returned
✗  Non-medical question               → REFUSED by Gate 1 (model never loaded)
✗  Medical question + non-medical img → REFUSED by Gate 2
✓  Metadata fields present on success
✓  GuardrailResult fields on refusal
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from guardrail.pipeline import GuardrailPipeline
from guardrail.result_types import ConfidenceResult, GuardrailResult, IntentResult


# ---------------------------------------------------------------------------
# Common mock factories
# ---------------------------------------------------------------------------

def _intent_pass(label: str = "medical question", score: float = 0.85) -> IntentResult:
    return IntentResult(passed=True, label=label, score=score)


def _intent_fail(label: str = "cooking", score: float = 0.05) -> IntentResult:
    return IntentResult(passed=False, label=label, score=score)


def _confidence_pass(prob: float = 0.70) -> ConfidenceResult:
    return ConfidenceResult(passed=True, top_prob=prob, threshold=0.30)


def _confidence_fail(prob: float = 0.08) -> ConfidenceResult:
    return ConfidenceResult(passed=False, top_prob=prob, threshold=0.30)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def pipeline():
    """
    Return a GuardrailPipeline in lazy mode with the actual model loading
    patched out.  The VQA model is never really loaded.
    """
    pipe = GuardrailPipeline.__new__(GuardrailPipeline)
    pipe.checkpoint = "fake.pth"
    pipe.config_path = "fake.yaml"
    pipe.text_encoder = "bert-base-uncased"
    pipe.text_decoder = "bert-base-uncased"
    pipe.intent_threshold = 0.55
    pipe.confidence_threshold = 0.30
    pipe.device = torch.device("cpu")
    pipe._model = None
    pipe._tokenizer = None
    pipe._transform = None
    pipe._gate2 = None

    # Stub intent classifier
    pipe._intent_clf = MagicMock()
    return pipe


# ---------------------------------------------------------------------------
# Helpers to patch _load_vqa_model and _gate2 on the fixture
# ---------------------------------------------------------------------------

def _attach_mocked_vqa(pipe, confidence_result: ConfidenceResult, vqa_answer: str = "yes"):
    """Patch the VQA loading and gate2 onto the pipeline fixture."""
    # Mark model as "loaded"
    pipe._model = MagicMock()
    pipe._tokenizer = MagicMock()
    pipe._transform = MagicMock()

    mock_gate2 = MagicMock()
    mock_gate2.check.return_value = confidence_result
    pipe._gate2 = mock_gate2

    return vqa_answer


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGuardrailPipeline:

    # ── Success path ─────────────────────────────────────────────────────

    @patch("guardrail.pipeline.predict", return_value="pleural effusion")
    def test_both_gates_pass_returns_answer(self, mock_predict, pipeline):
        """Medical question + medical image → answer returned, passed=True."""
        pipeline._intent_clf.classify.return_value = _intent_pass()
        _attach_mocked_vqa(pipeline, _confidence_pass(), "pleural effusion")
        pipeline._load_vqa_model = MagicMock()  # skip real loading

        result = pipeline.run("xray.jpg", "What abnormalities are present?")

        assert isinstance(result, GuardrailResult)
        assert result.passed is True
        assert result.answer == "pleural effusion"
        assert result.refusal_message is None
        assert result.gate_triggered == "none"

    @patch("guardrail.pipeline.predict", return_value="cardiomegaly")
    def test_success_result_has_metadata(self, mock_predict, pipeline):
        """Successful result must include timing metadata."""
        pipeline._intent_clf.classify.return_value = _intent_pass()
        _attach_mocked_vqa(pipeline, _confidence_pass(), "cardiomegaly")
        pipeline._load_vqa_model = MagicMock()

        result = pipeline.run("xray.jpg", "Is the heart enlarged?")

        assert "total_ms" in result.metadata
        assert "gate1_ms" in result.metadata
        assert "gate2_ms" in result.metadata
        assert "inference_ms" in result.metadata
        assert result.metadata["device"] == "cpu"

    @patch("guardrail.pipeline.predict", return_value="normal")
    def test_success_result_contains_gate_details(self, mock_predict, pipeline):
        """Successful result must include both gate result objects."""
        pipeline._intent_clf.classify.return_value = _intent_pass()
        _attach_mocked_vqa(pipeline, _confidence_pass(), "normal")
        pipeline._load_vqa_model = MagicMock()

        result = pipeline.run("xray.jpg", "Is this scan normal?")

        assert result.intent_result is not None
        assert result.confidence_result is not None
        assert result.intent_result.passed is True
        assert result.confidence_result.passed is True

    # ── Gate 1 refusal ───────────────────────────────────────────────────

    def test_non_medical_question_refused_by_gate1(self, pipeline):
        """Non-medical question → Gate 1 fires, model never loaded."""
        pipeline._intent_clf.classify.return_value = _intent_fail()
        pipeline._load_vqa_model = MagicMock()

        result = pipeline.run("xray.jpg", "What is the recipe for pasta?")

        assert result.passed is False
        assert result.gate_triggered == "intent"
        assert result.refusal_message is not None
        assert len(result.refusal_message) > 10
        assert result.answer is None

        # Critical: VQA model must NOT have been loaded
        pipeline._load_vqa_model.assert_not_called()

    def test_gate1_refusal_carries_intent_result(self, pipeline):
        """Gate 1 refusal result must contain the IntentResult for debugging."""
        intent_r = _intent_fail(label="entertainment", score=0.10)
        pipeline._intent_clf.classify.return_value = intent_r
        pipeline._load_vqa_model = MagicMock()

        result = pipeline.run("img.jpg", "Who won the Super Bowl?")

        assert result.intent_result is not None
        assert result.intent_result.label == "entertainment"
        assert result.confidence_result is None  # Gate 2 never ran

    def test_empty_question_refused_by_gate1(self, pipeline):
        """Empty string → Gate 1 refuses before any model work."""
        pipeline._intent_clf.classify.return_value = IntentResult(
            passed=False, label="<empty>", score=0.0
        )
        pipeline._load_vqa_model = MagicMock()

        result = pipeline.run("img.jpg", "")

        assert result.passed is False
        assert result.gate_triggered == "intent"
        pipeline._load_vqa_model.assert_not_called()

    # ── Gate 2 refusal ───────────────────────────────────────────────────

    def test_non_medical_image_refused_by_gate2(self, pipeline):
        """Medical question + non-medical image → Gate 2 fires."""
        pipeline._intent_clf.classify.return_value = _intent_pass()
        _attach_mocked_vqa(pipeline, _confidence_fail())
        pipeline._load_vqa_model = MagicMock()

        result = pipeline.run("cat_photo.jpg", "What abnormalities are visible?")

        assert result.passed is False
        assert result.gate_triggered == "confidence"
        assert result.refusal_message is not None
        assert result.answer is None

    def test_gate2_refusal_carries_both_gate_results(self, pipeline):
        """Gate 2 refusal must include both IntentResult and ConfidenceResult."""
        pipeline._intent_clf.classify.return_value = _intent_pass()
        _attach_mocked_vqa(pipeline, _confidence_fail(prob=0.05))
        pipeline._load_vqa_model = MagicMock()

        result = pipeline.run("photo.jpg", "Is this an X-ray?")

        assert result.intent_result is not None
        assert result.intent_result.passed is True
        assert result.confidence_result is not None
        assert result.confidence_result.passed is False
        assert result.confidence_result.top_prob == pytest.approx(0.05)

    # ── Refusal messages ─────────────────────────────────────────────────

    def test_intent_refusal_message_is_non_empty_string(self, pipeline):
        pipeline._intent_clf.classify.return_value = _intent_fail()
        pipeline._load_vqa_model = MagicMock()

        result = pipeline.run("img.jpg", "Recommend a restaurant.")
        assert isinstance(result.refusal_message, str)
        assert len(result.refusal_message) > 20

    def test_confidence_refusal_message_is_non_empty_string(self, pipeline):
        pipeline._intent_clf.classify.return_value = _intent_pass()
        _attach_mocked_vqa(pipeline, _confidence_fail())
        pipeline._load_vqa_model = MagicMock()

        result = pipeline.run("selfie.jpg", "Describe the pathology.")
        assert isinstance(result.refusal_message, str)
        assert len(result.refusal_message) > 20

    # ── num_beams / max_new_tokens forwarded ────────────────────────────

    @patch("guardrail.pipeline.predict", return_value="fracture")
    def test_inference_params_forwarded(self, mock_predict, pipeline):
        """num_beams and max_new_tokens must be forwarded to predict()."""
        pipeline._intent_clf.classify.return_value = _intent_pass()
        _attach_mocked_vqa(pipeline, _confidence_pass(), "fracture")
        pipeline._load_vqa_model = MagicMock()

        pipeline.run("xray.jpg", "Describe findings.", num_beams=5, max_new_tokens=30)

        call_kwargs = mock_predict.call_args.kwargs
        assert call_kwargs["num_beams"] == 5
        assert call_kwargs["max_new_tokens"] == 30
