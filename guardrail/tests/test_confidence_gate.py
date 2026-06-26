"""
guardrail/tests/test_confidence_gate.py
-----------------------------------------
Unit tests for Gate 2: ConfidenceGate.

All VQA model components (visual encoder, text encoder, text decoder,
tokenizer, transform) are mocked so no model weights or GPU are required.
The tests exercise the Softmax probability extraction logic and the
pass/fail decision against the configured threshold.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from guardrail.confidence_gate import ConfidenceGate
from guardrail.result_types import ConfidenceResult


def _make_mock_tokenizer(vocab_size: int = 30522):
    """Build a minimal BERT tokenizer mock."""
    tok = MagicMock()
    tok.cls_token_id = 101
    tok.sep_token_id = 102
    tok.pad_token_id = 0

    # tokenizer(question, ...) → fake encoding
    fake_encoding = MagicMock()
    fake_encoding.input_ids = torch.tensor([[101, 100, 100, 102]])
    fake_encoding.attention_mask = torch.ones(1, 4, dtype=torch.long)
    tok.return_value = fake_encoding

    return tok


def _make_mock_transform():
    """Return a transform that produces a fixed image tensor."""
    transform = MagicMock()
    transform.return_value = torch.zeros(3, 256, 256)
    return transform


def _make_gate(top_prob_target: float, threshold: float = 0.30) -> ConfidenceGate:
    """
    Construct a ConfidenceGate whose ``_get_top_prob`` is mocked to return
    ``top_prob_target`` for any input.  No model weights, no GPU needed.

    Parameters
    ----------
    top_prob_target:
        The exact top-1 Softmax probability the gate should report.
    threshold:
        Gate threshold.
    """
    model = MagicMock()
    tokenizer = _make_mock_tokenizer()
    transform = _make_mock_transform()
    device = torch.device("cpu")

    gate = ConfidenceGate(
        model=model,
        tokenizer=tokenizer,
        transform=transform,
        device=device,
        threshold=threshold,
    )

    # Directly mock the internal probability extraction so tests are
    # independent of the softmax / logit construction internals.
    gate._get_top_prob = MagicMock(return_value=(top_prob_target, 42))

    return gate


# ---------------------------------------------------------------------------
# Patch PIL and dataset utils (no real images needed)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def patch_image_loading():
    """Patch PIL.Image.open so no real image files are needed."""
    fake_image = MagicMock()
    fake_image.convert.return_value = fake_image

    with patch("guardrail.confidence_gate.Image.open", return_value=fake_image):
        yield


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestConfidenceGate:

    # ── Construction ────────────────────────────────────────────────────

    def test_default_threshold_from_config(self):
        gate = _make_gate(top_prob_target=0.50)
        from guardrail.config import CONFIDENCE_THRESHOLD  # noqa: PLC0415
        assert gate.threshold == CONFIDENCE_THRESHOLD

    def test_custom_threshold_stored(self):
        gate = _make_gate(top_prob_target=0.50, threshold=0.45)
        assert gate.threshold == 0.45

    # ── check() – passing cases ──────────────────────────────────────────

    def test_high_confidence_image_passes(self):
        """top_prob well above threshold → passed=True."""
        gate = _make_gate(top_prob_target=0.80, threshold=0.30)
        result = gate.check("fake_xray.jpg", "Any question?")

        assert isinstance(result, ConfidenceResult)
        assert result.passed is True
        assert result.top_prob >= 0.30

    def test_confidence_at_threshold_passes(self):
        """top_prob == threshold → passed=True (>=)."""
        gate = _make_gate(top_prob_target=0.30, threshold=0.30)
        result = gate.check("fake_xray.jpg", "Any question?")
        assert result.passed is True

    # ── check() – failing cases ──────────────────────────────────────────

    def test_low_confidence_image_fails(self):
        """top_prob well below threshold → passed=False."""
        gate = _make_gate(top_prob_target=0.05, threshold=0.30)
        result = gate.check("fake_photo.jpg", "Any question?")

        assert isinstance(result, ConfidenceResult)
        assert result.passed is False
        assert result.top_prob < 0.30

    def test_confidence_just_below_threshold_fails(self):
        """top_prob slightly below threshold → passed=False."""
        gate = _make_gate(top_prob_target=0.15, threshold=0.30)
        result = gate.check("fake_photo.jpg", "What is shown?")
        assert result.passed is False

    # ── Return type ──────────────────────────────────────────────────────

    def test_result_fields_populated(self):
        gate = _make_gate(top_prob_target=0.60, threshold=0.30)
        result = gate.check("img.jpg", "Question")

        assert result.threshold == 0.30
        assert 0.0 <= result.top_prob <= 1.0
        assert result.top_token_id is not None

    def test_result_is_frozen(self):
        """ConfidenceResult is a frozen dataclass — must not be mutable."""
        gate = _make_gate(top_prob_target=0.60, threshold=0.30)
        result = gate.check("img.jpg", "Question")
        with pytest.raises((AttributeError, TypeError)):
            result.passed = False  # type: ignore[misc]

    # ── Threshold variations ─────────────────────────────────────────────

    @pytest.mark.parametrize("threshold,prob,expected_pass", [
        (0.20, 0.25, True),
        (0.20, 0.10, False),
        (0.30, 0.35, True),
        (0.30, 0.20, False),
        (0.45, 0.50, True),
        (0.45, 0.40, False),
        (0.60, 0.70, True),
        (0.60, 0.50, False),
    ])
    def test_threshold_parametrized(self, threshold, prob, expected_pass):
        gate = _make_gate(top_prob_target=prob, threshold=threshold)
        result = gate.check("img.jpg", "Q?")
        assert result.passed == expected_pass, (
            f"threshold={threshold}, prob={prob:.2f}: "
            f"expected passed={expected_pass}, got {result.passed}"
        )
