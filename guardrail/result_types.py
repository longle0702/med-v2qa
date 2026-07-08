"""
guardrail/result_types.py
--------------------------
Typed result dataclasses for every stage of the dual-gate guardrail pipeline.
All fields are immutable after construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional


# ---------------------------------------------------------------------------
# Gate 1 – Text intent classification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IntentResult:
    """Result from Gate 1 (text-intent classifier)."""

    passed: bool
    """True when the query is classified as medical/clinical."""

    label: str
    """Top candidate label returned by the zero-shot classifier."""

    score: float
    """Confidence score for the medical-intent hypothesis (0–1)."""

    raw_scores: Dict[str, float] = field(default_factory=dict)
    """Full label → score mapping from the classifier (optional)."""

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[Gate-1 {status}] label='{self.label}' "
            f"score={self.score:.3f}"
        )


# ---------------------------------------------------------------------------
# Gate 2 – Softmax confidence check
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConfidenceResult:
    """Result from Gate 2 (MUMC Softmax confidence gate)."""

    passed: bool
    """True when the model's top-1 confidence exceeds the configured threshold."""

    top_prob: float
    """Raw top-1 Softmax probability at the first decoder step (0–1)."""

    threshold: float
    """The threshold value that was applied."""

    top_token_id: Optional[int] = None
    """Vocabulary ID of the most probable first token (diagnostic only)."""

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[Gate-2 {status}] top_prob={self.top_prob:.4f} "
            f"threshold={self.threshold:.4f}"
        )


# ---------------------------------------------------------------------------
# Gate 3 – Model Answer Confidence
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AnswerConfidenceResult:
    """Result from Gate 3 (Model Answer Confidence)."""

    passed: bool
    """True when the model's answer confidence exceeds the configured threshold."""

    confidence: float
    """Confidence score of the generated answer (0-1)."""

    threshold: float
    """The threshold value that was applied."""

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[Gate-3 {status}] confidence={self.confidence:.4f} "
            f"threshold={self.threshold:.4f}"
        )


# ---------------------------------------------------------------------------
# Full pipeline result
# ---------------------------------------------------------------------------

GateName = Literal["intent", "confidence", "answer_confidence", "none"]


@dataclass(frozen=True)
class GuardrailResult:
    """
    Unified result returned by ``GuardrailPipeline.run()``.

    When ``passed=True`` the ``answer`` field contains the VQA model's
    response.  When ``passed=False`` the ``refusal_message`` field
    contains the safe refusal text and ``gate_triggered`` identifies
    which gate fired.
    """

    passed: bool
    """True when both gates passed and the VQA model produced an answer."""

    answer: Optional[str] = None
    """VQA model answer (only set when ``passed=True``)."""

    refusal_message: Optional[str] = None
    """Safe refusal text shown to the user (only set when ``passed=False``)."""

    gate_triggered: GateName = "none"
    """Which gate fired: 'intent', 'confidence', or 'none' (success)."""

    intent_result: Optional[IntentResult] = None
    """Detailed Gate-1 result (always populated)."""

    confidence_result: Optional[ConfidenceResult] = None
    """Detailed Gate-2 result (populated when Gate 1 passed)."""

    answer_confidence_result: Optional[AnswerConfidenceResult] = None
    """Detailed Gate-3 result (populated when Gate 1 and Gate 2 passed)."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Arbitrary extra information (e.g. inference timing, device)."""

    def __str__(self) -> str:
        if self.passed:
            return f"[GuardrailResult PASSED] answer='{self.answer}'"
        return (
            f"[GuardrailResult REFUSED by {self.gate_triggered.upper()}] "
            f"reason='{self.refusal_message}'"
        )
