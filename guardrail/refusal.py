"""
guardrail/refusal.py
---------------------
Clinically-appropriate refusal messages and the helper that constructs
a ``GuardrailResult`` in refused state.

Both messages are deliberately non-alarming: they inform the user that
the input fell outside the system's scope without revealing internal
implementation details.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from guardrail.result_types import ConfidenceResult, GuardrailResult, IntentResult

# ---------------------------------------------------------------------------
# Refusal message catalogue
# ---------------------------------------------------------------------------

REFUSAL_MESSAGES: dict[str, str] = {
    "intent": (
        "I'm sorry, but I can only assist with medical and clinical questions. "
        "Your query does not appear to be related to healthcare, radiology, or "
        "clinical diagnosis. Please rephrase your question in a medical context "
        "and try again."
    ),
    "confidence": (
        "The uploaded image does not appear to be a recognisable medical scan. "
        "Please upload an appropriate clinical image and try again."
    ),
}


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_refusal(
    gate: str,
    intent_result: Optional["IntentResult"] = None,
    confidence_result: Optional["ConfidenceResult"] = None,
) -> "GuardrailResult":
    """
    Construct a ``GuardrailResult`` representing a refused request.

    Parameters
    ----------
    gate:
        Which gate triggered the refusal: ``"intent"`` or ``"confidence"``.
    intent_result:
        The Gate-1 result object (may be ``None`` if gate == "confidence"
        and we still want to attach the intent result).
    confidence_result:
        The Gate-2 result object (only relevant when gate == "confidence").

    Returns
    -------
    GuardrailResult
        A frozen result with ``passed=False`` and an appropriate refusal
        message.
    """
    # Import here to avoid circular imports
    from guardrail.result_types import GuardrailResult  # noqa: PLC0415

    message = REFUSAL_MESSAGES.get(gate, "Request refused by the safety guardrail.")

    return GuardrailResult(
        passed=False,
        answer=None,
        refusal_message=message,
        gate_triggered=gate,  # type: ignore[arg-type]
        intent_result=intent_result,
        confidence_result=confidence_result,
    )
