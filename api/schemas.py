"""
api/schemas.py
--------------
Pydantic response models for the Med-V²QA FastAPI endpoints.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    backend: Literal["pytorch"]
    device: str


# ---------------------------------------------------------------------------
# /predict
# ---------------------------------------------------------------------------

class GateDetail(BaseModel):
    """Per-gate detail included in the predict response."""

    # Gate 1
    intent_label: Optional[str] = None
    intent_score: Optional[float] = None
    intent_passed: Optional[bool] = None

    # Gate 2
    confidence_prob: Optional[float] = None
    confidence_threshold: Optional[float] = None
    confidence_passed: Optional[bool] = None


class PredictResponse(BaseModel):
    """
    Response for POST /predict.

    When ``passed=True`` the ``answer`` field contains the VQA model's answer.
    When ``passed=False`` the ``refusal_message`` explains which gate fired.
    """

    passed: bool = Field(..., description="True when both guardrail gates passed.")
    gate_triggered: Literal["intent", "confidence", "none"] = Field(
        ..., description="Which gate fired; 'none' on success."
    )

    answer: Optional[str] = Field(None, description="VQA answer (only set when passed=True).")
    refusal_message: Optional[str] = Field(
        None, description="Refusal text shown to the user (only set when passed=False)."
    )

    gate_detail: GateDetail = Field(default_factory=GateDetail)
    backend: Literal["pytorch"] = Field(
        ..., description="Inference backend used for this request."
    )
    latency_ms: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-stage timing breakdown in milliseconds.",
    )


# ---------------------------------------------------------------------------
# /triage
# ---------------------------------------------------------------------------

class TriageItem(BaseModel):
    """A single image in the triage result queue."""

    filename: str = Field(..., description="Original filename of the uploaded image.")
    score: float = Field(..., description="Abnormality score in [0, 1]; higher = more abnormal.")
    is_abnormal: bool = Field(..., description="True when score > 0.5.")
    refused: bool = Field(
        default=False,
        description="True when the image failed the guardrail image-confidence gate.",
    )


class TriageResponse(BaseModel):
    """
    Response for POST /triage.

    The queue is sorted descending by score (most critical first).
    """

    queue: List[TriageItem] = Field(..., description="Sorted triage queue.")
    total_images: int
    backend: Literal["pytorch"]
    latency_ms: float = Field(..., description="Total triage pipeline latency in ms.")


# ---------------------------------------------------------------------------
# /transcribe
# ---------------------------------------------------------------------------

class TranscribeResponse(BaseModel):
    """
    Response for POST /transcribe.

    ``transcript`` contains the Whisper-recognised text ready to populate
    the clinical query textarea on the frontend.
    """

    transcript: str = Field(..., description="Recognised speech text.")
    duration_ms: float = Field(..., description="End-to-end transcription latency in ms.")
