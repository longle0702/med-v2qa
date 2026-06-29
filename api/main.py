"""
api/main.py
-----------
Med-V²QA FastAPI application.

Endpoints
---------
GET  /health   — liveness / readiness probe
POST /predict  — single image + question → guardrail gates → VQA answer
POST /triage   — batch of images → sorted abnormality queue

The MUMC checkpoint is loaded **once** at startup into an ``InferenceEngine``
singleton, which is then injected into both ``GuardrailPipeline`` and
``BatchTriageService`` to avoid a double 2.2 GB model load.
"""
from __future__ import annotations

import io
import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from api.engine import InferenceEngine
from api.schemas import (
    GateDetail,
    HealthResponse,
    PredictResponse,
    TriageItem,
    TriageResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global singletons (populated in lifespan)
# ---------------------------------------------------------------------------
_engine: Optional[InferenceEngine] = None
_guardrail = None      # GuardrailPipeline
_triage = None         # BatchTriageService


# ---------------------------------------------------------------------------
# Lifespan — load model once on startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine, _guardrail, _triage

    logger.info("═" * 60)
    logger.info("  Med-V²QA API — startup")
    logger.info("═" * 60)

    # 1. Load shared inference engine (checkpoint + optional ONNX sessions)
    _engine = InferenceEngine()
    logger.info("InferenceEngine ready  (backend=%s, device=%s)",
                _engine.backend, _engine.device)

    # 2. Guardrail pipeline — inject pre-loaded model (no double load)
    from guardrail.pipeline import GuardrailPipeline  # noqa: PLC0415
    _guardrail = GuardrailPipeline(
        preloaded_model=_engine.model,
        preloaded_tokenizer=_engine.tokenizer,
        preloaded_transform=_engine.transform,
        preloaded_device=_engine.device,
    )
    logger.info("GuardrailPipeline ready.")

    # 3. Triage service — uses local CLIP model
    from triage.batch_sorter import BatchTriageService  # noqa: PLC0415
    _triage = BatchTriageService(device=_engine.device)
    logger.info("BatchTriageService ready.")


    logger.info("═" * 60)
    logger.info("  All components initialised — server is ready.")
    logger.info("═" * 60)

    yield  # ← server runs here

    logger.info("Shutting down Med-V²QA API.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Med-V²QA API",
    description=(
        "Interactive Clinical VQA Inference Engine with dual-gate guardrail "
        "and batch triage sorting. Powered by the MUMC architecture."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app.mount("/static", StaticFiles(directory=str(_FRONTEND_DIR)), name="static")


# ---------------------------------------------------------------------------
# Root — serve the clinical UI
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def root():
    """Serve the Med-V²QA clinical UI."""
    return FileResponse(str(_FRONTEND_DIR / "index.html"))


@app.get("/favicon.ico", include_in_schema=False)
@app.get("/apple-touch-icon.png", include_in_schema=False)
@app.get("/apple-touch-icon-precomposed.png", include_in_schema=False)
async def silence_icons():
    """Silence browser icon requests."""
    return Response(status_code=204)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_engine():
    if _engine is None:
        raise HTTPException(status_code=503, detail="Model not yet loaded.")


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness / readiness check",
    tags=["Utility"],
)
async def health():
    """Returns OK when the model is loaded and the server is ready."""
    _require_engine()
    return HealthResponse(
        status="ok",
        backend=_engine.backend,
        device=str(_engine.device),
    )


# ---------------------------------------------------------------------------
# POST /predict  — single VQA with guardrail
# ---------------------------------------------------------------------------

@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Guarded single-image VQA inference",
    tags=["Inference"],
)
async def predict(
    image: Annotated[UploadFile, File(description="Medical image (JPEG, PNG, TIFF …)")],
    question: Annotated[str, Form(description="Clinical question to ask about the image.")],
    num_beams: Annotated[int, Form(description="Beam width for decoder (default 3).")] = 3,
    max_new_tokens: Annotated[int, Form(description="Max answer tokens (default 20).")] = 20,
):
    """
    Run the dual-gate guardrail and (if both gates pass) generate a VQA answer.

    - **Gate 1 (intent)** — rejects non-medical questions before the image is processed.
    - **Gate 2 (confidence)** — rejects non-medical images based on Softmax confidence.
    - **VQA inference** — beam-search decoding via the MUMC model (ONNX-accelerated encoders).
    """
    _require_engine()
    t_start = time.perf_counter()

    # Save upload to a temp file (GuardrailPipeline takes a file path)
    image_bytes = await image.read()
    suffix = os.path.splitext(image.filename or ".jpg")[1] or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    try:
        result = _guardrail.run(
            image_path=tmp_path,
            question=question,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
        )
    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    total_ms = (time.perf_counter() - t_start) * 1000

    # Build gate detail
    gate_detail = GateDetail()
    if result.intent_result:
        ir = result.intent_result
        gate_detail = GateDetail(
            intent_label=ir.label,
            intent_score=round(ir.score, 4),
            intent_passed=ir.passed,
        )
    if result.confidence_result:
        cr = result.confidence_result
        gate_detail = GateDetail(
            intent_label=gate_detail.intent_label,
            intent_score=gate_detail.intent_score,
            intent_passed=gate_detail.intent_passed,
            confidence_prob=round(cr.top_prob, 4),
            confidence_threshold=cr.threshold,
            confidence_passed=cr.passed,
        )

    # Timing from guardrail metadata (falls back to total if not available)
    timing = result.metadata if result.metadata else {"total_ms": round(total_ms, 1)}

    return PredictResponse(
        passed=result.passed,
        gate_triggered=result.gate_triggered,
        answer=result.answer,
        refusal_message=result.refusal_message,
        gate_detail=gate_detail,
        backend=_engine.backend,
        latency_ms={k: round(float(v), 1) for k, v in timing.items() if isinstance(v, (int, float))},
    )


# ---------------------------------------------------------------------------
# POST /triage  — batch abnormality sorting
# ---------------------------------------------------------------------------

@app.post(
    "/triage",
    response_model=TriageResponse,
    summary="Batch triage — sort images by abnormality",
    tags=["Triage"],
)
async def triage(
    images: Annotated[
        List[UploadFile],
        File(description="Batch of medical images (up to 20 recommended)."),
    ],
):
    """
    Upload a batch of medical images (e.g. 5-10 X-rays). The system
    automatically scores each image for abnormality and returns the queue
    sorted so the most critical scans appear first.
    """
    _require_engine()

    if not images:
        raise HTTPException(status_code=422, detail="At least one image is required.")
    if len(images) > 50:
        raise HTTPException(status_code=422, detail="Maximum 50 images per batch.")

    t_start = time.perf_counter()

    # Save all uploads to temp files
    tmp_paths: list[str] = []
    filenames: list[str] = []
    try:
        for upload in images:
            raw = await upload.read()
            suffix = os.path.splitext(upload.filename or ".jpg")[1] or ".jpg"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(raw)
                tmp_paths.append(tmp.name)
                filenames.append(upload.filename or tmp.name)

        # ── Guardrail Gate 2: screen each image (non-medical → refused) ──
        valid_paths: list[str] = []
        refused_filenames: list[str] = []
        for path, fname in zip(tmp_paths, filenames):
            if _guardrail.check_image(path):
                valid_paths.append(path)
            else:
                logger.info("Triage: '%s' refused by guardrail (not a medical image).", fname)
                refused_filenames.append(fname)

        # ── Triage scoring on accepted images ────────────────────────────
        sorted_results = _triage.sort_batch(valid_paths) if valid_paths else []

    finally:
        for p in tmp_paths:
            try:
                os.unlink(p)
            except OSError:
                pass

    latency_ms = (time.perf_counter() - t_start) * 1000

    # Map temp paths back to original filenames
    path_to_filename = dict(zip(tmp_paths, filenames))

    # Accepted images — sorted by abnormality score (descending)
    queue = [
        TriageItem(
            filename=path_to_filename.get(item["image_path"], item["image_path"]),
            score=round(item["score"], 4),
            is_abnormal=item["is_abnormal"],
            refused=False,
        )
        for item in sorted_results
    ]

    # Refused images — appended at the end with score=0
    for fname in refused_filenames:
        queue.append(
            TriageItem(
                filename=fname,
                score=0.0,
                is_abnormal=False,
                refused=True,
            )
        )

    return TriageResponse(
        queue=queue,
        total_images=len(queue),
        backend=_engine.backend,
        latency_ms=round(latency_ms, 1),
    )
