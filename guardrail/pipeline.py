"""
guardrail/pipeline.py
----------------------
``GuardrailPipeline`` — the top-level orchestrator that wires together:

  Gate 1 (text intent)  →  Gate 2 (image confidence)  →  VQA inference

Short-circuit behaviour
-----------------------
* If Gate 1 **fails** (non-medical question), the image is never loaded
  and the VQA model never runs.  Return is immediate.
* If Gate 2 **fails** (non-medical image), the expensive beam-search
  inference is skipped.  Only one fast single-step forward pass was used.
* Only when **both gates pass** is the full ``predict()`` call made.

Usage
-----
::

    from guardrail.pipeline import GuardrailPipeline

    pipeline = GuardrailPipeline()          # uses config.py defaults
    result   = pipeline.run(
        image_path="path/to/xray.jpg",
        question="What abnormalities are visible in the chest X-ray?",
    )

    if result.passed:
        print(result.answer)
    else:
        print(result.refusal_message)
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import torch

from guardrail import config as cfg
from guardrail.confidence_gate import ConfidenceGate
from guardrail.intent_classifier import MedicalIntentClassifier
from guardrail.refusal import build_refusal
from guardrail.result_types import GuardrailResult



# Module-level imports so @patch("guardrail.pipeline.load_model") and
# @patch("guardrail.pipeline.predict") work in tests.
from inference import load_model, predict  # noqa: E402

logger = logging.getLogger(__name__)


class GuardrailPipeline:
    """
    Dual-gate medical guardrail pipeline.

    Parameters
    ----------
    checkpoint:
        Path to the ``.pth`` checkpoint file.
        Defaults to ``config.CHECKPOINT_PATH``.
    config_path:
        Path to the ``VQA.yaml`` configuration file.
        Defaults to ``config.CONFIG_PATH``.
    text_encoder:
        HuggingFace model name for the BERT encoder.
        Defaults to ``config.TEXT_ENCODER``.
    text_decoder:
        HuggingFace model name for the BERT decoder.
        Defaults to ``config.TEXT_DECODER``.
    intent_threshold:
        Minimum cumulative seed-token probability for Gate 1 to pass.
        Gate 1 uses the same MUMC .pth model — no extra weights needed.
        Defaults to ``config.INTENT_THRESHOLD``.
    confidence_threshold:
        Minimum Softmax top-1 probability to pass Gate 2.
        Defaults to ``config.CONFIDENCE_THRESHOLD``.
    device:
        Torch device string (``"cpu"``, ``"cuda"``, ``"mps"``).
        Auto-detected when ``None``.
    lazy:
        If ``True`` (default), the VQA model is loaded on first ``run()``
        call rather than at construction time.  Set to ``False`` to force
        eager loading (useful for server warm-up).
    """

    def __init__(
        self,
        checkpoint: str = cfg.CHECKPOINT_PATH,
        config_path: str = cfg.CONFIG_PATH,
        text_encoder: str = cfg.TEXT_ENCODER,
        text_decoder: str = cfg.TEXT_DECODER,
        intent_threshold: float = cfg.INTENT_THRESHOLD,
        confidence_threshold: float = cfg.CONFIDENCE_THRESHOLD,
        device: Optional[str] = None,
        lazy: bool = True,
        # ── Model injection (API server passes pre-loaded components) ──
        preloaded_model=None,
        preloaded_tokenizer=None,
        preloaded_transform=None,
        preloaded_device: Optional[torch.device] = None,
    ) -> None:
        self.checkpoint = checkpoint
        self.config_path = config_path
        self.text_encoder = text_encoder
        self.text_decoder = text_decoder
        self.intent_threshold = intent_threshold
        self.confidence_threshold = confidence_threshold

        # Device resolution
        if preloaded_device is not None:
            self.device = preloaded_device
        elif device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        logger.info("GuardrailPipeline device: %s", self.device)

        # Gate 1 — powered by the MUMC .pth model (no BART weights needed).
        # Model components are injected lazily once the VQA model is loaded;
        # until then the classifier bypasses Gate 1 (passes through).
        self._intent_clf = MedicalIntentClassifier(
            threshold=intent_threshold,
            device=self.device,
        )

        # VQA model components — populated lazily, eagerly, or injected.
        self._model = None
        self._tokenizer = None
        self._transform = None
        self._gate2: Optional[ConfidenceGate] = None

        # If the caller already loaded the model (e.g. API startup), inject it
        # into both Gate 1 and Gate 2 immediately.
        if preloaded_model is not None:
            self._model = preloaded_model
            self._tokenizer = preloaded_tokenizer
            self._transform = preloaded_transform
            # Inject model into Gate 1
            self._intent_clf.set_model(
                model=self._model,
                tokenizer=self._tokenizer,
                device=self.device,
            )
            # Gate 2 is CLIP-based — independent of the VQA model
            self._gate2 = ConfidenceGate(device=self.device)
            logger.info("GuardrailPipeline: using pre-loaded MUMC model + CLIP Gate 2.")
        elif not lazy:
            self._load_vqa_model()

    # ------------------------------------------------------------------
    # Private: model loading
    # ------------------------------------------------------------------

    def _load_vqa_model(self) -> None:
        """Load the MUMC VQA model (idempotent — safe to call multiple times)."""
        if self._model is not None:
            return

        logger.info(
            "Loading MUMC VQA model from '%s' on %s …",
            self.checkpoint,
            self.device,
        )

        self._model, self._tokenizer, self._transform = load_model(self.device)

        # Inject model into Gate 1 now that it is loaded
        self._intent_clf.set_model(
            model=self._model,
            tokenizer=self._tokenizer,
            device=self.device,
        )

        # Gate 2 is CLIP-based — independent of VQA model, loads its own weights lazily
        self._gate2 = ConfidenceGate(device=self.device)
        logger.info("MUMC VQA model loaded, Gate 1 and Gate 2 (CLIP) initialised.")


    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        image_path: str,
        question: str,
        num_beams: int = 5,
        max_new_tokens: int = 20,
    ) -> GuardrailResult:
        """
        Run the dual-gate guardrail and (if both pass) VQA inference.

        Parameters
        ----------
        image_path:
            Path to the medical image to analyse.
        question:
            User's clinical question.
        num_beams:
            Beam width for the VQA decoder (only used when both gates pass).
        max_new_tokens:
            Maximum answer tokens (only used when both gates pass).

        Returns
        -------
        GuardrailResult
            Unified result object — inspect ``.passed`` first.
        """
        t_start = time.perf_counter()

        # ── Gate 1: text intent ──────────────────────────────────────
        logger.info("Gate 1: classifying question intent …")
        t1 = time.perf_counter()
        intent_result = self._intent_clf.classify(question)
        t1_ms = (time.perf_counter() - t1) * 1000
        logger.info("%s  (%.0f ms)", intent_result, t1_ms)

        if not intent_result.passed:
            logger.info("Gate 1 FAILED → returning intent refusal.")
            return build_refusal(
                gate="intent",
                intent_result=intent_result,
            )

        # ── Lazy-load VQA model (only reached when Gate 1 passes) ────
        self._load_vqa_model()

        # ── Gate 2: image confidence ─────────────────────────────────
        logger.info("Gate 2: checking image confidence …")
        t2 = time.perf_counter()
        confidence_result = self._gate2.check(image_path, question)
        t2_ms = (time.perf_counter() - t2) * 1000
        logger.info("%s  (%.0f ms)", confidence_result, t2_ms)

        if not confidence_result.passed:
            logger.info("Gate 2 FAILED → returning confidence refusal.")
            return build_refusal(
                gate="confidence",
                intent_result=intent_result,
                confidence_result=confidence_result,
            )

        # ── Full VQA inference ───────────────────────────────────────
        logger.info("Both gates passed → running VQA inference …")
        t3 = time.perf_counter()
        answer = predict(
            model=self._model,
            tokenizer=self._tokenizer,
            transform=self._transform,
            image_path=image_path,
            question_str=question,
            device=self.device,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
        )
        t3_ms = (time.perf_counter() - t3) * 1000
        total_ms = (time.perf_counter() - t_start) * 1000
        logger.info(
            "VQA inference complete (%.0f ms).  Total pipeline: %.0f ms",
            t3_ms,
            total_ms,
        )

        return GuardrailResult(
            passed=True,
            answer=answer,
            refusal_message=None,
            gate_triggered="none",
            intent_result=intent_result,
            confidence_result=confidence_result,
            metadata={
                "gate1_ms": round(t1_ms, 1),
                "gate2_ms": round(t2_ms, 1),
                "inference_ms": round(t3_ms, 1),
                "total_ms": round(total_ms, 1),
                "device": str(self.device),
                "num_beams": num_beams,
                "max_new_tokens": max_new_tokens,
            },
        )

    def check_image(self, image_path: str) -> bool:
        """
        Screen a single image using the CLIP-based Gate 2.

        Used by the triage endpoint to filter non-medical images before scoring.
        Delegates directly to ``ConfidenceGate.check()`` so triage and
        ``/predict`` Gate 2 share **identical** classification logic.

        Returns
        -------
        bool
            ``True`` if the image passes (valid medical scan), ``False`` otherwise.
        """
        self._load_vqa_model()
        result = self._gate2.check(image_path)
        logger.info(
            "Triage CLIP screen [%s]: medical_score=%.4f threshold=%.2f — %s",
            image_path,
            result.top_prob,
            result.threshold,
            "PASS" if result.passed else "FAIL",
        )
        return result.passed
