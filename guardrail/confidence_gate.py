"""
guardrail/confidence_gate.py
-----------------------------
Gate 2 – CLIP-based Medical Image Classifier.

Uses ``openai/clip-vit-base-patch32`` (loaded from a local cache) to perform
zero-shot classification: it embeds the image and a set of medical vs.
non-medical text prompts and compares cosine similarities.

Design rationale
----------------
- CLIP is purpose-built for image-text alignment and produces robust
  zero-shot classifiers without any fine-tuning.
- Medical scans (X-rays, MRI, CT) produce high cosine similarity with
  prompts like "a chest X-ray" or "a medical scan", and low similarity
  with "a photo of a dog" or "a selfie".
- The gate is **independent** of the MUMC VQA model — it loads its own
  small (~600 MB) weights once and keeps them on CPU by default to avoid
  competing with the GPU-resident VQA model.
- The local model path is configured in ``config.CLIP_MODEL_PATH`` so
  no network access is required after the initial download.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from guardrail.config import CLIP_MODEL_PATH, CLIP_THRESHOLD
from guardrail.result_types import ConfidenceResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Medical vs. non-medical prompt sets
# ---------------------------------------------------------------------------

_MEDICAL_PROMPTS = [
    "a chest X-ray image",
    "a medical scan",
    "an MRI image",
    "a CT scan",
    "a radiological image",
    "a clinical medical image",
    "an ultrasound image",
    "a pathology slide",
    "a medical imaging study",
    "an X-ray of the lungs",
]

_NON_MEDICAL_PROMPTS = [
    "a photo of a dog",
    "a selfie photo",
    "a photo of nature",
    "a food photograph",
    "a street photo",
    "a photo of people",
    "a cartoon",
    "a screenshot",
    "a photo of a car",
    "a painting or artwork",
]


class ConfidenceGate:
    """
    Gate 2: CLIP-based zero-shot medical image classifier.

    Parameters
    ----------
    device:
        Torch device to run CLIP on.  Defaults to ``\"cpu\"`` to avoid
        competing with the GPU-resident VQA model.
    threshold:
        Minimum medical softmax score required to pass.
        Defaults to ``config.CLIP_THRESHOLD``.
    model_path:
        Local path to the saved CLIP model directory.
        Defaults to ``config.CLIP_MODEL_PATH``.
    """

    def __init__(
        self,
        device: torch.device = torch.device("cpu"),
        threshold: float = CLIP_THRESHOLD,
        model_path: str = CLIP_MODEL_PATH,
        # Legacy VQA-model args kept for API compatibility — unused by CLIP gate
        model=None,
        tokenizer=None,
        transform=None,
    ) -> None:
        self.device = device
        self.threshold = threshold
        self.model_path = model_path

        self._clip_model: Optional[CLIPModel] = None
        self._clip_processor: Optional[CLIPProcessor] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Lazily load the CLIP model and processor from the local cache."""
        if self._clip_model is not None:
            return
        logger.info(
            "Loading CLIP model from '%s' on device '%s' …",
            self.model_path,
            self.device,
        )
        self._clip_model = CLIPModel.from_pretrained(self.model_path)
        self._clip_model = self._clip_model.to(self.device)
        self._clip_model.eval()
        self._clip_processor = CLIPProcessor.from_pretrained(self.model_path)
        logger.info("CLIP model ready.")

    @torch.no_grad()
    def _classify(self, image_path: str) -> tuple[float, float]:
        """
        Compute zero-shot medical vs. non-medical probabilities.

        Returns
        -------
        (medical_score, non_medical_score)
            Softmax probabilities summed over medical / non-medical prompts.
        """
        image = Image.open(image_path).convert("RGB")
        all_prompts = _MEDICAL_PROMPTS + _NON_MEDICAL_PROMPTS

        inputs = self._clip_processor(
            text=all_prompts,
            images=image,
            return_tensors="pt",
            padding=True,
        )
        # Move all tensors to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self._clip_model(**inputs)

        # logits_per_image: (1, num_prompts) — similarity of image to each prompt
        logits = outputs.logits_per_image[0]           # (num_prompts,)
        probs = F.softmax(logits, dim=0)               # (num_prompts,)

        n_med = len(_MEDICAL_PROMPTS)
        medical_score = probs[:n_med].sum().item()
        non_medical_score = probs[n_med:].sum().item()

        return medical_score, non_medical_score

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, image_path: str, question: str = "") -> ConfidenceResult:
        """
        Evaluate whether the image is likely a valid medical scan.

        Parameters
        ----------
        image_path:
            Absolute or relative path to the image file.
        question:
            Unused — kept for API compatibility with the pipeline caller.

        Returns
        -------
        ConfidenceResult
            ``passed=True`` when medical_score ≥ threshold.
        """
        self._load()

        medical_score, non_medical_score = self._classify(image_path)
        passed = medical_score >= self.threshold

        logger.debug(
            "CLIP gate | medical=%.4f non_medical=%.4f threshold=%.4f passed=%s",
            medical_score,
            non_medical_score,
            self.threshold,
            passed,
        )

        return ConfidenceResult(
            passed=passed,
            top_prob=medical_score,
            threshold=self.threshold,
            top_token_id=None,
        )
