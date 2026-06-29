"""
api/engine.py
-------------
Shared inference engine for Med-V²QA.

Loads the MUMC VQA model once at startup and exposes:

* ONNX Runtime sessions for ``visual_encoder`` and ``text_encoder``
  (primary path, when ``onnx_models/`` is present).
* Raw PyTorch model as a transparent fallback when ONNX files are absent.

The engine is instantiated once by ``api/main.py`` and injected into both
``GuardrailPipeline`` and ``BatchTriageService`` so the 2.2 GB checkpoint
is never loaded more than once.

Public interface
----------------
``engine.backend``  → "onnx" | "pytorch"
``engine.model``    → loaded MUMC_VQA (always needed for decoder generate())
``engine.tokenizer``
``engine.transform``
``engine.device``
``engine.predict(image_bytes, question, num_beams, max_new_tokens) → str``
"""
from __future__ import annotations

import io
import logging
import os
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# Paths relative to project root (one level above api/)
_REPO_ROOT = Path(__file__).resolve().parent.parent
_ONNX_DIR = _REPO_ROOT / "onnx_models"
_VIS_ONNX = _ONNX_DIR / "visual_encoder.onnx"
_TXT_ONNX = _ONNX_DIR / "text_encoder.onnx"


class InferenceEngine:
    """
    Central MUMC inference engine used by all API endpoints.

    Parameters
    ----------
    device:
        Force a specific device string ("cpu", "cuda", "mps").
        Auto-detected when ``None``.
    """

    def __init__(self, device: Optional[str] = None) -> None:
        # ── Device ────────────────────────────────────────────────────
        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # ── Load PyTorch model (always needed for beam-search decoder) ─
        from inference import load_model  # noqa: PLC0415
        logger.info("Loading MUMC checkpoint on %s …", self.device)
        self.model, self.tokenizer, self.transform = load_model(self.device)
        logger.info("MUMC checkpoint loaded.")

        # Always use the PyTorch (.pth) backend — no ONNX sessions.
        self._vis_session = None
        self._txt_session = None
        self.backend: Literal["onnx", "pytorch"] = "pytorch"
        logger.info("Backend: PyTorch (.pth checkpoint).")

    # ------------------------------------------------------------------
    # Private: encode helpers
    # ------------------------------------------------------------------

    def _encode_image_tensor(self, image_bytes: bytes) -> torch.Tensor:
        """Decode bytes → normalised [1, 3, H, W] tensor on self.device."""
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return self.transform(image).unsqueeze(0).to(self.device)

    def _image_embeds(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Run visual encoder (ONNX or PyTorch) → [1, N, 768]."""
        if self.backend == "onnx":
            img_np = image_tensor.cpu().numpy().astype(np.float32)
            output = self._vis_session.run(None, {"image": img_np})
            return torch.from_numpy(output[0]).to(self.device)
        with torch.no_grad():
            return self.model.visual_encoder(image_tensor)

    def _question_hidden_states(
        self,
        image_embeds: torch.Tensor,
        image_atts: torch.Tensor,
        q_enc,
    ) -> torch.Tensor:
        """Run text encoder (ONNX or PyTorch) → last_hidden_state [1, L, 768]."""
        if self.backend == "onnx":
            feeds = {
                "input_ids":               q_enc.input_ids.cpu().numpy().astype(np.int64),
                "attention_mask":          q_enc.attention_mask.cpu().numpy().astype(np.int64),
                "encoder_hidden_states":   image_embeds.cpu().numpy().astype(np.float32),
                "encoder_attention_mask":  image_atts.cpu().numpy().astype(np.int64),
            }
            output = self._txt_session.run(None, feeds)
            return torch.from_numpy(output[0]).to(self.device)
        with torch.no_grad():
            out = self.model.text_encoder(
                q_enc.input_ids,
                attention_mask=q_enc.attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            return out.last_hidden_state

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        image_bytes: bytes,
        question: str,
        num_beams: int = 5,
        max_new_tokens: int = 20,
    ) -> str:
        """
        Run full VQA inference (encode image → encode question → beam-search).

        Parameters
        ----------
        image_bytes:
            Raw bytes of the uploaded image file.
        question:
            The user's clinical question string (pre-processing applied internally).
        num_beams:
            Beam width for the decoder.
        max_new_tokens:
            Maximum answer tokens to generate.

        Returns
        -------
        str
            The model's decoded answer.
        """
        from dataset.utils import pre_question  # noqa: PLC0415

        # 1. Image
        image_tensor = self._encode_image_tensor(image_bytes)

        # 2. Question
        processed_q = pre_question(question, 50)
        q_enc = self.tokenizer(
            [processed_q],
            padding="longest",
            truncation=True,
            max_length=25,
            return_tensors="pt",
        ).to(self.device)

        # 3. Encode via ONNX or PyTorch
        image_embeds = self._image_embeds(image_tensor)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(self.device)
        question_hidden = self._question_hidden_states(image_embeds, image_atts, q_enc)

        # 4. Beam-search decode (always PyTorch — generate() is dynamic)
        bos_ids = torch.tensor([[self.tokenizer.cls_token_id]], device=self.device)
        with torch.no_grad():
            outputs = self.model.text_decoder.generate(
                input_ids=bos_ids,
                encoder_hidden_states=question_hidden,
                encoder_attention_mask=q_enc.attention_mask,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                eos_token_id=self.tokenizer.sep_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        return self.tokenizer.decode(outputs[0][1:], skip_special_tokens=True).strip()
