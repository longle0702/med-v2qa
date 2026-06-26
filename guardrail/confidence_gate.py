"""
guardrail/confidence_gate.py
-----------------------------
Gate 2 – Softmax Confidence Gate.

Runs a **single greedy forward pass** (one decoder step only) through the
already-loaded MUMC model and reads the top-1 Softmax probability at the
first decoding position.  If that probability falls below the configured
threshold the image is considered non-medical and the request is refused.

Design rationale
----------------
- One decoder step is orders of magnitude faster than full beam-search
  inference, so Gate 2 adds minimal latency.
- We reuse the visual encoder and text encoder that are already resident
  in memory — no extra model weights are loaded.
- The Softmax distribution at the first BOS→token step is a reliable
  proxy for model confidence: a high-entropy distribution (low max prob)
  indicates the model has no strong prior over possible answers, which is
  the hallmark of out-of-domain inputs.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image

from guardrail.config import CONFIDENCE_THRESHOLD
from guardrail.result_types import ConfidenceResult

logger = logging.getLogger(__name__)


class ConfidenceGate:
    """
    Gate 2: checks image-level confidence via the MUMC decoder.

    Parameters
    ----------
    model:
        Loaded ``MUMC_VQA`` instance (already in eval mode).
    tokenizer:
        BERT tokenizer associated with the model.
    transform:
        torchvision transform pipeline used to pre-process images.
    device:
        Torch device the model lives on.
    threshold:
        Minimum top-1 Softmax probability required to pass.
        Defaults to ``config.CONFIDENCE_THRESHOLD``.
    """

    def __init__(
        self,
        model,
        tokenizer,
        transform,
        device: torch.device,
        threshold: float = CONFIDENCE_THRESHOLD,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.transform = transform
        self.device = device
        self.threshold = threshold

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_image(self, image_path: str) -> torch.Tensor:
        """Load and transform an image file into a batched tensor."""
        image = Image.open(image_path).convert("RGB")
        return self.transform(image).unsqueeze(0).to(self.device)

    def _encode_question(self, question: str):
        """Tokenise the question string."""
        from dataset.utils import pre_question  # noqa: PLC0415

        processed = pre_question(question, max_ques_words=50)
        return self.tokenizer(
            [processed],
            padding="longest",
            truncation=True,
            max_length=25,
            return_tensors="pt",
        ).to(self.device)

    @torch.no_grad()
    def _get_top_prob(
        self,
        image_tensor: torch.Tensor,
        q_enc,
    ) -> tuple[float, Optional[int]]:
        """
        Run one decoder step and return the top-1 Softmax probability.

        Returns
        -------
        (top_prob, top_token_id)
        """
        # 1. Visual encoding
        image_embeds = self.model.visual_encoder(image_tensor)
        image_atts = torch.ones(
            image_embeds.size()[:-1], dtype=torch.long
        ).to(self.device)

        # 2. Text (question) encoding with cross-attention on image
        question_output = self.model.text_encoder(
            q_enc.input_ids,
            attention_mask=q_enc.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        # 3. Single decoder step: BOS token → first output distribution
        bos_ids = torch.tensor(
            [[self.tokenizer.cls_token_id]], device=self.device
        )
        decoder_output = self.model.text_decoder(
            bos_ids,
            encoder_hidden_states=question_output.last_hidden_state,
            encoder_attention_mask=q_enc.attention_mask,
            return_dict=True,
            reduction="none",
        )

        # logits shape: (1, 1, vocab_size)
        logits = decoder_output.logits[:, 0, :]           # (1, vocab_size)
        probs = F.softmax(logits, dim=-1)                  # (1, vocab_size)
        top_prob, top_idx = probs.max(dim=-1)

        return top_prob.item(), top_idx.item()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, image_path: str, question: str) -> ConfidenceResult:
        """
        Evaluate whether the image is likely a valid medical scan.

        Parameters
        ----------
        image_path:
            Absolute or relative path to the image file.
        question:
            The user's question (used for cross-attention context).

        Returns
        -------
        ConfidenceResult
            ``passed=True`` when ``top_prob >= threshold``.
        """
        image_tensor = self._encode_image(image_path)
        q_enc = self._encode_question(question)

        top_prob, top_token_id = self._get_top_prob(image_tensor, q_enc)

        passed = top_prob >= self.threshold

        logger.debug(
            "Confidence gate | top_prob=%.4f threshold=%.4f passed=%s",
            top_prob,
            self.threshold,
            passed,
        )

        return ConfidenceResult(
            passed=passed,
            top_prob=top_prob,
            threshold=self.threshold,
            top_token_id=top_token_id,
        )
