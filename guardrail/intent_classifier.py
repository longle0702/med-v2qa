"""
guardrail/intent_classifier.py
--------------------------------
Gate 1 – Text Intent Classifier (VQA-model-based).

Uses the same MUMC .pth model already resident in memory for VQA inference.
No additional model weights (e.g. facebook/bart-large-mnli) are required.

How it works
------------
1. A blank (zero-filled) image tensor is fed to the visual encoder.
   This neutralises the visual signal so the decoder distribution is driven
   almost entirely by the *question text*.
2. The question is tokenised and passed through the text encoder with
   cross-attention on the blank image embeddings.
3. A single decoder step (BOS → first token) is run and the resulting
   logits are converted to a Softmax distribution over the BERT vocabulary.
4. We sum the probability mass assigned to a curated list of
   ``MEDICAL_SEED_TOKENS`` (radiology findings, anatomy terms, clinical
   adjectives).  A high cumulative mass signals that the model "thinks"
   the question sits in medical territory.
5. If the summed probability ≥ ``INTENT_THRESHOLD`` the gate passes.

Design notes
------------
- The classifier is **lazy** by default: the VQA model components are
  injected at construction time (when the API pre-loads the model), so
  Gate 1 never has to load its own weights.
- When called before the VQA model is ready (``model=None``), the gate
  passes with a neutral ``score=1.0`` so start-up is not blocked.
  The caller (``GuardrailPipeline``) ensures the model is loaded before
  Gate 2 runs.
- Token IDs for the seed tokens are resolved once at construction and
  cached as a ``torch.LongTensor`` for fast index_select lookups.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn.functional as F

from guardrail.config import (
    INTENT_THRESHOLD,
    MEDICAL_SEED_TOKENS,
)
from guardrail.result_types import IntentResult

logger = logging.getLogger(__name__)


class MedicalIntentClassifier:
    """
    Zero-cost text intent classifier powered by the MUMC VQA model.

    Parameters
    ----------
    model:
        Loaded ``MUMC_VQA`` instance in eval mode.  May be ``None`` at
        construction time — the gate will be bypassed (passes) until the
        model is injected via :meth:`set_model`.
    tokenizer:
        BERT tokenizer associated with the model.
    device:
        Torch device the model lives on.
    threshold:
        Minimum summed seed-token probability required to pass Gate 1.
        Defaults to ``config.INTENT_THRESHOLD``.
    image_size:
        Spatial resolution of the blank image tensor (must match the
        model's ``image_res`` config, typically 480).
    """

    def __init__(
        self,
        model=None,
        tokenizer=None,
        device: Optional[torch.device] = None,
        threshold: float = INTENT_THRESHOLD,
    ) -> None:
        self.threshold = threshold
        self.device = device or torch.device("cpu")

        self._model = model
        self._tokenizer = tokenizer
        self._seed_ids: Optional[torch.Tensor] = None

        if tokenizer is not None:
            self._build_seed_ids()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_seed_ids(self) -> None:
        """Resolve MEDICAL_SEED_TOKENS → vocab IDs (cached tensor)."""
        ids = []
        for token in MEDICAL_SEED_TOKENS:
            tok_ids = self._tokenizer.encode(token, add_special_tokens=False)
            if tok_ids:
                ids.append(tok_ids[0])  # take the first sub-word ID
        # Deduplicate while preserving order
        seen = set()
        unique_ids = []
        for i in ids:
            if i not in seen:
                seen.add(i)
                unique_ids.append(i)
        self._seed_ids = torch.tensor(unique_ids, dtype=torch.long, device=self.device)
        logger.debug(
            "Gate 1 seed vocab: %d unique token IDs for %d seed tokens.",
            len(unique_ids),
            len(MEDICAL_SEED_TOKENS),
        )

    def _blank_image(self) -> torch.Tensor:
        """Return a zero-filled image tensor sized to match the model's patch_embed."""
        img_size = self._model.visual_encoder.patch_embed.img_size
        H, W = (img_size, img_size) if isinstance(img_size, int) else img_size
        return torch.zeros(
            1, 3, H, W,
            dtype=torch.float32,
            device=self.device,
        )

    @torch.no_grad()
    def _score_question(self, question: str) -> tuple[float, str]:
        """
        Run the model with a blank image and compute the cumulative
        seed-token probability.

        Returns
        -------
        (seed_prob_sum, top_seed_token)
            ``seed_prob_sum`` — sum of softmax probabilities over all seed IDs.
            ``top_seed_token`` — the seed token with the highest individual prob.
        """
        # 1. Encode blank image
        image_tensor = self._blank_image()
        image_embeds = self._model.visual_encoder(image_tensor)
        image_atts = torch.ones(
            image_embeds.size()[:-1], dtype=torch.long
        ).to(self.device)

        # 2. Tokenise & encode question
        from dataset.utils import pre_question  # noqa: PLC0415
        processed = pre_question(question, max_ques_words=50)
        q_enc = self._tokenizer(
            [processed],
            padding="longest",
            truncation=True,
            max_length=25,
            return_tensors="pt",
        ).to(self.device)

        question_output = self._model.text_encoder(
            q_enc.input_ids,
            attention_mask=q_enc.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        # 3. Single decoder step: BOS → first-token distribution
        bos_ids = torch.tensor(
            [[self._tokenizer.cls_token_id]], device=self.device
        )
        decoder_output = self._model.text_decoder(
            bos_ids,
            encoder_hidden_states=question_output.last_hidden_state,
            encoder_attention_mask=q_enc.attention_mask,
            return_dict=True,
            reduction="none",
        )

        # logits: (1, 1, vocab_size) → (vocab_size,)
        logits = decoder_output.logits[0, 0, :]
        probs = F.softmax(logits, dim=-1)  # (vocab_size,)

        # 4. Sum probability mass over seed token IDs
        seed_probs = probs.index_select(0, self._seed_ids)  # (num_seeds,)
        seed_prob_sum = seed_probs.sum().item()

        # Identify the top contributing seed token (for diagnostics)
        top_local_idx = seed_probs.argmax().item()
        top_seed_token = MEDICAL_SEED_TOKENS[
            min(top_local_idx, len(MEDICAL_SEED_TOKENS) - 1)
        ]

        return seed_prob_sum, top_seed_token

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_model(self, model, tokenizer, device: torch.device) -> None:
        """
        Inject (or replace) the VQA model components after construction.

        Called by ``GuardrailPipeline`` once the MUMC model is loaded.
        """
        self._model = model
        self._tokenizer = tokenizer
        self.device = device
        self._build_seed_ids()

    def classify(self, question: str) -> IntentResult:
        """
        Classify *question* as medical or non-medical.

        Parameters
        ----------
        question:
            The raw question string submitted by the user.

        Returns
        -------
        IntentResult
            ``passed=True`` when cumulative seed-token probability ≥ threshold.
        """
        # Edge case: empty question
        cleaned = question.strip()
        if not cleaned:
            logger.debug("Empty question string → automatic refusal.")
            return IntentResult(
                passed=False,
                label="<empty>",
                score=0.0,
                raw_scores={},
            )

        # Model not yet loaded: pass through so the pipeline is not blocked
        # at start-up.  Gate 2 will catch non-medical images.
        if self._model is None or self._tokenizer is None:
            logger.warning(
                "Gate 1: VQA model not yet loaded — bypassing intent check."
            )
            return IntentResult(
                passed=True,
                label="<model-not-loaded>",
                score=1.0,
                raw_scores={},
            )

        seed_prob_sum, top_seed_token = self._score_question(cleaned)
        passed = seed_prob_sum >= self.threshold

        logger.debug(
            "Intent classify | top_seed='%s' seed_prob_sum=%.4f threshold=%.4f passed=%s",
            top_seed_token,
            seed_prob_sum,
            self.threshold,
            passed,
        )

        return IntentResult(
            passed=passed,
            label=top_seed_token,
            score=seed_prob_sum,
            raw_scores={},
        )

    def is_medical(self, question: str) -> bool:
        """
        Convenience wrapper that returns only the boolean gate decision.

        Parameters
        ----------
        question:
            The raw question string.

        Returns
        -------
        bool
            ``True`` when the question is classified as medical.
        """
        return self.classify(question).passed
