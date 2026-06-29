#!/usr/bin/env python3
"""
export_onnx.py
--------------
One-time script to export the two static MUMC sub-graphs to ONNX format.

Exports
-------
* onnx_models/visual_encoder.onnx
    Input  : image          [B, 3, 256, 256]  float32
    Output : image_embeds   [B, N, 768]        float32

* onnx_models/text_encoder.onnx
    Inputs : input_ids               [B, L]   int64
             attention_mask          [B, L]   int64
             encoder_hidden_states   [B, N, 768] float32
             encoder_attention_mask  [B, N]   int64
    Output : last_hidden_state       [B, L, 768] float32

Note: The text_decoder generate() loop is autoregressive and stays in
PyTorch. Only the two static encoder sub-graphs benefit from ONNX export.

Usage
-----
    python export_onnx.py
    python export_onnx.py --out-dir custom_onnx/ --opset 17
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_OUT_DIR = Path(__file__).parent / "onnx_models"
OPSET = 14


# ---------------------------------------------------------------------------
# Sub-graph wrappers
# ---------------------------------------------------------------------------

class VisualEncoderWrapper(torch.nn.Module):
    """Wraps model.visual_encoder so torch.onnx.export can trace it cleanly."""
    def __init__(self, model):
        super().__init__()
        self.visual_encoder = model.visual_encoder

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.visual_encoder(image)


class TextEncoderWrapper(torch.nn.Module):
    """
    Wraps model.text_encoder for ONNX export.

    Returns only ``last_hidden_state`` (a plain Tensor) so the ONNX graph
    has a single concrete output rather than a ModelOutput dataclass.
    """
    def __init__(self, model):
        super().__init__()
        self.text_encoder = model.text_encoder

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        out = self.text_encoder(
            input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=True,
        )
        return out.last_hidden_state


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def _export_visual_encoder(model, out_dir: Path, opset: int, device: torch.device) -> Path:
    logger.info("Exporting visual_encoder …")
    wrapper = VisualEncoderWrapper(model).eval().to(device)
    dummy = torch.randn(1, 3, 256, 256, device=device)

    out_path = out_dir / "visual_encoder.onnx"
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(out_path),
        opset_version=opset,
        input_names=["image"],
        output_names=["image_embeds"],
        dynamic_axes={
            "image":        {0: "batch"},
            "image_embeds": {0: "batch"},
        },
        do_constant_folding=True,
    )
    logger.info("  Saved → %s", out_path)
    return out_path


def _export_text_encoder(model, out_dir: Path, opset: int, device: torch.device) -> Path:
    logger.info("Exporting text_encoder …")
    wrapper = TextEncoderWrapper(model).eval().to(device)

    # Dummy inputs: batch=1, question_len=10, image_seq=197 (256px ViT patch grid)
    dummy_input_ids         = torch.ones(1, 10, dtype=torch.long, device=device)
    dummy_attention_mask    = torch.ones(1, 10, dtype=torch.long, device=device)
    dummy_enc_hidden        = torch.randn(1, 197, 768, device=device)
    dummy_enc_atts          = torch.ones(1, 197, dtype=torch.long, device=device)

    out_path = out_dir / "text_encoder.onnx"
    torch.onnx.export(
        wrapper,
        (dummy_input_ids, dummy_attention_mask, dummy_enc_hidden, dummy_enc_atts),
        str(out_path),
        opset_version=opset,
        input_names=[
            "input_ids",
            "attention_mask",
            "encoder_hidden_states",
            "encoder_attention_mask",
        ],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids":               {0: "batch", 1: "q_len"},
            "attention_mask":          {0: "batch", 1: "q_len"},
            "encoder_hidden_states":   {0: "batch", 1: "img_seq"},
            "encoder_attention_mask":  {0: "batch", 1: "img_seq"},
            "last_hidden_state":       {0: "batch", 1: "q_len"},
        },
        do_constant_folding=True,
    )
    logger.info("  Saved → %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_visual_encoder(pt_model, onnx_path: Path, device: torch.device) -> None:
    import onnxruntime as ort  # noqa: PLC0415

    logger.info("Validating visual_encoder …")
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    dummy_np = np.random.randn(1, 3, 256, 256).astype(np.float32)
    dummy_pt = torch.from_numpy(dummy_np).to(device)

    with torch.no_grad():
        pt_out = pt_model.visual_encoder(dummy_pt).cpu().numpy()
    ort_out = session.run(None, {"image": dummy_np})[0]

    max_diff = np.abs(pt_out - ort_out).max()
    logger.info("  max |PT − ONNX| = %.2e  %s", max_diff,
                "✓ PASS" if max_diff < 1e-4 else "✗ DIFF TOO LARGE")
    if max_diff >= 1e-4:
        raise RuntimeError(f"visual_encoder validation failed: max_diff={max_diff:.2e}")


def _validate_text_encoder(pt_model, onnx_path: Path, device: torch.device) -> None:
    import onnxruntime as ort  # noqa: PLC0415

    logger.info("Validating text_encoder …")
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    dummy_ids   = np.ones((1, 10), dtype=np.int64)
    dummy_atts  = np.ones((1, 10), dtype=np.int64)
    dummy_enc   = np.random.randn(1, 197, 768).astype(np.float32)
    dummy_eatts = np.ones((1, 197), dtype=np.int64)

    with torch.no_grad():
        pt_out = pt_model.text_encoder(
            torch.from_numpy(dummy_ids).to(device),
            attention_mask=torch.from_numpy(dummy_atts).to(device),
            encoder_hidden_states=torch.from_numpy(dummy_enc).to(device),
            encoder_attention_mask=torch.from_numpy(dummy_eatts).to(device),
            return_dict=True,
        ).last_hidden_state.cpu().numpy()

    ort_out = session.run(None, {
        "input_ids":              dummy_ids,
        "attention_mask":         dummy_atts,
        "encoder_hidden_states":  dummy_enc,
        "encoder_attention_mask": dummy_eatts,
    })[0]

    max_diff = np.abs(pt_out - ort_out).max()
    logger.info("  max |PT − ONNX| = %.2e  %s", max_diff,
                "✓ PASS" if max_diff < 1e-4 else "✗ DIFF TOO LARGE")
    if max_diff >= 1e-4:
        raise RuntimeError(f"text_encoder validation failed: max_diff={max_diff:.2e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Export MUMC encoders to ONNX.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR),
                        help=f"Output directory (default: {DEFAULT_OUT_DIR})")
    parser.add_argument("--opset",   type=int, default=OPSET,
                        help=f"ONNX opset version (default: {OPSET})")
    parser.add_argument("--no-validate", action="store_true",
                        help="Skip post-export validation with onnxruntime.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Export device: %s", device)

    # Load PyTorch model
    logger.info("Loading MUMC checkpoint …")
    from inference import load_model  # noqa: PLC0415
    model, _, _ = load_model(device)
    model.eval()

    # Export
    vis_path = _export_visual_encoder(model, out_dir, args.opset, device)
    txt_path = _export_text_encoder(model, out_dir, args.opset, device)

    # Validate (on CPU to avoid GPU↔CPU copy issues in comparison)
    if not args.no_validate:
        cpu_model, _, _ = load_model(torch.device("cpu"))
        cpu_model.eval()
        _validate_visual_encoder(cpu_model, vis_path, torch.device("cpu"))
        _validate_text_encoder(cpu_model, txt_path, torch.device("cpu"))

    logger.info("=" * 55)
    logger.info("  ONNX export complete.")
    logger.info("  visual_encoder : %s", vis_path)
    logger.info("  text_encoder   : %s", txt_path)
    logger.info("=" * 55)


if __name__ == "__main__":
    main()
