#!/usr/bin/env python3
"""
guardrail_demo.py
------------------
Command-line demonstration of the Dual-Gate Medical Guardrail.

Usage
-----
::

    # Should pass both gates → real VQA answer
    python guardrail_demo.py \\
        --image test_inference_data/open/synpic18319.jpg \\
        --question "Describe the lung abnormalities"

    # Should fail Gate 1 (non-medical text)
    python guardrail_demo.py \\
        --image test_inference_data/open/synpic18319.jpg \\
        --question "What is the capital of France?"

    # Should fail Gate 2 (non-medical image)
    python guardrail_demo.py \\
        --image /path/to/regular_photo.jpg \\
        --question "What findings are visible?"

    # Custom thresholds and beam width
    python guardrail_demo.py \\
        --image test_inference_data/open/synpic18319.jpg \\
        --question "Is there cardiomegaly?" \\
        --confidence-threshold 0.45 \\
        --intent-threshold 0.60 \\
        --num-beams 5 \\
        --max-new-tokens 30

Options
-------
--image                Path to the input image (required)
--question             Clinical question to ask (required)
--confidence-threshold Softmax top-1 floor for Gate 2 (default: 0.30)
--intent-threshold     Medical-intent score floor for Gate 1 (default: 0.55)
--num-beams            Beam width for VQA decoder (default: 3)
--max-new-tokens       Maximum answer tokens (default: 20)
--verbose              Enable DEBUG-level logging
"""

from __future__ import annotations

import argparse
import logging
import sys
import textwrap

from guardrail.pipeline import GuardrailPipeline

# ANSI colour helpers (disabled when output is not a TTY)
_USE_COLOUR = sys.stdout.isatty()

def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOUR else text

GREEN  = lambda t: _c(t, "32")
RED    = lambda t: _c(t, "31")
YELLOW = lambda t: _c(t, "33")
CYAN   = lambda t: _c(t, "36")
BOLD   = lambda t: _c(t, "1")
DIM    = lambda t: _c(t, "2")


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

BANNER = """\
╔══════════════════════════════════════════════════════╗
║   Med-V²QA  •  Dual-Gate Medical Guardrail  (MA8-2) ║
╚══════════════════════════════════════════════════════╝
"""


def _print_gate(label: str, passed: bool, detail: str) -> None:
    icon  = GREEN("✔") if passed else RED("✘")
    state = GREEN("PASS") if passed else RED("FAIL")
    print(f"  {icon}  {BOLD(label)}: {state}   {DIM(detail)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        prog="guardrail_demo",
        description="Dual-Gate Medical Guardrail demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python guardrail_demo.py --image xray.jpg --question "Any abnormalities?"
              python guardrail_demo.py --image photo.jpg --question "Who is in this picture?"
        """),
    )
    parser.add_argument("--image",    required=True, help="Path to the input image.")
    parser.add_argument("--question", required=True, help="Clinical question to ask.")
    parser.add_argument(
        "--confidence-threshold", type=float, default=0.30, metavar="FLOAT",
        help="Softmax top-1 probability floor for Gate 2 (default: 0.30).",
    )
    parser.add_argument(
        "--intent-threshold", type=float, default=0.55, metavar="FLOAT",
        help="Medical-intent score floor for Gate 1 (default: 0.55).",
    )
    parser.add_argument(
        "--num-beams", type=int, default=3,
        help="Beam width for the VQA decoder (default: 3).",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=20,
        help="Maximum answer tokens (default: 20).",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG-level logging.",
    )
    args = parser.parse_args()

    # Logging setup
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(levelname)-8s %(name)s: %(message)s",
    )

    print(CYAN(BANNER))
    print(f"  {BOLD('Image   :')} {args.image}")
    print(f"  {BOLD('Question:')} {args.question}")
    print(f"  {BOLD('Thresholds:')} intent={args.intent_threshold}  "
          f"confidence={args.confidence_threshold}")
    print()

    # Build and run pipeline
    pipe = GuardrailPipeline(
        intent_threshold=args.intent_threshold,
        confidence_threshold=args.confidence_threshold,
    )

    result = pipe.run(
        image_path=args.image,
        question=args.question,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
    )

    # ── Gate 1 display ────────────────────────────────────────────────
    if result.intent_result:
        ir = result.intent_result
        _print_gate(
            "Gate 1 (Intent)",
            ir.passed,
            f"label='{ir.label}'  score={ir.score:.3f}  "
            f"threshold={args.intent_threshold}",
        )
    else:
        print(f"  {DIM('Gate 1: not evaluated')}")

    # ── Gate 2 display ────────────────────────────────────────────────
    if result.confidence_result:
        cr = result.confidence_result
        _print_gate(
            "Gate 2 (Confidence)",
            cr.passed,
            f"top_prob={cr.top_prob:.4f}  "
            f"threshold={args.confidence_threshold}",
        )
    else:
        print(f"  {DIM('Gate 2: not evaluated (skipped by Gate 1 failure)')}")

    print()

    # ── Final outcome ─────────────────────────────────────────────────
    if result.passed:
        print(BOLD("─" * 54))
        print(BOLD(GREEN("✔  ACCEPTED — VQA answer:")))
        print(f"   {GREEN(result.answer)}")
        if result.metadata:
            print()
            print(DIM(
                f"   Timing → gate1={result.metadata['gate1_ms']:.0f}ms  "
                f"gate2={result.metadata['gate2_ms']:.0f}ms  "
                f"inference={result.metadata['inference_ms']:.0f}ms  "
                f"total={result.metadata['total_ms']:.0f}ms"
            ))
        print(BOLD("─" * 54))
        return 0
    else:
        print(BOLD("─" * 54))
        gate_label = result.gate_triggered.upper()
        print(BOLD(RED(f"✘  REFUSED by Gate {gate_label}:")))
        print(f"   {YELLOW(result.refusal_message)}")
        print(BOLD("─" * 54))
        return 1


if __name__ == "__main__":
    sys.exit(main())
