#!/usr/bin/env python3
"""
triage_demo.py
--------------
Command-line demonstration of the VQA-Driven Batch Triage Sorting.

Usage
-----
::

    python triage_demo.py --images test_inference_data/open/synpic18319.jpg test_inference_data/open/synpic32933.jpg /Users/alex0702/MyProjects/med-v2qa/test_inference_data/open/072.png

Options
-------
--images      Paths to input images (required, multiple allowed)
--verbose     Enable DEBUG-level logging
"""

import argparse
import logging
import sys

from triage.batch_sorter import BatchTriageService

# ANSI colour helpers
def BOLD(x: str) -> str: return f"\033[1m{x}\033[0m" if sys.stdout.isatty() else x
def RED(x: str) -> str: return f"\033[91m{x}\033[0m" if sys.stdout.isatty() else x
def GREEN(x: str) -> str: return f"\033[92m{x}\033[0m" if sys.stdout.isatty() else x
def CYAN(x: str) -> str: return f"\033[96m{x}\033[0m" if sys.stdout.isatty() else x
def DIM(x: str) -> str: return f"\033[2m{x}\033[0m" if sys.stdout.isatty() else x


BANNER = """\
╔══════════════════════════════════════════════════════╗
║   Med-V²QA  •  Batch Triage Sorting (MA8-3)          ║
╚══════════════════════════════════════════════════════╝"""


def main():
    parser = argparse.ArgumentParser(
        description="Demo for Batch Triage Sorting using MUMC VQA model.",
    )
    parser.add_argument("--images", nargs='+', required=True, help="Paths to input images.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable DEBUG-level logging.")
    
    args = parser.parse_args()

    # Logging setup
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(levelname)-8s %(name)s: %(message)s",
    )

    print(CYAN(BANNER))
    print(f"  {BOLD('Queue Size :')} {len(args.images)} images")
    print(f"  {BOLD('Images     :')}")
    for idx, path in enumerate(args.images):
        print(f"    {idx+1}. {DIM(path)}")
    print()

    # Build and run batch sorter
    triage = BatchTriageService()
    
    print("Running batch triage inference... (this may take a moment)\n")
    sorted_queue = triage.sort_batch(args.images)
    
    print("──────────────────────────────────────────────────────")
    print(BOLD("TRIAGE RESULTS (Sorted by Criticality)"))
    print("──────────────────────────────────────────────────────")
    
    for idx, item in enumerate(sorted_queue):
        path = item["image_path"]
        score = item["score"]
        is_abnormal = item["is_abnormal"]
        
        status_str = RED("ABNORMAL") if is_abnormal else GREEN("NORMAL")
        print(f"{idx+1:2d}. [{status_str}] Score: {score:.4f} | {DIM(path)}")
        
    print("──────────────────────────────────────────────────────")

if __name__ == "__main__":
    main()
