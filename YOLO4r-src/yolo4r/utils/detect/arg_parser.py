# utils/detect/args_parser.py
import argparse
import sys

from ..help_text import print_detect_help

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run object detection on video inputs and/or camera sources.",
        add_help=False   # <-- IMPORTANT
    )

    parser.add_argument("--test", action="store_true",
                        help="Use your test model directory...")
    parser.add_argument("--sources", nargs="*",
                        help="One or more camera/video sources")

    # Custom help routing
    if any(a in ("--help", "-h", "help") for a in sys.argv[1:]):
        print_detect_help()
        sys.exit(0)

    # Smart source defaulting
    if len(sys.argv) > 1 and not any(arg.startswith("--") for arg in sys.argv[1:]):
        args = parser.parse_args(["--sources"] + sys.argv[1:])
    else:
        args = parser.parse_args()

    if not args.sources:
        args.sources = ["usb0"]

    return args
