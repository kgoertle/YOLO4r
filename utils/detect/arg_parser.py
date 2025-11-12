import argparse
import sys

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run YOLO object detection using latest best.pt")
    parser.add_argument("--test", action="store_true", help="Use the test model (runs/test). Omit to use full model (runs/main).")
    parser.add_argument("--sources", nargs='*', help="List of sources, e.g. usb0 usb1 video.mp4. Defaults to ['usb0'] if omitted.")
    if len(sys.argv) > 1 and not any(arg.startswith("--") for arg in sys.argv[1:]):
        args = parser.parse_args(["--sources"] + sys.argv[1:])
    else:
        args = parser.parse_args()
    if not args.sources:
        args.sources = ["usb0"]

    return args
