import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run YOLO object detection using latest best.pt")
    parser.add_argument("--test", action="store_true", help="Use the test model (runs/test). Omit to use full model (runs/main).")
    parser.add_argument("--sources", nargs='*', help="List of sources, e.g. usb0 usb1 video.mp4. Defaults to ['usb0'] if omitted.")
    parser.add_argument("--lab", action="store_true", help="Enable lab mode (reports internal settings).")

    args = parser.parse_args()

    # default to usb0 when no sources specified
    if not args.sources:
        args.sources = ['usb0']

    return args
