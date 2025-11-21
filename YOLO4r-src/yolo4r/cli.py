# cli.py
import argparse
import sys

def print_global_help():
    print("""
YOLO4R - You Only Look Once for Research
==============================================

Available Commands:
  yolo4r train      Train, update, or resume a YOLO model.
  yolo4r detect     Run YOLO detection on one or more video/camera sources.
  yolo4r version    Show the YOLO4r version.
  yolo4r help       Show this help menu.

----------------------------------------------

Command Specific Help:
  yolo4r train help
  yolo4r detect help

----------------------------------------------
          
Examples:
  yolo4r train model=yolo11n architecture=custom_arch dataset=birds
  yolo4r train architecture=yolo12
  yolo4r train model=yolov8x name="best run ever!!" test
  yolo4r detect sources=usb0 usb1 usb2
  yolo4r detect test trailcam.mp4 trailcam2.mov

YOLO4r Documentation & Support:
  https://github.com/kgoertle/yolo4r
""")

def expand_key_value_args(argv):
    expanded = []

    mappings = {
        "name": "--name",
        "run": "--name",
        "run_name": "--name",

        "model": "--model",
        "weights": "--model",

        "update": "--update",

        "arch": "--arch",
        "architecture": "--arch",
        "backbone": "--arch",

        "data": "--dataset",
        "dataset": "--dataset",

        "labelstudio": "--labelstudio",
        "project": "--labelstudio",

        "sources": "--sources",
        "source": "--sources",

        "test": "--test",
    }

    boolean_true = {"1", "true", "yes", "on", ""}

    for arg in argv:

        # ---------- special case: plain "test" ----------
        if arg.lower() == "test":
            expanded.append("--test")
            continue

        # ---------- key=value pattern ----------
        if "=" in arg:
            key, value = arg.split("=", 1)
            key = key.lower()

            # special case: test=true / test=1
            if key == "test":
                if value.lower() in boolean_true:
                    expanded.append("--test")
                continue

            if key in mappings:
                expanded.append(mappings[key])
                expanded.append(value)
                continue

        expanded.append(arg)

    return expanded

def main():
    parser = argparse.ArgumentParser(
        prog="yolo4r",
        description="You Only Look Once for Research",
        add_help=True
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- TRAIN ----
    train_parser = subparsers.add_parser("train", help="Train or update a model.")
    train_parser.set_defaults(func="train")

    # ---- DETECT ----
    detect_parser = subparsers.add_parser("detect", help="Run YOLO detection.")
    detect_parser.set_defaults(func="detect")

    # ---- VERSION ----
    version_parser = subparsers.add_parser("version", help="Show YOLO4R version.")
    version_parser.set_defaults(func="version")

    # ---- HELP ----
    help_parser = subparsers.add_parser("help", help="Show all YOLO4R commands.")
    help_parser.set_defaults(func="help")

    # ---- Parse command (not sub-arguments) ----
    args, unknown = parser.parse_known_args()

    # Expand key=value into standard flags
    unknown = expand_key_value_args(unknown)

    # ROUTING
    if args.func == "train":
        if "--help" in unknown or "-h" in unknown or "help" in unknown:
            from .utils.help_text import print_train_help
            print_train_help()
            return

        from .train import main as train_main
        sys.argv = ["yolo4r-train"] + unknown
        return train_main()

    elif args.func == "detect":
        if "--help" in unknown or "-h" in unknown or "help" in unknown:
            from .utils.help_text import print_detect_help
            print_detect_help()
            return

        from .detect import main as detect_main
        sys.argv = ["yolo4r-detect"] + unknown
        return detect_main()

    elif args.func == "version":
        from .version import YOLO4R_VERSION
        print(f"YOLO4R {YOLO4R_VERSION}")
        return

    elif args.func == "help":
        return print_global_help()

    else:
        parser.print_help()
