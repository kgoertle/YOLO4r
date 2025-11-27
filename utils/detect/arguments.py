import argparse
import sys

from ..help_text import print_detect_help

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run object detection on video inputs and/or camera sources.",
        add_help=False
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Use your test model directory..."
    )

    parser.add_argument(
        "--sources",
        nargs="*",
        help="One or more camera/video sources"
    )

    parser.add_argument(
        "--model", "-m",
        type=str,
        help=(
            "Select a specific model run (folder inside runs/) "
            "or leave unset to use the default selection behavior."
        )
    )

    # ---- Custom help routing ----
    raw_args = sys.argv[1:]
    if any(a in ("--help", "-h", "help") for a in raw_args):
        print_detect_help()
        sys.exit(0)

    # ---- Normalize key=value forms like model=sparrows-v2 ----
    normalized = []
    for a in raw_args:
        if a.startswith("model=") or a.startswith("m="):
            _, v = a.split("=", 1)
            normalized.extend(["--model", v])
        else:
            normalized.append(a)

    # ---- Inject --sources when appropriate ----
    def inject_sources_if_needed(args_list):
        if not args_list:
            return args_list
        if "--sources" in args_list:
            return args_list
        if not any(a.startswith("--") for a in args_list):
            return ["--sources"] + args_list

        new = []
        i = 0
        while i < len(args_list):
            a = args_list[i]
            if a in ("--model", "-m"):
                new.append(a)
                if i + 1 < len(args_list):
                    new.append(args_list[i + 1])
                    i += 2
                else:
                    i += 1
            elif a == "--test":
                new.append(a)
                i += 1
            elif a.startswith("--"):
                new.append(a)
                i += 1
            else:
                new.append("--sources")
                new.extend(args_list[i:])
                break
        return new

    final_argv = inject_sources_if_needed(normalized)
    args = parser.parse_args(final_argv)
    if not args.sources:
        args.sources = ["usb0"]

    return args

