# utils/train/config.py
from pathlib import Path
import os, argparse, sys
from .io import (
    ensure_weights,
    ensure_yolo_yaml,
    normalize_model_name,
    FAMILY_TO_WEIGHTS,
    FAMILY_TO_YAML,
)

from .val_split import process_labelstudio_project

# ---- Base Paths ----
BASE_DIR = Path(os.getenv("YOLO_BASE_DIR", Path.cwd()))


def get_training_paths(dataset_folder: Path, test=False):
    """Return key directory paths for training and logging based on dataset folder."""
    return {
        "runs_root": BASE_DIR / "runs" / "test" if test else BASE_DIR / "runs",
        "logs_root": BASE_DIR / "logs" / "test" if test else BASE_DIR / "logs",
        "train_folder": dataset_folder / "train/images",
        "val_folder": dataset_folder / "val/images",
        "weights_folder": BASE_DIR / "weights",
        "models_folder": BASE_DIR / "models",
        "dataset_folder": dataset_folder,
    }

def get_args():
    """Parse and return command-line arguments for YOLO training."""
    parser = argparse.ArgumentParser(description="YOLO Training Script")

    # ------------- CORE TRAINING MODE FLAGS -------------
    mode_group = parser.add_mutually_exclusive_group(required=False)

    mode_group.add_argument(
        "--train",
        "--transfer-learning",
        "-t",
        action="store_true",
        help="Force transfer-learning mode.",
    )

    parser.add_argument(
        "--update",
        "--upgrade",
        "-u",
        type=str,
        nargs="?",
        const=True,
        help="Update an existing model run by folder name.",
    )

    mode_group.add_argument(
        "--scratch",
        "-s",
        action="store_true",
        help="Force scratch training from architecture.",
    )

    # ------------- MODEL + ARCHITECTURE SELECTION -------------
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        help="Pretrained weights (.pt) or family name for transfer learning.",
    )

    parser.add_argument(
        "--arch",
        "--architecture",
        "--backbone",
        "-a",
        "-b",
        type=str,
        help="YOLO architecture YAML (or family) for training from scratch.",
    )

    # ------------- ADDITIONAL FLAGS -------------
    parser.add_argument(
        "--resume",
        "-r",
        action="store_true",
        help="Resume training from latest last.pt",
    )

    parser.add_argument(
        "--test",
        "-T",
        action="store_true",
        help="Debug/testing mode (fast settings)",
    )

    parser.add_argument(
        "--dataset",
        "--data",
        "-d",
        type=str,
        default=None,
        help="Dataset folder inside ./data/",
    )

    parser.add_argument(
        "--name",
        "-n",
        type=str,
        default=None,
        help="Custom run name (defaults to timestamp)",
    )

    args = parser.parse_args()

    if not hasattr(args, "weights"):
        args.weights = None

    # ------------- DETERMINE TRAINING MODE (INITIAL) -------------
    if args.update:
        mode = "update"
    elif args.arch and not args.model:
        # Explicit architecture only → pure scratch
        mode = "scratch"
    elif args.model and not args.arch:
        # Model only → pure transfer-learning
        mode = "train"
    elif args.scratch:
        mode = "scratch"
    elif args.train:
        mode = "train"
    else:
        # Default to transfer-learning with a sensible default model
        mode = "train"

    # ------------- VALIDATION (NO MIXING -m AND -a) -------------
    if args.model and args.arch:
        print("[ERROR] Cannot use --model (.pt or family) and --arch (.yaml or family) simultaneously.")
        sys.exit(1)

    # ------------- MODEL / ARCH NAME VALIDATION -------------
    if args.model:
        m = args.model.lower()
        if not (m.endswith(".pt") or m in FAMILY_TO_WEIGHTS or m in FAMILY_TO_YAML):
            print(f"[ERROR] Unknown model name '{args.model}'.")
            print("[ERROR] Valid examples include:")
            print("       - yolov8, yolov8n.pt")
            print("       - yolo11, yolo11n.pt")
            print("       - yolo12, yolo12n.pt")
            print("       - yolo11-obb, yolov8-obb")
            sys.exit(1)

    if args.arch:
        a = args.arch.lower()
        if not (a.endswith(".yaml") or a in FAMILY_TO_YAML):
            print(f"[ERROR] Unknown architecture '{args.arch}'.")
            print("[ERROR] Valid architectures include:")
            print("       - yolov8, yolov8-obb")
            print("       - yolo11, yolo11-obb")
            print("       - yolo12, yolo12-obb")
            sys.exit(1)

    if args.update and args.arch:
        print("[ERROR] --update cannot be used with architecture selection.")
        sys.exit(1)

    # ------------- DATASET HANDLING -------------
    data_root = BASE_DIR / "data"
    data_root.mkdir(exist_ok=True)

    if args.dataset:
        dataset_folder = data_root / args.dataset
        if not dataset_folder.exists():
            print(f"[ERROR] Dataset folder not found: {dataset_folder}")
            sys.exit(1)
    else:
        ls_projects = [p for p in BASE_DIR.iterdir() if p.is_dir() and p.name.startswith("project-")]

        if ls_projects:
            newest = sorted(ls_projects, key=lambda x: x.stat().st_mtime, reverse=True)[0]
            print(f"[DATA] Found Label Studio project: {newest}")
            dataset_folder, DATA_YAML = process_labelstudio_project(newest, data_root)
        else:
            all_datasets = [d for d in data_root.iterdir() if d.is_dir()]
            if len(all_datasets) == 1:
                dataset_folder = all_datasets[0]
                print(f"[DATA] Auto-selected dataset: {dataset_folder.name}")
            else:
                print("[ERROR] Multiple datasets detected; specify with --dataset or --data.")
                print("Available datasets:", [d.name for d in all_datasets])
                sys.exit(1)

    DATA_YAML = dataset_folder / "data.yaml"
    if not DATA_YAML.exists():
        print(f"[ERROR] data.yaml not found in: {DATA_YAML}")
        sys.exit(1)

    # ------------- PATH SETUP -------------
    paths = get_training_paths(dataset_folder, test=args.test)
    paths["weights_folder"].mkdir(parents=True, exist_ok=True)
    paths["models_folder"].mkdir(parents=True, exist_ok=True)

    # ------------- WEIGHTS HANDLING (.pt or family name) -------------
    # Strict pairing for all families, plus a single special fallback for yolo12-obb
    if args.model:
        model_family, variant = normalize_model_name(args.model)

        if args.model == "yolo12-obb":
            print("[ERROR] yolo12-obb does not have pretrained weights. Use --arch yolo12-obb with --scratch.")
            sys.exit(1)

            args.weights = ensure_weights(
                paths["weights_folder"] / weight_name,
                model_type=fallback_family,
            )

        else:
            # NORMAL CASE FOR ALL OTHER MODELS
            weight_name = FAMILY_TO_WEIGHTS.get(model_family)
            if weight_name is None:
                print(f"[ERROR] No default weights registered for model family '{model_family}'.")
                sys.exit(1)

            args.weights = ensure_weights(
                paths["weights_folder"] / weight_name,
                model_type=model_family,
            )

    else:
        if mode != "scratch":
            default_family = "yolo11"
            weight_name = FAMILY_TO_WEIGHTS[default_family]
            args.weights = ensure_weights(
                paths["weights_folder"] / weight_name,
                model_type=default_family,
            )
        else:
            args.weights = None  # scratch mode: no weights needed

    # ------------- ARCHITECTURE HANDLING (strict pairing) -------------
    if args.arch:
        arch_family, _ = normalize_model_name(args.arch)
    elif args.model:
        arch_family, _ = normalize_model_name(args.model)
    else:
        arch_family = "yolo11"

    if args.model:
        model_family, _ = normalize_model_name(args.model)
        special_y12obb = (model_family == "yolo12-obb")
        if arch_family != model_family and not special_y12obb:
            print(f"[ERROR] Architecture '{arch_family}' does not match model family '{model_family}'.")
            sys.exit(1)

    yaml_name = FAMILY_TO_YAML.get(arch_family)
    if yaml_name is None:
        print(f"[ERROR] No architecture YAML registered for family '{arch_family}'.")
        sys.exit(1)

    model_yaml = ensure_yolo_yaml(
        paths["models_folder"] / yaml_name,
        model_type=arch_family,
    )

    if model_yaml is None:
        print(f"[ERROR] Failed to resolve architecture YAML for '{arch_family}'.")
        sys.exit(1)

    args.model_yaml = model_yaml
    if isinstance(args.weights, str) and args.weights.endswith(".pt"):
        args.weights = Path(args.weights)

    # ------------- ATTACH RESOLVED PATHS -------------
    args.DATA_YAML = DATA_YAML
    args.train_folder = paths["train_folder"]
    args.val_folder = paths["val_folder"]
    args.dataset_folder = dataset_folder

    return args, mode
