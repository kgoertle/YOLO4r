from pathlib import Path
import os
import argparse
import sys
from .io import ensure_weights, ensure_yolo_yaml
from .val_split import process_labelstudio_project

# ---- Base Paths ----
BASE_DIR = Path(os.getenv("YOLO_BASE_DIR", Path.cwd()))

def get_training_paths(dataset_folder: Path, test=False):
    """Return key directory paths for training and logging based on dataset folder."""
    return {
        "runs_root": BASE_DIR / "runs" / ("test" if test else "main"),
        "logs_root": BASE_DIR / "logs" / ("test" if test else "main"),
        "train_folder": dataset_folder / "train/images",
        "val_folder": dataset_folder / "val/images",
        "weights_folder": BASE_DIR / "weights",
        "models_folder": BASE_DIR / "models",
        "dataset_folder": dataset_folder
    }

def get_args():
    """Parse and return command-line arguments for YOLO training."""
    parser = argparse.ArgumentParser(description="YOLO Training Script")

    # ---- Mutually exclusive training modes ----
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--train", action="store_true", help="Transfer-learning training")
    group.add_argument("--update", action="store_true", help="Update weights from latest best.pt")
    group.add_argument("--scratch", action="store_true", help="Train from scratch on dataset")

    # ---- Other arguments ----
    parser.add_argument("--test", action="store_true", help="Debug mode for testing script")
    parser.add_argument("--resume", action="store_true", help="Resume from latest last.pt")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset folder in data/")
    parser.add_argument("--weights", type=str, default=None, help="Custom weights (.pt) for transfer learning")
    parser.add_argument("--model", type=str, default=None, help="Custom YOLO YAML for training from scratch")
    parser.add_argument("--name", type=str, default=None, help="Custom run name (defaults to timestamp)")

    args = parser.parse_args()

    # ---- Determine mode ----
    if args.update:
        mode = "update"
    elif args.scratch:
        mode = "scratch"
    else:
        mode = "train"

    # ---- Argument validation ----
    if args.weights and mode != "train":
        print("[ERROR] --weights can only be used with --train.")
        sys.exit(1)
    if args.model and mode != "scratch":
        print("[ERROR] --model can only be used with --scratch.")
        sys.exit(1)
    if args.weights and args.model:
        print("[ERROR] --weights and --model cannot be used together.")
        sys.exit(1)
    if args.weights and not args.weights.endswith(".pt"):
        print("[ERROR] --weights file must end with .pt")
        sys.exit(1)
    if args.model and not args.model.endswith(".yaml"):
        print("[ERROR] --model file must end with .yaml")
        sys.exit(1)

    # ---- Ensure data folder exists ----
    data_root = BASE_DIR / "data"
    data_root.mkdir(exist_ok=True)

    # Ensure all essential folders exist
    paths["weights_folder"].mkdir(parents=True, exist_ok=True)
    paths["models_folder"].mkdir(parents=True, exist_ok=True)
    
    # ---- Detect Label Studio projects in BASE_DIR if dataset not specified ----
    dataset_folder = None
    if args.dataset:
        dataset_folder = data_root / args.dataset
        if not dataset_folder.exists():
            print(f"[ERROR] Dataset folder not found: {dataset_folder}")
            sys.exit(1)
    else:
        # Auto-select or process Label Studio project
        ls_projects = [p for p in BASE_DIR.iterdir() if p.is_dir() and p.name.startswith("project-")]
        if ls_projects:
            newest_project = sorted(ls_projects, key=lambda x: x.stat().st_mtime, reverse=True)[0]
            print(f"[INFO] Found Label Studio project in base dir: {newest_project}")
            dataset_folder, DATA_YAML = process_labelstudio_project(newest_project, data_root)
        else:
            # No LS project, check for datasets in data/
            all_datasets = [d for d in data_root.iterdir() if d.is_dir()]
            if len(all_datasets) == 1:
                dataset_folder = all_datasets[0]
                print(f"[INFO] Auto-selected dataset: {dataset_folder.name}")
            else:
                print("[ERROR] Multiple datasets detected. Please specify --dataset")
                print("Available datasets:", [d.name for d in all_datasets])
                sys.exit(1)

    # ---- Define DATA_YAML ----
    DATA_YAML = dataset_folder / "data.yaml"
    if not DATA_YAML.exists():
        print(f"[ERROR] data.yaml not found in dataset folder: {DATA_YAML}")
        sys.exit(1)

    # ---- Resolve paths dynamically ----
    paths = get_training_paths(dataset_folder, test=args.test)

    # ---- Weights handling ----
    if args.weights:
        weights_path = Path(args.weights)
        if not weights_path.is_file():
            weights_path = paths["weights_folder"] / weights_path
        weights_path = ensure_weights(weights_path, model_type="yolo11n")
    else:
        weights_path = ensure_weights(paths["weights_folder"] / "yolo11n.pt", model_type="yolo11n")

    # ---- Model YAML handling ----
    if args.model:
        model_yaml = Path(args.model)
        if not model_yaml.is_file():
            model_yaml = paths["models_folder"] / model_yaml
        model_yaml = ensure_yolo_yaml(model_yaml, model_type=model_yaml.stem)
    else:
        model_yaml = ensure_yolo_yaml(paths["models_folder"] / "yolo11.yaml", model_type="yolo11")

    # ---- Attach resolved paths to args ----
    args.weights = weights_path
    args.model_yaml = model_yaml
    args.DATA_YAML = DATA_YAML
    args.train_folder = paths["train_folder"]
    args.val_folder = paths["val_folder"]
    args.dataset_folder = dataset_folder

    return args, mode

