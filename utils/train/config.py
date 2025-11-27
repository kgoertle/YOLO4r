# utils/train/config.py
from pathlib import Path
from importlib.resources import files
import os, argparse, sys, json, shutil
from datetime import datetime

from .io import (
    ensure_weights,
    ensure_yolo_yaml,
    normalize_model_name,
    FAMILY_TO_WEIGHTS,
    FAMILY_TO_YAML,
)
from .val_split import process_labelstudio_project
from ..help_text import print_detect_help, print_train_help
from ..console import (
    fmt_exit, fmt_info, fmt_model,
    fmt_warn, fmt_error, fmt_dataset, fmt_path, fmt_bold
)

# ---- CENTRALIZED PATHS (from utils.paths) ----
from ..paths import (
    BASE_DIR,
    DATA_DIR as DATA_ROOT,
    RUNS_DIR as RUNS_ROOT,
    LOGS_DIR as LOGS_ROOT,
    MODELS_DIR as MODELS_ROOT,
    WEIGHTS_DIR as WEIGHTS_ROOT,
    LS_ROOT,
    get_training_paths,
)

# -------- Example Project + Model Installation --------
def _install_examples():
    try:
        # ----------- Example Label Studio Project -----------
        pkg_example_ls = files("yolo4r") / "labelstudio-projects" / "example"
        target_ls = LS_ROOT / "example"

        if pkg_example_ls.is_dir() and not target_ls.exists():
            shutil.copytree(pkg_example_ls, target_ls, dirs_exist_ok=True)

        # ----------- Example Model Run -----------
        pkg_example_run = files("yolo4r") / "runs" / "sparrows"
        target_run = RUNS_ROOT / "sparrows"

        if pkg_example_run.is_dir() and not target_run.exists():
            shutil.copytree(pkg_example_run, target_run, dirs_exist_ok=True)

        # ----------- MODELS (architecture YAMLs) -----------
        pkg_models = files("yolo4r") / "models"
        target_models = MODELS_ROOT

        # Only copy if no user models exist yet
        if pkg_models.is_dir() and not any(target_models.iterdir()):
            shutil.copytree(pkg_models, target_models, dirs_exist_ok=True)

    except Exception as e:
        print(fmt_warn(f"Example installation failed: {e}"))

_install_examples()

def yaml_is_obb(yaml_path: Path) -> bool:
    try:
        with open(yaml_path, "r") as f:
            text = f.read().lower()
        return "obb" in text
    except Exception:
        return False

# -------- Helper: detect custom YAML vs official family --------
def is_custom_yaml(arch: str, models_dir: Path) -> bool:
    from .io import FAMILY_TO_YAML, normalize_model_name

    arch_lower = arch.lower()

    # Explicit YAML path/name
    if arch_lower.endswith(".yaml"):
        yaml_name = Path(arch_lower).name  # e.g., "yolo11.yaml" or "housesparrows-obb.yaml"

        # If this YAML name matches an official architecture
        if yaml_name in FAMILY_TO_YAML.values():
            return False

        # Otherwise treat as custom
        return True

    # No extension, so check if family is known
    family, _ = normalize_model_name(arch_lower)
    if family in FAMILY_TO_YAML:
        # Recognized official family
        return False

    # Not a known family, so must be custom
    return True

# -------- Label Studio helpers --------
def _find_labelstudio_projects(ls_root: Path):
    if not ls_root.exists():
        return []

    candidates = []
    for p in ls_root.iterdir():
        if not p.is_dir():
            continue
        img = p / "images"
        lbl = p / "labels"
        classes = p / "classes.txt"
        if img.is_dir() and lbl.is_dir() and classes.exists():
            candidates.append(p)
    return candidates

def _get_dataset_label_mode(dataset_folder: Path) -> str | None:
    meta_path = dataset_folder / "metadata.json"
    if not meta_path.exists():
        return None
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
        return meta.get("label_mode")
    except Exception:
        return None

def _family_is_obb(family: str | None) -> bool:
    return bool(family and family.endswith("-obb"))

# -------- Argument Parser --------
def get_args():
    parser = argparse.ArgumentParser(
        description="YOLO Training Script",
        add_help=False
    )

    # ---- Short-circuit help BEFORE validation triggers ----
    if any(arg in sys.argv for arg in ("--help", "-h", "help")):
        print_train_help()
        sys.exit(0)

    # ------------- CORE TRAINING MODE FLAGS -------------
    mode_group = parser.add_mutually_exclusive_group(required=False)

    mode_group.add_argument(
        "--train",
        "--transfer-learning",
        "-t",
        action="store_true",
        help="Initiate training using transfer-learning, which uses the COCO or DOTAv1 datasets for model weights to ensure better precision and recall. Using this call defaults to YOLO11n.pt.",
    )

    parser.add_argument(
        "--update",
        "--upgrade",
        "-u",
        type=str,
        nargs="?",
        const=True,
        help="Update an existing model run by specifying the folder name. This only works if your dataset has new images.",
    )

    mode_group.add_argument(
        "--scratch",
        "-s",
        action="store_true",
        help="Initiate training from scratch using a specific model architecture. This can include either official YOLO backbones OR a custom .yaml file. Using this call defaults to YOLO11.yaml.",
    )

    # ------------- MODEL + ARCHITECTURE SELECTION -------------
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        help="Initiate training using a specific model's weights for transfer-learning. Specifying a model will automatically download it if the file does not exist already and will also initiate a clean head using the appropriate model architecture.",
    )

    parser.add_argument(
        "--arch",
        "--architecture",
        "--backbone",
        "-a",
        "-b",
        type=str,
        help="Initiate training a model from scratch using a specific model architecture OR a custom model architecture. While this cannot be called if you're training using an official YOLO model, but can be used in tandem with custom model architectures.",
    )

    # ------------- ADDITIONAL FLAGS -------------
    parser.add_argument(
        "--resume",
        "-r",
        action="store_true",
        help="Resumes training from latest last.pt, which is useful if your training is interrupted, whether that be from the user or the system itself. So long as a last.pt file was produced, you should be able to resume training if anything happens.",
    )

    parser.add_argument(
        "--test",
        "-T",
        action="store_true",
        help="Debug/testing mode that lowers training parameters and places the runs/logs directories output in a 'test' folder. This should allow users to safely debug the pipeline as a whole.",
    )

    parser.add_argument(
        "--dataset",
        "--data",
        "-d",
        type=str,
        default=None,
        help="Allows you to call on a specific dataset folder inside ./data/",
    )

    parser.add_argument(
        "--name",
        "-n",
        type=str,
        default=None,
        help="As the name implies (pun unintended), allows you to set a custom run name. This also applies to your Label-Studio project folder if processed. If not set, this will default to the time the model was created.",
    )

    parser.add_argument(
        "--labelstudio",
        "--labelstudio-project",
        "--project",
        "-ls",
        type=str,
        default=None,
        help="Specify a Label-Studio project folder inside ~/.yolo4r/labelstudio-projects to process. This can be used independently from training to process a project export OR can be tagged with a training mode to automatically process and initiate training. If a specific folder is not called for, the most recent project folder will be processed.",
    )

    args = parser.parse_args()

    if not hasattr(args, "weights"):
        args.weights = None

    # ------------- DETERMINE TRAINING MODE (INITIAL) -------------
    if args.update:
        mode = "update"
    elif args.arch and not args.model:
        mode = "scratch"
    elif args.model and not args.arch:
        mode = "train"
    elif args.scratch:
        mode = "scratch"
    elif args.train:
        mode = "train"
    else:
        # Default: transfer learning
        mode = "train"

    # ------------- VALIDATION -------------
    custom_arch = bool(args.arch and is_custom_yaml(args.arch, MODELS_ROOT))

    # ---- Validate model name (if provided) ----
    if args.model:
        m = args.model.lower()

        if not m.endswith(".pt"):
            family, variant = normalize_model_name(m)

            # family must be one of the official families
            if family not in FAMILY_TO_YAML:
                print(fmt_error(f"Model family NOT recognized: '{fmt_bold(family)}'."))
                print(fmt_error("Valid model families include:"))
                print("       - yolov8, yolov8-obb")
                print("       - yolo11, yolo11-obb")
                print("       - yolo12, (yolo12-obb weights unreleased)")
                print("         'n', 'm', 'x', & 'l' variants are available.")
                print("         Tag any letter to the available model families if desired.")
                sys.exit(1)

            # variant must be n/s/m/l/x or None
            if variant not in {None, "n", "s", "m", "l", "x"}:
                print(fmt_error(f"Model variant NOT recognized '{fmt_bold(args.model)}'."))
                print("        Valid variants are: n, s, m, l, x")
                sys.exit(1)

    # ---- Validate architecture (if NOT custom YAML) ----
    if args.arch and not custom_arch:
        a = args.arch.lower()
        family, _ = normalize_model_name(a)
        if family not in FAMILY_TO_YAML:
            print(fmt_error(f"Model architecture NOT recognized '{fmt_bold(args.arch)}'."))
            print(fmt_error("Valid architectures include:"))
            print("       - yolov8, yolov8-obb")
            print("       - yolo11, yolo11-obb")
            print("       - yolo12, yolo12-obb")
            sys.exit(1)

    if args.update and args.arch:
        print(fmt_error("Update CANNOT be used with architecture selection."))
        sys.exit(1)

    # -------- Determine unified name for model + dataset --------
    if args.name:
        base_name = args.name.strip()
    else:
        base_name = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    # Safe-increment function
    def _increment_name(root: Path, name: str) -> str:
        proposed = name
        i = 1
        while (root / proposed).exists():
            proposed = f"{name}{i}"
            i += 1
        return proposed

    # Compute final resolved name
    final_name = _increment_name(BASE_DIR / "data", base_name)
    args.final_name = final_name
    args.name = final_name

    # ------------- DATASET HANDLING -------------
    data_root = BASE_DIR / "data"
    data_root.mkdir(exist_ok=True)

    # Ensure Label Studio root exists
    LS_ROOT.mkdir(exist_ok=True)

    # ------ USER SUPPLIED DATASET DIRECTLY ------
    if args.dataset:
        dataset_folder = data_root / args.dataset
        if not dataset_folder.exists():
            print(fmt_error(f"Dataset folder NOT found: {fmt_bold(dataset_folder)}"))
            sys.exit(1)

        DATA_YAML = dataset_folder / "data.yaml"
        if not DATA_YAML.exists():
            print(fmt_error(f"Data YAML NOT found in dataset folder: {fmt_bold(DATA_YAML)}"))
            sys.exit(1)

    # ------ USER EXPLICITLY REQUESTED LABEL STUDIO PROJECT ------
    elif args.labelstudio is not None:

        # ---- Determine whether training is requested ----
        training_requested = (
            args.train
            or args.scratch
            or args.update
            or args.test
            or args.model
            or args.arch
        )

        # ---- Process LS project first (dataset_folder now defined) ----
        if args.labelstudio is True or args.labelstudio == "":
            ls_projects = _find_labelstudio_projects(LS_ROOT)
            if not ls_projects:
                print(fmt_error("Label-Studio projects NOT found in labelstudio-projects/"))
                sys.exit(1)

            newest = sorted(ls_projects, key=lambda x: x.stat().st_mtime, reverse=True)[0]
            print(fmt_dataset(f"Using newest Label-Studio project: {fmt_path(newest)}"))
            dataset_folder, DATA_YAML = process_labelstudio_project(
                newest, data_root, dataset_name=args.final_name
            )

        else:
            specific = LS_ROOT / args.labelstudio
            if not specific.exists():
                print(fmt_error(f"Specified Label-Studio project NOT found: {fmt_path(specific)}"))
                sys.exit(1)

            print(fmt_dataset(f"Processing specified Label-Studio project: {fmt_path(specific)}"))
            dataset_folder, DATA_YAML = process_labelstudio_project(
                specific, data_root, dataset_name=args.final_name
            )

        # ---- EARLY EXIT: User only wanted dataset processing ----
        if not training_requested:
            print(fmt_dataset(f"Finished processing Label-Studio project: {fmt_path(dataset_folder)}"))
            print(fmt_info("Exiting after dataset creation."))
            sys.exit(0)

    # ------ NO LS REQUEST — USE LOCAL DATASETS ONLY ------
    else:
        all_datasets = [d for d in data_root.iterdir() if d.is_dir()]

        if len(all_datasets) == 0:
            print(fmt_error("No datasets exist. Provide dataset or use 'labelstudio' flags to process a project."))
            sys.exit(1)

        elif len(all_datasets) == 1:
            dataset_folder = all_datasets[0]
            DATA_YAML = dataset_folder / "data.yaml"
            print(fmt_dataset(f"Auto-selected dataset: {fmt_path(dataset_folder.name)}"))

        else:
            print(fmt_error("Multiple datasets detected; specify with 'dataset' or 'data' flags."))
            print("Available datasets:", [d.name for d in all_datasets])
            sys.exit(1)

    # ---- Dataset label mode (HBB vs OBB) ----
    label_mode = _get_dataset_label_mode(dataset_folder)
    dataset_is_obb = (label_mode == "obb") if label_mode is not None else None

    # ------------- PATH SETUP -------------
    paths = get_training_paths(dataset_folder, test=args.test)
    paths["weights_folder"].mkdir(parents=True, exist_ok=True)
    paths["models_folder"].mkdir(parents=True, exist_ok=True)

    # ------------- REQUESTED FAMILIES (MODEL / ARCH) -------------
    requested_model_family = None
    if args.model:
        requested_model_family, _ = normalize_model_name(args.model)

    if args.arch:
        if custom_arch:
            # Custom YAML: treat architecture family as unknown (for logging only)
            requested_arch_family = None
        else:
            requested_arch_family, _ = normalize_model_name(args.arch)
    elif requested_model_family:
        requested_arch_family = requested_model_family
    else:
        # Default: yolo11 family
        requested_arch_family = "yolo11"

    # -------- Official-only model/arch pairing (custom YAML exempt) --------
    if (
        not custom_arch
        and args.model
        and requested_arch_family
        and requested_model_family
    ):
        special_y12obb = (requested_model_family == "yolo12-obb")
        if requested_arch_family != requested_model_family and not special_y12obb:
            print(
                fmt_error(f"Model architecture '{fmt_bold(requested_arch_family)}' does NOT match model '{fmt_bold(requested_model_family)}'.")
            )
            sys.exit(1)

    # ------------- INITIAL EFFECTIVE FAMILIES -------------
    # Architecture family always comes from requested arch (or default yolo11)
    arch_family = requested_arch_family

    # Weight family is only used when not training from scratch
    weight_family = None
    if mode != "scratch":
        if requested_model_family:
            weight_family = requested_model_family
        else:
            # Default transfer-learning family when none specified → yolo11
            weight_family = "yolo11"

    # --------- OBB/HBB DATASET ENFORCEMENT (official families only) ---------
    if dataset_is_obb is not None and not custom_arch:
        arch_is_obb = _family_is_obb(arch_family)
        weight_is_obb = _family_is_obb(weight_family) if weight_family else None

        fallback_family = None

        if dataset_is_obb:
            if (arch_family and not arch_is_obb) or (weight_family and weight_is_obb is False):
                fallback_family = "yolo11-obb"
        else:
            if (arch_family and arch_is_obb) or (weight_family and weight_is_obb):
                fallback_family = "yolo11"

        if fallback_family:
            if arch_family != fallback_family:
                print(fmt_warn(f"Dataset is {label_mode.upper()} and does NOT match selection. Overriding architecture family to: {fmt_bold(fallback_family)}."))
                arch_family = fallback_family

            if mode != "scratch" and weight_family != fallback_family:
                print(fmt_warn(f"Dataset is {label_mode.upper()} and does NOT match selection. Overriding weight family to: {fmt_bold(fallback_family)}."))
                weight_family = fallback_family

    # ----------- ARCHITECTURE RESOLUTION (supports custom YAML) -----------
    if custom_arch:
        # Custom YAML path resolution, extension is optional
        arch_lower = args.arch.lower()
        candidates = []

        if arch_lower.endswith(".yaml"):
            candidates.append(Path(arch_lower))
            candidates.append(paths["models_folder"] / arch_lower)
        else:
            candidates.append(Path(arch_lower + ".yaml"))
            candidates.append(paths["models_folder"] / (arch_lower + ".yaml"))
            candidates.append(Path(arch_lower))
            candidates.append(paths["models_folder"] / arch_lower)

        model_yaml = None
        for cand in candidates:
            if cand.exists():
                model_yaml = cand
                break

        if not model_yaml:
            print(fmt_error(f"Custom model architecture YAML NOT found for '{args.arch}'. Tried:"))
            for cand in candidates:
                print(f"       - {cand}")
            sys.exit(1)

    else:
        # Official YOLO family architecture
        yaml_name = FAMILY_TO_YAML.get(arch_family)
        if yaml_name is None:
            print(fmt_error(f"Model architecture YAML NOT registered to '{fmt_bold(arch_family)}'."))
            sys.exit(1)

        model_yaml = ensure_yolo_yaml(
            paths["models_folder"] / yaml_name,
            model_type=arch_family,
        )

        if model_yaml is None:
            print(fmt_error(f"Failed to resolve model architecture YAML to '{fmt_bold(arch_family)}'."))
            sys.exit(1)

    print(fmt_model(f"Using model architecture YAML: {fmt_path(model_yaml)}"))

    # --------- Custom YAML OBB/HBB enforcement (after model_yaml resolution) ---------
    if custom_arch and dataset_is_obb is not None:
        arch_is_obb = yaml_is_obb(model_yaml)

        if dataset_is_obb and not arch_is_obb:
            print(fmt_error("OBB dataset requires an OBB-capable architecture. The specified YAML does not contain OBB layers."))
            sys.exit(1)

        if not dataset_is_obb and arch_is_obb:
            print(fmt_error("HBB dataset cannot be trained with an OBB architecture. The specified YAML contains OBB layers."))
            sys.exit(1)

    # ------------- WEIGHTS RESOLUTION (AFTER FALLBACK) -------------
    if mode != "scratch":
        if weight_family not in FAMILY_TO_WEIGHTS and weight_family != "yolo12-obb":
            print(fmt_error(f"Default weights are NOT registered for '{fmt_bold(weight_family)}'."))
            sys.exit(1)

        # args.model may carry variant info (yolo11m, yolo11x-obb, etc.)
        model_type_for_weights = args.model if args.model else weight_family
        args.weights = ensure_weights(
            paths["weights_folder"],
            model_type=model_type_for_weights,
        )
    else:
        args.weights = None

    args.model_yaml = model_yaml
    if isinstance(args.weights, str) and args.weights.endswith(".pt"):
        args.weights = Path(args.weights)

    # ------------- ATTACH RESOLVED PATHS -------------
    args.DATA_YAML = DATA_YAML
    args.train_folder = paths["train_folder"]
    args.val_folder = paths["val_folder"]
    args.dataset_folder = dataset_folder

    # Expose whether the architecture was custom to train.py
    args.custom_arch = custom_arch

    return args, mode
