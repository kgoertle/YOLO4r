# utils/paths.py
from pathlib import Path
from datetime import datetime
import re, os

# ------ BASE DIRECTORY ------
BASE_DIR = Path.home() / "YOLO4r"
BASE_DIR.mkdir(parents=True, exist_ok=True)

# ------ STANDARD FOLDER LAYOUT ------
DATA_DIR          = BASE_DIR / "data"
RUNS_DIR          = BASE_DIR / "runs"
LOGS_DIR          = BASE_DIR / "logs"
MODELS_DIR        = BASE_DIR / "models"
WEIGHTS_DIR       = BASE_DIR / "weights"
CONFIGS_DIR       = BASE_DIR / "configs"
LS_ROOT           = BASE_DIR / "labelstudio-projects"
WANDB_ROOT        = BASE_DIR / "wandb"

# ---- EXTRA CONSTANTS FOR DETECT HELPERS ----
DEFAULT_DATA_YAML   = MODELS_DIR / "data.yaml"
MEASURE_CONFIG_YAML = CONFIGS_DIR / "measure_config.yaml"

# Ensure folders exist
for d in [
    DATA_DIR,
    RUNS_DIR,
    LOGS_DIR,
    MODELS_DIR,
    WEIGHTS_DIR,
    CONFIGS_DIR,
    LS_ROOT,
    WANDB_ROOT,
]:
    d.mkdir(parents=True, exist_ok=True)

#  ------ RUNS / LOGS RESOLUTION ------
def get_runs_dir(test: bool = False) -> Path:
    return RUNS_DIR / "test" if test else RUNS_DIR


def get_logs_dir(test: bool = False) -> Path:
    return LOGS_DIR / "test" if test else LOGS_DIR

#  ------ TRAINING PATHS ------
def get_training_paths(dataset_folder: Path, test=False):
    runs_root = get_runs_dir(test)
    logs_root = get_logs_dir(test)

    return {
        "runs_root": runs_root,
        "logs_root": logs_root,
        "train_folder": dataset_folder / "train/images",
        "val_folder": dataset_folder / "val/images",
        "weights_folder": WEIGHTS_DIR,
        "models_folder": MODELS_DIR,
        "dataset_folder": dataset_folder,
    }

#  ------ MODEL CONFIG ------
def get_model_config_dir(model_name: str) -> Path:
    model_name = str(model_name).strip()
    cfg_dir = CONFIGS_DIR / model_name
    cfg_dir.mkdir(parents=True, exist_ok=True)
    return cfg_dir

#  ------ DATA.YAML DISCOVERY ------
def get_latest_dataset_yaml(printer=None):
    if not DATA_DIR.exists():
        if printer:
            printer.warn(f"Data directory missing: {DATA_DIR}")
        return None

    dataset_dirs = [d for d in DATA_DIR.iterdir() if d.is_dir()]
    if not dataset_dirs:
        if printer:
            printer.warn(f"Dataset(s) NOT found in: {DATA_DIR}")
        return None

    latest = max(dataset_dirs, key=lambda d: d.stat().st_mtime)
    yaml_path = latest / "data.yaml"

    if yaml_path.exists():
        return yaml_path

    if printer:
        printer.warn(f"Data YAML NOT found inside {latest}")
    return None

def get_model_data_yaml(model_folder: Path, printer=None):
    model_yaml = model_folder / "data.yaml"
    if model_yaml.exists():
        return model_yaml

    # fallback
    return get_latest_dataset_yaml(printer)

# ------ DETECTION OUTPUT PATHS ------
def get_detection_output_paths(
    weights_path,
    source_type,
    source_name,
    test_detect=False,
    base_time=None,
):
    weights_path = Path(weights_path)
    model_folder = weights_path.parent.parent 
    model_name   = model_folder.name

    logs_root = get_logs_dir(test_detect) / model_name / "measurements"

    # Timestamp for this detection run
    folder_time = base_time or datetime.now()
    run_ts = folder_time.strftime("%m-%d-%Y_%H-%M-%S")

    # Sanitize source name
    safe_name = re.sub(
        r"[^\w\-\.]",
        "_",
        Path(source_name).stem if source_type == "video" else source_name
    )

    # Folder structure
    if source_type == "video":
        base_folder = logs_root / "video-in" / safe_name / run_ts
    else:
        base_folder = logs_root / "camera-feeds" / safe_name / run_ts

    # Prevent overwriting identical timestamps
    original = base_folder
    suffix = 1
    while base_folder.exists():
        base_folder = original.parent / f"{run_ts}_{suffix}"
        suffix += 1

    # Subfolders
    video_folder        = base_folder / "recordings"
    scores_folder       = base_folder / "scores"
    counts_folder       = scores_folder / "counts"
    frame_counts_folder = scores_folder / "frame-counts"
    interactions_folder = scores_folder / "interactions"

    for d in [
        video_folder,
        scores_folder,
        counts_folder,
        frame_counts_folder,
        interactions_folder,
    ]:
        d.mkdir(parents=True, exist_ok=True)

    return {
        "video_folder": video_folder,
        "scores_folder": scores_folder,
        "counts": counts_folder,
        "frame-counts": frame_counts_folder,
        "interactions": interactions_folder,
        "metadata": scores_folder / f"{safe_name}_metadata.json",
        "safe_name": safe_name,
    }

# detect.py previously imported: get_output_folder
get_output_folder = get_detection_output_paths
