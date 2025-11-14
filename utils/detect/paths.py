# utils/detect/paths.py
from pathlib import Path
from datetime import datetime
import re

# ---------- PROJECT ROOT ----------
def find_project_root():
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "utils").exists() and (parent / "detect.py").exists():
            return parent
    return p.parents[2]

BASE_DIR = find_project_root()

# ---------- CORE DIRECTORIES ----------
RUNS_DIR_MAIN = BASE_DIR / "runs"
RUNS_DIR_TEST = RUNS_DIR_MAIN / "test"

LOGS_DIR = BASE_DIR / "logs"
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
CONFIGS_DIR = BASE_DIR / "configs"

for d in [RUNS_DIR_MAIN, LOGS_DIR, MODELS_DIR, DATA_DIR, CONFIGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DEFAULT_DATA_YAML = MODELS_DIR / "data.yaml"
CLASSES_CONFIG_YAML = CONFIGS_DIR / "classes_config.yaml"
MEASURE_CONFIG_YAML = CONFIGS_DIR / "measure_config.yaml"

# ---------- RUNS DIRECTORY RESOLUTION ----------
def get_runs_dir(test=False):
    return RUNS_DIR_TEST if test else RUNS_DIR_MAIN

# ---------- MODEL-SPECIFIC CONFIG ROOT ----------
def get_model_config_dir(model_name: str) -> Path:
    """
    Returns /configs/<model_name>/, guaranteed to exist.
    """
    model_name = str(model_name).strip()
    cfg_dir = CONFIGS_DIR / model_name
    cfg_dir.mkdir(parents=True, exist_ok=True)
    return cfg_dir

# ---------- DATASET YAML LOADING ----------
def get_latest_dataset_yaml(printer=None):
    """Returns newest dataset folder's data.yaml or None."""
    if not DATA_DIR.exists():
        if printer:
            printer.warn(f"Data directory {DATA_DIR} does not exist.")
        return None

    dataset_dirs = [d for d in DATA_DIR.iterdir() if d.is_dir()]
    if not dataset_dirs:
        if printer:
            printer.warn(f"No dataset folders found in {DATA_DIR}.")
        return None

    latest_dataset = max(dataset_dirs, key=lambda d: d.stat().st_mtime)
    yaml_path = latest_dataset / "data.yaml"

    if yaml_path.exists():
        return yaml_path

    if printer:
        printer.warn(f"No data.yaml found in {latest_dataset}")
    return None


def get_model_data_yaml(model_folder: Path, printer=None):
    """Returns the model run’s own data.yaml or falls back to dataset."""
    model_yaml = model_folder / "data.yaml"
    if model_yaml.exists():
        return model_yaml
    return get_latest_dataset_yaml(printer)

# ---------- OUTPUT FOLDER BUILDER ----------
def get_output_folder(weights_path, source_type, source_name, test_detect=False, base_time=None):
    weights_path = Path(weights_path)

    # Model folder is: /runs/<model_timestamp>/
    model_folder = weights_path.parent.parent
    model_timestamp = model_folder.name

    logs_root = (LOGS_DIR / "test" if test_detect else LOGS_DIR) / model_timestamp / "measurements"

    # Timestamp for this run
    folder_time = base_time or datetime.now()
    run_ts = folder_time.strftime("%m-%d-%Y_%H-%M-%S")

    # Clean name (video filename / usb0)
    safe_name = re.sub(r"[^\w\-\.]", "_",
                       Path(source_name).stem if source_type == "video" else source_name)

    # Folder structure
    if source_type == "video":
        base_folder = logs_root / "video-in" / safe_name / run_ts
    else:
        base_folder = logs_root / "camera-feeds" / safe_name / run_ts

    # Prevent overwriting
    original = base_folder
    suffix = 1
    while base_folder.exists():
        base_folder = original.parent / f"{run_ts}_{suffix}"
        suffix += 1

    # Subfolders
    video_folder = base_folder / "recordings"
    scores_folder = base_folder / "scores"
    counts_folder = scores_folder / "counts"
    frame_counts_folder = scores_folder / "frame-counts"
    interactions_folder = scores_folder / "interactions"

    # Create
    for d in [video_folder, scores_folder, counts_folder, frame_counts_folder, interactions_folder]:
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
