from pathlib import Path
from datetime import datetime
import re

BASE_DIR = Path(__file__).resolve().parents[2]

# ------ DIRECTORIES ------
RUNS_DIR_MAIN = BASE_DIR / "runs"
RUNS_DIR_TEST = BASE_DIR / "runs/test"

def get_runs_dir(test=False):
    """Return Path to runs directory based on mode."""
    return RUNS_DIR_TEST if test else RUNS_DIR_MAIN

def select_model_run(base_path, printer=None, choice_index=None, auto_select=False):
    """Return path to best.pt, auto-selecting latest or using choice_index."""
    base_path = Path(base_path)
    if not base_path.exists():
        return None

    model_dirs = sorted([d for d in base_path.iterdir() if d.is_dir()], reverse=True)
    if not model_dirs:
        return None

    if len(model_dirs) == 1 or auto_select:
        selected = model_dirs[0]
        if printer:
            printer.model_selected_info(selected.name) 
        return selected / "weights" / "best.pt"

    if choice_index and 1 <= choice_index <= len(model_dirs):
        selected = model_dirs[choice_index - 1]
        if printer:
            printer.model_selected_info(selected.name) 
        return selected / "weights" / "best.pt"

# ------ DATASET HANDLING ------
DATA_DIR = BASE_DIR / "data"

def get_latest_dataset_yaml(printer=None):
    if not DATA_DIR.exists():
        if printer:
            printer.warn(f"Data directory {DATA_DIR} does not exist.")
        return None

    dataset_dirs = [d for d in DATA_DIR.iterdir() if d.is_dir()]
    if not dataset_dirs:
        if printer:
            printer.warn(f"No dataset folders found in {DATA_DIR}.")
        return None

    # Sort by modification time, descending (latest first)
    latest_dataset = sorted(dataset_dirs, key=lambda d: d.stat().st_mtime, reverse=True)[0]
    data_yaml = latest_dataset / "data.yaml"

    if not data_yaml.exists():
        if printer:
            printer.warn(f"No data.yaml found in latest dataset {latest_dataset}.")
        return None

    return data_yaml

def get_model_data_yaml(model_folder: Path, printer=None):
    """Return the data.yaml stored with the model, or fallback to latest dataset."""
    model_yaml = model_folder / "data.yaml"
    if model_yaml.exists():
        return model_yaml
    # fallback
    return get_latest_dataset_yaml(printer)

def get_output_folder(weights_path, source_type, source_name, test_detect=False, base_time=None):
    weights_path = Path(weights_path)
    train_folder = weights_path.parent.parent
    model_timestamp = train_folder.name
    logs_root = BASE_DIR / ("logs/test" if test_detect else "logs") / model_timestamp / "measurements"

    folder_time = base_time or datetime.now()
    run_ts = folder_time.strftime("%m-%d-%Y_%H-%M-%S")

    # ------ Clean folder name ------
    safe_name = re.sub(r'[^\w\-\.]', '_', Path(source_name).stem if source_type == "video" else source_name)

    if source_type == "video":
        base_folder = logs_root / "video-in" / safe_name / run_ts
    else:
        base_folder = logs_root / "camera-feeds" / safe_name / run_ts

    # Avoid overwriting
    suffix = 1
    original_base = base_folder
    while base_folder.exists():
        base_folder = original_base.parent / f"{run_ts}_{suffix}"
        suffix += 1

    # ------ Main folders ------
    video_folder = base_folder / "recordings"
    scores_folder = base_folder / "scores"

    # ------ Measurement subfolders ------
    counts_folder = scores_folder / "counts"
    frame_counts_folder = scores_folder / "frame-counts"
    interactions_folder = scores_folder / "interactions"

    # Metadata file
    metadata_file = scores_folder / f"{safe_name}_metadata.json"

    # ------ Create directories ------
    for folder in [video_folder, scores_folder, counts_folder, frame_counts_folder, interactions_folder]:
        folder.mkdir(parents=True, exist_ok=True)

    return {
        "video_folder": video_folder,
        "scores_folder": scores_folder,
        "counts": counts_folder,
        "frame-counts": frame_counts_folder,
        "interactions": interactions_folder,
        "metadata": metadata_file,
        "safe_name": safe_name, 
    }
