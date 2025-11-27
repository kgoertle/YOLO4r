# utils/train/io.py
import requests
import json
from pathlib import Path
from typing import Optional
from ..console import (
    fmt_exit, fmt_info, fmt_model,
    fmt_warn, fmt_error, fmt_dataset
)

# -------- Model name normalization --------
def normalize_model_name(name: str) -> tuple[str, str | None]:
    base = name.lower().replace(".pt", "").replace(".yaml", "")

    # OBB flag
    is_obb = base.endswith("-obb")
    core = base[:-4] if is_obb else base  # strip "-obb" if present

    variant = None
    if core and core[-1] in {"n", "s", "m", "l", "x"}:
        family_core = core[:-1]
        variant = core[-1]
    else:
        family_core = core

    family = family_core + ("-obb" if is_obb else "")
    return family, variant

# -------- Model families to YAML --------
FAMILY_TO_YAML = {
    "yolov8":      "yolov8.yaml",
    "yolov8-obb":  "yolov8-obb.yaml",
    "yolo11":      "yolo11.yaml",
    "yolo11-obb":  "yolo11-obb.yaml",
    "yolo12":      "yolo12.yaml",
    "yolo12-obb":  "yolo12-obb.yaml",
}

# -------- Model families to default weights --------
FAMILY_TO_WEIGHTS = {
    "yolov8":      "yolov8n.pt",
    "yolov8-obb":  "yolov8n-obb.pt",
    "yolo11":      "yolo11n.pt",
    "yolo11-obb":  "yolo11n-obb.pt",
    "yolo12":      "yolo12n.pt",
    # NOTE: no official "yolo12n-obb.pt", so special-cased fallback
}

# -------- File Downloads --------
def download_file(url: str, dest_path: Path) -> Optional[Path]:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        print(fmt_info(f"Downloaded {dest_path}"))
        return dest_path
    except Exception as e:
        print(fmt_error(f"Failed downloading {url}: {e}"))
        return None

# -------- Ensure YAML (architecture) --------
def ensure_yolo_yaml(yolo_yaml_path: Path, model_type: str) -> Optional[Path]:
    from .io import FAMILY_TO_YAML, download_file, normalize_model_name

    family, _ = normalize_model_name(model_type)

    if family not in FAMILY_TO_YAML:
        print(fmt_error(f"Unsupported architecture family: '{model_type}' → '{family}'"))
        print(fmt_error(f"Supported families: {list(FAMILY_TO_YAML.keys())}"))
        return None

    if yolo_yaml_path.exists():
        return yolo_yaml_path

    yaml_filename = FAMILY_TO_YAML[family]
    yaml_urls = {
        "yolov8":      "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/models/v8/yolov8.yaml",
        "yolov8-obb":  "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/models/v8/yolov8-obb.yaml",
        "yolo11":      "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/models/11/yolo11.yaml",
        "yolo11-obb":  "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/models/11/yolo11-obb.yaml",
        "yolo12":      "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/models/12/yolo12.yaml",
        "yolo12-obb":  "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/models/12/yolo12-obb.yaml",
    }

    url = yaml_urls[family]
    print(fmt_info(f"Model architecture YAML not found, downloading '{family}' → {yolo_yaml_path}"))
    return download_file(url, yolo_yaml_path)

# -------- Ensure Weights --------
def ensure_weights(yolo_weights_path: Path, model_type: str) -> Optional[Path]:
    from .io import FAMILY_TO_WEIGHTS, download_file, normalize_model_name

    # Force weights path to a directory *first*
    if yolo_weights_path.suffix == ".pt":
        yolo_weights_path = yolo_weights_path.parent

    family, variant = normalize_model_name(model_type)
    is_obb = family.endswith("-obb")
    family_base = family[:-4] if is_obb else family
    variant = variant or "n"

    correct_name = f"{family_base}{variant}{'-obb' if is_obb else ''}.pt"
    dest_path = yolo_weights_path / correct_name

    # ----- EARLY EXIT -----
    if dest_path.is_file():
        return dest_path

    # ----- Handle special cases -----
    if family not in FAMILY_TO_WEIGHTS:
        if family == "yolo12-obb":
            print(fmt_warn(f"Pretrained OBB weights not found for '{family}'. Falling back to 'yolo12'."))
            correct_name = f"yolo12{variant}.pt"
            dest_path = yolo_weights_path / correct_name
            family = "yolo12"

            if dest_path.is_file():
                return dest_path
        else:
            print(fmt_error(f"No registered default weights for '{family}'"))
            return None

    # ----- URL lookup -----
    weight_urls = {
        "yolov8n.pt":  "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
        "yolov8s.pt":  "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt",
        "yolov8m.pt":  "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m.pt",
        "yolov8l.pt":  "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8l.pt",
        "yolov8x.pt":  "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x.pt",

        "yolov8n-obb.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-obb.pt",
        "yolov8s-obb.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-obb.pt",
        "yolov8m-obb.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m-obb.pt",
        "yolov8l-obb.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8l-obb.pt",
        "yolov8x-obb.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x-obb.pt",

        "yolo11n.pt":  "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
        "yolo11s.pt":  "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt",
        "yolo11m.pt":  "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt",
        "yolo11l.pt":  "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt",
        "yolo11x.pt":  "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt",

        "yolo11n-obb.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-obb.pt",
        "yolo11s-obb.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-obb.pt",
        "yolo11m-obb.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-obb.pt",
        "yolo11l-obb.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l-obb.pt",
        "yolo11x-obb.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-obb.pt",

        "yolo12n.pt":  "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12n.pt",
        "yolo12s.pt":  "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12s.pt",
        "yolo12m.pt":  "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12m.pt",
        "yolo12l.pt":  "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12l.pt",
        "yolo12x.pt":  "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12x.pt",
    }

    if correct_name not in weight_urls:
        print(fmt_error(f"No URL for {correct_name}"))
        return None

    print(fmt_info(f"Model weights not found, downloading '{model_type}' ({correct_name}) → {dest_path}"))
    return download_file(weight_urls[correct_name], dest_path)

# -------- Image Counting --------
def count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sum(len(list(folder.glob(f"*{e}"))) for e in exts)

# -------- Metadata Loading --------
def load_latest_metadata(logs_root: Path) -> Optional[dict]:
    if not logs_root.exists():
        return None
    latest, meta = 0, None
    for run in logs_root.iterdir():
        if not run.is_dir():
            continue
        p = run / "metadata.json"
        if p.exists() and (mtime := p.stat().st_mtime) > latest:
            latest = mtime
            try:
                meta = json.load(open(p, "r"))
            except Exception as e:
                print(fmt_warn(f"Failed to load metadata JSON file: {e}"))
    return meta
