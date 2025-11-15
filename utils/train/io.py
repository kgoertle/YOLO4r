# utils/train/io.py
import requests
import json
from pathlib import Path
from typing import Optional

# ---- Model name normalization ----
def normalize_model_name(name: str) -> tuple[str, str | None]:
    name = name.lower().replace(".pt", "").replace(".yaml", "")

    # OBB variants
    if "obb" in name:
        return name, None
    if name[-1] in {"n", "s", "m", "l", "x"}:
        return name[:-1], name[-1]

    return name, None

# ---- Model families → YAML ----
FAMILY_TO_YAML = {
    "yolov8":      "yolov8.yaml",
    "yolov8-obb":  "yolov8-obb.yaml",
    "yolo11":      "yolo11.yaml",
    "yolo11-obb":  "yolo11-obb.yaml",
    "yolo12":      "yolo12.yaml",
    "yolo12-obb":  "yolo12-obb.yaml",
}

# ---- Model families → default weights (.pt variant) ----
FAMILY_TO_WEIGHTS = {
    "yolov8":      "yolov8n.pt",
    "yolov8-obb":  "yolov8n-obb.pt",
    "yolo11":      "yolo11n.pt",
    "yolo11-obb":  "yolo11n-obb.pt",
    "yolo12":      "yolo12n.pt",
}

# ---- File Downloads ----
def download_file(url: str, dest_path: Path) -> Optional[Path]:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        print(f"[INFO] Downloaded {dest_path}")
        return dest_path
    except Exception as e:
        print(f"[ERROR] Failed downloading {url}: {e}")
        return None


# ---- Ensure YAML ----
def ensure_yolo_yaml(yolo_yaml_path: Path, model_type: str) -> Optional[Path]:
    """Download or return the correct model architecture YAML."""
    from .io import FAMILY_TO_YAML, download_file

    family, _ = normalize_model_name(model_type)

    if family not in FAMILY_TO_YAML:
        print(f"[ERROR] Unsupported architecture: '{model_type}'")
        print(f"[ERROR] Supported: {list(FAMILY_TO_YAML.keys())}")
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
    print(f"[DOWNLOAD] YAML not found, downloading '{family}' → {yolo_yaml_path}")
    return download_file(url, yolo_yaml_path)

# ---- Ensure Weights ----
def ensure_weights(yolo_weights_path: Path, model_type: str) -> Optional[Path]:
    from .io import FAMILY_TO_WEIGHTS, download_file

    # If file already exists — done
    if yolo_weights_path.exists():
        return yolo_weights_path

    family, variant = normalize_model_name(model_type)

    # Determine expected weight file
    default_weight_name = FAMILY_TO_WEIGHTS.get(family)

    # Handle architectures with no .pt available (yolo12-obb)
    if default_weight_name is None:
        print(f"[WARN] No pretrained OBB weights available for '{family}'.")
        fallback_name = FAMILY_TO_WEIGHTS.get(family.replace("-obb", ""))
        family = family.replace("-obb", "")
        default_weight_name = fallback_name
        print(f"[MODEL] Falling back to weights: {default_weight_name}")

    # Final resolved weight path
    dest_path = yolo_weights_path.parent / default_weight_name

    weight_urls = {
        "yolov8n.pt":     "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
        "yolov8n-obb.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-obb.pt",
        "yolo11n.pt":     "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
        "yolo11n-obb.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-obb.pt",
        "yolo12n.pt":     "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12n.pt",
    }

    if default_weight_name not in weight_urls:
        print(f"[ERROR] Could not determine weight URL for '{default_weight_name}'")
        return None

    print(f"[DOWNLOAD] Weights not found, downloading '{family}' → {dest_path}")
    return download_file(weight_urls[default_weight_name], dest_path)

# ---- Image Counting ----
def count_images(folder: Path) -> int:
    if not folder.exists(): return 0
    exts = {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}
    return sum(len(list(folder.glob(f"*{e}"))) for e in exts)

# ---- Metadata Loading ----
def load_latest_metadata(logs_root: Path) -> Optional[dict]:
    """Return latest metadata.json from logs_root."""
    if not logs_root.exists(): return None
    latest, meta = 0, None
    for run in logs_root.iterdir():
        if not run.is_dir(): continue
        p = run / "metadata.json"
        if p.exists() and (mtime := p.stat().st_mtime) > latest:
            latest = mtime
            try: 
                meta = json.load(open(p, "r"))
            except Exception as e:
                print(f"[WARN] Failed to load metadata.json: {e}")
    return meta
