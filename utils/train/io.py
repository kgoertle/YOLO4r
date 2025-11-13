import requests
import json
from pathlib import Path
from typing import Optional

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
def ensure_yolo_yaml(yolo_yaml_path: Path, model_type: str = "yolo11-obb") -> Optional[Path]:
    """Ensure the YAML exists locally, else download it based on model type."""
    if yolo_yaml_path.exists():
        return yolo_yaml_path

    urls = {
        "yolo11-obb": "https://raw.githubusercontent.com/ultralytics/ultralytics/bc3414ba1c172817150a1c31fa68479678215c1f/ultralytics/cfg/models/11/yolo11-obb.yaml",
        "yolo11": "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/models/11/yolo11.yaml",
    }

    url = urls.get(model_type, urls["yolo11"])
    return download_file(url, yolo_yaml_path)


# ---- Ensure Weights ----
def ensure_weights(yolo_weights_path: Path, model_type: str = "yolo11n-obb") -> Optional[Path]:
    """Ensure the weights exist locally, else download based on model type."""
    if yolo_weights_path.exists():
        return yolo_weights_path

    urls = {
        "yolo11n-obb": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-obb.pt",
        "yolo11n": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
    }

    url = urls.get(model_type, urls["yolo11n"])
    return download_file(url, yolo_weights_path)

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
