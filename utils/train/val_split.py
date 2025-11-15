# utils/train/val_split.py
import json
import random
import shutil
from pathlib import Path
from datetime import datetime
import yaml


# ---- Detect OBB vs HBB ----
def _detect_label_mode(lbl_folder: Path) -> str:
    """Return 'obb' if any label has >= 6 columns, else 'hbb'."""
    for lbl_file in lbl_folder.glob("*.txt"):
        try:
            with open(lbl_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    if len(parts) >= 6:
                        return "obb"
        except Exception:
            continue
    return "hbb"


# ---- Main LS → YOLO dataset processor ----
def process_labelstudio_project(project_folder: Path, data_root: Path, train_pct: float = 0.8, dataset_name: str | None = None):
    project_folder = Path(project_folder).resolve()
    data_root = Path(data_root).resolve()

    if not project_folder.exists():
        raise FileNotFoundError(f"Project folder not found: {project_folder}")

    # ---- Check for previously processed dataset ----
    for existing in data_root.iterdir():
        if not existing.is_dir():
            continue
        meta_path = existing / "metadata.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.load(open(meta_path, "r"))
            if meta.get("processed") and Path(meta.get("original_project")).resolve() == project_folder:
                data_yaml = existing / "data.yaml"
                if data_yaml.exists():
                    print(f"[INFO] Found existing processed dataset: {existing}")
                    return existing, data_yaml
        except Exception:
            pass

    # ---- Validate LS folder structure ----
    img_folder = project_folder / "images"
    lbl_folder = project_folder / "labels"
    classes_file = project_folder / "classes.txt"

    if not img_folder.is_dir() or not lbl_folder.is_dir() or not classes_file.exists():
        raise FileNotFoundError(
            f"Label Studio project must contain images/, labels/, and classes.txt: {project_folder}"
        )

    # ---- Detect label mode ----
    label_mode = _detect_label_mode(lbl_folder)
    print(f"[DATA] Detected label mode: {label_mode.upper()}")

    # ---- Determine output dataset folder ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if dataset_name:
        # Use user or timestamp-chosen dataset name
        base = dataset_name
    else:
        base = project_folder.name

    dataset_folder = data_root / base

    # Safe increment if exists
    if dataset_folder.exists():
        i = 1
        while (data_root / f"{base}{i}").exists():
            i += 1
        dataset_folder = data_root / f"{base}{i}"

    dataset_folder.mkdir(parents=True, exist_ok=True)

    # ---- Create train/val folder structure ----
    train_img = dataset_folder / "train/images"
    train_lbl = dataset_folder / "train/labels"
    val_img = dataset_folder / "val/images"
    val_lbl = dataset_folder / "val/labels"
    for p in (train_img, train_lbl, val_img, val_lbl):
        p.mkdir(parents=True, exist_ok=True)

    # ---- Gather and split images ----
    all_imgs = list(img_folder.glob("*"))
    if not all_imgs:
        raise RuntimeError(f"No images found in: {img_folder}")

    random.shuffle(all_imgs)
    split_idx = int(len(all_imgs) * train_pct)
    train_imgs = all_imgs[:split_idx]
    val_imgs = all_imgs[split_idx:]

    # ---- Copy image/label pairs ----
    def _copy_pairs(img_list, out_img_dir, out_lbl_dir):
        for img_path in img_list:
            shutil.copy2(img_path, out_img_dir / img_path.name)

            lbl_src = lbl_folder / f"{img_path.stem}.txt"
            if lbl_src.exists():
                shutil.copy2(lbl_src, out_lbl_dir / lbl_src.name)

    _copy_pairs(train_imgs, train_img, train_lbl)
    _copy_pairs(val_imgs, val_img, val_lbl)

    # ---- Create data.yaml ----
    with open(classes_file, "r") as f:
        names = [x.strip() for x in f.readlines() if x.strip()]

    if not names:
        raise RuntimeError(f"No class names found in: {classes_file}")

    data_yaml = dataset_folder / "data.yaml"
    yaml.dump(
        {
            "path": str(dataset_folder.resolve()),
            "train": str(train_img.resolve()),
            "val": str(val_img.resolve()),
            "nc": len(names),
            "names": names,
        },
        open(data_yaml, "w"),
        sort_keys=False,
    )

    # ---- Save metadata ----
    metadata = {
        "processed": True,
        "original_project": str(project_folder),
        "timestamp": timestamp,
        "label_mode": label_mode,
        "project_name": project_folder.name,
        "source_type": "labelstudio",
    }

    with open(dataset_folder / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"[INFO] Label Studio project processed: {dataset_folder}")
    return dataset_folder, data_yaml
