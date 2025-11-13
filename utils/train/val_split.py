import json
import random
import shutil
from pathlib import Path
from datetime import datetime
import yaml

def process_labelstudio_project(project_folder: Path, data_root: Path, train_pct: float = 0.8):
    project_folder = Path(project_folder).resolve()
    data_root = Path(data_root).resolve()

    if not project_folder.exists():
        raise FileNotFoundError(f"Project folder not found: {project_folder}")

    # ---- Check if this project was already processed ----
    for existing_dataset in data_root.iterdir():
        if not existing_dataset.is_dir():
            continue
        metadata_file = existing_dataset / "metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, "r") as f:
                    meta = json.load(f)
                if meta.get("processed") and Path(meta.get("original_project")).resolve() == project_folder:
                    print(f"[INFO] Found existing processed dataset for project: {existing_dataset}")
                    data_yaml = existing_dataset / "data.yaml"
                    if not data_yaml.exists():
                        raise FileNotFoundError(f"Existing dataset missing data.yaml: {data_yaml}")
                    return existing_dataset, data_yaml
            except Exception as e:
                print(f"[WARN] Could not read metadata.json in {existing_dataset}: {e}")

    # ---- Determine output folder for new split ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_folder = data_root / f"dataset_{timestamp}"
    dataset_folder.mkdir(parents=True, exist_ok=True)

    # ---- Create train/val folder structure ----
    train_img = dataset_folder / "train/images"
    train_lbl = dataset_folder / "train/labels"
    val_img = dataset_folder / "val/images"
    val_lbl = dataset_folder / "val/labels"
    for p in [train_img, train_lbl, val_img, val_lbl]:
        p.mkdir(parents=True, exist_ok=True)

    # ---- Split images/labels ----
    img_folder = project_folder / "images"
    lbl_folder = project_folder / "labels"

    all_imgs = list(img_folder.glob("*"))
    random.shuffle(all_imgs)
    split_idx = int(len(all_imgs) * train_pct)
    train_imgs, val_imgs = all_imgs[:split_idx], all_imgs[split_idx:]

    for img_path in train_imgs:
        lbl_path = lbl_folder / f"{img_path.stem}.txt"
        shutil.copy(img_path, train_img / img_path.name)
        if lbl_path.exists():
            shutil.copy(lbl_path, train_lbl / lbl_path.name)

    for img_path in val_imgs:
        lbl_path = lbl_folder / f"{img_path.stem}.txt"
        shutil.copy(img_path, val_img / img_path.name)
        if lbl_path.exists():
            shutil.copy(lbl_path, val_lbl / lbl_path.name)

    # ---- Create data.yaml ----
    classes_file = project_folder / "classes.txt"
    if not classes_file.exists():
        raise FileNotFoundError(f"Missing classes.txt in project folder: {project_folder}")
    with open(classes_file, "r") as f:
        names = [line.strip() for line in f.readlines()]
    nc = len(names)

    data_yaml = dataset_folder / "data.yaml"
    yaml_dict = {
        "path": str(dataset_folder.resolve()),
        "train": str(train_img.resolve()),
        "val": str(val_img.resolve()),
        "nc": nc,
        "names": names
    }
    with open(data_yaml, "w") as f:
        yaml.dump(yaml_dict, f, sort_keys=False)

    # ---- Save metadata ----
    meta = {
        "processed": True,
        "original_project": str(project_folder),
        "timestamp": timestamp
    }
    metadata_file = dataset_folder / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(meta, f, indent=4)

    print(f"[INFO] Label Studio project processed: {dataset_folder}")
    return dataset_folder, data_yaml
