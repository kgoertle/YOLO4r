# utils/detect/classes_config.py
import yaml
from pathlib import Path

# ---- PULL PATHS ----
from utils.detect.paths import DEFAULT_DATA_YAML, CLASSES_CONFIG_YAML, CONFIGS_DIR, get_latest_dataset_yaml

# ---- DEFAULT CLASS STATE ----
FOCUS_CLASSES = []
CONTEXT_CLASSES = []

# ---------- LOAD CLASSES FROM `data.yaml` FILE ----------
def load_data_yaml(yaml_path: Path):
    yaml_path = Path(yaml_path)

    if not yaml_path.exists():
        print(f"[WARN] data.yaml not found at {yaml_path}")
        return []

    try:
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f) or {}

        names = data.get("names")

        if isinstance(names, dict):
            # Convert keys to int → sort → then fetch by index
            normalized = {}
            for k, v in names.items():
                try:
                    ik = int(k)
                except Exception:
                    continue
                normalized[ik] = v

            ordered_keys = sorted(normalized.keys())
            return [normalized[k] for k in ordered_keys]

        if isinstance(names, list):
            # list format: ['M', 'F', ...]
            return names

        print(f"[WARN] Invalid 'names' field in {yaml_path}")
        return []

    except Exception as e:
        print(f"[ERROR] Failed to parse {yaml_path}: {e}")
        return []

# ---------- CLASS DEF HELPERS ----------
def _set_focus_classes(new_list):
    """Mutate FOCUS_CLASSES in-place so existing imports see updates."""
    global FOCUS_CLASSES
    FOCUS_CLASSES.clear()
    FOCUS_CLASSES.extend(new_list)


def _set_context_classes(new_list):
    """Mutate CONTEXT_CLASSES in-place so existing imports see updates."""
    global CONTEXT_CLASSES
    CONTEXT_CLASSES.clear()
    CONTEXT_CLASSES.extend(new_list)


# ---------- INITIALIZE CLASSES ----------
def initialize_classes(force_reload=False, data_yaml_path=None, printer=None):
    if data_yaml_path:
        detected = load_data_yaml(data_yaml_path)
        _set_focus_classes(detected)
  
    if CLASSES_CONFIG_YAML.exists() and not force_reload:
        with open(CLASSES_CONFIG_YAML, "r") as f:
            saved = yaml.safe_load(f) or {}

        _set_focus_classes(saved.get("FOCUS_CLASSES", []))
        _set_context_classes(saved.get("CONTEXT_CLASSES", []))

        if FOCUS_CLASSES:
            return

    if not CLASSES_CONFIG_YAML.exists():
        save_config()

    fallback_yaml = DEFAULT_DATA_YAML
    if printer:
        printer.warn("Using default model data.yaml…")

    latest_ds_yaml = get_latest_dataset_yaml(printer)
    if latest_ds_yaml:
        fallback_yaml = latest_ds_yaml

    detected = load_data_yaml(fallback_yaml)

    if not detected:
        print(f"[WARN] No classes found in fallback YAML: {fallback_yaml}")
        _set_focus_classes([])
        _set_context_classes([])
        return

    _set_focus_classes(detected)
    _set_context_classes([])

    print(f"[INIT] Initialized classes from: {fallback_yaml}")

# ---------- SAVE CLASS CONFIG ----------
def save_config():
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

    data = {
        "FOCUS_CLASSES": FOCUS_CLASSES,
        "CONTEXT_CLASSES": CONTEXT_CLASSES,
    }

    with open(CLASSES_CONFIG_YAML, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    print(f"[SAVE] Class configuration saved to {CLASSES_CONFIG_YAML}")

# ---------- RELOAD CONFIG ----------
def reload_config():
    if not CLASSES_CONFIG_YAML.exists():
        print("[WARN] No saved classes_config.yaml found.")
        return

    with open(CLASSES_CONFIG_YAML, "r") as f:
        saved = yaml.safe_load(f) or {}

    _set_focus_classes(saved.get("FOCUS_CLASSES", []))
    _set_context_classes(saved.get("CONTEXT_CLASSES", []))

# ---------- RESET FUNCTION ----------
def reset_to_data_yaml(printer=None):
    initialize_classes(force_reload=True, printer=printer)
