# utils/detect/classes_config.py
import yaml
from pathlib import Path

from utils.detect.paths import (
    DEFAULT_DATA_YAML,
    CONFIGS_DIR,
    get_latest_dataset_yaml,
    get_model_config_dir,
)

# ----- GLOBALS UPDATED IN-PLACE -----
FOCUS_CLASSES = []
CONTEXT_CLASSES = []

CURRENT_MODEL_NAME = None
CURRENT_MODEL_CONFIG = None  

# ---------- INTERNAL HELPERS ----------
def _set_focus_classes(new_list):
    global FOCUS_CLASSES
    FOCUS_CLASSES.clear()
    FOCUS_CLASSES.extend(new_list)

def _set_context_classes(new_list):
    global CONTEXT_CLASSES
    CONTEXT_CLASSES.clear()
    CONTEXT_CLASSES.extend(new_list)

def _model_config_path(model_name: str) -> Path:
    """Return /configs/<model>/classes_config.yaml"""
    model_dir = get_model_config_dir(model_name)
    return model_dir / "classes_config.yaml"

# ---------- LOAD CLASSES FROM data.yaml ----------
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
            normalized = {int(k): v for k, v in names.items() if str(k).isdigit()}
            return [normalized[k] for k in sorted(normalized.keys())]
        if isinstance(names, list):
            return names

        print(f"[WARN] Invalid 'names' field in {yaml_path}")
        return []

    except Exception as e:
        print(f"[ERROR] Failed to parse {yaml_path}: {e}")
        return []

# ---------- SAVE CONFIG ----------
def _save_model_config():
    global CURRENT_MODEL_CONFIG
    if not CURRENT_MODEL_CONFIG:
        print("[ERROR] Cannot save config — no model assigned.")
        return

    CURRENT_MODEL_CONFIG.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "FOCUS_CLASSES": FOCUS_CLASSES,
        "CONTEXT_CLASSES": CONTEXT_CLASSES,
    }

    with open(CURRENT_MODEL_CONFIG, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    print(f"[SAVE] Class configuration saved to {CURRENT_MODEL_CONFIG}")

# ---------- RELOAD EXISTING MODEL CONFIG ----------
def _reload_model_config(printer=None):
    """
    Reload the model-specific classes_config.yaml.
    If the YAML is corrupted or unreadable, show a clean error message,
    return False, and allow fallback to data.yaml.
    """

    if not CURRENT_MODEL_CONFIG or not CURRENT_MODEL_CONFIG.exists():
        return False

    try:
        with open(CURRENT_MODEL_CONFIG, "r") as f:
            saved = yaml.safe_load(f)

        if not isinstance(saved, dict):
            raise ValueError("Invalid format: expected mapping at root")

        _set_focus_classes(saved.get("FOCUS_CLASSES", []))
        _set_context_classes(saved.get("CONTEXT_CLASSES", []))
        return True

    except Exception as e:
        # ---- CLEAN ERROR MESSAGE ----
        if printer:
            printer.error(
                f"classes_config.yaml is corrupted for model '{CURRENT_MODEL_NAME}'."
            )
            printer.warn("Falling back to data.yaml and regenerating config...")
        else:
            print(f"[ERROR] Could not load classes_config.yaml: {e}")
            print(f"[ERROR] Path: {CURRENT_MODEL_CONFIG}")

        return False

# ---------- PUBLIC ENTRYPOINT ----------
def initialize_classes(model_name, data_yaml_path, force_reload=False, printer=None):
    global CURRENT_MODEL_NAME, CURRENT_MODEL_CONFIG
    CURRENT_MODEL_NAME = str(model_name)
    CURRENT_MODEL_CONFIG = _model_config_path(CURRENT_MODEL_NAME)

    if not force_reload and _reload_model_config(printer=printer):
        print(f"[INIT] Loaded {len(FOCUS_CLASSES)} classes from: {CURRENT_MODEL_CONFIG}")
        return
    detected = []
    if data_yaml_path and Path(data_yaml_path).exists():
        detected = load_data_yaml(data_yaml_path)
    if not detected:
        fallback_yaml = DEFAULT_DATA_YAML

        if printer:
            printer.warn("Using fallback dataset YAML…")

        latest_yaml = get_latest_dataset_yaml(printer)
        if latest_yaml:
            fallback_yaml = latest_yaml

        detected = load_data_yaml(fallback_yaml)

        if not detected:
            print(f"[WARN] Could not detect ANY classes. Using empty class list.")
            _set_focus_classes([])
            _set_context_classes([])
            return
    _set_focus_classes(detected)
    _set_context_classes([])

    _save_model_config()
