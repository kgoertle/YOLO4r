# utils/detect/classes_config.py
import yaml
from pathlib import Path
from ultralytics import YOLO

from ..paths import CONFIGS_DIR, get_model_config_dir

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

# ---------- SAVE CONFIG ----------
def _save_model_config(printer=None):
    global CURRENT_MODEL_CONFIG
    if not CURRENT_MODEL_CONFIG:
        if printer: 
            printer.error("Cannot save config — no model assigned.")
        return

    CURRENT_MODEL_CONFIG.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "FOCUS_CLASSES": FOCUS_CLASSES,
        "CONTEXT_CLASSES": CONTEXT_CLASSES,
    }

    with open(CURRENT_MODEL_CONFIG, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    if printer:
        printer.save(f"Class configuration saved to {CURRENT_MODEL_CONFIG}")

# ---------- RELOAD EXISTING MODEL CONFIG ----------
def _reload_model_config(printer=None):
    if not CURRENT_MODEL_CONFIG or not CURRENT_MODEL_CONFIG.exists():
        return False

    try:
        with open(CURRENT_MODEL_CONFIG, "r") as f:
            saved = yaml.safe_load(f)

        if not isinstance(saved, dict):
            raise ValueError("Invalid format for mapping classes.")

        _set_focus_classes(saved.get("FOCUS_CLASSES", []))
        _set_context_classes(saved.get("CONTEXT_CLASSES", []))
        return True

    except Exception as e:
        # ---- CLEAN ERROR MESSAGE ----
        if printer:
            printer.error(
                f"Class config YAML is corrupted for model '{CURRENT_MODEL_NAME}'."
            )
        else:
            if printer:
                printer.error(f"Could not load class config YAML: {e}")
                printer.error(f"Path: {CURRENT_MODEL_CONFIG}")

        return False

# ---------- PUBLIC ENTRYPOINT ----------
def initialize_classes(model_name, force_reload=False, printer=None, weights_path=None):
    global CURRENT_MODEL_NAME, CURRENT_MODEL_CONFIG

    CURRENT_MODEL_NAME = str(model_name)
    CURRENT_MODEL_CONFIG = _model_config_path(CURRENT_MODEL_NAME)

    # Helper for combined class list
    def all_classes():
        return FOCUS_CLASSES + CONTEXT_CLASSES

    # Determine model-display-name for messages
    p = Path(weights_path)
    if "runs" in p.parts:
        model_display = model_name
    else:
        model_display = p.stem
        if model_display[:-1].endswith(tuple("0123456789")) is False:
            model_display = model_display[:-1]

    # ----- Load existing YAML -----
    if not force_reload and CURRENT_MODEL_CONFIG.exists():
        if _reload_model_config(printer):
            classes_loaded = all_classes()
            if printer:
                printer.info(
                    f"Loaded {len(classes_loaded)} classes: {classes_loaded}"
                )
            return

        if printer:
            printer.warn("Class config YAML is invalid or unreadable. Regenerating...")

    try:
        mdl = YOLO(str(weights_path))
        names_dict = mdl.names or {}
        detected_classes = [names_dict[i] for i in sorted(names_dict.keys())]
    except Exception as e:
        if printer:
            printer.error(f"Failed to load classes: {e}")
        detected_classes = []

    # Assign detected classes
    _set_focus_classes(detected_classes)
    _set_context_classes([])

    # Save YAML
    CURRENT_MODEL_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    with open(CURRENT_MODEL_CONFIG, "w") as f:
        yaml.safe_dump({
            "FOCUS_CLASSES": FOCUS_CLASSES,
            "CONTEXT_CLASSES": CONTEXT_CLASSES
        }, f, sort_keys=False)

    # Clean saved path for message: configs/<model>/classes_config.yaml
    short_path = f"configs/{model_name}/classes_config.yaml"
    if printer:
        printer.info(f"Class configuration saved to: {short_path}")
        printer.info(
            f"Generated class config YAML with {len(detected_classes)} classes: {detected_classes}"
        )
