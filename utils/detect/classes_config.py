import yaml
from pathlib import Path

# ---------- GLOBALS ----------
BASE_DIR = Path(__file__).resolve().parent
DATA_YAML_PATH = BASE_DIR.parent.parent / "models" / "data.yaml"

# Create configs folder if missing
CONFIG_DIR = BASE_DIR.parent.parent / "configs"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_PATH = CONFIG_DIR / "classes_config.yaml"

# ---------- DEFAULT STATE ----------
FOCUS_CLASSES = []
CONTEXT_CLASSES = []


# ---------- CORE FUNCTIONS ----------
def load_data_yaml(data_yaml_path=DATA_YAML_PATH):
    if not data_yaml_path.exists():
        print(f"[WARN] data.yaml not found at {data_yaml_path}")
        return []

    try:
        with open(data_yaml_path, "r") as f:
            data = yaml.safe_load(f)

        if "names" in data:
            names = data["names"]
            if isinstance(names, dict):  # format: {0: "class", 1: "class"}
                return list(names.values())
            elif isinstance(names, list):  # format: ["class", "class"]
                return names

        print("[WARN] No valid 'names' field found in data.yaml.")
        return []

    except Exception as e:
        print(f"[ERROR] Failed to parse data.yaml: {e}")
        return []


def initialize_classes(force_reload=False, data_yaml_path=None):
    global FOCUS_CLASSES, CONTEXT_CLASSES

    # Load existing config if available
    if CONFIG_PATH.exists() and not force_reload:
        with open(CONFIG_PATH, "r") as f:
            saved = yaml.safe_load(f) or {}
        FOCUS_CLASSES = saved.get("FOCUS_CLASSES", [])
        CONTEXT_CLASSES = saved.get("CONTEXT_CLASSES", [])
        if FOCUS_CLASSES and not force_reload:
            return  # Already initialized

    # Otherwise, rebuild from data.yaml
    detected_classes = load_data_yaml(data_yaml_path=data_yaml_path or DATA_YAML_PATH)
    if not detected_classes:
        FOCUS_CLASSES = []
        CONTEXT_CLASSES = []
        print("[WARN] Could not initialize FOCUS_CLASSES (no classes found).")
        return

    FOCUS_CLASSES = detected_classes.copy()
    CONTEXT_CLASSES = []

    save_config()
    print(f"[INIT] FOCUS_CLASSES initialized with {len(FOCUS_CLASSES)} classes from {data_yaml_path or DATA_YAML_PATH}.")

def save_config():
    data = {
        "FOCUS_CLASSES": FOCUS_CLASSES,
        "CONTEXT_CLASSES": CONTEXT_CLASSES,
    }

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)  # ensure folder exists
    with open(CONFIG_PATH, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    print(f"[SAVE] Class configuration saved to {CONFIG_PATH}.")

def reload_config():
    global FOCUS_CLASSES, CONTEXT_CLASSES
    if not CONFIG_PATH.exists():
        print("[WARN] No saved classes_config.yaml found.")
        return

    with open(CONFIG_PATH, "r") as f:
        data = yaml.safe_load(f) or {}

    FOCUS_CLASSES = data.get("FOCUS_CLASSES", [])
    CONTEXT_CLASSES = data.get("CONTEXT_CLASSES", [])


def reset_to_data_yaml():
    initialize_classes(force_reload=True)


# ---------- Initialize on import ----------
initialize_classes()
