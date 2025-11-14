import yaml
from pathlib import Path

# ---------- GLOBALS ----------
BASE_DIR = Path(__file__).resolve().parent
DATA_YAML_PATH = BASE_DIR.parent.parent / "models" / "data.yaml"

CONFIG_DIR = BASE_DIR.parent.parent / "configs"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_PATH = CONFIG_DIR / "classes_config.yaml"

# ---------- DEFAULT STATE ----------
FOCUS_CLASSES = []
CONTEXT_CLASSES = []

# ---------- CORE FUNCTIONS ----------
def load_data_yaml(data_yaml_path):
    data_yaml_path = Path(data_yaml_path)

    if not data_yaml_path.exists():
        print(f"[WARN] data.yaml not found at {data_yaml_path}")
        return []

    try:
        with open(data_yaml_path, "r") as f:
            data = yaml.safe_load(f) or {}

        names = data.get("names")
        if isinstance(names, dict):
            return [names[k] for k in sorted(names.keys())]
        if isinstance(names, list):
            return names

        print(f"[WARN] No valid 'names' field in {data_yaml_path}")
        return []

    except Exception as e:
        print(f"[ERROR] Failed to parse {data_yaml_path}: {e}")
        return []

def initialize_classes(force_reload=False, data_yaml_path=None):
    global FOCUS_CLASSES, CONTEXT_CLASSES

    if data_yaml_path:
        detected_classes = load_data_yaml(Path(data_yaml_path))
        if not detected_classes:
            print(f"[WARN] Could not initialize FOCUS_CLASSES (no classes found in {data_yaml_path}).")
            FOCUS_CLASSES = []
            CONTEXT_CLASSES = []
            return

        FOCUS_CLASSES = detected_classes.copy()
        CONTEXT_CLASSES = []
        save_config()
        print(f"[INIT] Loaded {len(FOCUS_CLASSES)} classes from model data.yaml: {data_yaml_path}")
        return

    if CONFIG_PATH.exists() and not force_reload:
        with open(CONFIG_PATH, "r") as f:
            saved = yaml.safe_load(f) or {}
        FOCUS_CLASSES = saved.get("FOCUS_CLASSES", [])
        CONTEXT_CLASSES = saved.get("CONTEXT_CLASSES", [])
        if FOCUS_CLASSES:
            return

    # Full reset from global data.yaml
    detected_classes = load_data_yaml(DATA_YAML_PATH)
    if not detected_classes:
        print(f"[WARN] Could not initialize FOCUS_CLASSES (no classes found in {DATA_YAML_PATH}).")
        FOCUS_CLASSES = []
        CONTEXT_CLASSES = []
        return

    FOCUS_CLASSES = detected_classes.copy()
    CONTEXT_CLASSES = []
    save_config()
    print(f"[INIT] FOCUS_CLASSES initialized from default data.yaml ({DATA_YAML_PATH}).")

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
