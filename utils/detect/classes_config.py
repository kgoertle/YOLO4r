import yaml
from pathlib import Path

# ---------- GLOBALS ----------
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_YAML_PATH = BASE_DIR.parent.parent / "models" / "data.yaml"

# Create configs folder if missing
CONFIG_DIR = BASE_DIR.parent.parent / "configs"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_PATH = CONFIG_DIR / "classes_config.yaml"

# ---------- DEFAULT STATE ----------
FOCUS_CLASSES = []
CONTEXT_CLASSES = []

# ---------- CORE FUNCTIONS ----------
def load_data_yaml(data_yaml_path: Path):
    """Load class names from a data.yaml file."""
    if not data_yaml_path or not data_yaml_path.exists():
        print(f"[WARN] data.yaml not found at {data_yaml_path}")
        return []

    try:
        with open(data_yaml_path, "r") as f:
            data = yaml.safe_load(f)

        names = data.get("names", [])
        if isinstance(names, dict):
            return list(names.values())
        elif isinstance(names, list):
            return names

        print("[WARN] No valid 'names' field found in data.yaml.")
        return []

    except Exception as e:
        print(f"[ERROR] Failed to parse data.yaml: {e}")
        return []

def initialize_classes(data_yaml_path: Path = None, force_reload=False):
    """
    Initialize FOCUS_CLASSES and CONTEXT_CLASSES from a specific data.yaml
    (explicit path), or fallback to default if not provided.
    """
    global FOCUS_CLASSES, CONTEXT_CLASSES

    if data_yaml_path is None:
        data_yaml_path = DEFAULT_DATA_YAML_PATH

    if CONFIG_PATH.exists() and not force_reload:
        with open(CONFIG_PATH, "r") as f:
            saved = yaml.safe_load(f) or {}
        FOCUS_CLASSES = saved.get("FOCUS_CLASSES", [])
        CONTEXT_CLASSES = saved.get("CONTEXT_CLASSES", [])
        if FOCUS_CLASSES and not force_reload:
            return

    detected_classes = load_data_yaml(data_yaml_path)
    if not detected_classes:
        FOCUS_CLASSES, CONTEXT_CLASSES = [], []
        print(f"[WARN] Could not initialize FOCUS_CLASSES (no classes found in {data_yaml_path}).")
        return

    FOCUS_CLASSES = detected_classes.copy()
    CONTEXT_CLASSES = []

    save_config()
    print(f"[INIT] FOCUS_CLASSES initialized with {len(FOCUS_CLASSES)} classes from {data_yaml_path}.")

def save_config():
    """Persist current class configuration to YAML."""
    data = {"FOCUS_CLASSES": FOCUS_CLASSES, "CONTEXT_CLASSES": CONTEXT_CLASSES}
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    print(f"[SAVE] Class configuration saved to {CONFIG_PATH}.")

def reload_config():
    """Reload class configuration from saved YAML."""
    global FOCUS_CLASSES, CONTEXT_CLASSES
    if not CONFIG_PATH.exists():
        print("[WARN] No saved classes_config.yaml found.")
        return
    with open(CONFIG_PATH, "r") as f:
        data = yaml.safe_load(f) or {}
    FOCUS_CLASSES = data.get("FOCUS_CLASSES", [])
    CONTEXT_CLASSES = data.get("CONTEXT_CLASSES", [])

def reset_to_data_yaml():
    """Force re-initialize classes from default YAML."""
    initialize_classes(force_reload=True)

# ---------- Initialize on import using default YAML ----------
initialize_classes()
