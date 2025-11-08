from pathlib import Path
from datetime import datetime
import re

BASE_DIR = Path(__file__).resolve().parents[2]

def find_latest_best(base_path):
    base_path = Path(base_path)
    if not base_path.exists():
        return None
    dirs = [d for d in base_path.iterdir() if d.is_dir()]
    if not dirs:
        return None

    def parse_ts(name):
        try:
            return datetime.strptime(name, "%m-%d-%Y_%H-%M-%S")
        except Exception:
            return datetime.min

    latest = max(dirs, key=lambda d: parse_ts(d.name))
    pt = latest / "weights" / "best.pt"
    return pt if pt.exists() else None


def get_output_folder(weights_path, source_type, source_name, test_detect=False):
    """
    Returns a tuple of (video_folder, scores_folder) based on source_type.
    Each source gets its own timestamped folder to prevent overwriting.
    """
    train_folder = weights_path.parent.parent
    model_timestamp = train_folder.name
    logs_root = BASE_DIR / ("logs/test" if test_detect else "logs/main") / model_timestamp / "measurements"

    source_ts = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")  # timestamp for the source folder

    if source_type == "video":
        safe_name = re.sub(r'[^\w\-\.]', '_', Path(source_name).stem)
        video_folder = logs_root / "video-in" / f"{safe_name}_{source_ts}" / "recordings"
        scores_folder = logs_root / "video-in" / f"{safe_name}_{source_ts}" / "scores"
    else:  # USB / camera
        safe_name = re.sub(r'[^\w\-\.]', '_', source_name)
        video_folder = logs_root / f"{safe_name}_{source_ts}" / "recordings"
        scores_folder = logs_root / f"{safe_name}_{source_ts}" / "scores"

    video_folder.mkdir(parents=True, exist_ok=True)
    scores_folder.mkdir(parents=True, exist_ok=True)
    return video_folder, scores_folder
