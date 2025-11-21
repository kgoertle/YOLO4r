# utils/train/checkpoints.py
from pathlib import Path
from .io import ensure_weights

def check_checkpoint(runs_dir: Path, prefer_last=True):
    if not runs_dir.exists():
        return None

    # Sort subfolders by modified time (newest first)
    subfolders = sorted(
        [f for f in runs_dir.iterdir() if f.is_dir()],
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )

    for folder in subfolders:
        weights_dir = folder / "weights"
        if weights_dir.exists():
            filename = "last.pt" if prefer_last else "best.pt"
            candidate = weights_dir / filename
            if candidate.exists():
                return candidate

    return None

def get_checkpoint_and_resume(mode, resume_flag, runs_dir: Path,
                              default_weights=None, custom_weights=None,
                              update_folder=None):
    checkpoint = None

    # ---- Resume from last.pt ----
    if resume_flag:
        checkpoint = check_checkpoint(runs_dir, prefer_last=True)
        if not checkpoint:
            raise FileNotFoundError(f"No last.pt found for resuming in {runs_dir}")
        resume_flag = True

    # ---- Update mode: use best.pt ----
    elif mode == "update":
        if update_folder and isinstance(update_folder, str):
            target = runs_dir / update_folder / "weights" / "best.pt"
            if target.exists():
                checkpoint = target
            else:
                raise FileNotFoundError(f"[ERROR] best.pt not found in runs/{update_folder}/weights/")
        else:
            checkpoint = check_checkpoint(runs_dir, prefer_last=False)

        if not checkpoint:
            print(f"[WARN] No best.pt found. Falling back to default weights.")
            if default_weights:
                checkpoint = ensure_weights(
                    Path(default_weights),
                    model_type=str(default_weights)
                )

    # ---- Transfer-learning custom weights ----
    elif mode == "train" and custom_weights:
        checkpoint = ensure_weights(
            Path(custom_weights),
            model_type=str(custom_weights)
        )

    # ---- Fallback: default weights ----
    if checkpoint is None and default_weights:
        checkpoint = ensure_weights(
            Path(default_weights),
            model_type=str(default_weights)
        )

    return checkpoint, resume_flag
