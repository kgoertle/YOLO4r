# utils/train/wandb_logger.py
import wandb, warnings
from pathlib import Path
from ..paths import WANDB_ROOT

def init_wandb(run_name: str, project: str = "yolo-train", entity: str = "trevelline-lab"):
    """Initialize W&B tracking for a given run name, forcing logs into ~/.yolo4r/logs/wandb."""
    warnings.filterwarnings("ignore", category=UserWarning, message=".*reinit.*")

    wandb_dir = WANDB_ROOT
    run = wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        dir=str(wandb_dir),
        reinit=True
    )

    print(f"[INFO] W&B logging enabled for run: {run_name}")
    print(f"[INFO] W&B directory: {wandb_dir}")

    return run
