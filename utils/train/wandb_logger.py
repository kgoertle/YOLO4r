# utils/train/wandb_logger.py
import wandb, warnings
from pathlib import Path

from ..console import (
    fmt_exit, fmt_info, fmt_model,
    fmt_warn, fmt_error, fmt_dataset, fmt_path
)

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

    print(fmt_info(f"Logging enabled for run: {fmt_path(run_name)}"))
    print(fmt_info(f"W&B directory: {fmt_path(wandb_dir)}"))

    return run
