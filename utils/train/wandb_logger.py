# utils/train/wandb_logger.py
import wandb, warnings

def init_wandb(run_name: str, project: str = "yolo-train", entity: str = "trevelline-lab"):
    """Initialize W&B tracking for a given run name."""
    warnings.filterwarnings("ignore", category=UserWarning, message=".*reinit.*")
    run = wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        reinit=True
    )
    print(f"[INFO] W&B logging enabled for run: {run_name}")
    return run
