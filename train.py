import sys
import time
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

# ------ UTILITIES ------
from utils.train import (
    get_args,
    get_training_paths,
    ensure_weights,
    count_images,
    load_latest_metadata,
    get_checkpoint_and_resume,
    select_device,
    parse_results,
    save_quick_summary,
    save_metadata,
    process_labelstudio_project,
    init_wandb
)

# ------ TRAINING FUNCTION ------
def train_yolo(args, mode="train", checkpoint=None, resume_flag=False):
    """Orchestrates YOLO model training based on mode and arguments."""

    # ---- Validate dataset YAML ----
    if not args.DATA_YAML.exists():
        print(f"[ERROR] DATA_YAML not found: {args.DATA_YAML}")
        return

    reset_weights = mode == "scratch"
    epochs, imgsz = (10, 640) if args.test else (120, 640)
    if reset_weights and not args.test:
        epochs = 150

    total_imgs = count_images(args.train_folder) + count_images(args.val_folder)
    new_imgs = 0

    # ---- Update mode image check ----
    if mode == "update":
        logs_root = get_training_paths(args.DATA_YAML.parent, test=args.test)["logs_root"]
        prev_meta = load_latest_metadata(logs_root)
        prev_total = prev_meta.get("total_images_used", 0) if prev_meta else 0
        new_imgs = total_imgs - prev_total
        if new_imgs <= 0:
            print("[INFO] No new images detected. Skipping training.")
            return
        print(f"[INFO] {new_imgs} new images detected. Proceeding with update.")

    # ---- Model selection (scratch / transfer / update) ----
    if reset_weights:
        model_source = str(args.model_yaml)
        use_pretrained = False
    else:
        # Ensure checkpoint is a Path object if present
        if checkpoint:
            checkpoint = Path(checkpoint)
            model_source = str(checkpoint)
        else:
            # args.weights is already a Path from get_args()
            model_source = str(ensure_weights(args.weights))
        use_pretrained = True

    # ---- Device and batch settings ----
    device, batch_size, workers = select_device()
    timestamp = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    run_name = args.name or timestamp

    paths = get_training_paths(args.DATA_YAML.parent, test=args.test)
    run_folder = paths["runs_root"] / run_name
    log_dir = paths["logs_root"] / run_name

    print(f"[INFO] Initializing model from {'scratch' if not use_pretrained else 'weights'}: {model_source}")
    model = YOLO(model_source)

    try:
        init_wandb(run_name)
    except Exception as e:
        print(f"[WARN] Failed to initialize W&B: {e}")

    print(f"[INFO] Starting training mode: {mode}")
    start_time = time.time()

    try:
        model.train(
            data=str(args.DATA_YAML),
            model=model_source,
            epochs=epochs,
            resume=resume_flag,
            patience=10,
            imgsz=imgsz,
            batch=batch_size,
            workers=workers,
            project=str(paths["runs_root"]),
            name=run_name,
            exist_ok=False,
            pretrained=use_pretrained,
            device=device,
            augment=True,
            mosaic=True,
            mixup=True,
            fliplr=0.5,
            flipud=0.0,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=0.0,
            translate=0.1,
            plots=True,
            verbose=False,
            show=True,
            show_labels=True,
            show_conf=True
        )
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user. Partial results preserved.")
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        return

    elapsed = (time.time() - start_time) / 60
    print(f"[INFO] Training completed in {elapsed:.2f} minutes.")

    try:
        metrics = parse_results(run_folder) or {}
        save_quick_summary(log_dir, mode, epochs, metrics, new_imgs, total_imgs)
        save_metadata(log_dir, mode, epochs, new_imgs, total_imgs)
    except Exception as e:
        print(f"[ERROR] Failed to save post-training metadata: {e}")

    # ---- Copy data.yaml into model run folder ----
    try:
        run_weights_folder = run_folder / "weights"
        run_weights_folder.mkdir(parents=True, exist_ok=True)
        dst_yaml = run_folder / "data.yaml"
        if not dst_yaml.exists():
            shutil.copy(args.DATA_YAML, dst_yaml)
            print(f"[INFO] Copied dataset YAML to model folder: {dst_yaml}")
    except Exception as e:
        print(f"[WARN] Could not copy data.yaml to model folder: {e}")

# ------ MAIN ENTRY ------
def main():
    args, mode = get_args()

    checkpoint, resume_flag = None, args.resume
    try:
        checkpoint, resume_flag = get_checkpoint_and_resume(
            mode=mode,
            resume_flag=args.resume,
            runs_dir=get_training_paths(args.DATA_YAML.parent, test=args.test)["runs_root"],
            default_weights=args.weights,
            custom_weights=args.weights
        )
        if checkpoint:
            print(f"[INFO] Using checkpoint: {checkpoint}")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    train_yolo(args, mode=mode, checkpoint=checkpoint, resume_flag=resume_flag)


if __name__ == "__main__":
    main()
