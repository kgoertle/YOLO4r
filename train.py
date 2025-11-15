# train.py
import sys, time, shutil, wandb
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

# ------ UTILITIES ------
from utils.train import get_args, get_training_paths, ensure_weights, count_images, load_latest_metadata, get_checkpoint_and_resume, select_device, parse_results, save_quick_summary, save_metadata, init_wandb

# ------------- TRAINING FUNCTION -------------
def train_yolo(args, mode="train", checkpoint=None, resume_flag=False):
    """Orchestrates YOLO model training based on mode and arguments."""

    # ------------- VALIDATE DATASET YAML -------------
    if not args.DATA_YAML.exists():
        print(f"[ERROR] DATA_YAML not found: {args.DATA_YAML}")
        return

    reset_weights = (mode == "scratch")
    epochs, imgsz = (10, 640) if args.test else (3, 640)
    if reset_weights and not args.test:
        epochs = 150

    total_imgs = count_images(args.train_folder) + count_images(args.val_folder)
    new_imgs = 0

    # ------------- UPDATE MODE IMAGE CHECK -------------
    if mode == "update":
        paths = get_training_paths(args.DATA_YAML.parent, test=args.test)
        logs_root = paths["logs_root"] / args.dataset_folder.name

        prev_meta = load_latest_metadata(logs_root)
        prev_total = prev_meta.get("total_images_used", 0) if prev_meta else 0
        new_imgs = total_imgs - prev_total

        if new_imgs <= 0:
            print("[EXIT] No new images detected. Skipping training.")
            return

        print(f"[INFO] {new_imgs} new images detected. Proceeding with update.")

    # ------------- MODEL SOURCE SELECTION -------------
    if mode == "scratch":
        # Always use architecture only
        model_source = str(args.model_yaml)
        use_pretrained = False
        checkpoint = None

    else:
        # Normal transfer-learning / update logic
        if checkpoint:
            model_source = str(Path(checkpoint))
        else:
            # Weights were fully resolved in config.py
            model_source = str(ensure_weights(args.weights))
        use_pretrained = True

    # ------------- DEVICE + RUN NAME -------------
    device, batch_size, workers = select_device()
    timestamp = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    run_name = args.name or timestamp
    print(f"[MODEL] Model will be saved as: {run_name}")

    paths = get_training_paths(args.DATA_YAML.parent, test=args.test)

    # --- CLASS COUNT ENFORCEMENT FOR SCRATCH OBB TRAINING ---
    def enforce_nc_in_yaml(yaml_path, nc):
        """Ensure model yaml has correct nc for OBB models."""
        try:
            with open(yaml_path, "r") as f:
                cfg = yaml.safe_load(f)

            # Only enforce when YAML contains OBB head or nc mismatch
            if "obb" in yaml_path.name.lower() or cfg.get("nc", None) != nc:
                cfg["nc"] = nc
                tmp_path = yaml_path.parent / f"_tmp_{yaml_path.name}"
                with open(tmp_path, "w") as f:
                    yaml.safe_dump(cfg, f)
                return tmp_path

            return yaml_path

        except Exception as e:
            print(f"[WARN] Could not enforce nc in YAML: {e}")
            return yaml_path

    # ------------- MODEL INITIALIZATION -------------
    model = YOLO(model_source, task="detect")

    # ------------- W&B HANDLING -------------
    try:
        init_wandb(run_name)
    except Exception as e:
        print(f"[WARN] Failed to initialize W&B: {e}")

    start_time = time.time()

    # ------------- TRAINING CALL -------------
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
            augment=False,
            mosaic=False,
            mixup=True,
            fliplr=0.5,
            flipud=0.0,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=0.0,
            translate=0.1,
            plots=False,
            verbose=False,
            show=True,
            show_labels=True,
            show_conf=True,
        )
    except KeyboardInterrupt:
        print("\n[EXIT] Training interrupted by user. Partially completed results preserved.")
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        return

    elapsed = (time.time() - start_time) / 60
    print(f"[EXIT] Training completed in {elapsed:.2f} minutes.")

    # ------------- RUN DIRECTORY RESOLUTION -------------
    try:
        run_folder = Path(model.trainer.save_dir)
        run_name = run_folder.name
        log_dir = paths["logs_root"] / run_name
        log_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return

    # ------------- METRICS + METADATA SAVING -------------
    try:
        metrics = parse_results(run_folder) or {}

        weights_str = args.weights.name if args.weights else "n/a"
        arch_str = args.model_yaml.name if args.model_yaml else "n/a"

        save_quick_summary(
            log_dir=log_dir,
            mode=mode,
            epochs=epochs,
            metrics=metrics,
            new_imgs=new_imgs,
            total_imgs=total_imgs,
            weights_used=weights_str,
            arch_used=arch_str,
        )

        save_metadata(log_dir, mode, epochs, new_imgs, total_imgs)

    except Exception as e:
        print(f"[ERROR] Failed to save post-training metadata: {e}")

    # ------------- W&B SHUTDOWN -------------
    try:
        if wandb.run:
            wandb.finish()
            print("[EXIT] W&B run finalized cleanly.")
    except Exception as e:
        print(f"[WARN] Could not close W&B run cleanly: {e}")

    # ------------- COPY DATA.YAML INTO RUN FOLDER -------------
    try:
        run_weights_folder = run_folder / "weights"
        run_weights_folder.mkdir(parents=True, exist_ok=True)

        dst_yaml = run_folder / "data.yaml"
        if not dst_yaml.exists():
            shutil.copy(args.DATA_YAML, dst_yaml)
            print(f"[EXIT] Copied dataset YAML to model folder: {dst_yaml}")

    except Exception as e:
        print(f"[WARN] Could not copy data.yaml to model folder: {e}")


# ------------- MAIN ENTRY -------------
def main():
    args, mode = get_args()

    checkpoint, resume_flag = None, args.resume

    try:
        checkpoint, resume_flag = get_checkpoint_and_resume(
            mode=mode,
            resume_flag=args.resume,
            runs_dir=get_training_paths(args.DATA_YAML.parent, test=args.test)["runs_root"],
            default_weights=args.weights,
            custom_weights=args.weights,
            update_folder=args.update if isinstance(args.update, str) else None,
        )

        if mode == "update" and checkpoint:
            print(f"[MODEL] Updating model from: {checkpoint}")
        elif mode == "train":
            print(f"[MODEL] Training model from transfered weights: {args.weights}")
        elif mode == "scratch":
            print(f"[MODEL] Training model from scratch using model architecture: {args.model_yaml}")

        if resume_flag and checkpoint:
            print(f"[MODEL] Resuming model training from: {checkpoint}")

    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    train_yolo(args, mode=mode, checkpoint=checkpoint, resume_flag=resume_flag)


if __name__ == "__main__":
    main()
