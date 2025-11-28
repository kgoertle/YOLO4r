import sys, time, shutil, wandb, yaml, logging, os
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
from ultralytics.utils import LOGGER

# ------ UTILITIES ------
from .utils.train import (
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
    init_wandb,
)

# ------ CONSOLE UI ------
from .utils.console import (
    fmt_info, fmt_warn, fmt_error, fmt_model, fmt_dataset, fmt_train, fmt_exit, fmt_path,
    clear_terminal, print_training_header, print_training_footer,
    quiet_ultralytics_logs, quiet_wandb_logs,
    apply_ultralytics_patch, remove_ultralytics_patch, print_training_header_static
)

# ------------- TRAINING FUNCTION -------------
def train_yolo(args, mode="train", checkpoint=None, resume_flag=False):
    """Orchestrates YOLO model training based on mode and arguments."""

    # ------------- VALIDATE DATASET YAML -------------
    if not args.DATA_YAML.exists():
        print(fmt_error(f"DATA_YAML not found: {args.DATA_YAML}"))
        return

    # ------------- PATH SETUP -------------
    paths = get_training_paths(args.DATA_YAML.parent, test=args.test)

    # ------------- TRAINING PARAMETER SETUP -------------
    reset_weights = (mode == "scratch")
    epochs, imgsz = (10, 640) if args.test else (120, 640)
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
            print(fmt_exit("No new images detected. Skipping training."))
            return

        print(fmt_info(f"{new_imgs} new images detected. Proceeding with update."))

    # ------------- MODEL SOURCE SELECTION -------------
    custom_arch_supplied = getattr(args, "custom_arch", False)

    # Default flags
    use_pretrained = False
    model_source = None

    if mode == "scratch":
        # Always use pure YAML, no pretrained weights
        model_source = str(args.model_yaml)
        use_pretrained = False
        checkpoint = None

    else:
        # Prefer checkpoint if available (resume/update)
        if checkpoint:
            model_source = str(Path(checkpoint))
            use_pretrained = True

        # Otherwise, if we have resolved weights from get_args(), use those
        elif getattr(args, "weights", None):
            model_source = str(args.weights)
            use_pretrained = True

        # Final fallback: YAML only (Ultralytics decides weights)
        else:
            model_source = str(args.model_yaml)
            use_pretrained = True  # this just tells Ultralytics to use its default n-scale

    if model_source is None:
        print(fmt_error("Could not resolve a model source (weights or architecture)."))
        return

    # ------------- DEVICE + RUN NAME -------------
    device, batch_size, workers = select_device()
    timestamp = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    run_name = args.name or timestamp

    print(fmt_model(f"Saved as: {fmt_path(run_name)}"))

    # ------------- CONSOLE & LOGGER SETUP -------------
    quiet_ultralytics_logs()
    quiet_wandb_logs()

    print_training_header()

    # --- W&B init ---
    try:
        init_wandb(run_name)
    except Exception as e:
        print(fmt_warn(f"Failed to initialize W&B: {e}"))

    print_training_footer(
        model_source=model_source,
        dataset_name=args.dataset_folder.name,
        batch_size=batch_size,
        workers=workers,
        epochs=epochs,
    )

    # ULTRALYTICS MODEL INIT
    model = YOLO(model_source, task="detect")

    # Patch Ultralytics logger for clean epoch table output
    apply_ultralytics_patch()

    start_time = time.time()
    skip_completion = False

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
            show=False,
            show_labels=True,
            show_conf=True,
        )

    except KeyboardInterrupt:
        patched_info = LOGGER.info
        patched_info.interrupted = True
        remove_ultralytics_patch()
        skip_completion = True
        print_training_header_static()
        print(fmt_exit("Training interrupted by user. Partial results preserved."))

    except Exception as e:
        patched_info = LOGGER.info
        patched_info.interrupted = True
        remove_ultralytics_patch()
        print_training_header()
        print(fmt_error(f"Training failed: {e}"))
        return

    # ------------- AFTER TRAINING -------------
    elapsed = (time.time() - start_time) / 60

    remove_ultralytics_patch()

    if not skip_completion:
        print_training_header_static()
        print(fmt_exit(f"Training completed in {elapsed:.2f} minutes."))

    # ------------- RUN DIRECTORY RESOLUTION -------------
    try:
        run_folder = Path(model.trainer.save_dir)
        run_name = run_folder.name
        log_dir = paths["logs_root"] / run_name
        log_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return

    # ------------- METRICS + METADATA -------------
    try:
        metrics = parse_results(run_folder) or {}

        save_quick_summary(
            log_dir=log_dir,
            mode=mode,
            epochs=epochs,
            metrics=metrics,
            new_imgs=new_imgs,
            total_imgs=total_imgs,
            weights_used=args.weights.name if args.weights else "n/a",
            arch_used=args.model_yaml.name if args.model_yaml else "n/a",
        )

        save_metadata(log_dir, mode, epochs, new_imgs, total_imgs)

    except Exception as e:
        print(fmt_warn(f"Failed to save metadata JSON: {e}"))

    # ------------- COPY DATA.YAML INTO RUN FOLDER -------------
    try:
        dst_yaml = run_folder / "data.yaml"

        if not dst_yaml.exists():
            shutil.copy(args.DATA_YAML, dst_yaml)
            print(fmt_exit(f"Copied dataset YAML to: {fmt_path(dst_yaml)}"))

    except Exception as e:
        print(fmt_warn(f"Could not copy dataset YAML: {e}"))

    # ------------- W&B SHUTDOWN -------------
    try:
        if wandb.run:
            wandb.finish()
    except Exception as e:
        print(fmt_warn(f"Could not close W&B cleanly: {e}"))


# ------------- MAIN ENTRY -------------
def main():
    args, mode = get_args()

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
            print(fmt_model(f"Updating model from: {fmt_path(checkpoint)}"))
        elif mode == "train":
            print(fmt_model(f"Training model from transferred weights: {args.weights}"))
        elif mode == "scratch":
            print(fmt_model(f"Training from scratch using architecture: {args.model_yaml}"))
        if resume_flag and checkpoint:
            print(fmt_model(f"Resuming model from: {fmt_path(checkpoint)}"))

    except FileNotFoundError as e:
        print(fmt_error(str(e)))
        sys.exit(1)

    train_yolo(args, mode=mode, checkpoint=checkpoint, resume_flag=resume_flag)


if __name__ == "__main__":
    main()
