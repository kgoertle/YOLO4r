# detect.py
import sys, os, threading, time, platform, queue, json, cv2
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
from ultralytics import YOLO

# ----------- UTILITIES ---------------
from .utils.paths import get_runs_dir, get_output_folder, WEIGHTS_DIR
from .utils.detect.arguments import parse_arguments
from .utils.detect.printer import Printer
from .utils.detect.measurements import MeasurementConfig, Counter, Interactions, Aggregator, compute_counts_from_boxes
from .utils.detect.classes_config import initialize_classes
from .utils.detect.video_util import VideoSourceInfo, extract_video_metadata, extract_camera_metadata, VideoReader, create_video_writer, write_annotated_frame, extract_boxes_from_results
from .utils.detect.inference_util import InferenceWorker
from .utils.train.io import ensure_weights

# ---- SYSTEM ----
stop_event = threading.Event()
IS_MAC = platform.system() == "Darwin"
IS_LINUX = platform.system() == "Linux"
IS_PI = IS_LINUX and ("arm" in platform.machine() or "aarch64" in platform.machine())

# ---- THREADING MANAGER ----
class VideoProcessor:
    def __init__(
        self,
        weights_path,
        source,
        source_type,
        idx,
        total_sources,
        printer,
        test=False,
    ):
        self.weights_path = Path(weights_path)
        self.source = source
        self.source_type = source_type
        self.idx = idx
        self.total_sources = total_sources
        self.printer = printer
        self.test = test

        self.is_camera = source_type == "usb"
        self.source_display_name = (
            Path(source).stem if not self.is_camera else f"usb{source}"
        )

        # ---------- THREADING QUEUES ----------
        self.frame_queue = queue.Queue(maxsize=50)
        self.infer_queue = queue.Queue(maxsize=20)

        # Components
        self.reader = None
        self.infer_worker = None

        # Model / IO
        self.model = None
        self.is_obb_model = False
        self.cap = None
        self.out_writer = None

        # Measurement objects
        self.config = MeasurementConfig()
        self.counter = None
        self.aggregator = None
        self.interactions = None

        # Timing / metadata
        self.start_time = None
        self.fps_video = None
        self.total_frames = None
        self.frame_width = None
        self.frame_height = None

        self.paths = None
        self.out_file = None
        self.metadata_file = None

    def get_model_name_clean(self):
        p = self.weights_path
        if "runs" in p.parts:
            try:
                idx = p.parts.index("runs")
                model_name = p.parts[idx + 1]
                return model_name
            except Exception:
                return p.stem
        return p.stem

    # ----------- Initialization -----------
    def initialize(self):
        # Load model
        try:
            self.model = YOLO(str(self.weights_path))
            self.model.weights_path = self.weights_path
        except Exception as e:
            self.printer.model_fail(e)
            return False

        # Detect OBB capability
        try:
            test_frame = np.zeros((640, 640, 3), dtype=np.uint8)
            res = self.model.predict(test_frame, verbose=False, show=False)
            self.is_obb_model = hasattr(res[0], "obb") and res[0].obb is not None
        except Exception:
            self.is_obb_model = False

        # ---------- Capture init FIRST ----------
        try:
            if self.is_camera:
                backend = cv2.CAP_AVFOUNDATION if IS_MAC else cv2.CAP_V4L2
                self.cap = cv2.VideoCapture(int(self.source), backend)
            else:
                self.cap = cv2.VideoCapture(str(self.source))

            if not self.cap.isOpened():
                self.printer.open_capture_fail(self.source_display_name)
                return False
        except Exception:
            self.printer.open_capture_fail(self.source_display_name)
            return False

        # ---------- Unified VideoSourceInfo handling ----------
        if not self.is_camera:
            meta_dict = extract_video_metadata(self.source)
            src_info = VideoSourceInfo(
                meta_dict, is_camera=False, display_name=self.source_display_name
            )
        else:
            # Camera metadata pulled from the opened capture
            try:
                source_id = int(self.source)
            except ValueError:
                source_id = 0
            meta_dict = extract_camera_metadata(self.cap, source_id)
            src_info = VideoSourceInfo(
                meta_dict, is_camera=True, display_name=self.source_display_name
            )

        # Parse creation time (returns datetime)
        self.start_time = src_info.parse_creation_time()
        metadata = src_info.metadata
        metadata["creation_time_str"] = self.start_time.strftime("%H:%M:%S")

        # ---------- Output paths ----------
        self.paths = get_output_folder(
            self.weights_path,
            self.source_type,
            self.source if not self.is_camera else f"usb{self.source}",
            test_detect=self.test,
            base_time=self.start_time,
        )

        self.out_file = self.paths["video_folder"] / f"{self.paths['safe_name']}.mp4"
        self.metadata_file = self.paths["metadata"]

        with open(self.metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        # ---------- Dimensions & FPS ----------
        if not self.is_camera:
            ret, frame0 = self.cap.read()
            if not ret or frame0 is None:
                self.printer.read_frame_fail(self.source_display_name)
                return False

            self.frame_height, self.frame_width = frame0.shape[:2]

            # Reset capture back to the beginning
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        else:
            # For cameras, the metadata is just based on cv2 props anyway.
            self.frame_width = src_info.width or int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
            self.frame_height = src_info.height or int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

        # FPS
        if src_info.fps:
            self.fps_video = src_info.fps
        else:
            src_fps = self.cap.get(cv2.CAP_PROP_FPS)
            if not src_fps or src_fps <= 0 or np.isnan(src_fps):
                src_fps = 30.0
            self.fps_video = src_fps

        # ---------- TOTAL FRAMES FOR VIDEO SOURCES ----------
        if not self.is_camera:
            total = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
            try:
                total = int(total)
            except Exception:
                total = None
            self.total_frames = total if total and total > 0 else None
        else:
            self.total_frames = None

        # ---- VIDEO WRITER ----
        self.out_writer = create_video_writer(
            self.out_file,
            self.fps_video,
            self.frame_width,
            self.frame_height,
            self.source_display_name,
            self.printer,
            self.cap,
            self.source_type,
        )

        if self.out_writer is None:
            return False

        # Measurement objects
        self.counter = Counter(
            out_folder=self.paths["counts"],
            config=self.config,
            start_time=self.start_time,
        )
        self.aggregator = Aggregator(
            out_folder=self.paths["frame-counts"],
            config=self.config,
            start_time=self.start_time,
        )
        self.interactions = Interactions(
            out_folder=self.paths["interactions"],
            config=self.config,
            start_time=self.start_time,
            is_obb=self.is_obb_model,
        )

        # --- Wiring: reader + inference worker ---
        self.reader = VideoReader(
            cap=self.cap,
            frame_queue=self.frame_queue,
            infer_queue=self.infer_queue,
            is_camera=self.is_camera,
            source_display_name=self.source_display_name,
            global_stop_event=stop_event,
            printer=self.printer,
        )

        self.infer_worker = InferenceWorker(
            model=self.model,
            frame_queue=self.frame_queue,
            infer_queue=self.infer_queue,
            is_camera=self.is_camera,
            frame_width=self.frame_width,
            frame_height=self.frame_height,
            source_display_name=self.source_display_name,
            global_stop_event=stop_event,
            printer=self.printer,
        )

        return True

    # ---------- Writer / Processing Loop ----------
    def run(self):
        frame_count = 0
        prev_time = time.time()
        loop_start = time.time()

        self.reader.start()
        self.infer_worker.start()

        try:
            while not stop_event.is_set():
                try:
                    try:
                        item = self.infer_queue.get(timeout=0.1)
                    except queue.Empty:
                        if stop_event.is_set():
                            break
                        continue
                except queue.Empty:
                    continue

                # EOF marker
                if (
                    isinstance(item, tuple)
                    and len(item) == 2
                    and isinstance(item[0], str)
                    and item[0] == "EOF"
                ):
                    break

                frame_resized, results = item

                # ---------- Annotation ----------
                try:
                    annotated_tmp = results[0].plot() if results else frame_resized

                    # Only resize if the shapes don't match the writer's frame size.
                    h, w = annotated_tmp.shape[:2]
                    if (w, h) != (self.frame_width, self.frame_height):
                        annotated = cv2.resize(
                            annotated_tmp,
                            (self.frame_width, self.frame_height),
                            interpolation=cv2.INTER_LINEAR,
                        )
                    else:
                        annotated = annotated_tmp

                except Exception:
                    annotated = frame_resized

                # ---------- Extract boxes (delegated to video_utils) ----------
                names, boxes_list = extract_boxes_from_results(
                    results, self.is_obb_model
                )

                # ---------- Timestamp ----------
                video_ts = self.start_time + timedelta(
                    seconds=frame_count / self.fps_video
                )

                # ---------- Measurements ----------
                counts = compute_counts_from_boxes(boxes_list, names)

                self.counter.update_counts(boxes_list, names, video_ts)
                self.aggregator.push_frame_data(video_ts, counts_dict=counts)
                self.interactions.process_frame(boxes_list, names, video_ts)

                # ---------- Terminal status ----------
                fps_smooth, tstr, prev_time, _ = self.printer.format_time_fps(
                    frame_count,
                    prev_time,
                    loop_start,
                    fps_video=self.fps_video,
                    total_frames=self.total_frames,
                    source_type=self.source_type,
                )

                frame_count += 1
                if frame_count % 5 == 0:
                    self.printer.update_frame_status(
                        self.idx,
                        self.source_display_name,
                        frame_count,
                        fps_smooth,
                        counts,
                        tstr,
                    )

                # ---------- Write annotated frame ----------
                write_annotated_frame(self.out_writer, annotated)

        finally:
            if self.reader:
                self.reader.stop()
            if self.infer_worker:
                self.infer_worker.stop()

            if self.reader:
                self.reader.join(timeout=1.0)

            while True:
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    break

            if self.infer_worker:
                self.infer_worker.join(timeout=2.0)

            saved = [self.out_file, self.metadata_file]

            f1 = self.counter.save_results()
            if f1:
                saved.extend(f1)
            f2 = self.aggregator.save_interval_results()
            if f2:
                saved.append(f2)
            f3 = self.aggregator.save_session_summary()
            if f3:
                saved.append(f3)
            f4 = self.interactions.save_results()
            if f4:
                saved.extend(f4) if isinstance(f4, list) else saved.append(f4)

            self.printer.mark_source_complete(self.idx)
            self.interactions.finalize()
            self.printer.save_measurements(self.paths["scores_folder"], saved)

# ---- Main Entry ----
def main():
    args = parse_arguments()
    printer = Printer(total_sources=len(args.sources))

    runs_dir = get_runs_dir(test=args.test)
    selected = None

    # ---------- Explicit model selection via --model ----------
    if getattr(args, "model", None):
        model_arg = args.model.strip()
        candidate_dir = runs_dir / model_arg

        # Case 1 - runs/<model_run> directory
        if candidate_dir.is_dir():
            selected = candidate_dir
            printer.info(f"Initializing model: {candidate_dir.name}")

        else:
            model_path = Path(model_arg)

            # Case 2 - Explicit .pt path
            if model_path.suffix == ".pt":
                if not model_path.exists():
                    printer.error(f"Model file does not exist: {model_path}")
                    printer.exit("Detection aborted due to missing weights file.")
                    sys.exit(1)

                selected = model_path
                printer.info(
                    f"Initializing using explicit weight file: {model_path.name}"
                )

            else:
                # Case 3 - Official YOLO model name (use ensure_weights)

                # Create a temp placeholder path in weights directory
                placeholder = WEIGHTS_DIR / f"{model_arg}.pt"

                resolved = ensure_weights(placeholder, model_arg)

                if resolved is None or not resolved.exists():
                    printer.error(
                        f"Could NOT resolve or download model '{model_arg}'."
                    )
                    printer.exit("Detection aborted due to invalid model selection.")
                    sys.exit(1)

                selected = resolved
                printer.info(f"Initializing YOLO model: {resolved.name}")

    else:
        # ---- Default Behavior (no --model provided) ----
        model_dirs = sorted(
            [
                d
                for d in runs_dir.iterdir()
                if d.is_dir() and (args.test or d.name.lower() != "test")
            ],
            reverse=True,
        )

        if not model_dirs:

            # ---- FALLBACK to YOLO11n ----
            placeholder = WEIGHTS_DIR / "yolo11n.pt"
            resolved = ensure_weights(placeholder, "yolo11n")

            if resolved is None or not resolved.exists():
                printer.error("Failed to download or resolve YOLO11n fallback model.")
                printer.exit("Detection aborted due to missing fallback model.")
                sys.exit(1)

            selected = resolved
            printer.info(f"Using fallback model: {resolved.name}")

        else:
            # ---- Custom models available ----
            if len(model_dirs) == 1:
                selected = model_dirs[0]
                printer.info(f"Initializing model: {selected.name}")
            else:
                selected = printer.prompt_model_selection(
                    runs_dir, exclude_test=not args.test
                )
                if not selected:
                    sys.exit(1)

    # Terminal reporting
    selected = Path(selected)

    # ---- Model Name Reporting ----
    if selected.is_dir():
        model_name = selected.name
    elif selected.suffix == ".pt":
        model_name = selected.stem
    else:
        model_name = str(selected)

    printer.set_model_name(model_name)

    # ---- Weights Path ----
    if selected.suffix == ".pt":
        weights_path = selected
    else:
        best = selected / "weights" / "best.pt"
        if best.exists():
            weights_path = best
        else:
            pts = sorted(
                selected.rglob("*.pt"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if pts:
                weights_path = pts[0]
                printer.warn(f"No best.pt found — using: {weights_path.name}")
            else:
                printer.missing_weights(selected)
                sys.exit(1)

    if not weights_path.exists() or weights_path.stat().st_size == 0:
        printer.missing_weights(selected)
        sys.exit(1)

    # ---- GLOBAL Class Loading (correct indentation!) ----
    classes = initialize_classes(
        model_name=model_name,
        force_reload=False,
        printer=printer,
        weights_path=weights_path
    )
    printer.classes_loaded(classes)

    # ---- Build Processors ----
    processors = []
    for idx, src in enumerate(args.sources, start=1):
        s = str(src)

        if s.lower().startswith("usb"):
            source_type = "usb"
            try:
                source_id = int(s[3:])
            except ValueError:
                printer.warn(f"Invalid USB source '{s}' — must be like usb0, usb1")
                continue
        else:
            source_type = "video"
            source_id = s

        vp = VideoProcessor(
            weights_path,
            source_id,
            source_type,
            idx,
            len(args.sources),
            printer,
            test=args.test,
        )
        if vp.initialize():
            processors.append(vp)

    # ---- Start Threads ----
    threads = []
    for vp in processors:
        t = threading.Thread(target=vp.run, daemon=True)
        t.start()
        threads.append(t)

    try:
        while any(t.is_alive() for t in threads):
            time.sleep(0.1)
    except KeyboardInterrupt:
        printer.stop_signal_received(single_thread=(len(threads) == 1))
        stop_event.set()

        for t in threads:
            while t.is_alive():
                try:
                    t.join(timeout=0.5)
                except KeyboardInterrupt:
                    continue

    printer.release_all_writers()
    printer.all_threads_terminated()


if __name__ == "__main__":
    main()
