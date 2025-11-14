# detect.py
import sys, os, threading, time, platform, queue, json, cv2
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
from ultralytics import YOLO

# ----------- UTILITIES ---------------
from utils.detect.paths import get_runs_dir, get_output_folder, get_model_data_yaml
from utils.detect.arg_parser import parse_arguments
from utils.detect.printer import Printer
from utils.detect.measurements import MeasurementConfig, Counter, Interactions, Aggregator, compute_counts_from_boxes
from utils.detect.classes_config import initialize_classes
from utils.detect.video_metadata import extract_video_metadata, parse_creation_time

# ---- SYSTEM ----
stop_event = threading.Event()
IS_MAC = platform.system() == "Darwin"
IS_LINUX = platform.system() == "Linux"

# ----------- VIDEO PROCESSOR -----------
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
        data_yaml_path=None,
    ):
        self.weights_path = Path(weights_path)
        self.source = source
        self.source_type = source_type  # "video" or "usb"
        self.idx = idx
        self.total_sources = total_sources
        self.printer = printer
        self.test = test
        self.data_yaml_path = Path(data_yaml_path) if data_yaml_path else None

        self.is_camera = source_type == "usb"
        self.source_display_name = (
            Path(source).stem if not self.is_camera else f"usb{source}"
        )

        # Threading and buffers
        self.frame_queue = queue.Queue(maxsize=50)
        self.stop_reader = threading.Event()
        self.reader_thread = None

        # Model & I/O
        self.model = None
        self.is_obb_model = False
        self.cap = None
        self.out_writer = None

        # Measurements
        self.config = MeasurementConfig()
        self.counter = None
        self.aggregator = None
        self.interactions = None

        self.start_time = None
        self.fps_video = None
        self.total_frames = None

    # ----------- Initialization -----------
    def initialize(self):
        # ---- Load YOLO model ----
        try:
            self.model = YOLO(str(self.weights_path))
            self.model.weights_path = self.weights_path
            self.printer.model_init(self.weights_path)
        except Exception as e:
            self.printer.model_fail(e)
            return False

        # Detect OBB model
        try:
            test_frame = np.zeros((640, 640, 3), dtype=np.uint8)
            res = self.model.predict(test_frame, verbose=False, show=False)
            self.is_obb_model = hasattr(res[0], "obb") and res[0].obb is not None
        except Exception:
            self.is_obb_model = False

        # ---- Initialize classes ----
        if self.data_yaml_path and self.data_yaml_path.exists():
            initialize_classes(data_yaml_path=self.data_yaml_path)
        else:
            model_dir = self.weights_path.parent.parent
            model_yaml = get_model_data_yaml(model_dir, self.printer)
            initialize_classes(data_yaml_path=model_yaml)

        # ---- Metadata & Timestamps ----
        if not self.is_camera:
            metadata = extract_video_metadata(self.source)
        else:
            metadata = {
                "type": "camera",
                "source": str(self.source),
                "creation_time": datetime.now().isoformat(),
            }

        self.start_time = parse_creation_time(metadata) or datetime.now()
        metadata["creation_time_str"] = self.start_time.strftime("%H:%M:%S")

        # ---- Output folder construction ----
        self.paths = get_output_folder(
            self.weights_path,
            self.source_type,
            self.source if not self.is_camera else f"usb{self.source}",
            test_detect=self.test,
            base_time=self.start_time if not self.is_camera else None,
        )

        self.out_file = self.paths["video_folder"] / f"{self.paths['safe_name']}.mp4"
        self.metadata_file = self.paths["metadata"]

        with open(self.metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        # ---- Capture initialization ----
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

        # ---- Frame size ----
        if self.is_camera:
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
        else:
            ret, frame = self.cap.read()
            if not ret:
                self.printer.read_frame_fail(self.source_display_name)
                return False
            self.frame_height, self.frame_width = frame.shape[:2]
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # ---- FPS & total frames ----
        if not self.is_camera:
            self.fps_video = self.cap.get(cv2.CAP_PROP_FPS)
            if not self.fps_video or self.fps_video <= 0 or np.isnan(self.fps_video):
                self.fps_video = 20.0
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        else:
            self.fps_video = 20.0
            self.total_frames = None

        # ---- VideoWriter initialization ----
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.out_writer = cv2.VideoWriter(
            str(self.out_file),
            fourcc,
            self.fps_video,
            (self.frame_width, self.frame_height),
        )
        if not self.out_writer.isOpened():
            self.printer.warn(f"VideoWriter failed: {self.source_display_name}")
            return False

        # ---- Register writer ----
        self.printer.register_writer(
            self.out_file.name,
            self.out_writer,
            self.cap,
            self.source_type,
            self.out_file,
            display_name=self.source_display_name,
        )

        # ---- Measurement objects ----
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
        )

        return True

    # ---- Reader Thread ----
    def start_reader(self):
        """Thread that continuously reads frames and pushes into a queue."""

        def reader():
            while not self.stop_reader.is_set():
                try:
                    ret, frame = self.cap.read()
                except Exception:
                    ret, frame = False, None

                if not ret or frame is None:
                    time.sleep(0.02)
                    continue

                try:
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(frame)
                    except queue.Empty:
                        pass

        self.reader_thread = threading.Thread(target=reader, daemon=True)
        self.reader_thread.start()

    # ---- Main run loop ----
    def run(self):
        frame_count = 0
        prev_time = time.time()
        loop_start = time.time()

        self.start_reader()

        try:
            while not stop_event.is_set():
                try:
                    frame = self.frame_queue.get(timeout=0.5)
                except queue.Empty:
                    if self.stop_reader.is_set() and not self.is_camera:
                        break
                    continue

                frame_resized = cv2.resize(frame, (self.frame_width, self.frame_height))

                # ---- Inference ----
                try:
                    results = self.model.predict(
                        frame_resized, verbose=False, show=False, imgsz=640
                    )
                    annotated = results[0].plot() if results else frame_resized
                except Exception as e:
                    self.printer.inference_fail(self.source_display_name, e)
                    annotated = frame_resized

                # ---- Extract boxes ----
                names = {}
                boxes_list = []

                if results:
                    r = results[0]
                    names = r.names

                    if self.is_obb_model and hasattr(r, "obb") and r.obb is not None:
                        xywhr = r.obb.xywhr.cpu().numpy()
                        cls = r.obb.cls.cpu().numpy()
                        boxes_list = [[*b[:4], float(b[4]), int(c)] for b, c in zip(xywhr, cls)]

                    elif hasattr(r, "boxes") and r.boxes is not None:
                        xyxy = r.boxes.xyxy.cpu().numpy()
                        conf = r.boxes.conf.cpu().numpy()
                        cls = r.boxes.cls.cpu().numpy()
                        boxes_list = [
                            [float(x1), float(y1), float(x2), float(y2), float(cf), int(c)]
                            for (x1, y1, x2, y2), cf, c in zip(xyxy, conf, cls)
                        ]

                # Timestamp for this frame
                video_ts = self.start_time + timedelta(
                    seconds=frame_count / self.fps_video
                )

                # Counts
                counts = compute_counts_from_boxes(boxes_list, names)

                self.counter.update_counts(boxes_list, names, video_ts)
                self.aggregator.push_frame_data(video_ts, counts_dict=counts)
                self.interactions.process_frame(boxes_list, names, video_ts)

                # FPS display
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

                # write frame
                self.out_writer.write(annotated)

        finally:
            # Shutdown and cleanup
            self.stop_reader.set()
            if self.reader_thread:
                self.reader_thread.join()

            try:
                if self.cap: self.cap.release()
                if self.out_writer: self.out_writer.release()
            except Exception:
                pass

            saved = [self.out_file, self.metadata_file]

            # Measurement files
            f1 = self.counter.save_results()
            if f1: saved.extend(f1)
            f2 = self.aggregator.save_interval_results()
            if f2: saved.append(f2)
            f3 = self.aggregator.save_session_summary()
            if f3: saved.append(f3)
            f4 = self.interactions.save_results()
            if f4:
                if isinstance(f4, list):
                    saved.extend(f4)
                else:
                    saved.append(f4)
            self.interactions.finalize()
            self.printer.save_measurements(self.paths["scores_folder"], saved)

# ---- Main Entry ----
def main():
    args = parse_arguments()
    printer = Printer(total_sources=len(args.sources))

    runs_dir = get_runs_dir(test=args.test)

    # model directories (excluding "test" in normal mode)
    model_dirs = sorted(
        [
            d
            for d in runs_dir.iterdir()
            if d.is_dir() and (args.test or d.name.lower() != "test")
        ],
        reverse=True,
    )

    if not model_dirs:
        printer.missing_weights(runs_dir)
        sys.exit(1)

    # Selection menu
    if len(model_dirs) == 1:
        selected = model_dirs[0]
    else:
        selected = printer.prompt_model_selection(
            runs_dir, exclude_test=not args.test
        )

    if not selected:
        sys.exit(1)

    selected = Path(selected)

    # Weights path
    if selected.suffix == ".pt":
        weights_path = selected
    else:
        best = selected / "weights" / "best.pt"
        if best.exists():
            weights_path = best
        else:
            pts = sorted(selected.rglob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
            if pts:
                weights_path = pts[0]
                printer.warn(f"No best.pt found — using: {weights_path.name}")
            else:
                printer.missing_weights(selected)
                sys.exit(1)

    if not weights_path.exists() or weights_path.stat().st_size == 0:
        printer.missing_weights(selected)
        sys.exit(1)

    # Resolve dataset YAML for class loading
    data_yaml = get_model_data_yaml(selected, printer)
    if data_yaml is None:
        sys.exit(1)

    # Build processors
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
            data_yaml_path=data_yaml,
        )
        if vp.initialize():
            processors.append(vp)

    # Start threads
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
