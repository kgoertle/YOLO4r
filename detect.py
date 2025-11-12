import sys, os, threading, time, platform, queue, json, cv2
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
from ultralytics import YOLO

# ---- UTILITIES ----
from utils.detect.paths import get_runs_dir, get_output_folder
from utils.detect.video_rotation import get_rotation_angle, rotate_frame
from utils.detect.arg_parser import parse_arguments
from utils.detect.printer import Printer
from utils.detect.measurements import MeasurementConfig, Counter, Interactions, Aggregator, compute_counts_from_boxes
from utils.detect.classes_config import initialize_classes
from utils.detect.video_metadata import extract_video_metadata, parse_creation_time

# ------ GLOBALS ------
BASE_DIR = Path(__file__).resolve().parent
stop_event = threading.Event()
IS_MAC, IS_LINUX = platform.system() == "Darwin", platform.system() == "Linux"

# ----- DETECTION PROCESSOR -----
class VideoProcessor:
    def __init__(self, weights_path, source, source_type, idx, total_sources, printer, test=False, data_yaml_path=None):
        self.weights_path = weights_path
        self.source = source
        self.source_type = source_type
        self.idx = idx
        self.total_sources = total_sources
        self.printer = printer
        self.test = test
        self.data_yaml_path = data_yaml_path

        self.is_camera = source_type == "usb"
        self.source_display_name = Path(source).stem if not self.is_camera else f"usb{source}"

        self.frame_queue = queue.Queue(maxsize=50)
        self.stop_reader = threading.Event()
        self.reader_thread = None

        self.model = None
        self.is_obb_model = False
        self.cap = None
        self.out_writer = None
        self.rotation_angle = 0
        self.config = MeasurementConfig()
        self.counter = None
        self.aggregator = None
        self.interactions = None
        self.start_time = None
        self.fps_video = None
        self.total_frames = None

    def initialize(self):
        # ---- Initialize YOLO model ----
        try:
            self.model = YOLO(str(self.weights_path))
            self.model.weights_path = self.weights_path
            self.printer.model_init(self.weights_path)
        except Exception as e:
            self.printer.model_fail(e)
            return False

        # ---- Detect if model is OBB or standard ----
        try:
            dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
            dummy_results = self.model.predict(dummy_frame, verbose=False, show=False)
            if hasattr(dummy_results[0], "obb") and dummy_results[0].obb is not None:
                self.is_obb_model = True
            self.printer.info(f"{'OBB' if self.is_obb_model else 'Standard YOLO'} model detected.")
        except Exception:
            self.is_obb_model = False
            self.printer.info("Standard YOLO model assumed.")

        # ---- Video metadata ----
        metadata = extract_video_metadata(self.source) if not self.is_camera else {
            "type": "camera", "source": str(self.source), "creation_time": datetime.now().isoformat()
        }
        self.creation_dt = parse_creation_time(metadata)
        metadata["creation_time_str"] = self.creation_dt.strftime("%H:%M:%S") if self.creation_dt else "00:00:00"

        # ---- Output paths ----
        self.paths = get_output_folder(
            self.weights_path, self.source_type,
            self.source if not self.is_camera else f"usb{self.source}",
            test_detect=self.test, base_time=self.creation_dt if not self.is_camera else None
        )
        self.out_file = self.paths["video_folder"] / f"{self.source_display_name}.mp4"
        self.metadata_file = self.paths["metadata"]

        # Save metadata
        with open(self.metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        # ---- Initialize classes ----
        initialize_classes(data_yaml_path=self.data_yaml_path)

        # ---- Open video capture ----
        self.cap = cv2.VideoCapture(
            int(self.source) if self.is_camera else str(self.source),
            cv2.CAP_AVFOUNDATION if IS_MAC else cv2.CAP_V4L2 if IS_LINUX else 0
        ) if self.is_camera else cv2.VideoCapture(str(self.source))

        if not self.cap or not self.cap.isOpened():
            self.printer.open_capture_fail(self.source_display_name)
            return False

        # ---- Get rotation for files only ----
        self.rotation_angle = get_rotation_angle(self.source) if not self.is_camera else 0

        # ---- Determine frame width & height safely ----
        if self.is_camera:
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        else:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.printer.read_frame_fail(self.source_display_name)
                return False
            if self.rotation_angle in [90, 180, 270]:
                frame = rotate_frame(frame, self.rotation_angle)
            self.frame_height, self.frame_width = frame.shape[:2]
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # ---- FPS & total frames ----
        self.fps_video = self.cap.get(cv2.CAP_PROP_FPS) if not self.is_camera else 20.0
        if self.fps_video <= 0 or np.isnan(self.fps_video):
            self.fps_video = 20.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) if not self.is_camera else None

        # ---- Start time ----
        self.start_time = self.creation_dt if self.creation_dt else datetime.now()

        # ---- Video writer ----
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out_writer = cv2.VideoWriter(
            str(self.out_file),
            fourcc,
            self.fps_video,
            (self.frame_width, self.frame_height)
        )
        if not self.out_writer.isOpened():
            self.printer.error(f"Failed to open VideoWriter for {self.out_file}")
            return False

        self.printer.register_writer(
            self.out_file.name, self.out_writer, self.cap, self.source_type, self.out_file,
            display_name=self.source_display_name
        )

        # ---- Measurements ----
        self.counter = Counter(out_folder=self.paths["counts"], config=self.config, start_time=self.start_time)
        self.aggregator = Aggregator(out_folder=self.paths["frame-counts"], config=self.config, start_time=self.start_time)
        self.interactions = Interactions(out_folder=self.paths["interactions"], config=self.config, start_time=self.start_time)

        return True

    def start_reader(self):
        """Camera-safe frame reader thread"""
        def read_frames():
            while not self.stop_reader.is_set():
                try:
                    ret, frame = self.cap.read()
                except Exception as e:
                    self.printer.warn(f"VideoCapture read failed: {e}")
                    ret, frame = False, None

                if not ret or frame is None:
                    if not self.is_camera:
                        self.stop_reader.set()
                        break
                    time.sleep(0.05)
                    continue

                if hasattr(self, "frame_width") and hasattr(self, "frame_height"):
                    frame = cv2.resize(frame, (self.frame_width, self.frame_height))

                try:
                    self.frame_queue.put(frame, timeout=0.1)
                except queue.Full:
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put(frame, timeout=0.1)
                    except queue.Empty:
                        pass

        self.reader_thread = threading.Thread(target=read_frames, daemon=True)
        self.reader_thread.start()

    def run(self):
        frame_count, prev_time, fps_smooth, loop_start_time = 0, time.time(), 0.0, time.time()
        self.start_reader()

        try:
            while not stop_event.is_set():
                try:
                    frame = self.frame_queue.get(timeout=0.5)
                except queue.Empty:
                    if self.stop_reader.is_set() and not self.is_camera:
                        break
                    continue

                if self.rotation_angle in [90, 180, 270]:
                    frame = rotate_frame(frame, self.rotation_angle)

                # ---- Inference ----
                try:
                    results = self.model.predict(frame, verbose=False, show=False, imgsz=640)
                    annotated_frame = results[0].plot() if results and len(results) > 0 else frame
                except KeyboardInterrupt:
                    self.printer.stop_signal_received(single_thread=True)
                    stop_event.set()
                    annotated_frame = frame
                    break
                except Exception as e:
                    self.printer.inference_fail(self.source_display_name, e)
                    annotated_frame = frame

                # ---- Unified box extraction for OBB or standard YOLO ----
                current_boxes_list, names = [], {}
                if results and len(results) > 0:
                    res = results[0]
                    names = res.names

                    if self.is_obb_model and hasattr(res, "obb") and res.obb is not None:
                        boxes = res.obb.xywhr.cpu().numpy()
                        classes = res.obb.cls.cpu().numpy()
                        current_boxes_list = [[*b[:4], float(b[4]), int(c)] for b, c in zip(boxes, classes)]
                    elif hasattr(res, "boxes") and res.boxes is not None:
                        boxes = res.boxes.xyxy.cpu().numpy()
                        confs = res.boxes.conf.cpu().numpy()
                        classes = res.boxes.cls.cpu().numpy()
                        current_boxes_list = [
                            [float(x1), float(y1), float(x2), float(y2), float(conf), int(c)]
                            for (x1, y1, x2, y2), conf, c in zip(boxes, confs, classes)
                        ]

                video_time = self.start_time + timedelta(seconds=frame_count / self.fps_video)

                counts = compute_counts_from_boxes(current_boxes_list, names)
                self.counter.update_counts(current_boxes_list, names, timestamp=video_time)
                self.aggregator.push_frame_data(video_time, counts_dict=counts)
                self.interactions.process_frame(current_boxes_list, names, video_time)

                fps_smooth, time_info, prev_time, _ = self.printer.format_time_fps(
                    frame_count, prev_time, loop_start_time, fps_video=self.fps_video,
                    total_frames=self.total_frames, source_type=self.source_type
                )
                frame_count += 1

                if frame_count % 5 == 0:
                    self.printer.update_frame_status(
                        self.idx, self.source_display_name, frame_count, fps_smooth, counts, time_info
                    )

                self.out_writer.write(annotated_frame)

        finally:
            self.stop_reader.set()
            if hasattr(self, "cap") and self.cap:
                try: self.cap.release()
                except Exception: pass
            if self.reader_thread: self.reader_thread.join()

            saved_files = [self.out_file, self.metadata_file]
            counter_files = self.counter.save_results()
            if counter_files: saved_files.extend(counter_files)
            interval_file = self.aggregator.save_interval_results()
            if interval_file: saved_files.append(interval_file)
            summary_file = self.aggregator.save_session_summary()
            if summary_file: saved_files.append(summary_file)
            interactions_file = self.interactions.save_results()
            if interactions_file:
                saved_files.extend(interactions_file) if isinstance(interactions_file, list) else saved_files.append(interactions_file)
            self.interactions.finalize()
            self.printer.save_measurements(self.paths["scores_folder"], saved_files)


# ------ MAIN ENTRY ------
def main():
    args = parse_arguments()
    printer = Printer(total_sources=len(args.sources))

    runs_dir = get_runs_dir(test=args.test)
    model_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir()], reverse=True)

    if not model_dirs:
        printer.missing_weights(runs_dir)
        sys.exit(1)

    selected_model = model_dirs[0] if len(model_dirs) == 1 else printer.prompt_model_selection(runs_dir)
    if not selected_model: sys.exit(1)

    weights_path = selected_model / "weights" / "best.pt"
    if not weights_path.exists():
        printer.missing_weights(selected_model)
        sys.exit(1)

    dataset_data_yaml = selected_model / "data.yaml"
    if not dataset_data_yaml.exists(): dataset_data_yaml = None

    processors = []
    for idx, src in enumerate(args.sources, start=1):
        source_type = "usb" if str(src).lower().startswith("usb") else "video"
        source_id = int(src[3:]) if source_type == "usb" else src

        pproc = VideoProcessor(
            weights_path, source_id, source_type, idx, len(args.sources),
            printer, test=args.test, data_yaml_path=dataset_data_yaml
        )
        if not pproc.initialize(): continue
        processors.append(pproc)

    threads = []
    for proc in processors:
        t = threading.Thread(target=proc.run, daemon=True)
        t.start()
        threads.append(t)

    try:
        while any(t.is_alive() for t in threads):
            time.sleep(0.1)
    except KeyboardInterrupt:
        printer.stop_signal_received(single_thread=(len(threads) == 1))
        stop_event.set()
        # Robust join loop: safely waits for threads to exit
        for t in threads:
            while t.is_alive():
                try:
                    t.join(timeout=0.5)
                except KeyboardInterrupt:
                    # Ignore repeated Ctrl+C while waiting
                    continue

    printer.release_all_writers()
    printer.all_threads_terminated()


if __name__ == "__main__":
    main()
