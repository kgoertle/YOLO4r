import sys, os, threading, time, platform, queue, re
from pathlib import Path
from datetime import datetime
import numpy as np
import cv2
from ultralytics import YOLO
from pymediainfo import MediaInfo

# ---- UTILITIES ----
from utils.detect.paths import find_latest_best, get_output_folder
from utils.detect.video_rotation import get_rotation_angle, rotate_frame
from utils.detect.arg_parser import parse_arguments
from utils.detect.logger import DetectionDashboard
from utils.detect.temporal_aggregator import Aggregator
from utils.detect.basic_counts import log_bird_counts_summary, flush_remaining_interval
from utils.detect.interaction_metrics import InteractionMetrics

BASE_DIR = Path(__file__).resolve().parent
stop_event = threading.Event()
IS_MAC, IS_LINUX = platform.system() == "Darwin", platform.system() == "Linux"


def run_detection(weights_path, source, source_type, line_number, total_sources, dashboard, test=False):
    try:
        model = YOLO(str(weights_path))
        model.weights_path = weights_path
    except Exception as e:
        dashboard.log(f"[ERROR] Could not initialize model in thread: {e}")
        return
    
    # ---------- Prepare source names ----------
    raw_source_name = f"{source_type}{source}" if source_type == "usb" else str(source)
    display_name = raw_source_name if source_type == "usb" else Path(raw_source_name).stem
    safe_source_name = re.sub(r'[^\w\-\.]', '_', display_name)

    # ---------- Prepare output file ----------
    video_folder, scores_folder = get_output_folder(model.weights_path, source_type, raw_source_name, test)
    out_file = video_folder / f"{safe_source_name}.mp4"  # simplified name, no timestamp
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # ---------- Prepare measurement folders ----------
    basic_counts_folder = scores_folder / "counts"
    basic_counts_folder.mkdir(parents=True, exist_ok=True)

    aggregator_folder = scores_folder / "raw_detection"
    aggregator_folder.mkdir(parents=True, exist_ok=True)

    # ---------- Open capture ----------
    if source_type == "usb":
        cap = cv2.VideoCapture(int(source),
                               cv2.CAP_AVFOUNDATION if IS_MAC else cv2.CAP_V4L2 if IS_LINUX else 0)
    else:
        cap = cv2.VideoCapture(str(source))

    if not cap or not cap.isOpened():
        dashboard.log(f"[ERROR] Could not open {raw_source_name}!")
        return

    # ---------- Read first frame to get size ----------
    rotation_angle = get_rotation_angle(source) if source_type == "video" else 0
    ret, frame = cap.read()
    if not ret or frame is None:
        dashboard.log(f"[ERROR] Could not read a frame from {raw_source_name}")
        return
    if rotation_angle in [90, 180, 270] or frame.shape[0] > frame.shape[1]:
        frame = rotate_frame(frame, rotation_angle or 90)
    height, width = frame.shape[:2]

    # ---------- Register writer ----------
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    fps_video = fps_video if fps_video > 0 and not np.isnan(fps_video) else 20.0
    out_writer = cv2.VideoWriter(str(out_file), fourcc, fps_video, (width, height))
    dashboard.register_writer(raw_source_name, out_writer, cap, source_type, out_file)

    # ---------- Frame queue ----------
    frame_queue = queue.Queue(maxsize=10)
    stop_reader = threading.Event()

    # ---------- Initiate Aggregator ----------
    aggregator = Aggregator(interval_sec=5, session_sec=10)

    # ---------- Initiate Interaction Metrics ----------
    interaction_folder = scores_folder / "interactions"
    metrics = InteractionMetrics(interaction_folder)

    def read_frames():
        while not stop_reader.is_set():
            ret, frame = cap.read()
            if not ret:
                stop_reader.set()
                break
            try:
                frame_queue.put(frame, timeout=0.1)
            except queue.Full:
                pass

    reader_thread = threading.Thread(target=read_frames, daemon=True)
    reader_thread.start()

    frame_count, fps_smooth, prev_time = 0, 0, time.time()
    start_time = time.time()
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) if source_type == "video" else None

    try:
        while not stop_event.is_set():
            try:
                frame = frame_queue.get(timeout=0.5)
            except queue.Empty:
                if stop_reader.is_set():
                    break
                continue

            if rotation_angle in [90, 180, 270] or frame.shape[0] > frame.shape[1]:
                frame = rotate_frame(frame, rotation_angle or 90)

           # ---------- Inference ----------
            try:
                results = model.predict(frame, verbose=False, show=False, imgsz=640)
                annotated_frame = results[0].plot() if results and len(results) > 0 else frame
            except Exception as e:
                dashboard.log(f"[ERROR] Inference failed for {display_name}: {e}")
                annotated_frame = frame  # fallback to raw frame

            # ---------- Current Boxes ----------
            current_boxes_list = []
            if results and hasattr(results[0], "obb") and results[0].obb is not None:
                boxes = results[0].obb.xywhr.cpu().numpy()
                classes = results[0].obb.cls.cpu().numpy()
                current_boxes_list = [
                    [cx, cy, w, h, float(angle), int(cls)]
                    for cx, cy, w, h, angle, cls in zip(
                        boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3], boxes[:,4], classes
                    )
                ]

            # ---------- Class counts ----------
            names = results[0].names if results else {}
            males = sum(1 for b in current_boxes_list if names.get(b[5]) == "M")
            females = sum(1 for b in current_boxes_list if names.get(b[5]) == "F")
            other_objects = sum(1 for b in current_boxes_list if names.get(b[5]) not in ["M","F"])

            # ---------- Push to aggregator ----------
            other_counts_dict = {
                'Feeder': sum(1 for b in current_boxes_list if names.get(b[5]) == 'Feeder'),
                'Main_Perch': sum(1 for b in current_boxes_list if names.get(b[5]) == 'Main_Perch'),
                'Wooden_Perch': sum(1 for b in current_boxes_list if names.get(b[5]) == 'Wooden_Perch'),
                'Sky_Perch': sum(1 for b in current_boxes_list if names.get(b[5]) == 'Sky_Perch'),
                'Nesting_Box': sum(1 for b in current_boxes_list if names.get(b[5]) == 'Nesting_Box')
            }

            aggregator.push_frame_data(datetime.now(), males, females, other_counts_dict)

            # ---------- Log basic counts ----------
            log_bird_counts_summary(current_boxes_list, names, basic_counts_folder, interval_sec=aggregator.interval_sec)

            # ---------- Log interaction metrics ----------
            metrics.process_frame(current_boxes_list, names, timestamp=datetime.now())

            # ---------- Time and FPS ----------
            fps_smooth = 0.9 * fps_smooth + 0.1 * (1 / (time.time() - prev_time + 1e-6))
            prev_time = time.time()
            frame_count += 1

            if source_type == "video":
                elapsed_sec = frame_count / fps_video
                total_sec = total_frames / fps_video if total_frames else 0
                time_info = f"{int(elapsed_sec // 60):02d}:{int(elapsed_sec % 60):02d}/" \
                            f"{int(total_sec // 60):02d}:{int(total_sec % 60):02d}"
            else:
                elapsed_sec = int(time.time() - start_time)
                h, m = divmod(elapsed_sec // 60, 60)
                s = elapsed_sec % 60
                time_info = f"{h:02d}:{m:02d}:{s:02d}"

            # ---------- Update dashboard ----------
            if frame_count % 5 == 0:
                dashboard.update_line(
                    line_number,
                    f"[{display_name}] Frames:{frame_count} | FPS:{fps_smooth:.1f} | "
                    f"Males:{males} | Females:{females} | Objects:{other_objects} | Time:{time_info}"
                )

            # ---------- Write frame ----------
            out_writer.write(annotated_frame)
            time.sleep(0.001)

    finally:
        stop_reader.set()
        reader_thread.join()
        cap.release()
        dashboard.safe_release_writer(raw_source_name)

        # ---------- Save aggregator CSVs ----------
        interval_file, session_file = aggregator.save_results(aggregator_folder)
        if interval_file and session_file:
            dashboard.log(f"[SAVE] Interval results saved to: {interval_file}")
            dashboard.log(f"[SAVE] Session summary saved to: {session_file}")
        else:
            dashboard.log(f"[SAVE] No aggregator results to save for {display_name}.")

        # ---------- Flush any remaining counts ----------
        flush_remaining_interval(out_folder=basic_counts_folder)
        dashboard.log(f"[SAVE] Object counts saved to: {basic_counts_folder}")

        # ---------- Save interaction CSVs ----------
        interaction_file = metrics.finalize()
        if interaction_file:
            dashboard.log(f"[SAVE] Interaction metrics saved to: {interaction_file}")

# ------ MAIN FUNCTION -------
if __name__ == "__main__":
    args = parse_arguments()
    runs_dir = BASE_DIR / ("runs/test" if args.test else "runs/main")
    weights_path = find_latest_best(runs_dir)

    if not weights_path:
        print(f"[ERROR] Could not find a valid best.pt in {runs_dir}")
        sys.exit(1)

    print(f"[LOAD] Initializing model: {weights_path}.")
    model = YOLO(str(weights_path))
    model.weights_path = weights_path

    total_sources = len(args.sources)
    dashboard = DetectionDashboard(total_sources)

    threads = []
    if total_sources == 1:
        src = args.sources[0]
        source_type = "usb" if src.lower().startswith("usb") else "video"
        src_id = int(src[3:]) if source_type == "usb" else src
        try:
            run_detection(weights_path, src_id, source_type, 1, total_sources, dashboard, args.test)
        except KeyboardInterrupt:
            dashboard.log("[EXIT] Stop signal received. Terminating pipeline...")
            stop_event.set()
        finally:
            dashboard.safe_release_writer(src_id if source_type == "usb" else src)
    else:
        for i, src in enumerate(args.sources, start=1):
            source_type = "usb" if src.lower().startswith("usb") else "video"
            src_id = int(src[3:]) if source_type == "usb" else src
            t = threading.Thread(
                target=run_detection,
                args=(weights_path, src_id, source_type, i, total_sources, dashboard, args.test)
            )
            t.start()
            threads.append(t)

        try:
            while any(t.is_alive() for t in threads):
                for t in threads:
                    t.join(timeout=0.5)
        except KeyboardInterrupt:
            dashboard.log("[EXIT] Stop signal received. Terminating pipelines...")
            stop_event.set()
            for t in threads:
                t.join(timeout=2)

    dashboard.release_all_writers()
    dashboard.log("[EXIT] All detection threads safely terminated.")
