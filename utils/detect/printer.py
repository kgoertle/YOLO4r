import threading
import os
import re
from pathlib import Path
import time
from utils.detect.classes_config import FOCUS_CLASSES, CONTEXT_CLASSES

# ------ CENTRALIZED TERMINAL LOGGER ------
class Printer:
    def __init__(self, total_sources):
        self.total_sources = total_sources
        self.lock = threading.Lock()
        self.lines = [""] * total_sources
        self.active_writers = {}

        try:
            self.term_height = os.get_terminal_size().lines
        except OSError:
            self.term_height = 30

        self.start_line = max(1, self.term_height - total_sources + 1)

    # ------ Terminal Update Core ------
    def update_line(self, line_number, text):
        with self.lock:
            if self.lines[line_number - 1] != text:
                self.lines[line_number - 1] = text
                print(f"\033[{self.start_line + line_number - 1};0H\033[K{text}", end="", flush=True)

    def _emit(self, tag, message):
        with self.lock:
            print(f"\033[{self.start_line + self.total_sources};0H\033[K{tag} {message}", flush=True)

    # ------ Frame + Timing Utilities ------
    def format_time_fps(self, frame_count, prev_time, start_time, fps_video=None, total_frames=None, source_type="video"):
        current_time = time.time()
        instantaneous_fps = 1 / (current_time - prev_time + 1e-6)
        self.prev_fps_smooth = getattr(self, "prev_fps_smooth", instantaneous_fps)
        fps_smooth = 0.9 * self.prev_fps_smooth + 0.1 * instantaneous_fps
        self.prev_fps_smooth = fps_smooth
        prev_time = current_time

        if source_type == "video" and fps_video and total_frames:
            elapsed_sec = int(frame_count / fps_video)
            total_sec = int(total_frames / fps_video)
            time_info = f"{elapsed_sec//60:02d}:{elapsed_sec%60:02d}/{total_sec//60:02d}:{total_sec%60:02d}"
        else:
            elapsed_sec = int(current_time - start_time)
            time_info = f"{elapsed_sec//60:02d}:{elapsed_sec%60:02d}"

        return fps_smooth, time_info, prev_time, None  # ETA is now always None

    def update_frame_status(self, line_number, display_name, frame_count, fps_smooth, counts, time_info):
        counts_to_show = {cls: counts.get(cls, 0) for cls in FOCUS_CLASSES}
        if CONTEXT_CLASSES:
            objects_total = sum(counts.get(cls, 0) for cls in CONTEXT_CLASSES)
            counts_to_show["Objects"] = objects_total

        count_parts = [f"{cls}:{counts_to_show.get(cls, 0)}" for cls in FOCUS_CLASSES]
        if CONTEXT_CLASSES:
            count_parts.append(f"Objects:{counts_to_show.get('Objects', 0)}")

        text = f"[{display_name}] Frames:{frame_count} | FPS:{fps_smooth:.1f} | " \
               + " | ".join(count_parts) + f" | Time:{time_info}"
        self.update_line(line_number, text)

    # ------ Generic Logging ------
    def info(self, message): self._emit("[INFO]", message)
    def warn(self, message): self._emit("[WARN]", message)
    def error(self, message): self._emit("[ERROR]", message)
    def save(self, message_or_path):
        msg = f"Saved to: {message_or_path}" if isinstance(message_or_path, (Path, str)) else str(message_or_path)
        self._emit("[SAVE]", msg)
    def exit(self, message): self._emit("[EXIT]", message)

    # ------ Dedicated Messages ------
    def prompt_model_selection(self, runs_dir, exclude_test=False):
        """Prompt user to select a model run. Optionally exclude 'test' folder."""
        model_dirs = sorted(
            [d for d in runs_dir.iterdir() if d.is_dir() and (not exclude_test or d.name.lower() != "test")],
            reverse=True
        )
        if not model_dirs:
            self.missing_weights(runs_dir)
            return None
        self.model_prompt(model_dirs)
        while True:
            try:
                choice = input(f"Select a model run (1-{len(model_dirs)}): ")
            except KeyboardInterrupt:
                self.warn("Model selection interrupted by user. Exiting.")
                return None

            if choice.isdigit() and 1 <= int(choice) <= len(model_dirs):
                selected = model_dirs[int(choice) - 1]
                return selected / "weights" / "best.pt"
            else:
                self.info("Invalid selection, try again.")
    def missing_weights(self, runs_dir):
        self.error(f"No valid best.pt found in {runs_dir}. Please train a model first or place weights in the folder.")
    def model_init(self, weights_path):
        weights_path = Path(weights_path)
        try:
            runs_index = weights_path.parts.index("runs")
            short_path = Path(*weights_path.parts[runs_index:runs_index+3])
        except ValueError:
            short_path = weights_path.parent.parent  # fallback: just two levels up
        self.info(f"Initializing model: {short_path}")
    def model_fail(self, e): self.error(f"Could not initialize model in thread: {e}")
    def open_capture_fail(self, source_name): self.error(f"Could not open {source_name}!")
    def read_frame_fail(self, source_name): self.error(f"Could not read frame from {source_name}")
    def inference_fail(self, display_name, e): self.error(f"Inference failed for {display_name}: {e}")
    def save_aggregator_csv(self, file_path): self.save(file_path)
    def detection_complete(self, display_name): self.info(f"All detection outputs saved for {display_name}")
    def stop_signal_received(self, single_thread=True):
        msg = "Stop signal received. Terminating pipeline..." if single_thread else "Stop signal received. Terminating pipelines..."
        self.exit(msg)
    def skip_source(self, src):
        self.warn(f"Skipping source: {src}")
    def no_sources(self):
        self.warn("No valid sources to process.")
    def all_threads_terminated(self): self.exit("All detection threads safely terminated.")

    # ------ Dedicated Model Selection ------
    def model_prompt(self, model_dirs):
        self._emit("[MODEL]", f"{len(model_dirs)} models found in runs folder:")
        for i, d in enumerate(model_dirs, start=1):
            print(f"{i}. {d.name}")

    # ------ Recording initaliation ------
    def recording_initialized(self, timestamp):
        self._emit("[INFO]", f"Recording initialized at {timestamp}.")

    def save_measurements(self, base_dir, files):
        base_dir = Path(base_dir)
        try:
            # find "measurements" folder in the path
            meas_index = base_dir.parts.index("measurements")
            short_path = Path(*base_dir.parts[meas_index:])
        except ValueError:
            short_path = base_dir  # fallback

        self._emit("[SAVE]", f'Measurements saved to: "{short_path}"')
        if files:
            for f in files:
                print(f" - {Path(f).name}")


    # ------ Writer Management ------
    def register_writer(self, raw_source_name, writer, cap, source_type, out_file, display_name=None):
        safe_name = re.sub(r'[^\w\-]', '_', Path(out_file.name).stem) + out_file.suffix
        entry = {
            'writer': writer,
            'cap': cap,
            'source_type': source_type,
            'out_file': out_file,
            'source_name': raw_source_name,
            'display_name': display_name or raw_source_name
        }
        self.active_writers[safe_name] = entry

        # --- Generate timestamp dynamically ---
        from datetime import datetime
        timestamp = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
        self.recording_initialized(timestamp)

        return safe_name

    def safe_release_writer(self, raw_source_name):
        entry = self.active_writers.get(raw_source_name)
        if entry:
            writer, cap = entry['writer'], entry['cap']
            if writer:
                try:
                    writer.release()
                except Exception:
                    pass
            if cap:
                try:
                    cap.release()
                except Exception:
                    pass
            del self.active_writers[raw_source_name]

    def release_all_writers(self):
        for name in list(self.active_writers.keys()):
            self.safe_release_writer(name)
