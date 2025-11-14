# utils/detect/printer.py
import threading, os, re, time
from pathlib import Path
from datetime import datetime
from utils.detect.classes_config import FOCUS_CLASSES, CONTEXT_CLASSES, reload_config
reload_config()

# ---------- CENTRALIZED TERMINAL LOGGING ----------
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

        # Where dynamic status lines begin
        self.start_line = max(1, self.term_height - total_sources + 1)

    # ---------- Core terminal printing ----------
    def update_line(self, line_number, text):
        """Update a dedicated line in-place (source FPS & counts)."""
        with self.lock:
            if self.lines[line_number - 1] != text:
                self.lines[line_number - 1] = text
                print(
                    f"\033[{self.start_line + line_number - 1};0H\033[K{text}",
                    end="", flush=True
                )

    def _emit(self, tag, message):
        """Emit a tagged log line underneath the dynamic block."""
        with self.lock:
            print(
                f"\033[{self.start_line + self.total_sources};0H\033[K{tag} {message}",
                flush=True
            )

    # ---------- Logging shortcuts ----------
    def info(self, m):  self._emit("[INFO]", m)
    def warn(self, m):  self._emit("[WARN]", m)
    def error(self, m): self._emit("[ERROR]", m)
    def exit(self, m):  self._emit("[EXIT]", m)

    def save(self, m):
        """Generic save message."""
        if isinstance(m, (str, Path)):
            self._emit("[SAVE]", f"Saved to: {m}")
        else:
            self._emit("[SAVE]", str(m))

    # ---------- FPS + Time Formatting ----------
    def format_time_fps(self, frame_count, prev_time, start_time,
                        fps_video=None, total_frames=None, source_type="video"):

        now = time.time()
        instantaneous = 1 / (now - prev_time + 1e-6)

        # Smooth handled per-thread
        self.prev_fps_smooth = getattr(self, "prev_fps_smooth", instantaneous)
        fps_smooth = 0.9 * self.prev_fps_smooth + 0.1 * instantaneous
        self.prev_fps_smooth = fps_smooth

        prev_time = now

        # Time formatting
        if source_type == "video" and fps_video and total_frames:
            elapsed = int(frame_count / fps_video)
            total = int(total_frames / fps_video)
            time_str = f"{elapsed//60:02d}:{elapsed%60:02d}/{total//60:02d}:{total%60:02d}"
        else:
            elapsed = int(now - start_time)
            time_str = f"{elapsed//60:02d}:{elapsed%60:02d}"

        return fps_smooth, time_str, prev_time, None

    # ---------- Update per-source frame status line ----------
    def update_frame_status(self, line_number, display_name,
                            frame_count, fps_smooth, counts, time_str):

        # always show focus class counts
        focus_part = [f"{cls}:{counts.get(cls,0)}" for cls in FOCUS_CLASSES]

        parts = focus_part

        # context class support
        if CONTEXT_CLASSES:
            obj_total = sum(counts.get(c, 0) for c in CONTEXT_CLASSES)
            parts.append(f"Objects:{obj_total}")

        text = (
            f"[{display_name}] "
            f"Frames:{frame_count} | FPS:{fps_smooth:.1f} | "
            + " | ".join(parts)
            + f" | Time:{time_str}"
        )
        self.update_line(line_number, text)

    # ---------- Model Selection ----------
    def prompt_model_selection(self, runs_dir, exclude_test=False):
        model_dirs = sorted(
            [
                d for d in runs_dir.iterdir()
                if d.is_dir() and (not exclude_test or d.name.lower() != "test")
            ],
            reverse=True,
        )

        if not model_dirs:
            self.missing_weights(runs_dir)
            return None

        # Show list with pretty formatting
        self._emit("[MODEL]", f"{len(model_dirs)} models found in runs folder:")
        for i, d in enumerate(model_dirs, start=1):
            print(f" {i}. {d.name}")

        # Input loop
        while True:
            try:
                choice = input(f"Select a model run (1-{len(model_dirs)}): ").strip()
            except KeyboardInterrupt:
                self.warn("Model selection interrupted by user.")
                return None

            if choice.isdigit():
                choice = int(choice)
                if 1 <= choice <= len(model_dirs):
                    return model_dirs[choice - 1]

            self.warn("Invalid selection, try again.")

    # ---------- Model Initialization Messages ----------
    def model_init(self, weights_path):
        """Announces which model is being initialized (shortened path)."""
        weights_path = Path(weights_path)

        # Try to show "runs/<run>/<weights>" for clarity
        try:
            idx = weights_path.parts.index("runs")
            short = Path(*weights_path.parts[idx:idx+3])
        except ValueError:
            short = weights_path.parent.parent  # fallback

        self._emit("[INFO]", f"Initializing model: {short}")

    def model_fail(self, e):
        """Unified error message for model loading failure."""
        self._emit("[ERROR]", f"Could not initialize model: {e}")

    # ---------- Capture + inference errors ----------
    def open_capture_fail(self, src):
        self.error(f"Could not open source: {src}")

    def read_frame_fail(self, src):
        self.error(f"Could not read frame from {src}")

    def inference_fail(self, src, e):
        self.error(f"Inference failed for {src}: {e}")

    # ---------- Writer / save messages ----------
    def recording_initialized(self, ts):
        self._emit("[INFO]", f"Recording initialized at {ts}")

    def save_measurements(self, base_dir, files):
        """Print measurement summary."""
        base_dir = Path(base_dir)

        # Try to shorten path (measurements/<timestamp>/...)
        try:
            idx = base_dir.parts.index("measurements")
            short = Path(*base_dir.parts[idx:])
        except ValueError:
            short = base_dir

        self._emit("[SAVE]", f'Measurements saved to: "{short}"')

        for f in files:
            print(f" - {Path(f).name}")

    # ---------- Writer lifecycle ----------
    def register_writer(self, raw_name, writer, cap, source_type, out_file, display_name=None):

        safe_name = (
            re.sub(r"[^\w\-]", "_", Path(out_file.name).stem)
            + out_file.suffix
        )

        self.active_writers[safe_name] = {
            "writer": writer,
            "cap": cap,
            "source_type": source_type,
            "out_file": out_file,
            "source_name": raw_name,
            "display_name": display_name or raw_name
        }

        timestamp = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
        self.recording_initialized(timestamp)

        return safe_name

    def safe_release_writer(self, name):
        """Release writer and its capture safely."""
        entry = self.active_writers.get(name)
        if not entry:
            return

        writer = entry["writer"]
        cap = entry["cap"]

        if writer:
            try: writer.release()
            except Exception: pass

        if cap:
            try: cap.release()
            except Exception: pass

        self.active_writers.pop(name, None)

    def release_all_writers(self):
        """Release all active writers."""
        for name in list(self.active_writers.keys()):
            self.safe_release_writer(name)

    # ---------- Global shutdown / UI messages ----------
    def stop_signal_received(self, single_thread=True):
        msg = (
            "Stop signal received. Terminating pipeline..."
            if single_thread else
            "Stop signal received. Terminating pipelines..."
        )
        self.exit(msg)

    def skip_source(self, src):
        self.warn(f"Skipping source: {src}")

    def no_sources(self):
        self.warn("No valid sources provided.")

    def all_threads_terminated(self):
        self.exit("All detection threads safely terminated.")
