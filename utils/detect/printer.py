# utils/detect/printer.py

import os
import re
import time
import threading
from pathlib import Path
from datetime import datetime

from .classes_config import FOCUS_CLASSES, CONTEXT_CLASSES


class Printer:
    def __init__(self, total_sources: int):
        self.total_sources = total_sources
        self.lock = threading.Lock()

        # Per-source state (one dict per source index)
        self.sources = []
        for _ in range(total_sources):
            self.sources.append(
                {
                    "name": None,        # display name (e.g., "usb0", "skylight")
                    "frame_count": 0,
                    "fps": 0.0,
                    "time_str": "--:--",
                    "counts": {},        # class -> count
                    "completed": False,
                }
            )

        # Active video writers for release
        self.active_writers = {}

        # Logging state
        self.log_lines = []       # list of strings like "[INFO] message"
        cols, rows = self._get_term_size()
        self.max_logs = max(15, rows // 2)  # number of log lines to show at bottom

        # FPS smoothing (single bucket — matches previous behavior)
        self._fps_smooth = {}

        # Redraw throttling
        self.last_redraw_time = 0.0
        self.redraw_interval = 0.10  # seconds

        # Determine model name.
        self.model_name = None

        # Terminal update helpers
        self.freeze_ui = False
        self.final_exit_events = []   # store exit messages
        self.final_save_blocks = []   # store save groups
        self.in_shutdown = False      # freeze redraws during final summary
        self._classes_loaded_once = False
        self._recording_logged_sources = set()
        self._recording_initialized_once = False
        

    # --------- Internal Helpers ---------
    def _get_term_size(self):
        """Best-effort terminal size detection."""
        try:
            size = os.get_terminal_size()
            return size.columns, size.lines
        except OSError:
            return 120, 40

    def _append_log(self, tag: str, message: str):
        """Append a tagged log line to the log buffer."""
        # DO NOT include [MODEL] in the log lines — only the top UI sections
        line = f"{tag} {message}"

        with self.lock:
            self.log_lines.append(line)
            if len(self.log_lines) > self.max_logs:
                self.log_lines = self.log_lines[-self.max_logs:]
        self._maybe_redraw()

    # ---- Compute FPS & Time Strings ----
    def _format_time_str(
        self,
        frame_count: int,
        prev_time: float,
        start_time: float,
        fps_video: float = None,
        total_frames: int = None,
        source_type: str = "video",
        source_idx: int = 0,
    ):
        """
        Compute FPS (smoothed) and a time string.

        For video sources with known duration:
            "mm:ss/MM:SS | ETA:MM:SS"

        For camera sources:
            "mm:ss"
        """
        now = time.time()
        instantaneous = 1.0 / (now - prev_time + 1e-6)
        instantaneous = min(instantaneous, 60.0)

        prev_smooth = self._fps_smooth.get(source_idx, instantaneous)
        fps_smooth = 0.9 * prev_smooth + 0.1 * instantaneous
        fps_smooth = min(fps_smooth, 60.0)
        self._fps_smooth[source_idx] = fps_smooth

        # Fixed video with known total length
        eta_str = None
        if source_type == "video" and fps_video and total_frames:
            elapsed = frame_count / float(fps_video)
            total = total_frames / float(fps_video)
            remaining = max(0.0, total - elapsed)

            e_m, e_s = divmod(int(elapsed), 60)
            t_m, t_s = divmod(int(total), 60)
            r_m, r_s = divmod(int(remaining), 60)

            time_str = (
                f"{e_m:02d}:{e_s:02d}/{t_m:02d}:{t_s:02d} | "
                f"ETA:{r_m:02d}:{r_s:02d}"
            )
            eta_str = f"{r_m:02d}:{r_s:02d}"
        else:
            # Camera or unknown-length video
            elapsed = int(now - start_time)
            e_m, e_s = divmod(elapsed, 60)
            time_str = f"{e_m:02d}:{e_s:02d}"

        return fps_smooth, time_str, now, eta_str

    # ----- Build bullet lines for classes -----
        # ----- Build bullet lines for classes -----
    def _format_class_lines(self, counts: dict, width: int):
        # Always build a stable ordered list, even when counts is empty.
        display_entries = []

        # FOCUS classes first (in order)
        for cls in FOCUS_CLASSES:
            val = counts.get(cls, 0)
            display_entries.append(f"{cls}:{val}")

        # OBJECTS
        if CONTEXT_CLASSES:
            total_objects = counts.get("OBJECTS", 0)
            display_entries.append(f"OBJECTS:{total_objects}")

        # Safety: no entries means nothing to show
        if not display_entries:
            return ["  (no detections yet)"]

        # ---------- Determine columns ----------
        n = len(display_entries)

        # 1–5 → 1 column
        if n <= 5:
            cols = 1
        # 6–8 → 2 columns
        elif n <= 8:
            cols = 2
        # 9–15 → attempt 3 columns if terminal is wide enough
        elif n <= 15:
            cols = 3 if width > 90 else 2
        # 16+ → auto
        else:
            cols = max(1, width // 18)

        # Compute max label width
        max_len = max(len(e) for e in display_entries) + 4

        # Build rows
        lines = []
        for i in range(0, n, cols):
            row = display_entries[i:i + cols]
            padded = [e.ljust(max_len) for e in row]
            lines.append("  " + "".join(padded).rstrip())

        return lines

    # --- Control Terminal Locking ---
    def _redraw_locked(self):
        width, _ = self._get_term_size()

        # Clear screen & move cursor to top-left
        print("\033[H\033[J", end="")

        # Optional title
        title = "YOLO4r Detection"
        print(title[:width])
        print("-" * min(len(title), width))
        print()
        if self.model_name:
            print(f"[MODEL] {self.model_name}")
            print()

        # Render each source region
        for idx, src in enumerate(self.sources, start=1):
            name = src["name"] or f"source{idx}"
            frames = src["frame_count"]
            fps = src["fps"]
            time_str = src["time_str"]

            if src["completed"]:
                header = (
                    f"[{name}] Frames:-- | FPS:-- | Time:-- | ETA:--"
                )
            else:
                header = (
                    f"[{name}] Frames:{frames} | FPS:{fps:.1f} | Time:{time_str}"
                )

            print(header[:width])

            if src["completed"]:
                print("  F:-")
                print("  M:-")
                print("  OBJECTS:-")
            else:
                class_lines = self._format_class_lines(src["counts"], width)
                for ln in class_lines:
                    print(ln)

            print()  # blank line between sources

        # Separator above logs
        print("-" * width)
        print()

        # --- MODEL header for the log section itself ---
        if self.model_name:
            print(f"[MODEL] {self.model_name}")
            print()

        # Log lines
        if self.log_lines:
            for line in self.log_lines[-self.max_logs:]:
                print(line[:width])
        else:
            if self.log_lines:
                for line in self.log_lines[-self.max_logs:]:
                    print(line[:width])
            else:
                print("[INFO] Waiting for events...")

        # Ensure output is flushed
        print("", flush=True)

    def _maybe_redraw(self):
        """
        Throttled redraw entry point.
        """
        with self.lock:
            now = time.time()
            if now - self.last_redraw_time < self.redraw_interval:
                return
            self.last_redraw_time = now
            self._redraw_locked()
            
            if self.freeze_ui or self.in_shutdown:
                 return

    # ----------- API -----------
    # --- Generic logging ---

    def info(self, m: str):
        self._append_log("[INFO]", m)

    def warn(self, m: str):
        self._append_log("[WARN]", m)

    def error(self, m: str):
        self._append_log("[ERROR]", m)

    def exit(self, m: str):
        self._append_log("[EXIT]", m)

    def save(self, m):
        if isinstance(m, (str, Path)):
            self._append_log("[SAVE]", f"Saved to: {m}")
        else:
            self._append_log("[SAVE]", str(m))

    def save_measurements(self, base_dir, files):
        base_dir = Path(base_dir)
        try:
            source_name = base_dir.parent.parent.name
        except Exception:
            source_name = base_dir.name  

        title = f"Measurements for {source_name}"
        self.add_final_save_block(
            title=title,
            base_dir=base_dir,
            files=files,
        )

    # --- Log loaded classes ---
    def classes_loaded(self, classes_list):
        if classes_list is None:
            return  # nothing to log, do NOT crash
        if not self._classes_loaded_once:
            self.info(f"Loaded {len(classes_list)} classes: {classes_list}")
            self._classes_loaded_once = True

    # --- Store the active model name for UI display --
    def set_model_name(self, name: str):
        self.model_name = name

    # --- Model / weights helpers ---
    def missing_weights(self, runs_dir):
        runs_dir = Path(runs_dir)
        self.error("YOLO model weights NOT found.")
        self.warn(f'Expected to find at least one model directory inside: "{runs_dir}"')
        self.warn("Run training first OR copy a model into the runs folder.")
        self.exit("Detection aborted due to missing weights.")

    def model_init(self, weights_path):
        weights_path = Path(weights_path)
        try:
            idx = weights_path.parts.index("runs")
            short = Path(*weights_path.parts[idx:idx + 3])
        except ValueError:
            short = weights_path.parent.parent
        self.info(f"Initializing model: {short}")

    def model_fail(self, e: Exception):
        self.error(f"Could NOT initialize model: {e}")

    # --- FPS + timing API (called from VideoProcessor) ---

    # ----- Time Signature Helper ---
    def format_time_fps(
        self,
        frame_count,
        prev_time,
        start_time,
        fps_video=None,
        total_frames=None,
        source_type="video",
        source_idx=None,
    ):
        if source_idx is None:
            source_idx = 0

        fps_smooth, time_str, now, eta_str = self._format_time_str(
            frame_count=frame_count,
            prev_time=prev_time,
            start_time=start_time,
            fps_video=fps_video,
            total_frames=total_frames,
            source_type=source_type,
            source_idx=source_idx,
        )
        return fps_smooth, time_str, now, eta_str

    # --- UI update per frame ---
    def update_frame_status(
        self,
        line_number: int,
        display_name: str,
        frame_count: int,
        fps_smooth: float,
        counts: dict,
        time_str: str,
    ):
        idx = line_number - 1
        if idx < 0 or idx >= self.total_sources:
            return

        # Shallow copy counts to avoid mutation by caller
        counts_copy = dict(counts) if counts is not None else {}

        with self.lock:
            src = self.sources[idx]
            src["name"] = display_name
            src["frame_count"] = frame_count
            src["fps"] = fps_smooth
            src["time_str"] = time_str
            src["counts"] = counts_copy

        self._maybe_redraw()

    # --- Model selection prompt ---
    def prompt_model_selection(self, runs_dir, exclude_test=False):
        runs_dir = Path(runs_dir)

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

        # ---- Freeze UI redraws ----
        self.freeze_ui = True

        # Keep the header in logs
        self.info(f"{len(model_dirs)} models found in runs folder:")

        # Print model list BELOW UI
        print("\nAvailable models:")
        for i, d in enumerate(model_dirs, start=1):
            print(f"   {i}. {d.name}")
        print()

        try:
            while True:
                try:
                    choice = input(f"Select a model run (1-{len(model_dirs)}) or Ctrl+C to cancel: ").strip()
                except KeyboardInterrupt:
                    # ---- Graceful escape ----
                    print("\n[EXIT] Model selection cancelled by user.\n")
                    return None

                if choice.isdigit():
                    choice = int(choice)
                    if 1 <= choice <= len(model_dirs):
                        return model_dirs[choice - 1]

                self.warn("Invalid selection, try again.")

        finally:
            # Always restore UI redraw
            self.freeze_ui = False
            self._maybe_redraw()
   
    # --- Capture / inference errors ---
    def open_capture_fail(self, src):
        self.error(f"Could NOT open source: {src}")

    def read_frame_fail(self, src):
        self.error(f"Could NOT read frame from {src}")

    def inference_fail(self, src, e):
        self.error(f"Inference FAILED for {src}: {e}")

    # --- Time Initiatied ---
    def recording_initialized(self, ts: str):
        if not self._recording_initialized_once:
            self._recording_initialized_once = True
            self.info(f"Recording initialized at {ts}")

    # --- Register Writer --- 
    def register_writer(
        self,
        raw_name,
        writer,
        cap,
        source_type,
        out_file,
        display_name=None,
    ):
        safe_name = (
            re.sub(r"[^\w\-]", "_", Path(out_file.name).stem) +
            out_file.suffix
        )
        self.active_writers[safe_name] = {
            "writer": writer,
            "cap": cap,
            "source_type": source_type,
            "out_file": out_file,
            "source_name": raw_name,
            "display_name": display_name or raw_name,
        }
        ts = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
        src_key = display_name or raw_name
        self.recording_initialized(ts)
        return safe_name

    def safe_release_writer(self, name):
        entry = self.active_writers.get(name)
        if not entry:
            return
        try:
            entry["writer"].release()
        except Exception:
            pass
        try:
            entry["cap"].release()
        except Exception:
            pass
        self.active_writers.pop(name, None)

    def release_all_writers(self):
        for name in list(self.active_writers.keys()):
            self.safe_release_writer(name)

    # --- Shutdown ---

    def mark_source_complete(self, line_number):
        idx = line_number - 1
        if idx < 0 or idx >= self.total_sources:
            return

        with self.lock:
            src = self.sources[idx]
            src["completed"] = True
            src["frame_count"] = "--"
            src["fps"] = "--"
            src["time_str"] = "--"
            src["counts"] = {cls: "-" for cls in FOCUS_CLASSES}
            if CONTEXT_CLASSES:
                src["counts"]["OBJECTS"] = "-"

        self._append_log("[INFO]", f"Source '{src['name']}' completed.")
        self._maybe_redraw()

    def add_final_exit(self, msg: str):
        self.final_exit_events.append(f"[EXIT] {msg}")

    def add_final_save_block(self, title: str, base_dir: Path, files: list):
        try:
            idx = base_dir.parts.index("measurements")
            short = Path(*base_dir.parts[idx:])
        except ValueError:
            short = base_dir

        block = []
        block.append(f"[SAVE] {title}:")
        block.append(f'[SAVE] Measurements saved to: "{short}"')
        for f in files:
            name = Path(f).name
            block.append(f"      - {name}")
        self.final_save_blocks.append(block)

    def render_final_exit_block(self):
        width, _ = self._get_term_size()

        print("\n" + "-" * width + "\n")

        # --- EXIT messages first ---
        for line in self.final_exit_events:
            print(line)
        print()

        # --- MODEL header AFTER exit messages ---
        if self.model_name:
            print(f"[MODEL] {self.model_name}\n")

        # --- SAVE blocks per source ---
        for block in self.final_save_blocks:
            for line in block:
                print(line)
            print()

        print("[EXIT] All detection threads safely terminated.\n")

    def stop_signal_received(self, single_thread=True):
        msg = (
            "Stop signal received. Terminating pipeline..."
            if single_thread
            else "Stop signal received. Terminating pipelines..."
        )
        self.add_final_exit(msg)
        self.in_shutdown = True

    def skip_source(self, src):
        self.warn(f"Skipping source: {src}")

    def no_sources(self):
        self.warn("Valid source NOT provided.")

    def all_threads_terminated(self):
        self.render_final_exit_block()
