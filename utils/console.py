# utils/console.py
import os
import re
import time
import logging
import threading
from pathlib import Path
from datetime import datetime
from turtle import width
from ultralytics.utils import LOGGER

# ------------- COLORS -------------
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
ITALIC = "\033[3m"

RED = "\033[31m"
BRIGHT_RED = "\033[91m"

GREEN = "\033[32m"
BRIGHT_GREEN = "\033[92m"

YELLOW = "\033[33m"
BRIGHT_YELLOW = "\033[93m"

BLUE = "\033[34m"
BRIGHT_BLUE = "\033[94m"
BLUE_006B = "\033[38;5;33m"

MAGENTA = "\033[35m"
BRIGHT_MAGENTA = "\033[95m"

CYAN = "\033[36m"
BRIGHT_CYAN = "\033[96m"

WHITE = "\033[37m"

# ------------- FORMATTING HELPERS -------------
def fmt_info(msg):   return f"{BRIGHT_CYAN}{BOLD}info{RESET}: {WHITE}{msg}{RESET}"
def fmt_model(msg):  return f"{BRIGHT_YELLOW}{BOLD}model{RESET}: {WHITE}{msg}{RESET}"
def fmt_dataset(msg):return f"{BLUE}{BOLD}dataset{RESET}: {WHITE}{msg}{RESET}"
def fmt_train(msg):  return f"{GREEN}{BOLD}train{RESET}: {WHITE}{msg}{RESET}"
def fmt_exit(msg):   return f"{BRIGHT_RED}{BOLD}exit{RESET}: {WHITE}{msg}{RESET}"
def fmt_warn(msg):   return f"{YELLOW}{BOLD}warn{RESET}: {WHITE}{msg}{RESET}"
def fmt_error(msg):  return f"{RED}{BOLD}error{RESET}: {WHITE}{msg}{RESET}"
def fmt_save(msg):   return f"{BRIGHT_GREEN}{BOLD}save{RESET}: {WHITE}{msg}{RESET}"
def fmt_source(txt): return f"{BLUE_006B}{BOLD}{txt}{RESET}"
def fmt_label(txt):  return f"{WHITE}{BOLD}{txt}{RESET}"
def fmt_header(txt): return f"{WHITE}{BOLD}{txt}{RESET}"
def fmt_num(n):      return f"{WHITE}{BOLD}{n}{RESET}"
def fmt_path(path: str | Path) -> str: return f"{BRIGHT_MAGENTA}{BOLD}{path}{RESET}"
def fmt_bold(txt: str | Path) -> str: return f"{WHITE}{BOLD}{txt}{RESET}"

# ------------- TERMINAL HELPERS -------------
def clear_terminal():
    os.system("cls" if os.name == "nt" else "clear")

def print_divider(width=120):
    print("-" * width)

def print_training_header_static():
    print("\n" + "-" * 120 + "\n")

# ------------- ULTRALYTICS TRAINING LOG FILTERING / PATCHING -------------
class UltralyticsFilter(logging.Filter):
    noise_patterns = [
        r"Ultralytics.*Python",
        r"New .* available",
        r"engine/trainer",
        r"Transferred .* items",
        r"optimizer:",
        r"anchors:",
        r"Scanning .*labels",
        r"duplicate labels removed",
        r"ignoring corrupt",
        r"Freezing layer",
        r"summary:",
        r"Layers:",
        r"Model.*parameters",
        r"Using 0 dataloader workers",
        r"Image sizes",
        r"Creating cache",
        r"YOLO.*layers",
    ]

    def filter(self, record):
        msg = record.getMessage()
        return not any(re.search(p, msg) for p in self.noise_patterns)

def quiet_ultralytics_logs():
    ul = logging.getLogger("ultralytics")
    ul.setLevel(logging.INFO)
    ul.addFilter(UltralyticsFilter())

def quiet_wandb_logs():
    logging.getLogger("wandb").setLevel(logging.ERROR)


_original_info = LOGGER.info

def _is_epoch_line(msg: str) -> bool:
    return ("Epoch" in msg and "|" in msg) or ("%|" in msg)

def patched_info(msg, *a, **k):
    if getattr(patched_info, "interrupted", False):
        return

    lower = msg.lower()
    if "exit" in lower or "interrupt" in lower or "error" in lower:
        return _original_info(msg, *a, **k)

    if _is_epoch_line(msg):
        print_training_header()
        return _original_info(msg, *a, **k)

    if msg.strip() == "-" * 120:
        return

    suppress = (
        msg.startswith("train:")
        or msg.startswith("val:")
        or "Overriding model.yaml" in msg
        or "Fast image access" in msg
        or "ultralytics.nn" in msg
        or "Conv " in msg
        or "C2" in msg
        or "C3" in msg
        or "SPPF" in msg
        or "Concat" in msg
        or "Upsample" in msg
        or "parameters" in msg
        or msg.strip().startswith("from")
    )
    if suppress:
        return

    return _original_info(msg, *a, **k)

def apply_ultralytics_patch():
    LOGGER.info = patched_info

def remove_ultralytics_patch():
    LOGGER.info = _original_info

# ------------- TRAINING HEADER + FOOTER -------------
def print_training_header():
    clear_terminal()
    print(fmt_bold("YOLO4r Training"))
    print("----------------\n")

def print_training_footer(model_source, dataset_name, batch_size, workers, epochs):
    print(fmt_model(f"Model initializing: {fmt_path(model_source)}"))
    print(fmt_dataset(fmt_bold(dataset_name)))
    print(fmt_train(
        f"{WHITE}{BOLD}Epochs -{RESET} {epochs}    "
        f"{WHITE}{BOLD}Batch -{RESET} {batch_size}    "
        f"{WHITE}{BOLD}Workers -{RESET} {workers}"
    ))

    print()
    print_divider()
    print()

# -------------------------- UNIFIED CONSOLE UI CLASS --------------------------
class Console:
    """
    Unified UI for training + detection.
    Replaces Printer entirely.
    """

    # ------------- CONSTRUCTOR -------------
    def __init__(self, total_sources: int = 0):
        self.total_sources = total_sources
        self.lock = threading.Lock()

        # Per-source detection UI state
        self.sources = []
        for _ in range(total_sources):
            self.sources.append(
                {
                    "name": None,
                    "frame_count": 0,
                    "fps": 0.0,
                    "time_str": "--:--",
                    "counts": {},
                    "completed": False,
                    "source_type": None,  
                    "eta": "--",          
                }
            )

        self.log_lines = []
        cols, rows = self._get_term_size()
        self.max_logs = max(15, rows // 2)

        # FPS smoothing bucket
        self._fps_smooth = {}
        self.last_redraw_time = 0.0
        self.redraw_interval = 0.10
        
        self.model_name = None

        self.freeze_ui = False
        self.final_exit_events = []
        self.final_save_blocks = []
        self.in_shutdown = False

        self._classes_loaded_once = False
        self._recording_initialized_once = False
        self.all_classes = []

        # Active video writers
        self.active_writers = {}

    # ------------- INTERNAL HELPERS -------------
    def _get_term_size(self):
        try:
            size = os.get_terminal_size()
            return size.columns, size.lines
        except OSError:
            return 120, 40

    def _append_log(self, formatted: str):
        """formatted is already a fully formatted string (info:, warn:, etc.)"""
        with self.lock:
            self.log_lines.append(formatted)
            if len(self.log_lines) > self.max_logs:
                self.log_lines = self.log_lines[-self.max_logs:]
        self._maybe_redraw()

    # ------------- PUBLIC LOGGING API (DETECT + TRAIN) -------------
    def info(self, msg):   self._append_log(fmt_info(msg))
    def warn(self, msg):   self._append_log(fmt_warn(msg))
    def error(self, msg):  self._append_log(fmt_error(msg))
    def exit(self, msg):   self._append_log(fmt_exit(msg))
    def save(self, msg):   self._append_log(fmt_save(msg))

    # ------------- MODEL / WEIGHTS HELPERS -------------
    def model_fail(self, e):
        self.error(f"Could NOT initialize model: {e}")

    def missing_weights(self, runs_dir):
        runs_dir = Path(runs_dir)
        self.error("YOLO model weights NOT found.")
        self.warn(f'Expected to find at least one model directory inside: {fmt_path(runs_dir)}')
        self.warn("Run training first OR copy a model into the runs folder.")
        self.exit("Detection aborted due to missing weights.")

    def model_init(self, weights_path):
        weights_path = Path(weights_path)
        try:
            idx = weights_path.parts.index("runs")
            short = Path(*weights_path.parts[idx:idx+3])
        except ValueError:
            short = weights_path
        self._append_log(fmt_model(f"Initializing model: {fmt_path(short)}"))

    # ------------- MEASUREMENT SAVE WRAPPER -------------
    def save_measurements(self, base_dir, files):
        base_dir = Path(base_dir)
        try:
            source_name = base_dir.parent.parent.name
        except Exception:
            source_name = base_dir.name
        title = f"Measurements for {source_name}"
        self.add_final_save_block(title, base_dir, files)

    # ------------- VIDEO WRITER REGISTRATION -------------
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
        self.recording_initialized(ts)
        return safe_name

    def recording_initialized(self, ts: str):
        if not self._recording_initialized_once:
            self._recording_initialized_once = True
            self.info(f"Recording initialized at {ts}")

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

    # ------------- DETECTION UI -------------
    def set_model_name(self, name: str):
        self.model_name = name

    def _format_class_lines(self, counts: dict, width: int):
        from .detect.classes_config import FOCUS_CLASSES, CONTEXT_CLASSES

        display_entries = []

        # Are we in "placeholder" mode? (i.e., completed source with "--" everywhere)
        is_placeholder = bool(counts) and all(str(v) == "--" for v in counts.values())

        # ----- FOCUS CLASSES -----
        for c in FOCUS_CLASSES:
            if is_placeholder:
                val = "--"
            else:
                val = counts.get(c, 0)
            display_entries.append(f"{fmt_label(c)}: {val}")

        # ----- OBJECTS SUM -----
        if CONTEXT_CLASSES:
            if is_placeholder:
                obj_total = "--"
            else:
                obj_total = sum(counts.get(c, 0) for c in CONTEXT_CLASSES)
            display_entries.append(f"{fmt_label('OBJECTS')}: {obj_total}")

        if not display_entries:
            return ["  (no detections yet)"]

        # ----- DYNAMIC COLUMN LAYOUT -----
        n = len(display_entries)

        if n <= 5:
            cols = 1
        elif n <= 8:
            cols = 2
        elif n <= 15:
            cols = 3 if width > 90 else 2
        else:
            cols = max(1, width // 18)

        max_len = max(len(e) for e in display_entries) + 4

        lines = []
        for i in range(0, n, cols):
            row = display_entries[i:i + cols]
            row = [r.ljust(max_len) for r in row]
            lines.append("  " + "".join(row).rstrip())

        return lines

    # ------------- TIME / FPS HELPERS (PORT FROM PRINTER) -------------
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

        eta_str = None
        if source_type == "video" and fps_video and total_frames:
            elapsed = (frame_count + 1) / float(fps_video)
            total = total_frames / float(fps_video)
            remaining = max(0.0, total - elapsed)

            e_m, e_s = divmod(int(elapsed), 60)
            t_m, t_s = divmod(int(total), 60)
            r_m, r_s = divmod(int(remaining), 60)

            time_str = f"{e_m:02d}:{e_s:02d}/{t_m:02d}:{t_s:02d}"

            eta_str = f"{r_m:02d}:{r_s:02d}"
        else:
            elapsed = int(now - start_time)
            e_m, e_s = divmod(elapsed, 60)
            time_str = f"{e_m:02d}:{e_s:02d}"

        return fps_smooth, time_str, now, eta_str

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

    # ------------- REDRAW -------------
    def _redraw_locked(self):
        width, _ = self._get_term_size()

        print("\033[2J\033[H", end="")

        # Title
        print(fmt_bold("YOLO4r Detection"))
        print("-" * width)
        print()

        if self.model_name:
            print(fmt_model(self.model_name))
            print()

        # Per-source sections
        for idx, src in enumerate(self.sources, start=1):
            nm = src["name"] or f"source{idx}"
            name_fmt = fmt_source(nm)

            if src["completed"]:
                header = (
                    f"{name_fmt}: "
                    f"{fmt_label('Frames')}: -- | "
                    f"{fmt_label('FPS')}: -- | "
                    f"{fmt_label('Time')}: -- | "
                    f"{fmt_label('ETA')}: --"
                )
            else:
                eta_display = src.get("eta", "--")
                show_eta = not (src.get("source_type") == "usb")

                if show_eta:
                    header = (
                        f"{name_fmt}: "
                        f"{fmt_label('Frames')}: {src['frame_count']} | "
                        f"{fmt_label('FPS')}: {src['fps']:.1f} | "
                        f"{fmt_label('Time')}: {src['time_str']} | "
                        f"{fmt_label('ETA')}: {WHITE}{eta_display}{RESET}"
                    )
                else:
                    header = (
                        f"{name_fmt}: "
                        f"{fmt_label('Frames')}: {src['frame_count']} | "
                        f"{fmt_label('FPS')}: {src['fps']:.1f} | "
                        f"{fmt_label('Time')}: {src['time_str']}"
                    )

            print(header)

            for ln in self._format_class_lines(src["counts"], width):
                print(ln)

            print()

        print("-" * width)
        print()

        for ln in self.log_lines[-self.max_logs:]:
            print(ln[:width])

        print("", flush=True)

    def _maybe_redraw(self):
        if self.freeze_ui or self.in_shutdown:
            return
        with self.lock:
            now = time.time()
            if now - self.last_redraw_time < self.redraw_interval:
                return
            self.last_redraw_time = now
            self._redraw_locked()

    # ------------- DETECTION UPDATE API -------------
    def update_frame_status(
        self, idx, display_name, frame_count, fps_smooth, counts, time_str, eta=None
    ):
        i = idx - 1
        if i < 0 or i >= self.total_sources:
            return

        with self.lock:
            src = self.sources[i]
            src["name"] = display_name
            src["frame_count"] = frame_count
            src["fps"] = fps_smooth
            src["time_str"] = time_str
            src["counts"] = dict(counts)
            if eta is not None:
                src["eta"] = eta
            if src.get("source_type") is None:
                src["source_type"] = "usb" if display_name.startswith("usb") else "video"

        self._maybe_redraw()

    # ------------- SOURCE COMPLETION -------------
    def mark_source_complete(self, idx):
        i = idx - 1
        if i < 0 or i >= self.total_sources:
            return

        with self.lock:
            src = self.sources[i]
            src["completed"] = True
            src["frame_count"] = "--"
            src["fps"] = "--"
            src["time_str"] = "--"
            src["eta"] = "--"

            from .detect.classes_config import FOCUS_CLASSES, CONTEXT_CLASSES

            ordered_classes = list(FOCUS_CLASSES) + list(CONTEXT_CLASSES)
            placeholder = {cls: "--" for cls in ordered_classes}
            src["counts"] = placeholder

        self.info(f"Source '{fmt_bold(src['name'])}' completed.")
        self._redraw_locked()

    # ------------- ERROR HELPERS -------------
    def open_capture_fail(self, src):
        self.error(f"Could not open source: {fmt_path(src)}")

    def read_frame_fail(self, src):
        self.error(f"Could not read frame from {fmt_path(src)}")

    def inference_fail(self, src, e):
        self.error(f"Inference failed for {fmt_path(src)}: {e}")

    # ------------- SAVE BLOCKS -------------
    def add_final_save_block(self, title, base_dir: Path, files: list):
        try:
            idx = base_dir.parts.index("measurements")
            short = Path(*base_dir.parts[idx:])
        except ValueError:
            short = base_dir

        block = []
        block.append(fmt_save(f"{fmt_bold(title)}"))
        block.append(fmt_save(f"Saved to: {fmt_path(short)}"))

        for f in files:
            name = Path(f).name
            block.append(f"      - {name}")

        self.final_save_blocks.append(block)

    # ------------- FINAL EXIT BLOCK -------------
    def render_final_exit_block(self):
        width, _ = self._get_term_size()
        print("\n" + "-" * width + "\n")

        for line in self.final_exit_events:
            print(fmt_exit(line))
        print()

        if self.model_name:
            print(fmt_model(self.model_name) + "\n")

        for block in self.final_save_blocks:
            for ln in block:
                print(ln)
            print()

        print(fmt_exit("All detection threads safely terminated.") + "\n")

    # ------------- STOP SIGNAL -------------
    def stop_signal_received(self, single_thread=True):
        msg = (
            "Stop signal received. Terminating pipeline..."
            if single_thread
            else "Stop signal received. Terminating pipelines..."
        )
        self.final_exit_events.append(msg)
        self.in_shutdown = True

    def all_threads_terminated(self):
        self.freeze_ui = False
        self.render_final_exit_block()

    # ------------- CLASS LOAD LOGGING -------------
    def classes_loaded(self, classes_list):
        if classes_list is None:
            return
        self.all_classes = classes_list
        if not self._classes_loaded_once:
            self.info(f"Loaded {len(classes_list)} classes: {classes_list}")
            self._classes_loaded_once = True

    # ------------- MODEL SELECTION -------------
    def prompt_model_selection(self, runs_dir, exclude_test=False):
        runs_dir = Path(runs_dir)
        dirs = sorted(
            [d for d in runs_dir.iterdir()
            if d.is_dir() and (not exclude_test or d.name.lower() != "test")],
            reverse=True
        )

        if not dirs:
            self.missing_weights(runs_dir)
            return None

        self.freeze_ui = True
        self.info(f"{len(dirs)} models found:")

        # --- Bold header, not colon ---
        print("\n" + fmt_bold("Available models") + ":")

        # --- Bold numbers only ---
        for i, d in enumerate(dirs, 1):
            print(f"   {fmt_bold(i)}. {d.name}")
        print()

        try:
            while True:
                try:
                    ch = input(f"Select model (1-{len(dirs)}) or Ctrl+C: ").strip()
                except KeyboardInterrupt:
                    print("\n" + fmt_exit("Model selection cancelled") + "\n")
                    return None

                if ch.isdigit():
                    ch = int(ch)
                    if 1 <= ch <= len(dirs):
                        return dirs[ch - 1]

                self.warn("Invalid selection.")

        finally:
            self.freeze_ui = False
            self._maybe_redraw()

# ------------- SHARED INSTANCE FOR TRAINING -------------
console = Console(total_sources=0)
