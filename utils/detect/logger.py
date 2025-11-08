import threading, os
from pathlib import Path
import re

class DetectionDashboard:
    def __init__(self, total_sources):
        self.total_sources = total_sources
        self.lock = threading.Lock()
        self.lines = [""] * total_sources
        term_height = 30  # fallback if os.get_terminal_size() fails
        try:
            term_height = os.get_terminal_size().lines
        except OSError:
            pass
        self.start_line = max(1, term_height - total_sources + 1)
        self.active_writers = {}

    # ----- DASHBOARD -----
    def update_line(self, line_number, text):
        with self.lock:
            if self.lines[line_number - 1] != text:
                self.lines[line_number - 1] = text
                print(f"\033[{self.start_line + line_number - 1};0H\033[K{text}", end="", flush=True)

    # ----- GENERAL LOGGING -----
    def log(self, message):
        with self.lock:
            print(f"\033[{self.start_line + self.total_sources};0H\033[K{message}", flush=True)

    # ----- SMOOTHING REPORT -----
    def report_smoothing(self, args, user_set_flags):
        for param in ['smooth', 'dist_thresh', 'max_history']:
            value = getattr(args, param)
            user_set = user_set_flags.get(param, False)
            self.log(f"[INFO] {param} set by {'user' if user_set else 'default'} to {value}")
        print()

    # ----- WRITER LOGIC -----
    def register_writer(self, raw_source_name, writer, cap, source_type, out_file):
        safe_name = re.sub(r'[^\w\-]', '_', Path(out_file.name).stem) + out_file.suffix
        entry = {
            'writer': writer,
            'cap': cap,
            'source_type': source_type,
            'out_file': out_file,
            'safe_name': safe_name
        }
        with self.lock:
            self.active_writers[raw_source_name] = entry

    def safe_release_writer(self, raw_source_name):
        with self.lock:
            entry = self.active_writers.pop(raw_source_name, None)
        if entry is None:
            return

        writer = entry.get('writer')
        cap = entry.get('cap')
        source_type = entry.get('source_type')
        out_file = entry.get('out_file')

        try:
            if source_type in ["usb", "video"] and cap is not None:
                cap.release()
            elif source_type == "picamera" and cap is not None:
                try: cap.stop()
                except: pass
        except Exception as e:
            self.log(f"[WARN] Error releasing capture for {raw_source_name}: {e}")

        if writer is not None:
            try:
                writer.release()
            except Exception as e:
                self.log(f"[WARN] Error releasing writer for {raw_source_name}: {e}")
            self.log(f"[SAVE] Detection results saved to: {out_file}")

    def release_all_writers(self):
        # make a copy to avoid mutating dict during iteration
        for raw_source_name in list(self.active_writers.keys()):
            self.safe_release_writer(raw_source_name)

    def save_results(aggregator, current_boxes_list, names, out_folder, dashboard=None):
        interval_file, session_file = aggregator.save_results(out_folder)
        if dashboard:
            dashboard.log(f"[SAVE] Interval results saved to: {interval_file}")
            dashboard.log(f"[SAVE] Session summary saved to: {session_file}")

        if current_boxes_list and names:
            try:
                log_bird_counts_summary(current_boxes_list, names, out_folder)
                if dashboard:
                    dashboard.log(f"[SAVE] Basic counts summary saved to: {out_folder}")
            except Exception as e:
                if dashboard:
                    dashboard.log(f"[ERROR] Failed to save basic counts CSV: {e}")
