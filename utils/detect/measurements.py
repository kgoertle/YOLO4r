# utils/detect/measurements.py
import csv, math, yaml
import math
from pathlib import Path
from datetime import datetime, timedelta, timezone
from collections import defaultdict

# --- Import class lists + canonical config path ---
from utils.detect.classes_config import FOCUS_CLASSES, CONTEXT_CLASSES
from utils.detect.paths import MEASURE_CONFIG_YAML, CONFIGS_DIR

# ---------- CONFIG ----------
class MeasurementConfig:
    """Central configuration for all measurement parameters."""

    DEFAULTS = {
        "snapshot_interval_sec": 5,       # frequency of snapshot counts
        "avg_group_size": 3,              # grouping for average_counts.csv
        "interval_sec": 5,                # interval for aggregator
        "session_sec": 10,                # not used, but reserved
        "interaction_timeout_sec": 2.0,   # gap before ending an interaction
        "overlap_threshold": 0.1          # IoU threshold
    }

    def __init__(self, config_path=None):
        CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

        self.config_path = Path(config_path) if config_path else MEASURE_CONFIG_YAML

        # If missing → create with defaults
        if not self.config_path.exists():
            with open(self.config_path, "w") as f:
                yaml.safe_dump(self.DEFAULTS, f)

        # Load config
        with open(self.config_path, "r") as f:
            data = yaml.safe_load(f) or {}

        # Apply defaults if missing
        for k, v in self.DEFAULTS.items():
            setattr(self, k, data.get(k, v))

# ---------- COUNTING UTILITY ----------
def compute_counts_from_boxes(boxes, names):
    """Compute per-frame counts based on model output."""
    counts = {cls: 0 for cls in FOCUS_CLASSES}

    if CONTEXT_CLASSES:
        # Track full detail: each context class AND total
        for c in CONTEXT_CLASSES:
            counts[c] = 0
        counts["OBJECTS"] = 0

    for b in boxes:
        cls = names.get(b[5])
        if cls in FOCUS_CLASSES:
            counts[cls] += 1
        elif CONTEXT_CLASSES and cls in CONTEXT_CLASSES:
            counts[cls] += 1           
            counts["OBJECTS"] += 1    

    return add_ratio_to_counts(counts)

def add_ratio_to_counts(counts):
    """Add human-readable ratios only when context classes are enabled."""
    if not CONTEXT_CLASSES:
        return counts

    # Use only focus classes for ratio
    focus_values = [int(counts.get(cls, 0)) for cls in FOCUS_CLASSES]

    non_zero = [v for v in focus_values if v != 0]
    if len(non_zero) > 1:
        gcd_val = non_zero[0]
        for v in non_zero[1:]:
            gcd_val = math.gcd(gcd_val, v)
        if gcd_val > 1:
            focus_values = [v // gcd_val for v in focus_values]

    counts["RATIO"] = ":".join(str(v) for v in focus_values)
    return counts

# ---------- Counter ----------
# -- counts & averaged counts --
class Counter:
    def __init__(self, out_folder=None, config=None, start_time=None):
        self.out_folder = Path(out_folder) if out_folder else None
        self.config = config or MeasurementConfig()
        self.start_time = start_time

        self.snapshot_buffer = []
        self.last_snapshot = None
        self.creation_ref = None
        self.group_number = 1

    def update_counts(self, boxes, names, timestamp=None):
        """Record a count snapshot if enough time passed."""
        now = timestamp or datetime.now()

        if self.last_snapshot and (now - self.last_snapshot).total_seconds() < self.config.snapshot_interval_sec:
            return

        counts = compute_counts_from_boxes(boxes, names)

        # Convert system timestamp → video timestamp
        if self.start_time:
            if not self.creation_ref:
                self.creation_ref = now
            elapsed = (now - self.creation_ref).total_seconds()
            video_ts = self.start_time + timedelta(seconds=elapsed)
        else:
            video_ts = now

        self.snapshot_buffer.append((video_ts, counts))
        self.last_snapshot = now

    # ---- Helper: compute averages ----
    def _compute_averages(self):
        if not self.snapshot_buffer:
            return []

        group_size = self.config.avg_group_size
        averages = []

        for i in range(0, len(self.snapshot_buffer), group_size):
            block = self.snapshot_buffer[i:i + group_size]

            summed = defaultdict(float)
            for _, c in block:
                for cls, val in c.items():
                    if cls not in ("RATIO",):
                        summed[cls] += val

            divisor = len(block)
            avg_counts = {cls: summed[cls] / divisor for cls in summed}
            avg_counts = add_ratio_to_counts(avg_counts)

            midpoint = block[0][0] + (block[-1][0] - block[0][0]) / 2
            averages.append({
                "Group": self.group_number,
                "Time": midpoint.strftime("%H:%M:%S"),
                "Counts": avg_counts
            })
            self.group_number += 1

        return averages

    def save_results(self):
        """Save counts.csv and average_counts.csv."""
        if not self.out_folder:
            return None

        self.out_folder.mkdir(parents=True, exist_ok=True)
        saved = []

        all_cols = (
            FOCUS_CLASSES +
            (["OBJECTS"] if CONTEXT_CLASSES else []) +
            (["RATIO"] if CONTEXT_CLASSES else [])
        )

        # SNAPSHOT CSV
        f_snap = self.out_folder / "counts.csv"
        with open(f_snap, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["TIME"] + all_cols)
            for ts, c in self.snapshot_buffer:
                row = [ts.strftime("%H:%M:%S")] + [
                    c.get(cls, "") for cls in all_cols
                ]
                w.writerow(row)
        saved.append(f_snap)

        # AVERAGE CSV
        averages = self._compute_averages()
        if averages:
            f_avg = self.out_folder / "average_counts.csv"
            with open(f_avg, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["GROUP", "TIME"] + all_cols)
                for a in averages:
                    row = [a["Group"], a["Time"]] + [
                        a["Counts"].get(cls, "") for cls in all_cols
                    ]
                    w.writerow(row)
            saved.append(f_avg)

        return saved

# ---------- INTERACTIONS ----------
class Interactions:
    def __init__(self, out_folder=None, config=None, start_time=None):
        self.out_folder = Path(out_folder) if out_folder else None
        self.config = config or MeasurementConfig()
        self.start_time = start_time

        self.active = {}     # active interactions
        self.records = []    # completed interactions
        self.ref_time = None # reference system timestamp

    def _normalize(self, dt):
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

    def _video_time(self, ts):
        if not self.start_time or not self.ref_time:
            return ts

        delta = (self._normalize(ts) - self._normalize(self.ref_time)).total_seconds()
        return self.start_time + timedelta(seconds=delta)

    @staticmethod
    def _overlap(a, b, threshold):
        """Compute IoU-style overlap."""
        ax1, ay1, ax2, ay2 = a[0]-a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2
        bx1, by1, bx2, by2 = b[0]-b[2]/2, b[1]-b[3]/2, b[0]+b[2]/2, b[1]+b[3]/2

        iw = max(0, min(ax2, bx2) - max(ax1, bx1))
        ih = max(0, min(ay2, by2) - max(ay1, by1))
        if iw == 0 or ih == 0:
            return False
        inter = iw * ih
        union = a[2]*a[3] + b[2]*b[3] - inter
        return inter / union > threshold

    def process_frame(self, boxes, names, ts):
        if not self.ref_time:
            self.ref_time = ts

        video_ts = self._video_time(ts)

        birds = [b for b in boxes if names.get(b[5]) in FOCUS_CLASSES]
        objs  = [b for b in boxes if CONTEXT_CLASSES and names.get(b[5]) in CONTEXT_CLASSES]

        active_now = set()

        if CONTEXT_CLASSES:
            for b in birds:
                bname = names.get(b[5])
                for o in objs:
                    oname = names.get(o[5])
                    if b is o:
                        continue
                    if self._overlap(b, o, self.config.overlap_threshold):
                        pair = (bname, oname)
                        active_now.add(pair)
                        self._activate(pair, video_ts)
        else:
            # bird-bird interactions
            for i, b1 in enumerate(birds):
                for j, b2 in enumerate(birds):
                    if j <= i:
                        continue
                    name1, name2 = names.get(b1[5]), names.get(b2[5])
                    if name1 == name2:
                        continue
                    if self._overlap(b1, b2, self.config.overlap_threshold):
                        pair = tuple(sorted((name1, name2)))
                        active_now.add(pair)
                        self._activate(pair, video_ts)

        self._finalize_inactive(active_now, video_ts)

    def _activate(self, pair, ts):
        if pair not in self.active:
            self.active[pair] = {"start": ts, "last": ts}
        else:
            self.active[pair]["last"] = ts

    def _finalize_inactive(self, active_now, ts):
        """Finish interactions which have timed out."""
        ended = []
        for pair, info in self.active.items():
            if pair not in active_now and (ts - info["last"]).total_seconds() >= self.config.interaction_timeout_sec:
                self._record(pair, info["start"], info["last"])
                ended.append(pair)
        for p in ended:
            del self.active[p]

    def finalize(self):
        """Force-finish all active interactions at end of video."""
        for pair, info in self.active.items():
            self._record(pair, info["start"], info["last"])
        self.active.clear()
        return self.records

    def _record(self, pair, start, end):
        dur = round((end - start).total_seconds(), 2)
        if dur <= 0:
            return

        if CONTEXT_CLASSES:
            entry = {
                "TIME0": start.strftime("%H:%M:%S"),
                "TIME1": end.strftime("%H:%M:%S"),
                "FOCUS": pair[0],
                "CONTEXT": pair[1],
                "DURATION": dur,
            }
        else:
            entry = {
                "TIME0": start.strftime("%H:%M:%S"),
                "TIME1": end.strftime("%H:%M:%S"),
                "CLASS1": pair[0],
                "CLASS2": pair[1],
                "DURATION": dur,
            }

        self.records.append(entry)

    def save_results(self):
        if not self.records or not self.out_folder:
            return None

        self.out_folder.mkdir(parents=True, exist_ok=True)
        out_file = self.out_folder / "interactions.csv"

        if CONTEXT_CLASSES:
            headers = ["TIME0", "TIME1", "FOCUS", "CONTEXT", "DURATION"]
        else:
            headers = ["TIME0", "TIME1", "CLASS1", "CLASS2", "DURATION"]

        with open(out_file, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=headers)
            w.writeheader()
            for r in sorted(self.records, key=lambda x: x["TIME0"]):
                w.writerow(r)

        return out_file

# ---------- Aggregator ----------
# -- interval aggregation & summary --
class Aggregator:
    def __init__(self, out_folder, config=None, start_time=None):
        self.out_folder = Path(out_folder)
        self.config = config or MeasurementConfig()
        self.start_time = start_time

        self.frame_data = []  # (timestamp, counts)
        self.intervals = []

    def push_frame_data(self, timestamp, current_boxes_list=None, names=None, counts_dict=None):
        if counts_dict is None and current_boxes_list and names:
            counts_dict = compute_counts_from_boxes(current_boxes_list, names)

        counts_dict = dict(counts_dict)
        counts_dict.pop("RATIO", None)

        self.frame_data.append((timestamp, counts_dict))

    def aggregate_intervals(self):
        if not self.frame_data:
            return []

        self.frame_data.sort(key=lambda x: x[0])
        intervals = []
        interval_counts = defaultdict(list)

        interval_start = self.frame_data[0][0]
        interval_end = interval_start + timedelta(seconds=self.config.interval_sec)

        for ts, counts in self.frame_data:
            if ts >= interval_end:
                intervals.append(self._finalize_interval(interval_start, interval_counts))
                interval_counts.clear()
                interval_start = interval_end
                interval_end = interval_start + timedelta(seconds=self.config.interval_sec)

            for cls, val in counts.items():
                interval_counts[cls].append(val)

        if interval_counts:
            intervals.append(self._finalize_interval(interval_start, interval_counts))

        self.intervals = intervals
        return intervals

    def _finalize_interval(self, start_ts, interval_counts):
        summed = {cls: sum(vals) for cls, vals in interval_counts.items()}

        if CONTEXT_CLASSES:
            obj_total = sum(summed.get(c, 0) for c in CONTEXT_CLASSES)
            summed = {cls: summed.get(cls, 0) for cls in FOCUS_CLASSES}
            summed["OBJECTS"] = obj_total

        summed = add_ratio_to_counts(summed)

        midpoint = start_ts + timedelta(seconds=self.config.interval_sec / 2)
        return {
            "TIME": midpoint.strftime("%H:%M:%S"),
            "Counts": summed
        }

    def save_interval_results(self):
        intervals = self.aggregate_intervals()
        if not intervals:
            return None

        self.out_folder.mkdir(parents=True, exist_ok=True)
        out_file = self.out_folder / "interval_results.csv"

        all_cols = (
            FOCUS_CLASSES +
            (["OBJECTS"] if CONTEXT_CLASSES else []) +
            (["RATIO"] if CONTEXT_CLASSES else [])
        )

        with open(out_file, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["TIME"] + all_cols)
            for iv in intervals:
                row = [iv["TIME"]] + [
                    iv["Counts"].get(cls, "") for cls in all_cols
                ]
                w.writerow(row)

        return out_file

    def save_session_summary(self):
        if not self.frame_data:
            return None

        session_totals = defaultdict(float)
        session_rates = defaultdict(list)

        for _, counts in self.frame_data:
            for cls, val in counts.items():
                session_totals[cls] += val
                session_rates[cls].append(val / self.config.interval_sec)

        # Merge context → OBJECTS
        if CONTEXT_CLASSES:
            obj_total = sum(session_totals.pop(c, 0) for c in CONTEXT_CLASSES)
            session_totals["OBJECTS"] = obj_total

            obj_rates = []
            for c in CONTEXT_CLASSES:
                obj_rates.extend(session_rates.pop(c, []))
            session_rates["OBJECTS"] = obj_rates or [0.0]

        focus_total = sum(session_totals.get(cls, 0) for cls in FOCUS_CLASSES) or 1.0

        summary_rows = []
        for cls, total in session_totals.items():
            rates = session_rates.get(cls, [])
            mean_rate = sum(rates) / len(rates) if rates else 0.0
            std_dev = (
                math.sqrt(sum((r - mean_rate) ** 2 for r in rates) / len(rates))
                if rates else 0.0
            )

            if cls in FOCUS_CLASSES:
                prop = total / focus_total
            else:
                prop = "n/a"

            summary_rows.append({
                "CLASS": cls,
                "TOTAL_COUNT": round(total, 3),
                "AVG_RATE": round(mean_rate, 3),
                "STD_DEV": round(std_dev, 3),
                "PROP": prop if isinstance(prop, str) else round(prop, 3)
            })

        # Save
        out_file = self.out_folder / "session_summary.csv"
        self.out_folder.mkdir(parents=True, exist_ok=True)

        with open(out_file, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["CLASS","TOTAL_COUNT","AVG_RATE","STD_DEV","PROP"])
            w.writeheader()
            for row in summary_rows:
                w.writerow(row)

        return out_file
