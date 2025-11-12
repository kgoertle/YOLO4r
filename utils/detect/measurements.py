import csv
import math
from pathlib import Path
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from utils.detect.classes_config import FOCUS_CLASSES, CONTEXT_CLASSES
import yaml

# ---------------- CONFIGURATION ----------------
class MeasurementConfig:
    """Central configuration for counts, intervals, and interactions."""
    DEFAULTS = {
        "snapshot_interval_sec": 5,
        "avg_group_size": 3,
        "interval_sec": 5,
        "session_sec": 10,
        "interaction_timeout_sec": 2.0,
        "overlap_threshold": 0.1
    }

    CONFIG_FOLDER = Path("configs")
    CONFIG_FILE = CONFIG_FOLDER / "measure_config.yaml"

    def __init__(self, config_file=None):
        self.CONFIG_FOLDER.mkdir(parents=True, exist_ok=True)
        config_path = Path(config_file) if config_file else self.CONFIG_FILE

        if not config_path.exists():
            with open(config_path, "w") as f:
                yaml.safe_dump(self.DEFAULTS, f)

        with open(config_path, "r") as f:
            data = yaml.safe_load(f) or {}

        for k, v in self.DEFAULTS.items():
            setattr(self, k, data.get(k, v))

# ---------------- UTILITY ----------------
def compute_counts_from_boxes(boxes, names):
    counts = {cls: 0 for cls in FOCUS_CLASSES}
    if CONTEXT_CLASSES:
        counts["OBJECTS"] = 0

    for b in boxes:
        cls = names.get(b[5])
        if cls in FOCUS_CLASSES:
            counts[cls] += 1
        elif CONTEXT_CLASSES and cls in CONTEXT_CLASSES:
            counts[cls] = counts.get(cls, 0) + 1
            counts["OBJECTS"] += 1

    # Simplified RATIO
    counts = add_ratio_to_counts(counts)
    return counts

def add_ratio_to_counts(counts):
    if CONTEXT_CLASSES:
        focus_values = [counts.get(cls, 0) for cls in FOCUS_CLASSES[:4]]
        # Convert to integers
        focus_values = [int(v) for v in focus_values]
        # Compute GCD of all non-zero values
        non_zero_values = [v for v in focus_values if v != 0]
        if non_zero_values:
            gcd_value = non_zero_values[0]
            for v in non_zero_values[1:]:
                gcd_value = math.gcd(gcd_value, v)
            if gcd_value > 1:
                focus_values = [v // gcd_value for v in focus_values]
        counts["RATIO"] = ":".join(str(v) for v in focus_values)
    return counts

# ---------------- COUNTER ----------------
class Counter:
    def __init__(self, out_folder=None, config=None, start_time=None):
        self.out_folder = Path(out_folder) if out_folder else None
        self.config = config or MeasurementConfig()
        self.start_time = start_time
        self.snapshot_buffer = []
        self.avg_group_counter = 1
        self.last_snapshot_time = None
        self.creation_ref = None

    def update_counts(self, current_boxes_list, names, timestamp=None):
        now = timestamp or datetime.now()
        if self.last_snapshot_time and (now - self.last_snapshot_time).total_seconds() < self.config.snapshot_interval_sec:
            return

        counts = compute_counts_from_boxes(current_boxes_list, names)
        counts = add_ratio_to_counts(counts)

        if self.start_time:
            if not self.creation_ref:
                self.creation_ref = now
            elapsed = (now - self.creation_ref).total_seconds()
            ts = self.start_time + timedelta(seconds=elapsed)
        else:
            ts = now

        self.snapshot_buffer.append((ts, counts))
        self.last_snapshot_time = now

    def flush_remaining(self):
        averages = []
        if not self.snapshot_buffer:
            return averages

        group_size = self.config.avg_group_size
        for i in range(0, len(self.snapshot_buffer), group_size):
            group = self.snapshot_buffer[i:i + group_size]
            summed = {cls: 0 for cls in FOCUS_CLASSES + (["OBJECTS"] if CONTEXT_CLASSES else [])}

            for _, counts in group:
                for cls, val in counts.items():
                    if CONTEXT_CLASSES and cls in CONTEXT_CLASSES:
                        summed["OBJECTS"] += val
                    elif cls in summed:
                        summed[cls] += val

            divisor = len(group) or 1
            avg_counts = {cls: summed.get(cls, 0)/divisor for cls in summed}
            avg_counts = add_ratio_to_counts(avg_counts)

            midpoint = (group[0][0] + (group[-1][0] - group[0][0])/2).strftime("%H:%M:%S")
            averages.append({"Group": self.avg_group_counter, "Time": midpoint, "Counts": avg_counts})
            self.avg_group_counter += 1

        return averages

    def save_results(self):
        if not self.out_folder:
            return None
        self.out_folder.mkdir(parents=True, exist_ok=True)

        all_cols = FOCUS_CLASSES + (["OBJECTS"] if CONTEXT_CLASSES else []) + (["RATIO"] if CONTEXT_CLASSES else [])
        saved_files = []

        # --- Snapshot CSV ---
        snapshot_file = self.out_folder / "counts.csv"
        with open(snapshot_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["TIME"] + all_cols)
            for ts, counts in self.snapshot_buffer:
                row = [counts.get(cls,0) if cls not in ["RATIO","OBJECTS"] else
                       counts.get("RATIO","") if cls=="RATIO" else
                       sum(counts.get(c,0) for c in CONTEXT_CLASSES) 
                       for cls in all_cols]
                writer.writerow([ts.strftime("%H:%M:%S")] + row)
        saved_files.append(snapshot_file)

        # --- Averaged CSV ---
        averages = self.flush_remaining()
        if averages:
            avg_file = self.out_folder / "average_counts.csv"
            with open(avg_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["INTERVAL","TIME"] + all_cols)
                for avg in averages:
                    row = [avg["Group"], avg["Time"]] + [
                        round(avg["Counts"].get(cls,0),3) if cls not in ["RATIO","OBJECTS"] else
                        avg["Counts"].get("RATIO","") if cls=="RATIO" else
                        round(avg["Counts"].get("OBJECTS",0),3)
                        for cls in all_cols
                    ]
                    writer.writerow(row)
            saved_files.append(avg_file)

        return saved_files

# ---------------- INTERACTIONS ----------------
class Interactions:
    def __init__(self, out_folder=None, config=None, start_time=None):
        self.out_folder = Path(out_folder) if out_folder else None
        self.config = config or MeasurementConfig()
        self.start_time = start_time
        self.active = {}
        self.records = []
        self.ref_time = None  # system time of first frame

    def _normalize_dt(self, dt):
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt

    def _to_video_time(self, current_ts):
        if not self.start_time or not self.ref_time:
            return current_ts
        ref = self._normalize_dt(self.ref_time)
        cur = self._normalize_dt(current_ts)
        creation = self._normalize_dt(self.start_time)
        elapsed = (cur - ref).total_seconds()
        return creation + timedelta(seconds=elapsed)

    def process_frame(self, boxes, names, timestamp):
        if not self.ref_time:
            self.ref_time = timestamp
        video_ts = self._to_video_time(timestamp)

        birds = [b for b in boxes if names.get(b[5]) in FOCUS_CLASSES]
        objects = [b for b in boxes if CONTEXT_CLASSES and names.get(b[5]) in CONTEXT_CLASSES]
        overlaps = set()

        if CONTEXT_CLASSES:
            for bird in birds:
                bird_name = names.get(bird[5])
                for obj in objects:
                    obj_name = names.get(obj[5])
                    if bird is obj or bird_name==obj_name:
                        continue
                    if self._is_overlapping(bird,obj,self.config.overlap_threshold):
                        pair = (bird_name,obj_name)
                        overlaps.add(pair)
                        self._update_active(pair,video_ts)
        else:
            for i, bird in enumerate(birds):
                for j, other in enumerate(birds):
                    if j <= i:
                        continue
                    b1, b2 = names.get(bird[5]), names.get(other[5])
                    if b1 == b2: continue
                    if self._is_overlapping(bird,other,self.config.overlap_threshold):
                        pair = tuple(sorted((b1,b2)))
                        overlaps.add(pair)
                        self._update_active(pair,video_ts)

        self._finalize_inactive(overlaps, video_ts)

    def _update_active(self, pair, ts):
        if pair not in self.active:
            self.active[pair] = {"start": ts, "last_seen": ts}
        else:
            self.active[pair]["last_seen"] = ts

    def _finalize_inactive(self, overlaps, ts):
        ended = []
        for pair, times in list(self.active.items()):
            if pair not in overlaps and (ts - times["last_seen"]).total_seconds() >= self.config.interaction_timeout_sec:
                self._record_interaction(pair, times["start"], times["last_seen"])
                ended.append(pair)
        for k in ended:
            del self.active[k]

    def finalize(self):
        for pair, times in self.active.items():
            self._record_interaction(pair, times["start"], times["last_seen"])
        self.active.clear()
        return self.records

    def _record_interaction(self, pair, start, end):
        duration = round((end-start).total_seconds(),2)
        if duration<=0: return
        if CONTEXT_CLASSES:
            self.records.append({"TIME0":start.strftime("%H:%M:%S"), "TIME1":end.strftime("%H:%M:%S"),
                                 "FOCUS":pair[0],"CONTEXT":pair[1],"DURATION":duration})
        else:
            self.records.append({"TIME0":start.strftime("%H:%M:%S"), "TIME1":end.strftime("%H:%M:%S"),
                                 "CLASS1":pair[0],"CLASS2":pair[1],"DURATION":duration})

    def save_results(self):
        if not self.records or not self.out_folder:
            return None
        self.out_folder.mkdir(parents=True, exist_ok=True)
        out_file = self.out_folder / "interactions.csv"
        fieldnames = ["TIME0","TIME1","FOCUS","CONTEXT","DURATION"] if CONTEXT_CLASSES else ["TIME0","TIME1","CLASS1","CLASS2","DURATION"]
        with open(out_file,"w",newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in sorted(self.records,key=lambda x:x["TIME0"]):
                writer.writerow({k:v for k,v in r.items() if k in fieldnames})
        return out_file

    @staticmethod
    def _is_overlapping(box1, box2, threshold=0.1):
        if box1 is box2:
            return False
        x1_min, x1_max = box1[0]-box1[2]/2, box1[0]+box1[2]/2
        y1_min, y1_max = box1[1]-box1[3]/2, box1[1]+box1[3]/2
        x2_min, x2_max = box2[0]-box2[2]/2, box2[0]+box2[2]/2
        y2_min, y2_max = box2[1]-box2[3]/2, box2[1]+box2[3]/2
        inter_area = max(0,min(x1_max,x2_max)-max(x1_min,x2_min)) * max(0,min(y1_max,y2_max)-max(y1_min,y2_min))
        if inter_area==0: return False
        union_area = box1[2]*box1[3] + box2[2]*box2[3] - inter_area
        return inter_area/union_area > threshold


# ---------------- AGGREGATOR ----------------
class Aggregator:
    def __init__(self, out_folder, config=None, start_time=None):
        self.out_folder = Path(out_folder)
        self.config = config or MeasurementConfig()
        self.start_time = start_time
        self.frame_data = []
        self.intervals = []

    def push_frame_data(self, timestamp, current_boxes_list=None, names=None, counts_dict=None):
        if counts_dict is None and current_boxes_list and names:
            counts_dict = compute_counts_from_boxes(current_boxes_list, names)
        # Remove any previous RATIO before storing numeric counts
        counts_dict.pop("RATIO", None)
        self.frame_data.append((timestamp, counts_dict))


    def aggregate_intervals(self):
        if not self.frame_data:
            return []
        self.frame_data.sort(key=lambda x: x[0])
        interval_start = self.frame_data[0][0]
        interval_end = interval_start + timedelta(seconds=self.config.interval_sec)
        interval_counts = defaultdict(list)
        intervals = []

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

    def _finalize_interval(self, interval_start, interval_counts):
        # Only sum numeric values
        summed_counts = {}
        for cls, vals in interval_counts.items():
            numeric_vals = [v for v in vals if isinstance(v, (int, float))]
            summed_counts[cls] = sum(numeric_vals)

        if CONTEXT_CLASSES:
            # Compute OBJECTS as sum of context classes
            objects_sum = sum(summed_counts.pop(c, 0) for c in CONTEXT_CLASSES)
            summed_counts["OBJECTS"] = objects_sum

        # Compute RATIO after summing numeric values
        summed_counts = add_ratio_to_counts(summed_counts)

        midpoint = interval_start + timedelta(seconds=self.config.interval_sec / 2)
        midpoint_str = midpoint.strftime("%H:%M:%S")
        return {"TIME": midpoint_str, "Counts": summed_counts}

    def save_interval_results(self):
        intervals = self.aggregate_intervals()
        if not intervals:
            return None
        self.out_folder.mkdir(parents=True, exist_ok=True)
        out_file = self.out_folder / "interval_results.csv"

        all_cols = FOCUS_CLASSES + (["OBJECTS"] if CONTEXT_CLASSES else []) + (["RATIO"] if CONTEXT_CLASSES else [])
        with open(out_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["TIME"] + all_cols)
            for interval in intervals:
                row = [interval["TIME"]] + [round(interval["Counts"].get(cls,0),3) if cls not in ["RATIO","OBJECTS"] 
                                            else interval["Counts"].get("RATIO","") if cls=="RATIO" 
                                            else round(interval["Counts"].get("OBJECTS",0),3)
                                            for cls in all_cols]
                writer.writerow(row)
        return out_file

    def save_session_summary(self):
        if not self.frame_data:
            return None

        session_counts = defaultdict(float)
        session_rates = defaultdict(list)

        # Accumulate counts and rates
        for _, counts in self.frame_data:
            for cls, val in counts.items():
                session_counts[cls] += val
                session_rates[cls].append(val / self.config.interval_sec if self.config.interval_sec > 0 else 0)

        # Handle CONTEXT_CLASSES
        if CONTEXT_CLASSES:
            objects_total = sum(session_counts.pop(c, 0) for c in CONTEXT_CLASSES)
            session_counts["OBJECTS"] = objects_total

            object_rates = []
            for c in CONTEXT_CLASSES:
                object_rates.extend(session_rates.pop(c, []))
            session_rates["OBJECTS"] = object_rates or [0.0]

        # Compute total only over FOCUS_CLASSES for PROP
        focus_total = sum(session_counts.get(cls, 0) for cls in FOCUS_CLASSES) or 1.0

        summary_list = []
        for cls, total in session_counts.items():
            rates = session_rates.get(cls, [])
            if rates:
                mean_rate = sum(rates) / len(rates)
                std_dev = math.sqrt(sum((r - mean_rate) ** 2 for r in rates) / len(rates))
            else:
                mean_rate = 0.0
                std_dev = 0.0

            # PROP: only for focus classes, OBJECTS gets "n/a"
            if cls in FOCUS_CLASSES:
                prop = total / focus_total
            else:
                prop = "n/a"

            summary_list.append({
                "CLASS": cls,
                "TOTAL_COUNT": round(total, 3),
                "AVG_RATE": round(mean_rate, 3),
                "STD_DEV": round(std_dev, 3),
                "PROP": prop if isinstance(prop, str) else round(prop, 3)
            })

        # Save CSV
        out_file = self.out_folder / "session_summary.csv"
        self.out_folder.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["CLASS","TOTAL_COUNT","AVG_RATE","STD_DEV","PROP"])
            writer.writeheader()
            for row in summary_list:
                writer.writerow(row)

        return out_file
