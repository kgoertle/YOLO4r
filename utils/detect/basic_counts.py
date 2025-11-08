import csv
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict


# ------ SETTINGS ------
DEFAULT_INTERVAL_SEC = 5 # set to desired interval for average
AVG_GROUP_SIZE = 3  # number of snapshots to average over

BIRD_CLASSES = ["M", "F"]
OBJECT_CLASSES = ["Feeder", "Main_Perch", "Wooden_Perch", "Sky_Perch", "Nesting_Box"]

# ------ COUNT LOGIC ------
_last_seen_counts = defaultdict(int)
_last_snapshot_time = None
_interval_sec = DEFAULT_INTERVAL_SEC
_snapshot_buffer = []  # holds last N snapshots for averaging
_completed_snapshots = []  # list of (timestamp, counts)
_avg_group_counter = 1  # sequential numbering of grouped averages

# ------ MAIN FUNCTIONS ------
def log_bird_counts_summary(current_boxes_list, names, out_folder=None, interval_sec=None):
    """
    Calls this function for each frame. 
    It collects snapshot counts every 'interval_sec' seconds and writes them to CSV. It also computes grouped averages across AVG_GROUP_SIZE snapshots.
    """
    global _last_seen_counts, _last_snapshot_time, _interval_sec, _snapshot_buffer, _completed_snapshots

    if interval_sec is not None:
        _interval_sec = interval_sec

    now = datetime.now()

    # Count detections in this frame (snapshot)
    latest_counts = defaultdict(int)
    for b in current_boxes_list:
        cls_name = names.get(b[5])
        if cls_name in BIRD_CLASSES + OBJECT_CLASSES:
            latest_counts[cls_name] += 1
    _last_seen_counts = latest_counts

    if _last_snapshot_time is None:
        _last_snapshot_time = now
        _write_snapshot_csv(now, latest_counts, out_folder)
        _snapshot_buffer.append(latest_counts)
        _completed_snapshots.append((now, latest_counts))
        return

    # Take new snapshot only every X seconds
    if (now - _last_snapshot_time).total_seconds() >= _interval_sec:
        _write_snapshot_csv(now, latest_counts, out_folder)
        _snapshot_buffer.append(latest_counts)
        _completed_snapshots.append((now, latest_counts))
        _last_snapshot_time = now

        # Write grouped averages every AVG_GROUP_SIZE snapshots
        if len(_snapshot_buffer) >= AVG_GROUP_SIZE:
            _write_grouped_average_csv(_snapshot_buffer, out_folder)
            _snapshot_buffer = []


def flush_remaining_interval(out_folder=None):
    global _snapshot_buffer

    if _snapshot_buffer:
        _write_grouped_average_csv(_snapshot_buffer, out_folder)
        _snapshot_buffer = []

# ------ HELPERS ------
def _write_snapshot_csv(timestamp, latest_counts, out_folder):
    """
    Write a single snapshot (current detection counts).
    """
    out_folder = Path(out_folder) if out_folder else Path.cwd()
    out_folder.mkdir(parents=True, exist_ok=True)
    csv_file = out_folder / "counts.csv"
    file_exists = csv_file.is_file()

    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            header = ["Time"] + BIRD_CLASSES + OBJECT_CLASSES
            writer.writerow(header)
        row = [timestamp.strftime("%H:%M:%S")] + [
            latest_counts.get(cls, 0) for cls in BIRD_CLASSES + OBJECT_CLASSES
        ]
        writer.writerow(row)
        f.flush()


def _write_grouped_average_csv(snapshots, out_folder):
    """
    Compute average counts across 'AVG_GROUP_SIZE' snapshots.
    """
    global _avg_group_counter

    if not snapshots:
        return

    out_folder = Path(out_folder) if out_folder else Path.cwd()
    out_folder.mkdir(parents=True, exist_ok=True)
    csv_file = out_folder / "average_counts.csv"
    file_exists = csv_file.is_file()

    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            header = ["Group", "Start Time", "End Time"] + BIRD_CLASSES + OBJECT_CLASSES
            writer.writerow(header)

        start_time = datetime.now().strftime("%H:%M:%S")
        end_time = start_time

        # Sum up counts
        summed = defaultdict(float)
        for counts in snapshots:
            for cls, val in counts.items():
                summed[cls] += val

        # Average across snapshots
        avg_counts = {cls: summed[cls] / len(snapshots) for cls in BIRD_CLASSES + OBJECT_CLASSES}

        row = [
            _avg_group_counter,
            snapshots[0].get("Time", start_time),
            snapshots[-1].get("Time", end_time)
        ] + [round(avg_counts.get(cls, 0), 1) for cls in BIRD_CLASSES + OBJECT_CLASSES]

        writer.writerow(row)
        f.flush()
        _avg_group_counter += 1
