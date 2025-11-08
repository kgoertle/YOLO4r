import csv
from datetime import datetime
from pathlib import Path

class InteractionMetrics:
    """
    Tracks and logs bird-object interactions with duration.
    Each interaction is defined as a period of continuous overlap between one bird (M/F) and one object (Feeder, Main_Perch, etc.).
    """
    def __init__(self, output_folder: Path):
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)

        # Stores currently active interactions:
        # (bird_name, object_name), value = {'start': datetime, 'last_seen': datetime}
        self.active_interactions = {}

        # Stores completed interaction records (for CSV output)
        self.records = []

        # Configurable timeout in seconds: if not overlapping for this long, end the interaction
        self.timeout_sec = 2.0

    def process_frame(self, boxes, names, timestamp: datetime):
        """
        Process a single frame of detections and update active interactions.
        """
        birds = [b for b in boxes if names.get(b[5]) in ("M", "F")]
        objects = [b for b in boxes if names.get(b[5]) not in ("M", "F")]

        # Track currently overlapping pairs for this frame
        current_overlaps = set()

        for bird in birds:
            bx, by, bw, bh, _, bcls = bird
            bird_name = names.get(bcls, "")
            for obj in objects:
                ox, oy, ow, oh, _, ocls = obj
                obj_name = names.get(ocls, "")
                if self._is_overlapping(bird, obj):
                    pair_key = (bird_name, obj_name)
                    current_overlaps.add(pair_key)

                    if pair_key not in self.active_interactions:
                        # Start a new interaction
                        self.active_interactions[pair_key] = {
                            "start": timestamp,
                            "last_seen": timestamp
                        }
                    else:
                        # Update last seen time
                        self.active_interactions[pair_key]["last_seen"] = timestamp

        # Check for ended interactions (no longer overlapping)
        ended_pairs = []
        for pair_key, times in list(self.active_interactions.items()):
            if pair_key not in current_overlaps:
                # If not seen recently, consider it ended
                elapsed = (timestamp - times["last_seen"]).total_seconds()
                if elapsed > self.timeout_sec:
                    start = times["start"]
                    end = times["last_seen"]
                    duration = round((end - start).total_seconds(), 2)
                    if duration > 0:
                        self.records.append({
                            "Initial_time": start.strftime("%H:%M:%S"),
                            "Final_time": end.strftime("%H:%M:%S"),
                            "Bird": pair_key[0],
                            "Object": pair_key[1],
                            "Duration": duration
                        })
                    ended_pairs.append(pair_key)

        # Remove ended pairs
        for k in ended_pairs:
            del self.active_interactions[k]

    def _is_overlapping(self, box1, box2, threshold: float = 0.1):
        """
        Approximate overlap check (axis-aligned bounding boxes).
        Returns True if intersection area / union area > threshold.
        """
        x1_min = box1[0] - box1[2] / 2
        x1_max = box1[0] + box1[2] / 2
        y1_min = box1[1] - box1[3] / 2
        y1_max = box1[1] + box1[3] / 2

        x2_min = box2[0] - box2[2] / 2
        x2_max = box2[0] + box2[2] / 2
        y2_min = box2[1] - box2[3] / 2
        y2_max = box2[1] + box2[3] / 2

        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)

        inter_w = max(0, inter_xmax - inter_xmin)
        inter_h = max(0, inter_ymax - inter_ymin)
        inter_area = inter_w * inter_h
        if inter_area == 0:
            return False

        box1_area = box1[2] * box1[3]
        box2_area = box2[2] * box2[3]
        union_area = box1_area + box2_area - inter_area
        return (inter_area / union_area) > threshold

    def finalize(self):
        """
        Save all recorded interactions to CSV.
        Any still-active interactions are also closed and written out.
        """
        # Finalize any still-active pairs
        for pair_key, times in self.active_interactions.items():
            start = times["start"]
            end = times["last_seen"]
            duration = round((end - start).total_seconds(), 2)
            if duration > 0:
                self.records.append({
                    "Initial_time": start.strftime("%H:%M:%S"),
                    "Final_time": end.strftime("%H:%M:%S"),
                    "Bird": pair_key[0],
                    "Object": pair_key[1],
                    "Duration": duration
                })

        if not self.records:
            return None

        out_path = self.output_folder / f"interaction_metrics.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["Initial_time", "Final_time", "Bird", "Object", "Duration"])
            writer.writeheader()
            writer.writerows(self.records)
        return out_path
