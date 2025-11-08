from pathlib import Path
from datetime import datetime, timedelta
import csv
import math

class Aggregator:
    def __init__(self, interval_sec=5, session_sec=10):
        """
        interval_sec: length of interval aggregation in seconds
        session_sec: length of session aggregation in seconds (for testing)
        """
        self.interval_sec = interval_sec
        self.session_sec = session_sec

        # Stores raw frame-level counts: list of tuples (timestamp, M, F, O)
        self.frame_data = []

        # Stores aggregated interval metrics
        self.intervals = []

    def push_frame_data(self, timestamp, males, females, other_counts):
        """
        Push frame-level detection counts into the aggregator
        """
        o_count = sum(other_counts.get(c, 0) for c in ['Feeder', 'Main_Perch', 'Wooden_Perch', 'Sky_Perch', 'Nesting_Box'])
        self.frame_data.append((timestamp, males, females, o_count))

    def _aggregate_intervals(self):
        """
        Convert frame-level data into interval-level aggregated counts.
        Each interval is self.interval_sec long.
        """
        if not self.frame_data:
            return

        self.intervals = []
        start_idx = 0
        while start_idx < len(self.frame_data):
            start_time = self.frame_data[start_idx][0]
            end_time = start_time + timedelta(seconds=self.interval_sec)
            interval_counts = []

            idx = start_idx
            while idx < len(self.frame_data) and self.frame_data[idx][0] < end_time:
                interval_counts.append(self.frame_data[idx][1:4])  # (M, F, O)
                idx += 1

            if interval_counts:
                self._save_interval(start_time, end_time, interval_counts)

            start_idx = idx

    def _aggregate_session(self):
        """
        Aggregate completed intervals into session metrics, including:
        - Total count per class
        - Mean detection rate per second
        - Standard deviation of rate
        - Normalized proportion of total counts
        """
        if not self.intervals:
            return []

        total_counts_all_classes = sum(
            sum(i[f'{cls}_count'] for cls in ['M', 'F', 'O']) for i in self.intervals
        )

        session_summary = []
        for cls in ['M', 'F', 'O']:
            counts = [i[f'{cls}_rate'] for i in self.intervals]
            total_count = sum(i[f'{cls}_count'] for i in self.intervals)
            mean_rate = sum(counts) / len(counts)
            std_dev = math.sqrt(sum((r - mean_rate) ** 2 for r in counts) / (len(counts) - 1)) if len(counts) > 1 else 0.0
            normalized = total_count / total_counts_all_classes if total_counts_all_classes > 0 else 0.0

            session_summary.append({
                'Class': cls,
                'Total_Count': total_count,
                'Mean_Rate_per_sec': round(mean_rate, 3),
                'Std_Dev_Rate': round(std_dev, 3),
                'Normalized_Proportion': round(normalized, 3)
            })

        return session_summary

    def _save_interval(self, start_time, end_time, counts):
        """
        Aggregate counts for an interval and append to intervals list.
        """
        total_m = sum(c[0] for c in counts)
        total_f = sum(c[1] for c in counts)
        total_o = sum(c[2] for c in counts)

        # rate per second
        rate_m = total_m / self.interval_sec
        rate_f = total_f / self.interval_sec
        rate_o = total_o / self.interval_sec

        # M:F ratio
        mf_ratio = rate_m / rate_f if rate_f > 0 else float('inf')

        self.intervals.append({
            'start': start_time,
            'end': end_time,
            'M_count': total_m,
            'F_count': total_f,
            'O_count': total_o,
            'M_rate': rate_m,
            'F_rate': rate_f,
            'O_rate': rate_o,
            'MF_ratio': mf_ratio
        })

    def save_results(self, out_folder):
        """
        Aggregate interval and session results, save both as CSVs, and return their file paths.
        """
        out_folder = Path(out_folder)
        out_folder.mkdir(parents=True, exist_ok=True)

        # 1. Aggregate frame data into intervals
        self._aggregate_intervals()
        if not self.intervals:
            return None, None

        # 2. Save interval-level results
        interval_file = out_folder / "interval_results.csv"
        with interval_file.open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Interval_Start', 'Interval_End', 'Class', 'Count', 'Rate_per_sec', 'MF_Ratio'])
            for interval in self.intervals:
                for cls in ['M', 'F', 'O']:
                    writer.writerow([
                        interval['start'].strftime("%H:%M:%S"),
                        interval['end'].strftime("%H:%M:%S"),
                        cls,
                        interval[f'{cls}_count'],
                        round(interval[f'{cls}_rate'], 3),
                        round(interval['MF_ratio'], 3) if cls in ['M', 'F'] else ''
                    ])

        # 3. Aggregate and save session summary
        session_data = self._aggregate_session()
        if not session_data:
            return interval_file, None

        session_file = out_folder / "session_summary.csv"
        with session_file.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=session_data[0].keys())
            writer.writeheader()
            writer.writerows(session_data)

        return interval_file, session_file