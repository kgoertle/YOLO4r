# video_util.py
import os, cv2, time, queue, json, platform, threading, subprocess
from pathlib import Path
from datetime import datetime, timezone

# ---------- VideoSourceInfo ----------
class VideoSourceInfo:
    def __init__(self, metadata: dict, is_camera: bool, display_name: str):
        self.metadata = metadata
        self.is_camera = is_camera
        self.display_name = display_name

        self.width = metadata.get("width")
        self.height = metadata.get("height")
        self.fps = metadata.get("fps")
        self.creation_time = metadata.get("creation_time_used")
        self.creation_dt = None

    def parse_creation_time(self):
        ts = self.metadata.get("creation_time_used")
        if not ts:
            self.creation_dt = datetime.now()
            return self.creation_dt

        # Normalize variants like "Z"
        ts = ts.strip().replace("Z", "+00:00")

        for fmt in (
            "%Y-%m-%dT%H:%M:%S.%f%z",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
        ):
            try:
                dt = datetime.strptime(ts, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                self.creation_dt = dt
                return dt
            except ValueError:
                continue

        # ISO fallback
        try:
            dt = datetime.fromisoformat(ts)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            self.creation_dt = dt
            return dt
        except Exception:
            self.creation_dt = datetime.now()
            return self.creation_dt


# ---------- Filename Parsing ----------
def parse_filename_time(video_path):
    stem = Path(video_path).stem
    parts = stem.split("-")
    if len(parts) == 3:
        try:
            h, m, s = map(int, parts)
            return datetime.strptime(f"{h:02d}:{m:02d}:{s:02d}", "%H:%M:%S").time()
        except Exception:
            return None
    return None


# ---------- Metadata Extraction ----------
def extract_video_metadata(video_path: str or Path) -> dict:
    video_path = Path(video_path)
    metadata = {
        "type": "video",
        "source": str(video_path),
    }

    # ----- FFprobe extraction -----
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,codec_name,avg_frame_rate,duration",
            "-show_entries", "format_tags=creation_time",
            "-of", "json",
            str(video_path),
        ]
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        data = json.loads(result.stdout)

        if "streams" in data and len(data["streams"]) > 0:
            s = data["streams"][0]
            metadata["width"] = s.get("width")
            metadata["height"] = s.get("height")
            metadata["codec"] = s.get("codec_name")

            # FPS extraction
            fps_str = s.get("avg_frame_rate", "0/1")
            try:
                num, den = map(float, fps_str.split("/"))
                metadata["fps"] = round(num / den, 3) if den != 0 and num != 0 else None
            except Exception:
                metadata["fps"] = None

        # Embedded creation time
        format_tags = data.get("format", {}).get("tags", {})
        metadata["creation_time_embedded"] = format_tags.get("creation_time")

    except Exception as e:
        metadata["ffprobe_error"] = str(e)

    # ---------- Filesystem creation time ----------
    try:
        stat = os.stat(video_path)
        creation_ts = getattr(stat, "st_birthtime", stat.st_mtime)
        creation_fs = datetime.fromtimestamp(creation_ts).isoformat()
        metadata["creation_time_filesystem"] = creation_fs
    except Exception as e:
        metadata["creation_time_filesystem"] = None
        metadata["fs_time_error"] = str(e)

    # ---------- Filename timestamp ----------
    ft = parse_filename_time(video_path)
    if ft:
        metadata["creation_time_filename"] = ft.strftime("%H:%M:%S")
    else:
        metadata["creation_time_filename"] = None

    # ---------- Platform detection ----------
    is_mac = platform.system() == "Darwin"
    is_linux = platform.system() == "Linux"
    is_pi = is_linux and ("arm" in platform.machine() or "aarch64" in platform.machine())

    creation_fs = metadata.get("creation_time_filesystem")
    creation_file = metadata.get("creation_time_filename")
    creation_emb = metadata.get("creation_time_embedded")

    # ---------- Priority Selection ----------
    if creation_file:
        if is_mac and creation_fs:
            date_part = creation_fs.split("T")[0]
        elif is_pi:
            date_part = datetime.now().strftime("%Y-%m-%d")
        elif creation_fs:
            if isinstance(creation_fs, str) and "T" in creation_fs:
                date_part = creation_fs.split("T")[0]
            else:
                date_part = datetime.now().strftime("%Y-%m-%d")
        else:
            date_part = datetime.now().strftime("%Y-%m-%d")

        metadata["creation_time_used"] = f"{date_part}T{creation_file}"

    elif is_mac and creation_fs:
        metadata["creation_time_used"] = creation_fs

    elif creation_emb:
        metadata["creation_time_used"] = creation_emb

    elif creation_fs:
        metadata["creation_time_used"] = creation_fs

    else:
        metadata["creation_time_used"] = datetime.now().isoformat()
        metadata["creation_time_error"] = (
            "No usable timestamp from filename/fs/embedded. Using system time."
        )

    metadata["extracted_at"] = datetime.now().isoformat()
    return metadata


# ---------- Metadata Extraction (Camera via cv2) ----------
def extract_camera_metadata(cap, source_id: int) -> dict:
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    return {
        "type": "camera",
        "source": f"usb{source_id}",
        "width": int(width) if width else 640,
        "height": int(height) if height else 480,
        "fps": round(fps if fps and fps > 0 else 30.0, 3),
        "started_at": datetime.now().isoformat(),
        "creation_time_used": datetime.now().isoformat(),
    }


# ---------- Timestamp Parser (standalone compatibility) ----------
def parse_creation_time(metadata: dict):
    ts = metadata.get("creation_time_used")
    if not ts:
        return None
    ts = ts.strip().replace("Z", "+00:00")

    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
    ):
        try:
            dt = datetime.strptime(ts, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue

    try:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


# ---------- Video Reader ----------
class VideoReader:
    """
    Frame reader for a single source.
    Handles:
    - reading frames
    - sending EOF to inference queue
    - realtime camera frame dropping
    """

    def __init__(
        self,
        cap,
        frame_queue: queue.Queue,
        infer_queue: queue.Queue,
        is_camera: bool,
        source_display_name: str,
        global_stop_event,
        printer,
    ):
        self.cap = cap
        self.frame_queue = frame_queue
        self.infer_queue = infer_queue
        self.is_camera = is_camera
        self.source_display_name = source_display_name
        self.global_stop_event = global_stop_event
        self.printer = printer

        self._stop_local = threading.Event()
        self._thread = None

    def start(self):
        if not self._thread or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def stop(self):
        self._stop_local.set()

    def join(self, timeout=None):
        if self._thread:
            self._thread.join(timeout=timeout)

    def _run(self):
        while not self._stop_local.is_set():
            ret, frame = self.cap.read()

            if not ret or frame is None:
                if self.is_camera:
                    time.sleep(0.01)
                    continue

                # EOF
                try:
                    self.infer_queue.put(("EOF", None), timeout=0.1)
                except Exception:
                    pass
                break

            try:
                self.frame_queue.put(frame, timeout=0.02)
            except queue.Full:
                if self.is_camera:
                    # Drop frame
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        self.frame_queue.put(frame, timeout=0.02)
                    except queue.Full:
                        pass
                else:
                    # Block for video files
                    while not self.global_stop_event.is_set():
                        try:
                            self.frame_queue.put(frame, timeout=0.1)
                            break
                        except queue.Full:
                            continue


# ---------- Video Writer ----------
def create_video_writer(
    out_file: Path,
    fps: float,
    frame_width: int,
    frame_height: int,
    source_display_name: str,
    printer,
    cap,
    source_type: str,
):
    """
    Create and register a cv2.VideoWriter for annotated output.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_file), fourcc, fps, (frame_width, frame_height))

    if not writer.isOpened():
        printer.warn(f"VideoWriter failed: {source_display_name}")
        return None

    printer.register_writer(
        out_file.name,
        writer,
        cap,
        source_type,
        out_file,
        display_name=source_display_name,
    )
    return writer


def write_annotated_frame(writer, frame):
    if writer:
        writer.write(frame)


# ---------- Box Extraction from Ultralytics Results ----------
def extract_boxes_from_results(results, is_obb_model: bool):
    names, boxes_list = {}, []
    if not results:
        return names, boxes_list

    r = results[0]
    names = r.names

    try:
        if is_obb_model and hasattr(r, "obb") and r.obb is not None:
            xyxy = r.obb.xyxy.cpu().numpy()
            conf = r.obb.conf.cpu().numpy()
            cls = r.obb.cls.cpu().numpy()
        else:
            xyxy = r.boxes.xyxy.cpu().numpy()
            conf = r.boxes.conf.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy()

        boxes_list = [
            [
                float(x1),
                float(y1),
                float(x2),
                float(y2),
                float(cf),
                int(c),
            ]
            for (x1, y1, x2, y2), cf, c in zip(xyxy, conf, cls)
        ]
    except Exception:
        # If something is weird, just fall back to empty boxes
        names, boxes_list = r.names, []

    return names, boxes_list
