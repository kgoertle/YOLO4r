# utils/detect/video_metadata.py
import os, subprocess, json, cv2, platform
from datetime import datetime, timezone
from pathlib import Path


def parse_filename_time(video_path):
    """
    Extract HH-MM-SS from a filename like 06-48-01.mp4
    Returns a naive datetime.time object or None.
    """
    stem = Path(video_path).stem
    parts = stem.split("-")
    if len(parts) == 3:
        try:
            h, m, s = map(int, parts)
            return datetime.strptime(f"{h:02d}:{m:02d}:{s:02d}", "%H:%M:%S").time()
        except:
            return None
    return None


def extract_video_metadata(video_path):
    video_path = Path(video_path)
    metadata = {
        "type": "video",
        "source": str(video_path),
    }

    # ---- Get FFprobe Metadata ----
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,codec_name,avg_frame_rate,duration",
            "-show_entries", "format_tags=creation_time",
            "-of", "json",
            str(video_path)
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        data = json.loads(result.stdout)

        if "streams" in data and len(data["streams"]) > 0:
            s = data["streams"][0]
            metadata["width"] = s.get("width")
            metadata["height"] = s.get("height")
            metadata["codec"] = s.get("codec_name")

            fps_str = s.get("avg_frame_rate", "0/1")
            try:
                num, den = map(float, fps_str.split("/"))
                metadata["fps"] = round(num / den, 3) if den != 0 and num != 0 else None
            except Exception:
                metadata["fps"] = None

        # Embedded creation time from ffmpeg (often null for OpenCV-generated MP4s)
        format_tags = data.get("format", {}).get("tags", {})
        metadata["creation_time_embedded"] = format_tags.get("creation_time")

    except Exception as e:
        metadata["ffprobe_error"] = str(e)

    # ---- Filesystem creation time (date only; time is inaccurate after copying to macOS) ----
    try:
        stat = os.stat(video_path)
        creation_ts = getattr(stat, "st_birthtime", stat.st_mtime)
        creation_time_fs = datetime.fromtimestamp(creation_ts).isoformat()
        metadata["creation_time_filesystem"] = creation_time_fs
    except Exception as e:
        metadata["creation_time_filesystem"] = None
        metadata["fs_time_error"] = str(e)

    # ---- Extract time from filename (highest priority) ----
    filename_time = parse_filename_time(video_path)
    if filename_time:
        metadata["creation_time_filename"] = filename_time.strftime("%H:%M:%S")
    else:
        metadata["creation_time_filename"] = None

    # ---- Choose preferred creation time (platform-aware) ----

    # System detection
    is_mac = platform.system() == "Darwin"
    is_linux = platform.system() == "Linux"
    is_pi = is_linux and ("arm" in platform.machine() or "aarch64" in platform.machine())

    creation_fs = metadata.get("creation_time_filesystem")
    creation_file = metadata.get("creation_time_filename")
    creation_emb = metadata.get("creation_time_embedded")

    # ---- PRIORITY 1: Filename timestamp (common for both Mac and Pi) ----
    if creation_file:
        # Determine date part safely
        if is_mac and creation_fs:
            # macOS: st_birthtime is valid
            date_part = creation_fs.split("T")[0]
        elif is_pi:
            # Raspberry Pi: filesystem timestamps not reliable → use today's date
            date_part = datetime.now().strftime("%Y-%m-%d")
        elif creation_fs:
            # Generic Linux server: use filesystem mtime if present
            if isinstance(creation_fs, str) and "T" in creation_fs:
                date_part = creation_fs.split("T")[0]
            else:
                date_part = datetime.now().strftime("%Y-%m-%d")
        else:
            # Last fallback
            date_part = datetime.now().strftime("%Y-%m-%d")

        metadata["creation_time_used"] = f"{date_part}T{creation_file}"

    # ---- PRIORITY 2: Filesystem creation (Mac only; ext4 doesn't support it) ----
    elif is_mac and creation_fs:
        metadata["creation_time_used"] = creation_fs

    # ---- PRIORITY 3: Embedded metadata (FFmpeg tags) ----
    elif creation_emb:
        metadata["creation_time_used"] = creation_emb

    # ---- PRIORITY 4: Generic Linux (server) fallback to mtime ----
    elif creation_fs:
        metadata["creation_time_used"] = creation_fs

    # ---- PRIORITY 5: Absolute final fallback ----
    else:
        metadata["creation_time_used"] = datetime.now().isoformat()
        metadata["creation_time_error"] = (
            "No usable timestamp (filename/fs/embedded). Used current system time."
        )

    # ---- Extraction timestamp ----
    metadata["extracted_at"] = datetime.now().isoformat()

    return metadata


def extract_camera_metadata(cap, source_id):
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    return {
        "type": "camera",
        "source": f"usb{source_id}",
        "width": int(width),
        "height": int(height),
        "fps": round(fps if fps > 0 else 30, 3),
        "started_at": datetime.now().isoformat()
    }


# ---- Time Handling ----
def parse_creation_time(metadata):
    ts = metadata.get("creation_time_used")
    if not ts:
        return None

    # Normalize variants
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
