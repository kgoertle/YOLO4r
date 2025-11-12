import os, subprocess, json, cv2
from datetime import datetime, timezone
from pathlib import Path

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
            # FPS from fraction (e.g. "30/1")
            fps_str = s.get("avg_frame_rate", "0/1")
            try:
                num, den = map(float, fps_str.split("/"))
                metadata["fps"] = round(num / den, 3) if den != 0 and num != 0 else None
            except Exception:
                metadata["fps"] = None

        # Get embedded creation time
        format_tags = data.get("format", {}).get("tags", {})
        metadata["creation_time_embedded"] = format_tags.get("creation_time")

    except Exception as e:
        metadata["ffprobe_error"] = str(e)

    # ---- Get file creation time ----
    try:
        stat = os.stat(video_path)
        creation_ts = getattr(stat, "st_birthtime", stat.st_mtime)
        creation_time_fs = datetime.fromtimestamp(creation_ts).isoformat()
        metadata["creation_time_filesystem"] = creation_time_fs
    except Exception as e:
        metadata["creation_time_filesystem"] = None
        metadata["fs_time_error"] = str(e)

    # ---- Choose the preferred creation time ----
    preferred_time = metadata.get("creation_time_filesystem") or metadata.get("creation_time_embedded")
    metadata["creation_time_used"] = preferred_time

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
    ts = ts.strip().replace("Z", "+00:00")  # convert UTC marker to offset

    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%f%z",  # ISO with microseconds + timezone
        "%Y-%m-%dT%H:%M:%S%z",     # ISO without microseconds + timezone
        "%Y-%m-%dT%H:%M:%S.%f",    # ISO without timezone
        "%Y-%m-%d %H:%M:%S",       # space-separated
    ):
        try:
            dt = datetime.strptime(ts, fmt)
            # If no timezone info, assume local system timezone
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue

    # Fallback to fromisoformat for odd variants
    try:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None