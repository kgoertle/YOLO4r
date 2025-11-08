import cv2
from pymediainfo import MediaInfo

def get_rotation_angle(video_path, print_lock=None):
    """
    Extract the rotation angle from video metadata using pymediainfo.
    Returns 0 if unavailable.
    """
    try:
        media_info = MediaInfo.parse(str(video_path))
        for track in media_info.tracks:
            if track.track_type == "Video":
                rotation = getattr(track, "rotation", 0)
                if rotation:
                    return int(float(rotation))
    except Exception as e:
        if print_lock:
            with print_lock:
                print(f"[WARN] Could not read rotation from {video_path}: {e}")
        else:
            print(f"[WARN] Could not read rotation from {video_path}: {e}")
    return 0


def rotate_frame(frame, angle):
    """
    Rotate frame by 90, 180, or 270 degrees.
    If no matching angle, return frame unchanged.
    """
    if angle == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame
