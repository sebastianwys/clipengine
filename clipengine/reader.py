# reader.py
# frame access for analysis. windows decode at native fps, downscaled to
# the analysis width. consecutive frames keep optical-flow displacements
# small enough for farneback to track even fast whips. only ever called
# on materialized files; the catalog guards eviction upstream.

from typing import Optional

import cv2
import numpy as np

from clipengine import config


class ClipReadError(Exception):
    """raised when a clip cannot be opened or yields no frames."""


def _downscale(frame: np.ndarray, width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= width:
        return frame
    scale = width / w
    return cv2.resize(frame, (width, max(2, round(h * scale))),
                      interpolation=cv2.INTER_AREA)


def _open(path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise ClipReadError(f"decoder could not open {path}")
    return cap


def _fps(cap: cv2.VideoCapture) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1.0 or fps > 240.0:
        return 30.0
    return fps


def read_window(path, start_s: float,
                duration_s: float = config.WINDOW_SECONDS,
                width: int = config.ANALYSIS_WIDTH,
                max_frames: int = config.MAX_WINDOW_FRAMES
                ) -> tuple[list[np.ndarray], float]:
    """decode consecutive downscaled bgr frames starting at start_s."""
    cap = _open(path)
    try:
        fps = _fps(cap)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        want = min(max_frames, max(2, round(duration_s * fps)))
        start_frame = max(0, round(start_s * fps))
        if total > 0:
            start_frame = min(start_frame, max(0, total - want))
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames: list[np.ndarray] = []
        while len(frames) < want:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(_downscale(frame, width))
        if len(frames) < 2 and start_frame > 0:
            # long-gop seek can overshoot near the tail; retry from zero
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            while len(frames) < want:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(_downscale(frame, width))
        if len(frames) < 2:
            raise ClipReadError(f"no decodable frames in {path}")
        return frames, fps
    finally:
        cap.release()


def read_pair(path, t_s: float, width: int = config.ANALYSIS_WIDTH
              ) -> Optional[tuple[np.ndarray, np.ndarray, float]]:
    """two consecutive frames at t_s for spot flow probes. returns none on
    seek or decode failure so callers can skip the position."""
    cap = _open(path)
    try:
        fps = _fps(cap)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        target = max(0, round(t_s * fps))
        if total > 0:
            target = min(target, max(0, total - 2))
        if target > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok1, f1 = cap.read()
        ok2, f2 = cap.read()
        if not (ok1 and ok2):
            return None
        return _downscale(f1, width), _downscale(f2, width), fps
    finally:
        cap.release()


def read_frame(path, t_s: float, width: int) -> Optional[np.ndarray]:
    """single frame at t_s, used for thumbnails."""
    pair = read_pair(path, t_s, width)
    return pair[0] if pair else None
