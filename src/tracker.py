import threading
import time
from dataclasses import dataclass

import mediapipe as mp
import numpy as np

_mp_holistic = mp.solutions.holistic

PALM_LANDMARKS = [0, 5, 9, 13, 17]


@dataclass
class TrackingResult:
    right_hand_pos: tuple | None
    left_hand_pos: tuple | None
    right_hand_landmarks: list | None
    left_hand_landmarks: list | None
    shoulders: tuple | None
    pose_landmarks: object | None   # raw MediaPipe NormalizedLandmarkList
    frame_time: float


class HandTracker:
    def __init__(self, min_detection_confidence: float = 0.6, min_tracking_confidence: float = 0.5):
        self._lock = threading.Lock()
        self._holistic = _mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame_rgb: np.ndarray) -> TrackingResult:
        t = time.perf_counter()
        with self._lock:
            results = self._holistic.process(frame_rgb)

        right_pos = _extract_hand_pos(results.right_hand_landmarks, mirror=False)
        left_pos = _extract_hand_pos(results.left_hand_landmarks, mirror=False)
        shoulders = _extract_shoulders(results.pose_landmarks)

        return TrackingResult(
            right_hand_pos=right_pos,
            left_hand_pos=left_pos,
            right_hand_landmarks=results.right_hand_landmarks,
            left_hand_landmarks=results.left_hand_landmarks,
            shoulders=shoulders,
            pose_landmarks=results.pose_landmarks,
            frame_time=t,
        )

    def close(self):
        self._holistic.close()


def _extract_hand_pos(landmarks, mirror: bool = True) -> tuple | None:
    if landmarks is None:
        return None
    pts = [landmarks.landmark[i] for i in PALM_LANDMARKS]
    x = float(np.mean([p.x for p in pts]))
    y = float(np.mean([p.y for p in pts]))
    if mirror:
        x = 1.0 - x
    return (x, y)


def _extract_shoulders(pose_landmarks) -> tuple | None:
    if pose_landmarks is None:
        return None
    lm = pose_landmarks.landmark
    r = lm[12]
    l = lm[11]
    return ((r.x, r.y), (l.x, l.y))


def compute_grid_bounds(shoulders: tuple | None, frame_shape: tuple) -> tuple:
    """Returns (x0, y0, x1, y1) — fixed full-frame grid, independent of body position."""
    H, W = frame_shape[:2]
    return (0, 0, W, H)


def get_hand_cell(pos: tuple | None, grid_bounds: tuple) -> int:
    """Returns 0-8 cell index (row-major) or -1 if outside grid. grid_bounds must be 6-tuple."""
    if pos is None:
        return -1
    return cell_from_normalized(pos, grid_bounds)


def cell_from_normalized(pos: tuple, grid_bounds: tuple) -> int:
    """Map normalized (0-1) frame position to 0-8 grid cell. grid_bounds must be 6-tuple."""
    if len(grid_bounds) < 6:
        return -1
    x0, y0, x1, y1, W, H = grid_bounds

    px = pos[0] * W
    py = pos[1] * H

    if px < x0 or px > x1 or py < y0 or py > y1:
        return -1

    col = int((px - x0) / (x1 - x0) * 3)
    row = int((py - y0) / (y1 - y0) * 3)
    col = min(col, 2)
    row = min(row, 2)
    return row * 3 + col


def compute_grid_bounds_with_shape(shoulders: tuple | None, frame_shape: tuple) -> tuple:
    """Returns 6-tuple (x0, y0, x1, y1, W, H) for get_hand_cell compatibility."""
    H, W = frame_shape[:2]
    x0, y0, x1, y1 = compute_grid_bounds(shoulders, frame_shape)
    return (x0, y0, x1, y1, W, H)


CELL_NAMES = {
    0: "Top-Left", 1: "Top-Center", 2: "Top-Right",
    3: "Mid-Left", 4: "Center", 5: "Mid-Right",
    6: "Bot-Left", 7: "Bot-Center", 8: "Bot-Right",
}
