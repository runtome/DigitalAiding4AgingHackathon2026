import os
import threading
import time
import urllib.request
from dataclasses import dataclass

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# Optional GPU support via torch (falls back to CPU gracefully)
try:
    import torch
    _USE_GPU = torch.cuda.is_available()
except ImportError:
    _USE_GPU = False

print(f"HandTracker: {'GPU (CUDA)' if _USE_GPU else 'CPU'} delegate")

# Model .task files download alongside this module on first run
_DIR = os.path.dirname(os.path.abspath(__file__))
_MODELS = {
    "hand": (
        os.path.join(_DIR, "hand_landmarker.task"),
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    ),
    "pose": (
        os.path.join(_DIR, "pose_landmarker_full.task"),
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
    ),
    "face": (
        os.path.join(_DIR, "face_landmarker.task"),
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    ),
}


def _ensure_models():
    for name, (path, url) in _MODELS.items():
        if not os.path.exists(path):
            print(f"  Downloading {name} model → {path}")
            urllib.request.urlretrieve(url, path)


PALM_LANDMARKS = [0, 5, 9, 13, 17]


@dataclass
class TrackingResult:
    right_hand_pos: tuple | None
    left_hand_pos: tuple | None
    right_hand_landmarks: list | None   # 21 NormalizedLandmark (Tasks API)
    left_hand_landmarks: list | None
    face_landmarks: list | None         # 478 NormalizedLandmark, first detected face
    shoulders: tuple | None
    pose_landmarks: list | None         # 33 NormalizedLandmark, first detected pose
    frame_time: float


def _make_base(model_path: str, gpu: bool) -> python.BaseOptions:
    delegate = python.BaseOptions.Delegate.GPU if gpu else python.BaseOptions.Delegate.CPU
    return python.BaseOptions(model_asset_path=model_path, delegate=delegate)


def _try_create(create_fn, make_opts_fn, name: str):
    """Try GPU delegate first, silently fall back to CPU."""
    if _USE_GPU:
        try:
            task = create_fn(make_opts_fn(gpu=True))
            print(f"  {name}: GPU")
            return task
        except Exception:
            print(f"  {name}: GPU failed → CPU")
    task = create_fn(make_opts_fn(gpu=False))
    if not _USE_GPU:
        print(f"  {name}: CPU")
    return task


class HandTracker:
    def __init__(self, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        _ensure_models()
        self._lock = threading.Lock()
        conf_d, conf_t = min_detection_confidence, min_tracking_confidence

        def hand_opts(gpu):
            return vision.HandLandmarkerOptions(
                base_options=_make_base(_MODELS["hand"][0], gpu),
                running_mode=vision.RunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=conf_d,
                min_tracking_confidence=conf_t,
            )

        def pose_opts(gpu):
            return vision.PoseLandmarkerOptions(
                base_options=_make_base(_MODELS["pose"][0], gpu),
                running_mode=vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=conf_d,
                min_tracking_confidence=conf_t,
            )

        def face_opts(gpu):
            return vision.FaceLandmarkerOptions(
                base_options=_make_base(_MODELS["face"][0], gpu),
                running_mode=vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=conf_d,
                min_tracking_confidence=conf_t,
            )

        print("Initializing MediaPipe Tasks landmarkers:")
        self._hand_lm = _try_create(vision.HandLandmarker.create_from_options, hand_opts, "Hand")
        self._pose_lm = _try_create(vision.PoseLandmarker.create_from_options, pose_opts, "Pose")
        self._face_lm = _try_create(vision.FaceLandmarker.create_from_options, face_opts, "Face")

    def process(self, frame_rgb: np.ndarray) -> TrackingResult:
        """Process one frame. frame_rgb must be an RGB uint8 array (Gradio webcam convention)."""
        t = time.perf_counter()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        with self._lock:
            hand_res = self._hand_lm.detect(mp_image)
            pose_res = self._pose_lm.detect(mp_image)
            face_res = self._face_lm.detect(mp_image)

        # Map hands to person-anatomical left/right.
        # The frame is selfie-mirrored before being passed here, so the Tasks API's
        # perspective-based handedness label is flipped: model "Left" = person's right.
        right_lm, left_lm = None, None
        for i, lm_list in enumerate(hand_res.hand_landmarks):
            label = hand_res.handedness[i][0].category_name
            if label == "Left":   # mirrored image → person's right hand
                right_lm = lm_list
            else:                  # mirrored image → person's left hand
                left_lm = lm_list

        pose_lm = pose_res.pose_landmarks[0] if pose_res.pose_landmarks else None
        face_lm = face_res.face_landmarks[0] if face_res.face_landmarks else None

        return TrackingResult(
            right_hand_pos=_extract_hand_pos(right_lm),
            left_hand_pos=_extract_hand_pos(left_lm),
            right_hand_landmarks=right_lm,
            left_hand_landmarks=left_lm,
            face_landmarks=face_lm,
            shoulders=_extract_shoulders(pose_lm),
            pose_landmarks=pose_lm,
            frame_time=t,
        )

    def close(self):
        self._hand_lm.close()
        self._pose_lm.close()
        self._face_lm.close()


def _extract_hand_pos(landmarks) -> tuple | None:
    if landmarks is None:
        return None
    pts = [landmarks[i] for i in PALM_LANDMARKS]
    return (float(np.mean([p.x for p in pts])), float(np.mean([p.y for p in pts])))


def _extract_shoulders(pose_landmarks) -> tuple | None:
    if pose_landmarks is None:
        return None
    r = pose_landmarks[12]
    l = pose_landmarks[11]
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
