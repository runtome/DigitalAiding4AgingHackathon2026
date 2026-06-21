"""Standalone assessment window — launched as a subprocess by AssessmentRunner.

Running as a subprocess gives this script its own main thread, which is required
for cv2.imshow to work on Windows.  The parent process (Gradio app) passes
assessment parameters via CLI args and reads the result via a pickle file.
"""
import sys
import os

# Ensure the project root (parent of this file's directory) is on sys.path so
# that `from src.X import …` works whether this script is launched with
# `python src/run_assessment.py` or `python -m src.run_assessment`.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import argparse
import copy
import pickle
import time
import winsound

import cv2
import numpy as np


def _loading_frame(msg: str) -> np.ndarray:
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(img, msg, (60, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 220, 100), 2, cv2.LINE_AA)
    return img


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration",  type=int,   required=True)
    parser.add_argument("--hand",                  required=True)
    parser.add_argument("--name",      default="")
    parser.add_argument("--age",       type=int,   default=0)
    parser.add_argument("--output",                required=True)
    parser.add_argument("--stopfile",              required=True)
    args = parser.parse_args()

    _TITLE      = "Assessment — Live View"
    _COUNTDOWN  = 3.0

    # ── Show a window immediately (before heavy imports load) ────────────────
    cv2.namedWindow(_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(_TITLE, 960, 720)
    cv2.imshow(_TITLE, _loading_frame("Loading camera..."))
    cv2.waitKey(1)

    # ── Heavy imports (MediaPipe models load here) ───────────────────────────
    from src.tracker import HandTracker, compute_grid_bounds_with_shape
    from src.game_engine import GameEngine
    from src.visualizer import draw_countdown_overlay, draw_overlay

    cv2.imshow(_TITLE, _loading_frame("Initializing tracker..."))
    cv2.waitKey(1)

    tracker = HandTracker()
    engine  = GameEngine()

    state = engine.start(args.duration, args.hand)
    state.participant_name = args.name or ""
    state.participant_age  = args.age if args.age > 0 else None

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cv2.imshow(_TITLE, _loading_frame("Camera not found!"))
        cv2.waitKey(2000)
        cv2.destroyAllWindows()
        tracker.close()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    countdown_start = time.perf_counter()

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            # Stop signal: parent creates this file to request early stop
            if os.path.exists(args.stopfile):
                break

            # User closed the window
            if cv2.getWindowProperty(_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                break

            frame_bgr = cv2.flip(frame_bgr, 1)   # selfie / mirror view
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            tracking  = tracker.process(frame_rgb)
            grid_bounds = compute_grid_bounds_with_shape(tracking.shoulders, frame_bgr.shape)
            now = time.perf_counter()

            # ── Countdown ────────────────────────────────────────────────────
            if state.phase == "countdown":
                remaining_cd = max(0.0, _COUNTDOWN - (now - countdown_start))
                if remaining_cd <= 0:
                    state = copy.copy(state)
                    state.events = list(state.events)
                    state.phase = "running"
                    state.start_time = now
                    state.target_start_time = now
                out = draw_countdown_overlay(
                    frame_bgr, tracking.pose_landmarks,
                    int(remaining_cd) + 1 if remaining_cd > 0 else 0,
                    tracking.right_hand_landmarks,
                    tracking.left_hand_landmarks,
                    tracking.face_landmarks,
                )

            # ── Running ──────────────────────────────────────────────────────
            elif state.phase == "running":
                state, result = engine.process_frame(
                    state,
                    tracking.right_hand_pos,
                    tracking.left_hand_pos,
                    grid_bounds,
                    now,
                )
                out = draw_overlay(
                    frame_bgr, state, result, grid_bounds,
                    tracking.right_hand_pos, tracking.left_hand_pos,
                    tracking.pose_landmarks,
                    tracking.right_hand_landmarks,
                    tracking.left_hand_landmarks,
                    tracking.face_landmarks,
                )
                if result.event == "hit":
                    winsound.PlaySound("SystemAsterisk", winsound.SND_ALIAS | winsound.SND_ASYNC)
                if state.phase == "done":
                    cv2.imshow(_TITLE, out)
                    cv2.waitKey(800)   # linger so user sees the "done" overlay
                    break

            else:
                break

            cv2.imshow(_TITLE, out)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):   # q or ESC
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()
        # Write final state for the parent process to pick up
        with open(args.output, "wb") as fh:
            pickle.dump(state, fh)


if __name__ == "__main__":
    main()
