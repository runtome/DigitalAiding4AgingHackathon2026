import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="starlette")

import copy
import threading
import time

import cv2
import gradio as gr
import numpy as np

_stop_event = threading.Event()   # set by Stop button; cleared by stream

from src.game_engine import GameEngine, GameState
from src.motion_analyzer import MotionAnalyzer
from src.report_generator import export_csv, export_pdf
from src.tracker import (
    HandTracker,
    compute_grid_bounds_with_shape,
    CELL_NAMES,
)
from src.visualizer import (
    draw_countdown_overlay,
    draw_overlay,
    draw_preview_overlay,
    make_accuracy_chart,
    make_dominance_chart,
    make_gantt_chart,
    make_lnu_gauge,
    make_motor_age_gauge,
    make_quality_chart,
    make_speed_chart,
    make_summary_markdown,
)

_tracker = HandTracker()
_engine = GameEngine()


# ─── Unified stream handler (runs every frame regardless of phase) ──────────

_COUNTDOWN_S = 3  # seconds of "3-2-1" before test starts


def _bgr_to_gradio(bgr_frame: np.ndarray) -> np.ndarray:
    """Convert an BGR overlay frame to the RGB format Gradio expects.

    Gradio's webcam Image component applies CSS scaleX(-1) to the displayed
    element, which would double-flip our Python-mirrored frame and produce a
    non-mirrored display.  We flip back to raw orientation here so that
    Gradio's single CSS flip delivers the correct selfie/mirror view.
    """
    return cv2.cvtColor(cv2.flip(bgr_frame, 1), cv2.COLOR_BGR2RGB)


def _make_done(state, frame_bgr, grid_bounds, pose_ok, tracking):
    """Transition state to done and return preview frame."""
    _stop_event.clear()
    s = copy.copy(state) if state is not None else state
    if s is not None:
        s.events = list(s.events)
        s.phase = "done"
    out = draw_preview_overlay(
        frame_bgr, grid_bounds, pose_ok,
        tracking.pose_landmarks, tracking.right_hand_landmarks,
        tracking.left_hand_landmarks, tracking.face_landmarks,
    )
    return _bgr_to_gradio(out), s, gr.update(), gr.update(), gr.update()


def unified_stream(frame: np.ndarray, state: GameState):
    """Single handler for preview / countdown / assessment phases."""
    _no_update = (gr.update(), state, gr.update(), gr.update(), gr.update())
    try:
        if frame is None or not isinstance(frame, np.ndarray):
            return _no_update

        # Flip for selfie/mirror-view processing; all coordinates and cell detection
        # use this mirrored space.  _bgr_to_gradio() flips back before returning so
        # Gradio's CSS scaleX(-1) on the webcam element gives the final mirror display.
        frame = cv2.flip(frame, 1)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        tracking = _tracker.process(frame)  # MediaPipe expects RGB
        grid_bounds = compute_grid_bounds_with_shape(tracking.shoulders, frame.shape)
        pose_ok = tracking.shoulders is not None
        now = time.perf_counter()

        # Stop button was pressed — end immediately from any active phase
        if _stop_event.is_set() and state is not None and state.phase in ("countdown", "running"):
            return _make_done(state, frame_bgr, grid_bounds, pose_ok, tracking)

        # ── Countdown phase ──────────────────────────────────────────────────
        if state is not None and state.phase == "countdown":
            remaining = max(0.0, _COUNTDOWN_S - (now - state.start_time))
            if remaining <= 0:
                state = copy.copy(state)
                state.events = list(state.events)
                state.phase = "running"
                state.start_time = now
                state.target_start_time = now
            out = draw_countdown_overlay(
                frame_bgr, tracking.pose_landmarks,
                int(remaining) + 1 if remaining > 0 else 0,
                tracking.right_hand_landmarks, tracking.left_hand_landmarks, tracking.face_landmarks,
            )
            return _bgr_to_gradio(out), state, gr.update(), gr.update(), gr.update()

        # ── Preview (idle / analyzed) ────────────────────────────────────────
        if state is None or state.phase not in ("running", "done"):
            out = draw_preview_overlay(
                frame_bgr, grid_bounds, pose_ok,
                tracking.pose_landmarks, tracking.right_hand_landmarks,
                tracking.left_hand_landmarks, tracking.face_landmarks,
            )
            return _bgr_to_gradio(out), state, gr.update(), gr.update(), gr.update()

        # ── Running ──────────────────────────────────────────────────────────
        if state.phase == "running":
            state, result = _engine.process_frame(
                state,
                tracking.right_hand_pos,
                tracking.left_hand_pos,
                grid_bounds,
                now,
            )
            out = draw_overlay(
                frame_bgr, state, result, grid_bounds,
                tracking.right_hand_pos, tracking.left_hand_pos,
                tracking.pose_landmarks, tracking.right_hand_landmarks,
                tracking.left_hand_landmarks, tracking.face_landmarks,
            )
            target_name = CELL_NAMES.get(state.current_target, "—")
            return _bgr_to_gradio(out), state, float(result.remaining_s), int(result.hit_count), target_name

        # ── Done (waiting for poll_timer) ────────────────────────────────────
        out = draw_preview_overlay(
            frame_bgr, grid_bounds, pose_ok,
            tracking.pose_landmarks, tracking.right_hand_landmarks,
            tracking.left_hand_landmarks, tracking.face_landmarks,
        )
        return _bgr_to_gradio(out), state, gr.update(), gr.update(), gr.update()

    except Exception:
        import traceback
        traceback.print_exc()
        return _no_update


# ─── Control handlers ────────────────────────────────────────────────────────

def handle_start(duration, hand_side, name, age):
    try:
        dur = int(duration)
    except Exception:
        dur = 60
    state = _engine.start(dur, hand_side)
    state.participant_name = name or ""
    state.participant_age = int(age) if age else None
    return (
        state,
        gr.update(visible=False),  # hide setup_panel
        gr.update(visible=True),   # show assessment_panel
        gr.update(visible=False),  # hide analysis_panel
        gr.update(visible=False),  # hide report_panel
    )


def handle_stop(state: GameState):
    _stop_event.set()          # signal the stream to stop on its next frame
    if state is not None and state.phase in ("running", "countdown"):
        s = copy.copy(state)
        s.events = list(s.events)
        s.phase = "done"
        return s
    return state


# ─── Game-over polling ────────────────────────────────────────────────────────

_NO_UPDATE_COUNT = 14  # number of outputs in check_game_over

def check_game_over(state: GameState):
    no_op = tuple([gr.update()] * (_NO_UPDATE_COUNT - 1) + [state])
    if state is None:
        return no_op

    # Backup auto-stop: if stream missed the final frame
    if state.phase == "running":
        elapsed = time.perf_counter() - state.start_time
        if elapsed >= state.duration_s:
            state = copy.copy(state)
            state.events = list(state.events)
            state.phase = "done"

    if state.phase != "done":
        return no_op

    # Mark as analyzed so timer doesn't re-run
    state = copy.copy(state)
    state.events = state.events
    state.phase = "analyzed"

    analysis = MotionAnalyzer().analyze(state)

    summary_md = make_summary_markdown(analysis, state)
    speed_fig   = make_speed_chart(state.events)
    acc_fig     = make_accuracy_chart(state.events)
    qual_fig    = make_quality_chart(state.events, analysis)
    dom_fig     = make_dominance_chart(analysis)
    lnu_fig     = make_lnu_gauge(analysis.lnu_score, analysis.lnu_risk)
    motor_fig   = make_motor_age_gauge(analysis.motor_age, state.participant_age)
    gantt_fig   = make_gantt_chart(state.events, state.duration_s)

    return (
        analysis,          # analysis_res
        summary_md,        # summary_md
        speed_fig,         # speed_plot
        acc_fig,           # accuracy_plot
        qual_fig,          # quality_plot
        dom_fig,           # dominance_plot
        lnu_fig,           # lnu_gauge_plot
        motor_fig,         # motor_age_plot
        gantt_fig,         # gantt_plot
        gr.update(visible=False),  # assessment_panel
        gr.update(visible=True),   # analysis_panel
        gr.update(visible=True),   # report_panel
        gr.update(visible=True),   # setup_panel (allow restart)
        state,
    )


# ─── Export ──────────────────────────────────────────────────────────────────

def handle_export(state: GameState, analysis):
    if state is None or not state.events or analysis is None:
        return None, None
    charts_for_pdf = {
        "Speed Analysis": make_speed_chart(state.events),
        "Accuracy by Zone": make_accuracy_chart(state.events),
        "LNU Risk": make_lnu_gauge(analysis.lnu_score, analysis.lnu_risk),
        "Motor Age": make_motor_age_gauge(analysis.motor_age, state.participant_age),
        "Gantt": make_gantt_chart(state.events, state.duration_s),
    }
    csv_path = export_csv(state.events)
    pdf_path = export_pdf(analysis, charts_for_pdf, state)
    return csv_path, pdf_path


# ─── Gradio UI ───────────────────────────────────────────────────────────────

CSS = """
#main-cam { border-radius: 12px; }
#main-cam img, #main-cam video {
    width: 100% !important;
    max-height: 480px !important;
    object-fit: contain !important;
}
.gr-group { border-radius: 10px; padding: 12px; }
"""

with gr.Blocks(title="Upper Limb Dexterity Assessment") as demo:

    game_state   = gr.State(value=None)
    analysis_res = gr.State(value=None)

    gr.Markdown(
        "# Upper Limb Dexterity Assessment\n"
        "*AI-Powered Reaching Task — Sarcopenia & Learned Non-Use Screening*"
    )
    gr.Markdown(
        "**How to use:** Click the **⏺ Record** button on the camera below to start the webcam, "
        "then click **Start Assessment**.\n\n"
        "> **Screening tool only.** Results are experimental indicators, not clinical diagnoses. "
        "Consult a healthcare professional for medical evaluation."
    )

    # Single component: webcam feed in, processed frame out (overlay drawn server-side)
    webcam = gr.Image(
        sources=["webcam"],
        type="numpy",
        streaming=True,
        label="Camera Feed  (click ⏺ Record first)",
        elem_id="main-cam",
        height=480,
    )

    # Setup Panel
    with gr.Group(visible=True) as setup_panel:
        gr.Markdown("### ⚙️ Test Setup\nStand 2–3 m from camera so your full body and hands are visible.")
        with gr.Row():
            with gr.Column(scale=1):
                duration_radio = gr.Radio(
                    choices=[("30 sec", 30), ("1 min", 60), ("2 min", 120), ("3 min", 180), ("5 min", 300)],
                    value=60,
                    label="Test Duration",
                )
                hand_radio = gr.Radio(
                    choices=[("Right Hand", "right"), ("Left Hand", "left"), ("Both Hands", "both")],
                    value="right",
                    label="Hands to Assess",
                    info="Both Hands measures free-choice reaching. LNU risk requires Both Hands with ≥3 hits per hand.",
                )
            with gr.Column(scale=1):
                participant_name = gr.Textbox(label="Name (optional)", placeholder="e.g. John Doe")
                participant_age  = gr.Number(label="Age (optional)", precision=0, minimum=1, maximum=120)
        start_btn = gr.Button("▶  Start Assessment", variant="primary", size="lg")

    # Assessment Panel
    with gr.Group(visible=False) as assessment_panel:
        gr.Markdown("### 🎯 Reach toward the **green highlighted zone**")
        with gr.Row():
            remaining_display = gr.Number(label="⏱ Time Remaining (s)", interactive=False)
            hits_display      = gr.Number(label="✅ Hits", interactive=False)
            target_display    = gr.Textbox(label="Current Target Zone", interactive=False)
        stop_btn = gr.Button("⏹  Stop Early", variant="stop")

    # Analysis Panel
    with gr.Group(visible=False) as analysis_panel:
        gr.Markdown("### 📊 Analysis Results")
        with gr.Tabs():
            with gr.Tab("Summary"):
                summary_md = gr.Markdown()
                with gr.Row():
                    lnu_gauge_plot = gr.Plot(label="LNU Risk Score")
                    motor_age_plot = gr.Plot(label="Motor Age Estimate")
            with gr.Tab("Speed"):
                speed_plot = gr.Plot()
            with gr.Tab("Accuracy"):
                accuracy_plot = gr.Plot()
            with gr.Tab("Movement Quality"):
                quality_plot = gr.Plot()
            with gr.Tab("Hand Dominance"):
                dominance_plot = gr.Plot()

    # Report Panel
    with gr.Group(visible=False) as report_panel:
        gr.Markdown("### 📄 Report & Export")
        gantt_plot = gr.Plot(label="Reaching Events Timeline")
        with gr.Row():
            export_btn   = gr.Button("📥 Generate PDF & CSV Report", variant="primary")
            download_csv = gr.File(label="CSV Data")
            download_pdf = gr.File(label="PDF Report")

    # Timer for game-over detection
    poll_timer = gr.Timer(value=1.0, active=True)

    # ─── Event Wiring ────────────────────────────────────────────────────────

    webcam.stream(
        fn=unified_stream,
        inputs=[webcam, game_state],
        outputs=[webcam, game_state, remaining_display, hits_display, target_display],
        stream_every=0.15,
        concurrency_limit=1,
    )

    start_btn.click(
        fn=handle_start,
        inputs=[duration_radio, hand_radio, participant_name, participant_age],
        outputs=[game_state, setup_panel, assessment_panel, analysis_panel, report_panel],
    )

    stop_btn.click(
        fn=handle_stop,
        inputs=[game_state],
        outputs=[game_state],
    )

    poll_timer.tick(
        fn=check_game_over,
        inputs=[game_state],
        outputs=[
            analysis_res,
            summary_md,
            speed_plot,
            accuracy_plot,
            quality_plot,
            dominance_plot,
            lnu_gauge_plot,
            motor_age_plot,
            gantt_plot,
            assessment_panel,
            analysis_panel,
            report_panel,
            setup_panel,
            game_state,
        ],
    )

    export_btn.click(
        fn=handle_export,
        inputs=[game_state, analysis_res],
        outputs=[download_csv, download_pdf],
    )

if __name__ == "__main__":
    demo.launch(share=False, theme=gr.themes.Soft(), css=CSS)
