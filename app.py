import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="starlette")

import copy

import gradio as gr

from src.assessment_runner import AssessmentRunner
from src.game_engine import GameEngine, GameState
from src.motion_analyzer import MotionAnalyzer
from src.report_generator import export_pdf
from src.tracker import HandTracker
from src.visualizer import (
    make_accuracy_chart,
    make_dominance_chart,
    make_dominance_prediction_md,
    make_event_log_df,
    make_gantt_chart,
    make_lnu_gauge,
    make_motor_age_gauge,
    make_quality_chart,
    make_speed_chart,
    make_speed_normal_dist_chart,
    make_summary_markdown,
)

_tracker = HandTracker()
_engine  = GameEngine()
_runner  = AssessmentRunner(_tracker, _engine)


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
        gr.update(visible=False),       # hide setup_panel
        gr.update(visible=True),        # show assessment_panel
        gr.update(visible=False),       # hide analysis_panel
        gr.update(visible=False),       # hide prediction_panel
        gr.update(visible=False),       # hide report_panel
        gr.update(interactive=True),    # re-enable open_video_btn for this new game
    )


def handle_open_video(state: GameState):
    """Launch the cv2 assessment window.  Disables this button while running."""
    if state is not None:
        _runner.start(state)
    return gr.update(interactive=False)


def handle_stop(state: GameState):
    _runner.stop()
    return state


# ─── Game-over polling ────────────────────────────────────────────────────────

_NO_UPDATE_COUNT = 18   # number of outputs in check_game_over


def check_game_over(state: GameState):
    no_op = tuple([gr.update()] * (_NO_UPDATE_COUNT - 1) + [state])
    if state is None or not _runner.is_done:
        return no_op

    # Consume the runner's final state (clears is_done so this block runs only once)
    final = _runner.final_state
    _runner.final_state = None
    if final is None:
        return no_op

    state = copy.copy(final)
    state.events = list(state.events)
    state.phase = "analyzed"

    analysis      = MotionAnalyzer().analyze(state)
    summary_md    = make_summary_markdown(analysis, state)
    speed_fig     = make_speed_chart(state.events)
    speed_nd_fig  = make_speed_normal_dist_chart(state.events)
    acc_fig       = make_accuracy_chart(state.events)
    qual_fig      = make_quality_chart(state.events, analysis)
    dom_fig       = make_dominance_chart(analysis)
    lnu_fig       = make_lnu_gauge(analysis.lnu_score, analysis.lnu_risk)
    motor_fig     = make_motor_age_gauge(analysis.motor_age, state.participant_age)
    gantt_fig     = make_gantt_chart(state.events, state.duration_s)
    event_log_df  = make_event_log_df(state.events, state.start_time)
    dom_pred_md   = make_dominance_prediction_md(analysis)

    return (
        analysis,
        summary_md,
        speed_fig,
        speed_nd_fig,
        acc_fig,
        qual_fig,
        dom_fig,
        lnu_fig,
        motor_fig,
        gantt_fig,
        event_log_df,                   # event_log_table
        gr.update(visible=False),       # assessment_panel
        gr.update(visible=True),        # analysis_panel
        gr.update(visible=True),        # prediction_panel
        dom_pred_md,                    # dominant_prediction_md
        gr.update(visible=True),        # report_panel
        gr.update(visible=True),        # setup_panel (allow restart)
        state,
    )


# ─── Export ──────────────────────────────────────────────────────────────────

def handle_export(state: GameState, analysis):
    if state is None or not state.events or analysis is None:
        return None
    charts_for_pdf = {
        "Speed Analysis":   make_speed_chart(state.events),
        "Accuracy by Zone": make_accuracy_chart(state.events),
        "LNU Risk":         make_lnu_gauge(analysis.lnu_score, analysis.lnu_risk),
        "Motor Age":        make_motor_age_gauge(analysis.motor_age, state.participant_age),
        "Gantt":            make_gantt_chart(state.events, state.duration_s),
    }
    return export_pdf(analysis, charts_for_pdf, state)


# ─── Gradio UI ───────────────────────────────────────────────────────────────

CSS = """
.gr-group { border-radius: 10px; padding: 12px; }
.gradio-container { font-size: 16px; }
.gradio-container p,
.gradio-container label,
.gradio-container .label-wrap span,
.gradio-container .prose { font-size: 16px !important; }
.gradio-container button { font-size: 15px !important; }
.gradio-container input,
.gradio-container select { font-size: 15px !important; }
"""

with gr.Blocks(title="Upper Limb Dexterity Assessment") as demo:

    game_state   = gr.State(value=None)
    analysis_res = gr.State(value=None)

    gr.Markdown(
        "# Upper Limb Dexterity Assessment\n"
        "*AI-Powered Reaching Task — Sarcopenia & Learned Non-Use Screening*"
    )

    # ── Setup Panel ──────────────────────────────────────────────────────────
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
                    value="both",
                    label="Hands to Assess",
                    info="Both Hands measures free-choice reaching. LNU risk requires Both Hands with ≥3 hits per hand.",
                )
            with gr.Column(scale=1):
                participant_name = gr.Textbox(label="Name (optional)", placeholder="e.g. John Doe")
                participant_age  = gr.Number(label="Age (optional)", precision=0, minimum=1, maximum=120, value=45)
        start_btn = gr.Button("▶  Start Assessment", variant="primary", size="lg")

    # ── Assessment Panel ─────────────────────────────────────────────────────
    with gr.Group(visible=False) as assessment_panel:
        gr.Markdown(
            "### 🎯 Ready to begin\n"
            "Click **Open Live Video** to launch the camera window and start the 3-2-1 countdown."
        )
        open_video_btn = gr.Button("🎥  Open Live Video", variant="primary", size="lg")
        stop_btn       = gr.Button("⏹  Stop Early", variant="stop")

    # ── Analysis Panel ───────────────────────────────────────────────────────
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
                speed_normal_dist_plot = gr.Plot(label="Reaction Time Normal Distribution Analysis")
            with gr.Tab("Accuracy"):
                accuracy_plot = gr.Plot()
            with gr.Tab("Movement Quality"):
                quality_plot = gr.Plot()
            with gr.Tab("Hand Dominance"):
                dominance_plot = gr.Plot()

    # ── Dominant Hand Prediction Panel ───────────────────────────────────────
    with gr.Group(visible=False) as prediction_panel:
        dominant_prediction_md = gr.Markdown()

    # ── Report Panel ─────────────────────────────────────────────────────────
    with gr.Group(visible=False) as report_panel:
        gr.Markdown("### 📄 Report & Export")
        gantt_plot = gr.Plot(label="Reaching Events Timeline")
        event_log_table = gr.Dataframe(
            label="Event Log",
            interactive=False,
            wrap=True,
        )
        with gr.Row():
            export_btn   = gr.Button("📥 Generate PDF Report", variant="primary")
            download_pdf = gr.File(label="PDF Report")

    # Timer polls for assessment completion every second
    poll_timer = gr.Timer(value=1.0, active=True)

    # ── Event Wiring ─────────────────────────────────────────────────────────

    start_btn.click(
        fn=handle_start,
        inputs=[duration_radio, hand_radio, participant_name, participant_age],
        outputs=[game_state, setup_panel, assessment_panel, analysis_panel, prediction_panel, report_panel, open_video_btn],
    )

    open_video_btn.click(
        fn=handle_open_video,
        inputs=[game_state],
        outputs=[open_video_btn],
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
            speed_normal_dist_plot,
            accuracy_plot,
            quality_plot,
            dominance_plot,
            lnu_gauge_plot,
            motor_age_plot,
            gantt_plot,
            event_log_table,
            assessment_panel,
            analysis_panel,
            prediction_panel,
            dominant_prediction_md,
            report_panel,
            setup_panel,
            game_state,
        ],
    )

    export_btn.click(
        fn=handle_export,
        inputs=[game_state, analysis_res],
        outputs=[download_pdf],
    )

if __name__ == "__main__":
    demo.launch(share=False, theme=gr.themes.Soft(), css=CSS)
