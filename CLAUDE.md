# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI-powered upper limb dexterity assessment tool for aging adults (DigitalAiding4Aging Hackathon 2026). Participants reach toward highlighted zones on a 3×3 grid via webcam; the system measures reaction time, accuracy, movement quality, tremor, and detects Learned Non-Use (LNU) asymmetry between hands.

## Running the App

```bash
pip install -r requirements.txt
python app.py
```

The Gradio UI launches at `http://localhost:7860`. The webcam requires clicking the **⏺ Record** button in the UI before starting an assessment.

## Architecture

| File | Responsibility |
|---|---|
| `app.py` | Gradio UI wiring; all event handlers (`unified_stream`, `handle_start`, `handle_stop`, `check_game_over`, `handle_export`) |
| `src/tracker.py` | MediaPipe Holistic wrapper → `TrackingResult`; grid-to-cell mapping |
| `src/game_engine.py` | State machine (`GameState`); frame-by-frame hit/timeout logic |
| `src/motion_analyzer.py` | Post-game metrics (speed, accuracy, quality, LNU, motor age) → `AnalysisResult` |
| `src/visualizer.py` | OpenCV frame overlays (countdown, running, preview); all Plotly charts |
| `src/report_generator.py` | CSV and PDF export via pandas + fpdf2 |

### Data flow

```
Webcam frame (BGR numpy)
  → cv2.flip (selfie mirror)
  → HandTracker.process() → TrackingResult
  → GameEngine.process_frame() → (GameState, FrameResult)
  → draw_overlay() → annotated frame returned to Gradio
  [on game over via poll_timer]
  → MotionAnalyzer.analyze() → AnalysisResult
  → visualizer chart functions → Plotly figures shown in tabs
```

### State machine

`GameState.phase` transitions: `idle` → `countdown` (3 s) → `running` → `done` → `analyzed`

`GameState` follows a **copy-on-write** pattern: every mutation in `process_frame` does `state = copy.copy(state)` before modifying, because Gradio passes state by value across frames.

### Key constants and behaviors

- `_COUNTDOWN_S = 3` — seconds of 3-2-1 before test starts
- `MIN_DWELL_FRAMES = 3` — consecutive frames hand must stay in target cell to register a hit
- `TARGET_TIMEOUT_S = 5.0` — seconds before an unvisited target is marked a miss and replaced
- `_CELL_WEIGHTS` — corner cells (0,2,6,8) weighted 1.5×, center cell (4) weighted 0.5× for random target selection; last target excluded from selection
- The grid is always full-frame (`0, 0, W, H`), not body-relative
- `compute_grid_bounds_with_shape` returns a **6-tuple** `(x0, y0, x1, y1, W, H)`; `cell_from_normalized` requires this 6-tuple form
- `_stop_event` is a module-level `threading.Event` used to cross-thread the Stop button signal into the stream callback

### Clinical metrics

- **LNU (Learned Non-Use)**: weighted composite of use asymmetry (50%), reaction-time asymmetry (30%), and quality asymmetry (20%) between hands. Score 0–100; thresholds Low <33, Moderate 33–67, High >67.
- **Motor Age**: estimated from composite score and tremor power; formula in `_compute_motor_age`.
- **Tremor Power**: FFT of horizontal hand trajectory resampled to 10 Hz; power in 3–7 Hz band as % of total.
- **Jerk Score**: Normalized Jerk Score (NJS) from third derivative of position; lower NJS → higher score.

### Gradio streaming notes

- `webcam.stream` fires every 0.15 s (`stream_every=0.15`) with `concurrency_limit=1`
- A `gr.Timer(value=1.0)` polls `check_game_over` for the done→analyzed transition, since the stream may miss the final frame
- Chart generation and `MotionAnalyzer.analyze()` happen inside `check_game_over`, not in the stream
