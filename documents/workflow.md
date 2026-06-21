# Workflow

## Purpose

This application runs an AI-powered upper limb dexterity assessment. A participant reaches toward randomly highlighted cells in a 3x3 webcam grid. The system records hit/miss events, reaction times, movement trajectories, hand usage, movement quality, learned non-use risk, dominant-hand prediction, and estimated motor age.

The tool is an experimental screening aid only. It is not a clinical diagnostic device.

## High-Level Runtime Flow

1. User launches the app with `python app.py`.
2. Gradio opens the browser UI at the local server.
3. User enters setup information:
   - Test duration: 30 seconds, 1 minute, 2 minutes, 3 minutes, or 5 minutes.
   - Hands to assess: right, left, or both.
   - Optional participant name.
   - Optional participant age.
4. User clicks Start Assessment.
5. The app creates a `GameState` in countdown phase and shows the assessment panel.
6. User clicks Open Live Video.
7. `AssessmentRunner` starts `src/run_assessment.py` as a subprocess.
8. The subprocess opens a fullscreen OpenCV camera window.
9. MediaPipe hand, pose, and face landmarkers initialize.
10. A 3-second countdown is displayed.
11. The game enters running phase.
12. Each video frame is processed:
    - Camera frame is mirrored.
    - MediaPipe detects hands, pose, and face landmarks.
    - Hand positions are mapped into the 3x3 grid.
    - `GameEngine.process_frame()` checks whether the active hand entered the target cell.
    - The overlay draws grid, target, hands, skeleton, timer, hit count, and progress bar.
13. A hit is registered when the hand remains in the target cell for 3 consecutive frames.
14. A miss is registered if the current target is not hit within 5 seconds.
15. After every hit or miss, a new weighted random target is selected.
16. The assessment ends when:
    - Duration expires.
    - User presses Stop Early.
    - User closes the OpenCV window.
    - User presses `q` or `Esc` in the OpenCV window.
17. The subprocess writes the final `GameState` to a temporary pickle file.
18. The Gradio app polls every second with `gr.Timer`.
19. When the final state is available, `MotionAnalyzer` computes metrics.
20. `visualizer.py` builds summary markdown, Plotly charts, gauges, event log table, and dominance prediction text.
21. Results panels become visible in the Gradio UI.
22. User can generate CSV and PDF reports.

## Module Workflow

### `app.py`

`app.py` is the main application entry point. It owns:

- Gradio UI layout.
- Global `HandTracker`, `GameEngine`, and `AssessmentRunner` objects.
- Setup, start, open-video, stop, polling, analysis, and export handlers.
- Visibility state for setup, assessment, analysis, prediction, and report panels.

Important handlers:

| Handler | Role |
|---|---|
| `handle_start()` | Creates a new game state with duration, hand mode, participant name, and age. |
| `handle_open_video()` | Starts the OpenCV subprocess through `AssessmentRunner`. |
| `handle_stop()` | Signals the subprocess to stop early. |
| `check_game_over()` | Polls for subprocess completion, analyzes final state, and populates charts/results. |
| `handle_export()` | Exports event data to CSV and summary/charts to PDF. |

### `src/assessment_runner.py`

`AssessmentRunner` isolates OpenCV camera display from the Gradio process. This is especially important on Windows because `cv2.imshow()` is more reliable when it owns its own process and main thread.

Workflow:

1. Creates a temporary output pickle file path.
2. Creates a temporary stop-signal file path.
3. Launches `src/run_assessment.py` through `subprocess.Popen`.
4. Parent process keeps polling `is_done`.
5. When the pickle file appears, parent loads the final `GameState`.
6. If user stops early, parent creates the stop-signal file.

### `src/run_assessment.py`

This is the standalone live assessment process.

Workflow:

1. Parses CLI arguments from `AssessmentRunner`.
2. Opens a fullscreen, always-on-top OpenCV window.
3. Imports heavy MediaPipe components after showing an initial loading window.
4. Creates `HandTracker` and `GameEngine`.
5. Opens webcam device `0`.
6. Sets camera target resolution to 1280x720.
7. Runs a 3-second countdown.
8. Processes frames until done or stopped.
9. Draws all live overlays.
10. Plays a Windows system sound on successful hits.
11. Shows a finish screen for 2 seconds when duration expires.
12. Releases camera, closes tracker/window, and writes final state to pickle.

### `src/tracker.py`

`HandTracker` wraps MediaPipe Tasks.

It manages:

- Hand landmarker.
- Pose landmarker.
- Face landmarker.
- Optional CUDA detection through `torch`, falling back to CPU.
- First-run download of `.task` model files if missing.

Frame output is a `TrackingResult` containing:

- Right hand palm center.
- Left hand palm center.
- Hand landmarks.
- Pose landmarks.
- Face landmarks.
- Shoulder landmarks.
- Frame timestamp.

Grid mapping:

- The current implementation uses the full camera frame as the 3x3 grid.
- Normalized hand coordinates are converted to row-major cell IDs from `0` to `8`.
- Display labels show these as zones `1` to `9`.

### `src/game_engine.py`

`GameEngine` owns assessment state transitions and event recording.

Game phases:

| Phase | Meaning |
|---|---|
| `idle` | No active game. |
| `countdown` | Game was created but live countdown is still running. |
| `running` | Frames are evaluated for hit/miss events. |
| `done` | Assessment duration ended or process stopped. |
| `analyzed` | Final state has been analyzed in the Gradio app. |

Hit rules:

- Current target is a 0-based grid cell.
- The participant must dwell inside the target for `MIN_DWELL_FRAMES = 3`.
- A target times out after `TARGET_TIMEOUT_S = 5.0`.
- In `both` mode, either hand can hit the target.
- In single-hand mode, only the selected hand is evaluated.

Target selection:

- Uses weighted random selection.
- Corners have higher weight: `1.5`.
- Edges have weight: `1.0`.
- Center has lower weight: `0.5`.
- The previous target is excluded.
- Cells currently occupied by a hand are excluded when possible.

Event data recorded per target:

- Event ID.
- Target cell.
- Hand used.
- Target shown timestamp.
- Hit timestamp.
- Reaction time in milliseconds.
- Success flag.
- Movement trajectory.
- Path length.
- Direct distance.

### `src/motion_analyzer.py`

`MotionAnalyzer.analyze()` converts raw game events into an `AnalysisResult`.

It computes:

- Reaction-time lists per hand.
- Mean, median, best, and worst reaction time.
- Speed score.
- Hit rate and accuracy score.
- Hit totals per grid cell.
- Path efficiency.
- Jerk score.
- Tremor power.
- Movement quality score.
- Composite score.
- Dominant-hand prediction.
- Learned non-use risk.
- Motor age estimate.

### `src/visualizer.py`

This module renders both live OpenCV overlays and post-assessment Plotly outputs.

Live OpenCV rendering:

- Countdown overlay.
- 3x3 grid.
- Current target highlight.
- Hit and miss flash effects.
- Pose skeleton.
- Hand skeletons.
- Face mesh.
- Hand position dots.
- Remaining time.
- Hit count.
- Progress bar.

Post-assessment rendering:

- Summary markdown.
- Speed box plot.
- Reaction-time distribution and statistics.
- Accuracy by target zone.
- Movement quality chart.
- Hand dominance radar chart.
- LNU gauge.
- Motor age gauge.
- Reaching timeline/Gantt chart.
- Event log table.
- Dominant-hand prediction markdown.

### `src/report_generator.py`

This module exports results.

CSV export:

- Writes one row per event.
- Includes event ID, target cell, hand, success, reaction time, path length, direct distance, target shown time, and hit time.

PDF export:

- Creates a summary report with participant metadata and hand-level metrics.
- Adds dominance, LNU, motor age, and disclaimer text.
- Attempts to embed Plotly charts as PNG images through Kaleido.
- Adds a methodology page.

### `src/cv2_popup.py`

This file defines `PopupDisplay`, a threaded OpenCV popup helper with a bounded frame queue. It is present in the codebase but the current main workflow uses the subprocess-based `AssessmentRunner` and `run_assessment.py` path instead.

## User Workflow

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the app:

```bash
python app.py
```

3. Open the Gradio URL, normally:

```text
http://localhost:7860
```

4. Fill in setup details.
5. Click Start Assessment.
6. Click Open Live Video.
7. Stand 2-3 meters from the camera.
8. Reach toward the highlighted zones until the assessment ends.
9. Review Summary, Speed, Accuracy, Movement Quality, Hand Dominance, Prediction, Timeline, and Event Log.
10. Generate CSV and PDF report if needed.

## Data Flow

```text
Camera frame
  -> OpenCV mirror flip
  -> RGB conversion
  -> MediaPipe hand/pose/face detection
  -> normalized hand positions
  -> 3x3 grid cell mapping
  -> GameEngine hit/miss evaluation
  -> GameState event list
  -> final pickle from subprocess to Gradio parent
  -> MotionAnalyzer metrics
  -> visualizer charts and tables
  -> optional CSV/PDF export
```

## Stop and Completion Flow

Normal completion:

```text
duration expires
  -> GameState.phase = done
  -> finish screen appears
  -> final state written to output pickle
  -> Gradio poll loads final state
  -> analysis panels shown
```

Early stop:

```text
user clicks Stop Early
  -> AssessmentRunner creates stop-signal file
  -> subprocess sees file
  -> loop exits
  -> final state written to output pickle
  -> Gradio poll loads final state
```

Window/key stop:

```text
user closes OpenCV window or presses q/Esc
  -> subprocess loop exits
  -> final state written to output pickle
  -> Gradio poll loads final state
```

## Important Runtime Notes

- The application assumes webcam index `0`.
- The live assessment uses Windows-specific APIs in `run_assessment.py`:
  - `ctypes.windll.user32` for always-on-top.
  - `winsound` for hit feedback.
- MediaPipe model files are stored in `src`.
- If model files are missing, `tracker.py` attempts to download them from Google Cloud Storage.
- The app imports and creates a global `HandTracker` in `app.py`, while the subprocess also creates its own tracker.
- The subprocess is the active live assessment path.
- The final state transfer uses a temporary pickle file, not sockets or shared memory.

