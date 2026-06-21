# Specification Sheet

## Product Name

DigitalAiding4Aging Hackathon 2026 - AI-Powered Upper Limb Dexterity Assessment

## Product Summary

A local webcam-based assessment application for screening upper limb dexterity. The participant reaches toward randomly highlighted zones on a 3x3 grid while the system tracks hand movement with MediaPipe and calculates speed, accuracy, quality, learned non-use risk, dominant-hand prediction, and estimated motor age.

## Intended Use

The application is intended for experimental screening, demonstration, and hackathon use. It provides functional movement indicators from a reaching task. It is not intended to diagnose disease, replace clinical evaluation, or make medical decisions.

## Technology Stack

| Area | Technology |
|---|---|
| Language | Python 3.10+ |
| Web UI | Gradio 6.19.0 |
| Computer vision | OpenCV 4.10.0.84 |
| Landmark detection | MediaPipe 0.10.21 Tasks API |
| Numeric processing | NumPy |
| Data tables | pandas |
| Signal analysis | SciPy FFT and interpolation |
| Charts | Plotly |
| PDF export | fpdf2 |
| Chart image export | Kaleido 0.2.1 |

## Dependency Specification

Defined in `requirements.txt`:

| Package | Version |
|---|---|
| `gradio` | `6.19.0` |
| `mediapipe` | `0.10.21` |
| `opencv-python` | `4.10.0.84` |
| `numpy` | `>=1.26,<2.0` |
| `pandas` | `>=2.2` |
| `scipy` | `>=1.13` |
| `plotly` | `>=5.22` |
| `fpdf2` | `>=2.8` |
| `Pillow` | `>=10.3` |
| `kaleido` | `0.2.1` |

## Platform Requirements

| Requirement | Specification |
|---|---|
| Operating system | Current implementation is Windows-oriented for live assessment window behavior. |
| Python | 3.10 or newer. |
| Camera | Webcam accessible as OpenCV device `0`. |
| Browser | Modern browser supported by Gradio. |
| Display | Fullscreen OpenCV live view. |
| Network | Required on first run if MediaPipe `.task` models are missing. |

## Windows-Specific Features

`src/run_assessment.py` uses:

- `ctypes.windll.user32` to set the OpenCV window as always-on-top.
- `winsound.PlaySound()` to play a hit sound.

These calls are Windows-specific. Cross-platform use would require fallback implementations.

## Application Entry Points

| File | Role |
|---|---|
| `app.py` | Main Gradio application and event orchestration. |
| `src/run_assessment.py` | Standalone OpenCV live assessment subprocess. |
| `src/assessment_runner.py` | Parent-process subprocess manager and result loader. |

## Source Module Specification

| Module | Responsibility |
|---|---|
| `config.py` | Central scoring constants and thresholds. |
| `src/tracker.py` | MediaPipe model loading, landmark detection, hand position extraction, grid mapping. |
| `src/game_engine.py` | Game state, target selection, hit/miss detection, event recording. |
| `src/motion_analyzer.py` | Post-assessment metric calculations. |
| `src/visualizer.py` | OpenCV overlays, Plotly charts, markdown summaries, event tables. |
| `src/report_generator.py` | CSV and PDF export. |
| `src/cv2_popup.py` | Threaded OpenCV popup helper; not used by the current main subprocess workflow. |

## MediaPipe Models

The code expects these files in `src`:

| Model File | Purpose |
|---|---|
| `hand_landmarker.task` | Detects up to two hands and 21 hand landmarks per hand. |
| `pose_landmarker_full.task` | Detects body pose landmarks. |
| `face_landmarker.task` | Detects face landmarks. |

If a model file does not exist, `tracker.py` downloads it from MediaPipe model URLs.

## Assessment Configuration

| Setting | Value |
|---|---|
| Grid | 3x3 full-frame grid |
| Camera index | `0` |
| Camera target resolution | 1280x720 |
| Countdown duration | 3 seconds |
| Target timeout | 5 seconds |
| Hit dwell requirement | 3 consecutive frames |
| Target labels | Zones 1-9 |
| Internal target IDs | Cells 0-8, row-major |
| Duration options | 30, 60, 120, 180, 300 seconds |
| Hand modes | right, left, both |

## Grid Cell Mapping

Internal cell IDs are row-major:

| Cell ID | UI Zone | Name |
|---|---:|---|
| `0` | 1 | Top-Left |
| `1` | 2 | Top-Center |
| `2` | 3 | Top-Right |
| `3` | 4 | Mid-Left |
| `4` | 5 | Center |
| `5` | 6 | Mid-Right |
| `6` | 7 | Bot-Left |
| `7` | 8 | Bot-Center |
| `8` | 9 | Bot-Right |

## Target Selection Weights

| Cell Type | Weight |
|---|---:|
| Corner cells | 1.5 |
| Edge cells | 1.0 |
| Center cell | 0.5 |

The engine excludes the previous target. It also excludes cells currently occupied by a hand when possible.

## Game State Data Specification

`GameState` fields:

| Field | Type | Meaning |
|---|---|---|
| `phase` | `str` | Current phase: idle, countdown, running, done, analyzed. |
| `duration_s` | `int` | Assessment duration in seconds. |
| `start_time` | `float` | Perf-counter timestamp for assessment start. |
| `current_target` | `int` | Current 0-based target cell. |
| `target_start_time` | `float` | Perf-counter timestamp when current target appeared. |
| `hand_side` | `str` | Selected mode: right, left, both. |
| `events` | `list` | Recorded target events. |
| `trajectory_buffer` | `list` | Recent hand trajectory points for current target. |
| `participant_name` | `str` | Optional participant name. |
| `participant_age` | `int \| None` | Optional participant age. |
| `last_target` | `int` | Previous target cell. |
| `dwell_count` | `int` | Consecutive frames inside target. |
| `flash_frames` | `int` | Remaining overlay flash frames. |
| `flash_color` | `str` | none, green, or red. |
| `_event_counter` | `int` | Internal event ID counter. |

## Event Data Specification

Each recorded event contains:

| Field | Type | Meaning |
|---|---|---|
| `event_id` | `int` | Sequential event number starting from 0. |
| `target_cell` | `int` | Target cell ID from 0 to 8. |
| `hand` | `str` | right or left. |
| `target_shown_at` | `float` | Perf-counter timestamp when target appeared. |
| `hit_at` | `float \| None` | Timestamp of successful hit, or `None` for miss. |
| `reaction_time_ms` | `float \| None` | Hit latency in milliseconds, or `None` for miss. |
| `trajectory` | `list` | Hand trajectory points collected during the target attempt. |
| `success` | `bool` | `True` for hit, `False` for timeout/miss. |
| `path_length` | `float` | Sum of movement distances between trajectory points. |
| `direct_distance` | `float` | Straight-line distance from first to last trajectory point. |

Trajectory point:

| Field | Type | Meaning |
|---|---|---|
| `t` | `float` | Perf-counter timestamp. |
| `x` | `float` | Normalized horizontal hand position. |
| `y` | `float` | Normalized vertical hand position. |

## Analysis Result Specification

`AnalysisResult` contains:

| Category | Fields |
|---|---|
| Reaction time | `rt_right`, `rt_left`, mean, median, best, worst per hand |
| Speed | `speed_score_right`, `speed_score_left` |
| Accuracy | hit rates, accuracy scores, hits/total per cell |
| Quality | path efficiency, jerk score, tremor power, quality score per hand |
| Composite | `composite_right`, `composite_left` |
| Dominance | `dominant_hand` |
| LNU | `lnu_score`, `lnu_risk` |
| Motor age | `motor_age` |
| Chart lists | efficiency, jerk, and tremor lists per hand |

## Scoring Constants

Defined in `config.py`.

### Speed

| Constant | Default | Meaning |
|---|---:|---|
| `SPEED_RT_BEST_MS` | 350 | Mean RT at or below this receives speed score 100. |
| `SPEED_RT_RANGE_MS` | 450 | RT range over which speed score declines to 0. |

Formula:

```text
Speed = clamp(100 - (MeanRT - SPEED_RT_BEST_MS) / SPEED_RT_RANGE_MS * 100, 0, 100)
```

### Quality Weights

| Constant | Default |
|---|---:|
| `QUALITY_WEIGHT_EFFICIENCY` | 0.4 |
| `QUALITY_WEIGHT_JERK` | 0.4 |
| `QUALITY_WEIGHT_TREMOR` | 0.2 |

Formula:

```text
Quality = 0.4 * PathEfficiency + 0.4 * JerkScore + 0.2 * (100 - TremorPower)
```

### Jerk

| Constant | Default | Meaning |
|---|---:|---|
| `JERK_LOG_MULTIPLIER` | 15 | Sensitivity multiplier for normalized jerk score. |
| `JERK_MIN_TRAJ_PTS` | 5 | Minimum trajectory points needed. |
| `JERK_DEFAULT_SCORE` | 50.0 | Fallback score for short trajectories. |

### Tremor

| Constant | Default | Meaning |
|---|---:|---|
| `TREMOR_FS_HZ` | 10.0 | Resampling frequency. |
| `TREMOR_BAND_LO_HZ` | 3.0 | Tremor band lower bound. |
| `TREMOR_BAND_HI_HZ` | 7.0 | Tremor band upper bound. |
| `TREMOR_MIN_DURATION_S` | 0.5 | Minimum trajectory duration. |
| `TREMOR_MIN_POINTS` | 8 | Minimum trajectory points. |

### Composite

| Constant | Default |
|---|---:|
| `COMPOSITE_WEIGHT_SPEED` | 0.4 |
| `COMPOSITE_WEIGHT_ACCURACY` | 0.3 |
| `COMPOSITE_WEIGHT_QUALITY` | 0.3 |

Formula:

```text
Composite = 0.4 * Speed + 0.3 * Accuracy + 0.3 * Quality
```

### Dominance

| Constant | Default | Meaning |
|---|---:|---|
| `DOMINANCE_MARGIN` | 2 | Minimum composite-score gap needed to declare right or left dominance. |

Rules:

| Condition | Result |
|---|---|
| Right composite > left composite + margin | right |
| Left composite > right composite + margin | left |
| Difference within margin | tie |
| Not both-hands mode | N/A |

### Learned Non-Use

| Constant | Default |
|---|---:|
| `LNU_WEIGHT_USE_ASYM` | 0.5 |
| `LNU_WEIGHT_RT_ASYM` | 0.3 |
| `LNU_WEIGHT_QUAL_ASYM` | 0.2 |
| `LNU_THRESHOLD_LOW` | 33 |
| `LNU_THRESHOLD_HIGH` | 67 |
| `LNU_MIN_BILATERAL_HITS` | 3 |

Formula:

```text
LNU = 0.5 * UseAsymmetry + 0.3 * RTAsymmetry + 0.2 * QualityAsymmetry
```

Risk tiers:

| Score | Risk |
|---:|---|
| Less than 33 | Low |
| 33 to less than 67 | Moderate |
| 67 or higher | High |

LNU is only computed in both-hands mode when each hand has at least 3 successful hits.

### Motor Age

| Constant | Default |
|---|---:|
| `MOTOR_AGE_BASE_MIN` | 20 |
| `MOTOR_AGE_RANGE` | 65 |
| `MOTOR_AGE_TREMOR_THRESH` | 10 |
| `MOTOR_AGE_TREMOR_DIVISOR` | 5 |

Formula:

```text
MotorAge = MOTOR_AGE_BASE_MIN
         + (100 - BestComposite) / 100 * MOTOR_AGE_RANGE
         + max(0, TremorPower - MOTOR_AGE_TREMOR_THRESH) / MOTOR_AGE_TREMOR_DIVISOR
```

For single-hand mode, the selected hand is used. For both-hands mode, the best composite score and lower tremor value are used.

## UI Specification

### Setup Panel

Inputs:

- Test duration radio.
- Hand mode radio.
- Name textbox.
- Age number input.
- Start Assessment button.

### Assessment Panel

Controls:

- Open Live Video button.
- Stop Early button.

### Analysis Panel

Tabs:

- Summary.
- Speed.
- Accuracy.
- Movement Quality.
- Hand Dominance.

### Prediction Panel

Displays dominant-hand prediction markdown.

### Report Panel

Displays:

- Reaching Events Timeline.
- Event Log table.
- Generate PDF & CSV Report button.
- CSV download.
- PDF download.

## Live Video Overlay Specification

The OpenCV assessment window displays:

- Mirrored webcam feed.
- 3x3 grid.
- Target zone highlight.
- Zone numbers.
- Hit cell flash.
- Timeout red flash.
- Hand dots labeled `R` and `L`.
- Pose skeleton.
- Hand skeletons.
- Face mesh.
- Current zone instruction.
- Hit count.
- Remaining seconds.
- Bottom progress bar.

## Export Specification

### CSV

Generated by `export_csv()`.

Columns:

- `event_id`
- `target_cell`
- `hand`
- `success`
- `reaction_time_ms`
- `path_length`
- `direct_distance`
- `target_shown_at`
- `hit_at`

### PDF

Generated by `export_pdf()`.

Contents:

- Title and participant metadata.
- Performance summary table.
- Dominant hand.
- LNU risk.
- Estimated motor age.
- Screening disclaimer.
- Embedded charts when Plotly image export succeeds.
- Methodology page.

## Temporary File Specification

The subprocess workflow uses temporary files:

| File Type | Purpose |
|---|---|
| `assessment_result_*.pkl` | Final pickled `GameState` written by subprocess and read by parent. |
| `assessment_stop_*.stop` | Stop-signal file created by parent to request early subprocess termination. |
| `dexterity_*.csv` | Exported CSV report. |
| `dexterity_report_*.pdf` | Exported PDF report. |

## Known Constraints

- The app is designed around local execution.
- Webcam index is hardcoded to `0`.
- Live assessment uses Windows-only APIs.
- MediaPipe model download requires network access if model files are missing.
- The current grid covers the full frame and does not dynamically resize around the participant body.
- Medical claims are limited to screening indicators and must be interpreted cautiously.

## Run Commands

Install:

```bash
pip install -r requirements.txt
```

Run:

```bash
python app.py
```

Default local URL:

```text
http://localhost:7860
```

