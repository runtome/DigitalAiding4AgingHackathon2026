# DigitalAiding4Aging Hackathon 2026

## AI-Powered Upper Limb Dexterity Assessment

ระบบประเมินความคล่องแคล่วของแขนส่วนบนด้วย AI สำหรับผู้สูงอายุ

---

## Overview

An AI-powered reaching-task tool that assesses upper limb dexterity via webcam. The participant stands 2–3 m from the camera and reaches toward highlighted zones on a 3×3 grid. The system measures:

| Metric | Description |
|---|---|
| **Reaction Time** | Speed from target appearing to hand entering the zone |
| **Accuracy** | Percentage of targets successfully reached |
| **Movement Quality** | Path efficiency, movement smoothness, and tremor |
| **LNU Risk** | Left–Right asymmetry indicating Learned Non-Use |
| **Motor Age** | Estimated functional motor age from composite performance |
| **Dominant Hand** | Predicted dominant hand from bilateral composite score difference |

> **Screening tool only.** Results are experimental indicators, not clinical diagnoses. Consult a healthcare professional for medical evaluation.

---

## Requirements

- Python 3.10+
- Webcam (USB or built-in)
- 4 GB RAM minimum
- Modern browser (Chrome, Edge, Firefox)

### Key Packages

| Package | Version | Role |
|---|---|---|
| `gradio` | 6.19.0 | Web UI |
| `mediapipe` | 0.10.21 | Hand & pose tracking |
| `opencv-python` | 4.10.0.84 | Video processing & assessment window |
| `numpy` | ≥ 1.26, < 2.0 | Numerical computation |
| `pandas` | ≥ 2.2 | Event log table & data export |
| `plotly` | ≥ 5.22 | Analysis charts |
| `scipy` | ≥ 1.13 | Signal analysis (FFT, interpolation, statistics) |
| `fpdf2` | ≥ 2.8 | PDF report generation |
| `Pillow` | ≥ 10.3 | Image handling |
| `kaleido` | 0.2.1 | Plotly → PNG conversion for PDF charts |

---

## Installation

```bash
pip install -r requirements.txt
```

---

## How to Run

```bash
python app.py
```

Open your browser at `http://localhost:7860`, then follow these steps:

1. **Fill in test setup** — Choose duration, hands to assess, name, and age (default 45)
2. **Start Assessment** — Click **▶ Start Assessment** (3-2-1 countdown begins)
3. **Open Live Video** — Click **🎥 Open Live Video** to launch the fullscreen assessment window
4. **Reach targets** — Reach your hand to each highlighted yellow zone; it turns green on success with a chime sound
5. **View results** — After time expires the window closes automatically and the analysis panel appears
6. **Export report** — Click **📥 Generate PDF Report** to download the PDF
7. **Restart** — Click **▶ Start Assessment** again to run another session

> **Tip:** Stand 2–3 m from the camera so your full body and both hands are visible.

---

## Architecture

```
app.py                    Gradio UI, event handlers, poll timer
config.py                 All tunable scoring thresholds (see below)
src/tracker.py            HandTracker (MediaPipe), grid cell mapping
src/game_engine.py        GameState machine, hit/timeout detection
src/motion_analyzer.py    Post-game metrics (speed, accuracy, quality, LNU, motor age)
src/visualizer.py         OpenCV overlays, Plotly charts, event log table, statistical analysis
src/report_generator.py   PDF report export (with embedded charts)
src/run_assessment.py     Standalone subprocess for fullscreen cv2 window (Windows)
src/assessment_runner.py  Subprocess launcher & IPC (pickle + sentinel file)
```

### Data Flow

```
Webcam frame (BGR)
  → cv2.flip (selfie mirror)
  → HandTracker.process() → TrackingResult
  → GameEngine.process_frame() → (GameState, FrameResult)
  → draw_overlay() → annotated frame in cv2 window
  [on game over, via gr.Timer poll]
  → MotionAnalyzer.analyze() → AnalysisResult
  → Plotly charts + event log table shown in Gradio
```

### Grid Layout

The 3×3 grid covers the full camera frame:

```
┌──────────┬──────────┬──────────┐
│ Top-Left │Top-Center│ Top-Right│  cells 1, 2, 3
├──────────┼──────────┼──────────┤
│ Mid-Left │  Center  │ Mid-Right│  cells 4, 5, 6
├──────────┼──────────┼──────────┤
│ Bot-Left │Bot-Center│ Bot-Right│  cells 7, 8, 9
└──────────┴──────────┴──────────┘
```

Corner cells (1, 3, 7, 9) are weighted 1.5× for selection; center (5) is weighted 0.5×.

---

## Clinical Metrics

### 1. Speed Score (0–100)

Linear mapping of mean reaction time to a 0–100 score.  
Default range: **500 ms → score 100**, **2000 ms → score 0** (calibrated for reaching tasks with aging adults).  
Configurable via `SPEED_RT_BEST_MS` and `SPEED_RT_RANGE_MS` in `config.py`.

### 2. Accuracy Score (0–100)

Percentage of targets successfully reached within the timeout window.  
Computed per hand and per grid cell.

### 3. Movement Quality Score (0–100)

Weighted blend of three sub-scores:

```
Quality = 0.4 × PathEfficiency + 0.4 × JerkScore + 0.2 × (100 − TremorPower)
```

- **Path Efficiency** — ratio of direct distance to actual path length, displayed as 0–100% (100% = perfectly straight)
- **Jerk Score** — smoothness from Dimensionless Jerk (NJS) computed on Savitzky-Golay-smoothed trajectory; lower NJS → higher score (0–100)
- **Tremor Power** — FFT energy in the 4.5–7 Hz band as % of total power; only computed for reaches lasting ≥ 1.5 s (shorter reaches return 0 — too few FFT points for reliable frequency resolution)

### 4. Composite Score (0–100)

```
Composite = 0.4 × Speed + 0.3 × Accuracy + 0.3 × Quality
```

Used as the primary per-hand performance index.

### 5. Dominant Hand Prediction

Compares the composite scores of both hands:

- Right composite − Left composite > `DOMINANCE_MARGIN` → **Right-Handed**
- Left composite − Right composite > `DOMINANCE_MARGIN` → **Left-Handed**
- Difference ≤ `DOMINANCE_MARGIN` → **Inconclusive / Ambidextrous**

Default `DOMINANCE_MARGIN = 2` (raise to 5–10 to require a clearer performance gap).

### 6. LNU Risk Score (0–100)

Learned Non-Use asymmetry index — requires Both Hands mode with ≥ 3 hits per hand:

```
LNU = 0.5 × UseAsymmetry + 0.3 × RTAsymmetry + 0.2 × QualityAsymmetry
```

| Score | Risk Level |
|---|---|
| 0 – 32 | Low |
| 33 – 66 | Moderate |
| 67 – 100 | High |

### 7. Motor Age Estimate

Estimates a functional motor age from performance:

```
MotorAge = 20 + (100 − BestComposite) / 100 × 65 + TremorAdjustment
TremorAdjustment = max(0, TremorPower − 10) / 5
```

Range: 20 (excellent) to ~85+ (severe impairment).  
Compared to the participant's chronological age when provided.

### 8. Reaction Time Normal Distribution Analysis (Speed Tab)

Shown below the boxplot in the Speed tab. For each hand it fits a Gaussian PDF to the RT samples and displays:

- **Overlaid curves** with colour-coded mean lines (dashed)
- **Descriptive statistics table** — N, μ, median, σ, σ², min, max, 95% CI, CV%
- **Normality test** — Shapiro-Wilk W statistic and p-value per hand
- **Hypothesis test** — Paired t-test (both normal) or Wilcoxon Signed-Rank (otherwise), with test statistic, p-value, and significance verdict
- **Cohen's d** — pooled effect size with Negligible / Small / Medium / Large classification
- **Summary card** — faster hand, mean difference (ms), % difference, statistical significance, effect size

All thresholds (CI level, α, Cohen's d tiers) are configurable in `config.py`.

---

## Configuration (`config.py`)

All scoring thresholds are centralised in `config.py` at the project root.  
**Edit this file to tune the assessment without touching source code.**

### Speed Scoring

| Constant | Default | Meaning |
|---|---|---|
| `SPEED_RT_BEST_MS` | `500` | Reaction time (ms) at or below which the speed score is 100. Calibrated for reaching tasks: fast adults reach in ~500 ms. |
| `SPEED_RT_RANGE_MS` | `1500` | The ms span over which the score falls from 100 to 0. Worst RT = `SPEED_RT_BEST_MS + SPEED_RT_RANGE_MS` = 2000 ms → score 0. |

Formula: `score = clamp(100 − (RT − SPEED_RT_BEST_MS) / SPEED_RT_RANGE_MS × 100, 0, 100)`

**Example:** RT = 1138 ms → `100 − (1138 − 500) / 1500 × 100 ≈ 57`

---

### Quality Component Weights

| Constant | Default | Meaning |
|---|---|---|
| `QUALITY_WEIGHT_EFFICIENCY` | `0.4` | Weight of path efficiency in the quality score (40%). |
| `QUALITY_WEIGHT_JERK` | `0.4` | Weight of movement smoothness (jerk score) in the quality score (40%). |
| `QUALITY_WEIGHT_TREMOR` | `0.2` | Weight of tremor (inverted: applied as `100 − TremorPower`) in the quality score (20%). |

Must sum to 1.0.

---

### Jerk Score

| Constant | Default | Meaning |
|---|---|---|
| `JERK_LOG_MULTIPLIER` | `6` | Sensitivity of the jerk score. Higher value → score drops faster as NJS increases. Formula: `score = clamp(100 − log(1+NJS) × JERK_LOG_MULTIPLIER, 0, 100)`. Calibrated for Savitzky-Golay-smoothed trajectories from webcam data. |
| `JERK_MIN_TRAJ_PTS` | `5` | Minimum trajectory points required to compute jerk. Fewer points → fallback score. |
| `JERK_DEFAULT_SCORE` | `50.0` | Fallback jerk score when trajectory is too short to compute NJS. |

---

### Tremor Analysis

| Constant | Default | Meaning |
|---|---|---|
| `TREMOR_FS_HZ` | `10.0` | Resample frequency (Hz) for uniform-time signal before FFT. |
| `TREMOR_BAND_LO_HZ` | `4.5` | Lower bound of the tremor frequency band (Hz). Set to 4.5 Hz to exclude normal arm-swing frequency (1–4 Hz) which otherwise causes false tremor readings. |
| `TREMOR_BAND_HI_HZ` | `7.0` | Upper bound of the tremor frequency band (Hz). The 4.5–7 Hz range targets pathological tremor (Parkinson's, essential tremor). |
| `TREMOR_MIN_DURATION_S` | `1.5` | Minimum trajectory duration (s) to compute tremor. Shorter reaches return 0 — fewer than ~15 FFT points gives unreliable frequency resolution. |
| `TREMOR_MIN_POINTS` | `15` | Minimum trajectory points to compute tremor. Fewer → tremor power = 0. |

---

### Composite Score Weights

| Constant | Default | Meaning |
|---|---|---|
| `COMPOSITE_WEIGHT_SPEED` | `0.4` | Weight of speed score in the composite (40%). |
| `COMPOSITE_WEIGHT_ACCURACY` | `0.3` | Weight of accuracy score in the composite (30%). |
| `COMPOSITE_WEIGHT_QUALITY` | `0.3` | Weight of quality score in the composite (30%). |

Must sum to 1.0.

---

### Dominant Hand Prediction

| Constant | Default | Meaning |
|---|---|---|
| `DOMINANCE_MARGIN` | `2` | Minimum composite score gap required to declare a dominant hand. Gap ≤ this → "Inconclusive / Ambidextrous". Raise to 5–10 for a stricter threshold. |

---

### LNU (Learned Non-Use) Risk

| Constant | Default | Meaning |
|---|---|---|
| `LNU_WEIGHT_USE_ASYM` | `0.5` | Weight of usage asymmetry (hit count difference between hands) in the LNU score. |
| `LNU_WEIGHT_RT_ASYM` | `0.3` | Weight of reaction-time asymmetry between hands. |
| `LNU_WEIGHT_QUAL_ASYM` | `0.2` | Weight of composite quality asymmetry between hands. |
| `LNU_THRESHOLD_LOW` | `33` | LNU scores below this → "Low" risk. |
| `LNU_THRESHOLD_HIGH` | `67` | LNU scores at or above this → "High" risk. Between LOW and HIGH → "Moderate". |
| `LNU_MIN_BILATERAL_HITS` | `3` | Minimum successful hits **per hand** required to compute LNU. Fewer → N/A. |

---

### Motor Age Estimate

| Constant | Default | Meaning |
|---|---|---|
| `MOTOR_AGE_BASE_MIN` | `20` | Estimated motor age when composite = 100 (excellent). |
| `MOTOR_AGE_RANGE` | `65` | Age span: motor age ranges from 20 to 85 as composite falls from 100 to 0. |
| `MOTOR_AGE_TREMOR_THRESH` | `10` | Tremor power (%) below which no age penalty is added. |
| `MOTOR_AGE_TREMOR_DIVISOR` | `5` | Each 5% of excess tremor adds 1 estimated year to motor age. |

Formula:
```
MotorAge = MOTOR_AGE_BASE_MIN
         + (100 − BestComposite) / 100 × MOTOR_AGE_RANGE
         + max(0, TremorPower − MOTOR_AGE_TREMOR_THRESH) / MOTOR_AGE_TREMOR_DIVISOR
```

---

### Statistical Analysis

| Constant | Default | Meaning |
|---|---|---|
| `STATS_CI_LEVEL` | `0.95` | Confidence interval level for the descriptive stats table (e.g., 0.95 = 95% CI). Change to 0.90 or 0.99 as needed. |
| `STATS_NORMALITY_ALPHA` | `0.05` | p-value threshold for the Shapiro-Wilk test. p > this → distribution is considered normal → Paired t-test is used. |
| `STATS_SIGNIFICANCE_ALPHA` | `0.05` | p-value threshold for the hypothesis test. p < this → result is "statistically significant". |
| `COHEN_D_NEGLIGIBLE` | `0.2` | Cohen's d below this → "Negligible" effect size. |
| `COHEN_D_SMALL` | `0.5` | Cohen's d below this → "Small" effect size. |
| `COHEN_D_MEDIUM` | `0.8` | Cohen's d below this → "Medium" effect size; at or above → "Large". |

---

## License

MIT License — Developed for the DigitalAiding4Aging Hackathon 2026.
