# Calculation Reference

This document explains the main calculation formulas used by the assessment system.

The application UI in `app.py` does not calculate these formulas directly. It calls `MotionAnalyzer().analyze(state)`, and the main formulas are implemented in `src/motion_analyzer.py`. Raw event data, including reaction time, is recorded in `src/game_engine.py`.

## 1. Reaction Time

Reaction time is calculated for each successful target hit.

```text
ReactionTime_ms = (hit_time - target_start_time) * 1000
```

Where:

- `target_start_time` is the time when the target appeared.
- `hit_time` is the time when the hit was registered.
- The result is converted from seconds to milliseconds.

A hit is registered only after the hand stays in the target cell for 3 consecutive frames.

```text
MIN_DWELL_FRAMES = 3
```

Mean reaction time per hand:

```text
MeanRT_hand = average(successful reaction_time_ms for that hand)
```

Speed score:

```text
SpeedScore = clamp(
  100 - (MeanRT - SPEED_RT_BEST_MS) / SPEED_RT_RANGE_MS * 100,
  0,
  100
)
```

Current configuration:

```text
SPEED_RT_BEST_MS = 500
SPEED_RT_RANGE_MS = 1500
```

Meaning:

- 500 ms or faster gives a speed score of 100.
- 2000 ms or slower gives a speed score of 0.

## 2. Accuracy

Accuracy is the percentage of successful hits out of all target attempts for each hand.

```text
HitRate_hand = successful_events_hand / total_events_hand * 100
```

```text
AccuracyScore_hand = HitRate_hand
```

If there are no events for that hand:

```text
AccuracyScore_hand = 0
```

The system also counts hits and total attempts for each grid cell, but the main accuracy score is calculated per hand.

## 3. Movement Quality

Movement quality combines three measurements:

- Path efficiency
- Jerk score
- Tremor power

### 3.1 Path Efficiency

Path efficiency measures how direct the hand movement was.

```text
PathEfficiency = direct_distance / path_length
```

Where:

- `direct_distance` is the straight-line distance from the first trajectory point to the last point.
- `path_length` is the total travelled distance along the trajectory.

If `path_length` is almost zero:

```text
PathEfficiency = 1.0
```

Average path efficiency is converted to a 0-100 score:

```text
PathEfficiencyScore = mean(PathEfficiencyList) * 100
```

### 3.2 Jerk Score

Jerk score measures movement smoothness. Lower jerk means smoother movement.

The system first calculates normalized jerk:

```text
NJS = 0.5 * integral(jerk_magnitude^2 over time) * (duration^5 / distance^2)
```

Then it converts normalized jerk into a score:

```text
JerkScore = clamp(
  100 - log(1 + NJS) * JERK_LOG_MULTIPLIER,
  0,
  100
)
```

Current configuration:

```text
JERK_LOG_MULTIPLIER = 6
JERK_MIN_TRAJ_PTS = 5
JERK_DEFAULT_SCORE = 50
```

If the trajectory is too short, the fallback jerk score is used:

```text
JerkScore = 50
```

### 3.3 Tremor Power

Tremor power is calculated from the frequency content of the hand trajectory.

```text
TremorPower = tremor_band_power / total_positive_frequency_power * 100
```

Current tremor frequency band:

```text
TREMOR_BAND_LO_HZ = 4.5
TREMOR_BAND_HI_HZ = 7.0
```

If the trajectory is too short, tremor power returns 0.

Current minimum requirements:

```text
TREMOR_MIN_DURATION_S = 1.5
TREMOR_MIN_POINTS = 15
```

### 3.4 Final Movement Quality Score

The final movement quality score is:

```text
MovementQuality = clamp(
  0.4 * PathEfficiencyScore
  + 0.4 * JerkScore
  + 0.2 * (100 - TremorPower),
  0,
  100
)
```

Current configuration:

```text
QUALITY_WEIGHT_EFFICIENCY = 0.4
QUALITY_WEIGHT_JERK = 0.4
QUALITY_WEIGHT_TREMOR = 0.2
```

## 4. Composite Score

The composite score combines speed, accuracy, and movement quality.

```text
Composite = clamp(
  0.4 * SpeedScore
  + 0.3 * AccuracyScore
  + 0.3 * MovementQuality,
  0,
  100
)
```

Current configuration:

```text
COMPOSITE_WEIGHT_SPEED = 0.4
COMPOSITE_WEIGHT_ACCURACY = 0.3
COMPOSITE_WEIGHT_QUALITY = 0.3
```

The composite score is used by:

- Dominant hand prediction
- LNU risk calculation
- Motor age estimation

## 5. LNU Risk

LNU means Learned Non-Use.

LNU is only calculated when both hands are assessed.

Required condition:

```text
hand_side = "both"
right successful hits >= 3
left successful hits >= 3
```

If this condition is not met:

```text
LNU = N/A
```

### 5.1 Use Asymmetry

Use asymmetry measures whether one hand was used much more than the other.

```text
UseAsymmetry = (1 - min(right_hits, left_hits) / max(right_hits, left_hits)) * 100
```

### 5.2 Reaction-Time Asymmetry

Reaction-time asymmetry compares average reaction time between hands.

```text
RTAsymmetry = abs(RightMeanRT - LeftMeanRT) / max(RightMeanRT, LeftMeanRT) * 100
```

### 5.3 Quality Asymmetry

In the current implementation, this is calculated from the composite score difference.

```text
QualityAsymmetry = abs(RightComposite - LeftComposite)
```

### 5.4 Final LNU Score

```text
LNU = clamp(
  0.5 * UseAsymmetry
  + 0.3 * RTAsymmetry
  + 0.2 * QualityAsymmetry,
  0,
  100
)
```

Current configuration:

```text
LNU_WEIGHT_USE_ASYM = 0.5
LNU_WEIGHT_RT_ASYM = 0.3
LNU_WEIGHT_QUAL_ASYM = 0.2
```

LNU risk level:

```text
LNU < 33        -> Low
33 <= LNU < 67 -> Moderate
LNU >= 67      -> High
```

Current thresholds:

```text
LNU_THRESHOLD_LOW = 33
LNU_THRESHOLD_HIGH = 67
```

## 6. Motor Age

Motor age is estimated from the best composite score and tremor power.

For right-hand mode:

```text
best_composite = RightComposite
best_tremor = RightTremorPower
```

For left-hand mode:

```text
best_composite = LeftComposite
best_tremor = LeftTremorPower
```

For both-hands mode:

```text
best_composite = max(RightComposite, LeftComposite)
best_tremor = min(RightTremorPower, LeftTremorPower)
```

Base age:

```text
BaseAge = MOTOR_AGE_BASE_MIN + (100 - best_composite) / 100 * MOTOR_AGE_RANGE
```

Tremor adjustment:

```text
TremorAdjustment = max(0, best_tremor - MOTOR_AGE_TREMOR_THRESH) / MOTOR_AGE_TREMOR_DIVISOR
```

Final motor age:

```text
MotorAge = round(BaseAge + TremorAdjustment, 1)
```

Current configuration:

```text
MOTOR_AGE_BASE_MIN = 20
MOTOR_AGE_RANGE = 65
MOTOR_AGE_TREMOR_THRESH = 10
MOTOR_AGE_TREMOR_DIVISOR = 5
```

Expanded formula:

```text
MotorAge = round(
  20 + (100 - best_composite) / 100 * 65
  + max(0, best_tremor - 10) / 5,
  1
)
```

## 7. Dominant Hand

Dominant hand prediction is only calculated in both-hands mode.

For single-hand mode:

```text
DominantHand = "N/A"
```

For both-hands mode:

```text
if RightComposite > LeftComposite + DOMINANCE_MARGIN:
    DominantHand = "right"

elif LeftComposite > RightComposite + DOMINANCE_MARGIN:
    DominantHand = "left"

else:
    DominantHand = "tie"
```

Current configuration:

```text
DOMINANCE_MARGIN = 2
```

Meaning:

- Right hand is dominant if its composite score is more than 2 points higher than the left hand.
- Left hand is dominant if its composite score is more than 2 points higher than the right hand.
- Otherwise, the result is a tie.
