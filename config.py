# config.py — tune all assessment scoring thresholds here without touching source code

# ── Speed Scoring ──────────────────────────────────────────────────────────────
# Linear map: BEST_MS → score 100, (BEST_MS + RANGE_MS) → score 0
SPEED_RT_BEST_MS   = 350   # reaction time (ms) that earns a perfect speed score
SPEED_RT_RANGE_MS  = 450   # ms span over which score falls from 100 to 0

# ── Quality Component Weights ─────────────────────────────────────────────────
# Must sum to 1.0
QUALITY_WEIGHT_EFFICIENCY = 0.4
QUALITY_WEIGHT_JERK       = 0.4
QUALITY_WEIGHT_TREMOR     = 0.2   # applied as (100 - tremor_score)

# ── Jerk Score ────────────────────────────────────────────────────────────────
# score = clamp(100 - log1p(NJS) * JERK_LOG_MULTIPLIER, 0, 100)
JERK_LOG_MULTIPLIER  = 15
JERK_MIN_TRAJ_PTS    = 5      # minimum trajectory points to compute jerk
JERK_DEFAULT_SCORE   = 50.0   # fallback when trajectory is too short

# ── Tremor ────────────────────────────────────────────────────────────────────
TREMOR_FS_HZ          = 10.0   # resample frequency in Hz
TREMOR_BAND_LO_HZ     = 3.0    # tremor band lower bound (Hz)
TREMOR_BAND_HI_HZ     = 7.0    # tremor band upper bound (Hz)
TREMOR_MIN_DURATION_S = 0.5    # minimum trajectory duration (s) to compute tremor
TREMOR_MIN_POINTS     = 8      # minimum trajectory points to compute tremor

# ── Composite Score Weights ───────────────────────────────────────────────────
# Must sum to 1.0
COMPOSITE_WEIGHT_SPEED    = 0.4
COMPOSITE_WEIGHT_ACCURACY = 0.3
COMPOSITE_WEIGHT_QUALITY  = 0.3

# ── Dominance ─────────────────────────────────────────────────────────────────
# Composite score difference required to declare one hand dominant (not a tie)
DOMINANCE_MARGIN = 2

# ── LNU (Learned Non-Use) ────────────────────────────────────────────────────
# Weights for the three asymmetry components (must sum to 1.0)
LNU_WEIGHT_USE_ASYM    = 0.5
LNU_WEIGHT_RT_ASYM     = 0.3
LNU_WEIGHT_QUAL_ASYM   = 0.2
# Risk tier thresholds (score 0–100)
LNU_THRESHOLD_LOW      = 33   # score < this  → "Low" risk
LNU_THRESHOLD_HIGH     = 67   # score ≥ this  → "High" risk
# Minimum bilateral hits required to compute LNU (per hand)
LNU_MIN_BILATERAL_HITS = 3

# ── Motor Age Estimate ────────────────────────────────────────────────────────
# base_age = BASE_MIN + (100 - composite) / 100 * AGE_RANGE
MOTOR_AGE_BASE_MIN        = 20    # estimated motor age when composite = 100
MOTOR_AGE_RANGE           = 65    # span: motor age ranges from BASE_MIN to BASE_MIN+RANGE
# Tremor age penalty: (tremor_power - THRESH) / DIVISOR  (clamped to ≥ 0)
MOTOR_AGE_TREMOR_THRESH   = 10    # tremor power % below which no age penalty
MOTOR_AGE_TREMOR_DIVISOR  = 5
