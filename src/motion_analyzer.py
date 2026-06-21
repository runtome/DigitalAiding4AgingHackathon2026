from dataclasses import dataclass, field

import numpy as np
from scipy.fft import fft, fftfreq
from scipy.interpolate import interp1d

from config import (
    SPEED_RT_BEST_MS, SPEED_RT_RANGE_MS,
    QUALITY_WEIGHT_EFFICIENCY, QUALITY_WEIGHT_JERK, QUALITY_WEIGHT_TREMOR,
    JERK_LOG_MULTIPLIER, JERK_MIN_TRAJ_PTS, JERK_DEFAULT_SCORE,
    TREMOR_FS_HZ, TREMOR_BAND_LO_HZ, TREMOR_BAND_HI_HZ,
    TREMOR_MIN_DURATION_S, TREMOR_MIN_POINTS,
    COMPOSITE_WEIGHT_SPEED, COMPOSITE_WEIGHT_ACCURACY, COMPOSITE_WEIGHT_QUALITY,
    DOMINANCE_MARGIN,
    LNU_WEIGHT_USE_ASYM, LNU_WEIGHT_RT_ASYM, LNU_WEIGHT_QUAL_ASYM,
    LNU_THRESHOLD_LOW, LNU_THRESHOLD_HIGH, LNU_MIN_BILATERAL_HITS,
    MOTOR_AGE_BASE_MIN, MOTOR_AGE_RANGE,
    MOTOR_AGE_TREMOR_THRESH, MOTOR_AGE_TREMOR_DIVISOR,
)

_MIN_BILATERAL_HITS = LNU_MIN_BILATERAL_HITS


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


@dataclass
class AnalysisResult:
    # Raw reaction times per hand
    rt_right: list = field(default_factory=list)
    rt_left: list = field(default_factory=list)

    # Speed (0-100)
    speed_score_right: float = 0.0
    speed_score_left: float = 0.0
    rt_mean_right: float = 0.0
    rt_mean_left: float = 0.0
    rt_median_right: float = 0.0
    rt_median_left: float = 0.0
    rt_best_right: float = 0.0
    rt_best_left: float = 0.0
    rt_worst_right: float = 0.0
    rt_worst_left: float = 0.0

    # Accuracy (0-100)
    hit_rate_right: float = 0.0
    hit_rate_left: float = 0.0
    accuracy_score_right: float = 0.0
    accuracy_score_left: float = 0.0
    hits_per_cell: dict = field(default_factory=dict)
    total_per_cell: dict = field(default_factory=dict)

    # Quality (0-100)
    path_efficiency_right: float = 0.0
    path_efficiency_left: float = 0.0
    jerk_score_right: float = 0.0
    jerk_score_left: float = 0.0
    tremor_power_right: float = 0.0
    tremor_power_left: float = 0.0
    quality_score_right: float = 0.0
    quality_score_left: float = 0.0

    # Composite & Dominance
    composite_right: float = 0.0
    composite_left: float = 0.0
    dominant_hand: str = "unknown"

    # LNU — None when assessment doesn't have valid bilateral data
    lnu_score: float | None = None
    lnu_risk: str = "N/A"

    # Motor Age
    motor_age: float | None = None

    # Per-trial quality lists for charts
    efficiency_list_right: list = field(default_factory=list)
    efficiency_list_left: list = field(default_factory=list)
    jerk_list_right: list = field(default_factory=list)
    jerk_list_left: list = field(default_factory=list)
    tremor_list_right: list = field(default_factory=list)
    tremor_list_left: list = field(default_factory=list)


class MotionAnalyzer:
    def analyze(self, state) -> AnalysisResult:
        events = state.events if state else []
        hand_side = state.hand_side if state else "both"
        r = AnalysisResult()

        right_events = [e for e in events if e["hand"] == "right"]
        left_events = [e for e in events if e["hand"] == "left"]

        _compute_speed(r, right_events, left_events)
        _compute_accuracy(r, right_events, left_events, events)
        _compute_quality(r, right_events, left_events)
        _compute_composite(r)
        _compute_dominance(r, hand_side)
        _compute_lnu(r, right_events, left_events, hand_side)
        _compute_motor_age(r, state.participant_age if state else None, hand_side)

        return r


def _compute_speed(r: AnalysisResult, right_ev, left_ev):
    rt_r = [e["reaction_time_ms"] for e in right_ev if e["success"] and e["reaction_time_ms"]]
    rt_l = [e["reaction_time_ms"] for e in left_ev if e["success"] and e["reaction_time_ms"]]
    r.rt_right = rt_r
    r.rt_left = rt_l

    for rts, prefix in [(rt_r, "right"), (rt_l, "left")]:
        if rts:
            mean = float(np.mean(rts))
            setattr(r, f"rt_mean_{prefix}", mean)
            setattr(r, f"rt_median_{prefix}", float(np.median(rts)))
            setattr(r, f"rt_best_{prefix}", float(np.min(rts)))
            setattr(r, f"rt_worst_{prefix}", float(np.max(rts)))
            score = _clamp(100 - (mean - SPEED_RT_BEST_MS) / SPEED_RT_RANGE_MS * 100, 0, 100)
            setattr(r, f"speed_score_{prefix}", score)


def _compute_accuracy(r: AnalysisResult, right_ev, left_ev, all_events):
    def _rate(evs):
        total = len(evs)
        hits = sum(1 for e in evs if e["success"])
        return (hits / total * 100) if total > 0 else 0.0

    r.hit_rate_right = _rate(right_ev)
    r.hit_rate_left = _rate(left_ev)
    r.accuracy_score_right = r.hit_rate_right
    r.accuracy_score_left = r.hit_rate_left

    hits_per_cell: dict[int, int] = {i: 0 for i in range(9)}
    total_per_cell: dict[int, int] = {i: 0 for i in range(9)}
    for e in all_events:
        c = e["target_cell"]
        if 0 <= c <= 8:
            total_per_cell[c] += 1
            if e["success"]:
                hits_per_cell[c] += 1
    r.hits_per_cell = hits_per_cell
    r.total_per_cell = total_per_cell


def _compute_quality(r: AnalysisResult, right_ev, left_ev):
    for evs, prefix in [(right_ev, "right"), (left_ev, "left")]:
        success_evs = [e for e in evs if e["success"] and e["trajectory"]]
        eff_list, jerk_list, tremor_list = [], [], []
        for e in success_evs:
            path_len = e["path_length"]
            direct = e["direct_distance"]
            eff = direct / path_len if path_len > 1e-6 else 1.0
            eff_list.append(_clamp(eff, 0, 1))
            jerk_list.append(_compute_jerk_score(e["trajectory"]))
            tremor_list.append(_compute_tremor_power(e["trajectory"]))

        setattr(r, f"efficiency_list_{prefix}", eff_list)
        setattr(r, f"jerk_list_{prefix}", jerk_list)
        setattr(r, f"tremor_list_{prefix}", tremor_list)

        eff_score    = float(np.mean(eff_list) * 100) if eff_list else 50.0
        jerk_score   = float(np.mean(jerk_list)) if jerk_list else 50.0
        tremor_score = float(np.mean(tremor_list)) if tremor_list else 0.0

        setattr(r, f"path_efficiency_{prefix}", eff_score)
        setattr(r, f"jerk_score_{prefix}", jerk_score)
        setattr(r, f"tremor_power_{prefix}", tremor_score)

        quality = (
            QUALITY_WEIGHT_EFFICIENCY * eff_score
            + QUALITY_WEIGHT_JERK * jerk_score
            + QUALITY_WEIGHT_TREMOR * (100 - tremor_score)
        )
        setattr(r, f"quality_score_{prefix}", _clamp(quality, 0, 100))


def _compute_jerk_score(traj: list) -> float:
    if len(traj) < JERK_MIN_TRAJ_PTS:
        return JERK_DEFAULT_SCORE
    times = np.array([p["t"] for p in traj])
    xs = np.array([p["x"] for p in traj])
    ys = np.array([p["y"] for p in traj])

    dt = np.diff(times)
    dt = np.where(dt < 1e-6, 1e-6, dt)

    vx = np.diff(xs) / dt
    vy = np.diff(ys) / dt
    if len(vx) < 2:
        return JERK_DEFAULT_SCORE

    dt2 = dt[1:]
    ax = np.diff(vx) / dt2
    ay = np.diff(vy) / dt2
    if len(ax) < 2:
        return JERK_DEFAULT_SCORE

    dt3 = dt2[1:]
    jx = np.diff(ax) / dt3
    jy = np.diff(ay) / dt3

    duration = float(times[-1] - times[0])
    dx = xs[-1] - xs[0]
    dy = ys[-1] - ys[0]
    distance = float((dx * dx + dy * dy) ** 0.5)

    if duration < 1e-6 or distance < 1e-6:
        return JERK_DEFAULT_SCORE

    jerk_mag = np.sqrt(jx ** 2 + jy ** 2)
    njs = 0.5 * np.trapz(jerk_mag ** 2, times[3:]) * (duration ** 5 / distance ** 2)
    return float(_clamp(100 - np.log1p(njs) * JERK_LOG_MULTIPLIER, 0, 100))


def _compute_tremor_power(traj: list) -> float:
    if len(traj) < TREMOR_MIN_POINTS:
        return 0.0
    times = np.array([p["t"] for p in traj])
    xs = np.array([p["x"] for p in traj])

    duration = times[-1] - times[0]
    if duration < TREMOR_MIN_DURATION_S:
        return 0.0

    n_samples = max(8, int(duration * TREMOR_FS_HZ))
    t_uniform = np.linspace(times[0], times[-1], n_samples)
    try:
        f_interp = interp1d(times, xs, kind="linear", bounds_error=False, fill_value="extrapolate")
        x_uniform = f_interp(t_uniform)
    except Exception:
        return 0.0

    N = len(x_uniform)
    freqs = fftfreq(N, 1.0 / TREMOR_FS_HZ)
    power = np.abs(fft(x_uniform)) ** 2

    pos_mask    = freqs > 0
    tremor_mask = (np.abs(freqs) >= TREMOR_BAND_LO_HZ) & (np.abs(freqs) <= TREMOR_BAND_HI_HZ)

    total_power = np.sum(power[pos_mask])
    tremor_p    = np.sum(power[tremor_mask])

    if total_power < 1e-10:
        return 0.0
    return float(_clamp(tremor_p / total_power * 100, 0, 100))


def _compute_composite(r: AnalysisResult):
    for prefix in ("right", "left"):
        speed = getattr(r, f"speed_score_{prefix}")
        acc   = getattr(r, f"accuracy_score_{prefix}")
        qual  = getattr(r, f"quality_score_{prefix}")
        comp  = (
            COMPOSITE_WEIGHT_SPEED * speed
            + COMPOSITE_WEIGHT_ACCURACY * acc
            + COMPOSITE_WEIGHT_QUALITY * qual
        )
        setattr(r, f"composite_{prefix}", _clamp(comp, 0, 100))


def _compute_dominance(r: AnalysisResult, hand_side: str):
    if hand_side != "both":
        r.dominant_hand = "N/A"
        return
    if r.composite_right > r.composite_left + DOMINANCE_MARGIN:
        r.dominant_hand = "right"
    elif r.composite_left > r.composite_right + DOMINANCE_MARGIN:
        r.dominant_hand = "left"
    else:
        r.dominant_hand = "tie"


def _compute_lnu(r: AnalysisResult, right_ev, left_ev, hand_side: str):
    if hand_side != "both":
        r.lnu_score = None
        r.lnu_risk = "N/A"
        return

    n_r = sum(1 for e in right_ev if e["success"])
    n_l = sum(1 for e in left_ev if e["success"])

    if n_r < _MIN_BILATERAL_HITS or n_l < _MIN_BILATERAL_HITS:
        r.lnu_score = None
        r.lnu_risk = "N/A"
        return

    max_n = max(n_r, n_l, 1)
    min_n = min(n_r, n_l)
    use_asym = (1 - min_n / max_n) * 100

    rt_r = r.rt_mean_right if r.rt_mean_right > 0 else 999
    rt_l = r.rt_mean_left  if r.rt_mean_left  > 0 else 999
    max_rt  = max(rt_r, rt_l, 1)
    rt_asym = abs(rt_r - rt_l) / max_rt * 100

    q_asym = abs(r.composite_right - r.composite_left)

    lnu = (
        LNU_WEIGHT_USE_ASYM  * use_asym
        + LNU_WEIGHT_RT_ASYM * rt_asym
        + LNU_WEIGHT_QUAL_ASYM * q_asym
    )
    r.lnu_score = float(_clamp(lnu, 0, 100))

    if r.lnu_score < LNU_THRESHOLD_LOW:
        r.lnu_risk = "Low"
    elif r.lnu_score < LNU_THRESHOLD_HIGH:
        r.lnu_risk = "Moderate"
    else:
        r.lnu_risk = "High"


def _compute_motor_age(r: AnalysisResult, actual_age: int | None, hand_side: str = "both"):
    if hand_side == "right":
        best_composite = r.composite_right
        best_tremor    = r.tremor_power_right
    elif hand_side == "left":
        best_composite = r.composite_left
        best_tremor    = r.tremor_power_left
    else:
        best_composite = max(r.composite_right, r.composite_left)
        best_tremor    = min(r.tremor_power_right, r.tremor_power_left)

    base_age   = MOTOR_AGE_BASE_MIN + (100 - best_composite) / 100 * MOTOR_AGE_RANGE
    tremor_adj = max(0.0, best_tremor - MOTOR_AGE_TREMOR_THRESH) / MOTOR_AGE_TREMOR_DIVISOR
    r.motor_age = round(base_age + tremor_adj, 1)
