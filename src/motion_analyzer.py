from dataclasses import dataclass, field

import numpy as np
from scipy.fft import fft, fftfreq
from scipy.interpolate import interp1d


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

    # LNU
    lnu_score: float = 0.0
    lnu_risk: str = "Low"

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
        r = AnalysisResult()

        right_events = [e for e in events if e["hand"] == "right"]
        left_events = [e for e in events if e["hand"] == "left"]

        _compute_speed(r, right_events, left_events)
        _compute_accuracy(r, right_events, left_events, events)
        _compute_quality(r, right_events, left_events)
        _compute_composite(r)
        _compute_dominance(r)
        _compute_lnu(r, right_events, left_events)
        _compute_motor_age(r, state.participant_age if state else None)

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
            score = _clamp(100 - (mean - 350) / 450 * 100, 0, 100)
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
            traj = e["trajectory"]
            path_len = e["path_length"]
            direct = e["direct_distance"]
            eff = direct / path_len if path_len > 1e-6 else 1.0
            eff_list.append(_clamp(eff, 0, 1))
            jerk_list.append(_compute_jerk_score(traj))
            tremor_list.append(_compute_tremor_power(traj))

        setattr(r, f"efficiency_list_{prefix}", eff_list)
        setattr(r, f"jerk_list_{prefix}", jerk_list)
        setattr(r, f"tremor_list_{prefix}", tremor_list)

        eff_score = float(np.mean(eff_list) * 100) if eff_list else 50.0
        jerk_score = float(np.mean(jerk_list)) if jerk_list else 50.0
        tremor_score = float(np.mean(tremor_list)) if tremor_list else 0.0

        setattr(r, f"path_efficiency_{prefix}", eff_score)
        setattr(r, f"jerk_score_{prefix}", jerk_score)
        setattr(r, f"tremor_power_{prefix}", tremor_score)

        quality = 0.4 * eff_score + 0.4 * jerk_score + 0.2 * (100 - tremor_score)
        setattr(r, f"quality_score_{prefix}", _clamp(quality, 0, 100))


def _compute_jerk_score(traj: list) -> float:
    if len(traj) < 5:
        return 50.0
    times = np.array([p["t"] for p in traj])
    xs = np.array([p["x"] for p in traj])
    ys = np.array([p["y"] for p in traj])

    dt = np.diff(times)
    dt = np.where(dt < 1e-6, 1e-6, dt)

    vx = np.diff(xs) / dt
    vy = np.diff(ys) / dt
    if len(vx) < 2:
        return 50.0

    dt2 = dt[1:]
    ax = np.diff(vx) / dt2
    ay = np.diff(vy) / dt2
    if len(ax) < 2:
        return 50.0

    dt3 = dt2[1:]
    jx = np.diff(ax) / dt3
    jy = np.diff(ay) / dt3

    duration = float(times[-1] - times[0])
    dx = xs[-1] - xs[0]
    dy = ys[-1] - ys[0]
    distance = float((dx * dx + dy * dy) ** 0.5)

    if duration < 1e-6 or distance < 1e-6:
        return 50.0

    jerk_mag = np.sqrt(jx ** 2 + jy ** 2)
    njs = 0.5 * np.trapz(jerk_mag ** 2, times[3:]) * (duration ** 5 / distance ** 2)
    score = _clamp(100 - np.log1p(njs) * 15, 0, 100)
    return float(score)


def _compute_tremor_power(traj: list, fs: float = 10.0) -> float:
    if len(traj) < 8:
        return 0.0
    times = np.array([p["t"] for p in traj])
    xs = np.array([p["x"] for p in traj])

    duration = times[-1] - times[0]
    if duration < 0.5:
        return 0.0

    n_samples = max(8, int(duration * fs))
    t_uniform = np.linspace(times[0], times[-1], n_samples)
    try:
        f_interp = interp1d(times, xs, kind="linear", bounds_error=False, fill_value="extrapolate")
        x_uniform = f_interp(t_uniform)
    except Exception:
        return 0.0

    N = len(x_uniform)
    freqs = fftfreq(N, 1.0 / fs)
    power = np.abs(fft(x_uniform)) ** 2

    pos_mask = freqs > 0
    tremor_mask = (np.abs(freqs) >= 3) & (np.abs(freqs) <= 7)

    total_power = np.sum(power[pos_mask])
    tremor_p = np.sum(power[tremor_mask])

    if total_power < 1e-10:
        return 0.0
    ratio = tremor_p / total_power
    return float(_clamp(ratio * 100, 0, 100))


def _compute_composite(r: AnalysisResult):
    for prefix in ("right", "left"):
        speed = getattr(r, f"speed_score_{prefix}")
        acc = getattr(r, f"accuracy_score_{prefix}")
        qual = getattr(r, f"quality_score_{prefix}")
        comp = 0.4 * speed + 0.3 * acc + 0.3 * qual
        setattr(r, f"composite_{prefix}", _clamp(comp, 0, 100))


def _compute_dominance(r: AnalysisResult):
    if r.composite_right > r.composite_left + 2:
        r.dominant_hand = "right"
    elif r.composite_left > r.composite_right + 2:
        r.dominant_hand = "left"
    else:
        r.dominant_hand = "tie"


def _compute_lnu(r: AnalysisResult, right_ev, left_ev):
    n_r = sum(1 for e in right_ev if e["success"])
    n_l = sum(1 for e in left_ev if e["success"])
    total = n_r + n_l

    if total == 0:
        r.lnu_score = 50.0
        r.lnu_risk = "Moderate"
        return

    max_n = max(n_r, n_l, 1)
    min_n = min(n_r, n_l)
    use_asym = (1 - min_n / max_n) * 100

    rt_r = r.rt_mean_right if r.rt_mean_right > 0 else 999
    rt_l = r.rt_mean_left if r.rt_mean_left > 0 else 999
    max_rt = max(rt_r, rt_l, 1)
    rt_asym = abs(rt_r - rt_l) / max_rt * 100

    q_asym = abs(r.composite_right - r.composite_left)

    lnu = 0.5 * use_asym + 0.3 * rt_asym + 0.2 * q_asym
    r.lnu_score = float(_clamp(lnu, 0, 100))

    if r.lnu_score < 33:
        r.lnu_risk = "Low"
    elif r.lnu_score < 67:
        r.lnu_risk = "Moderate"
    else:
        r.lnu_risk = "High"


def _compute_motor_age(r: AnalysisResult, actual_age: int | None):
    best_composite = max(r.composite_right, r.composite_left)
    best_tremor = min(r.tremor_power_right, r.tremor_power_left)
    base_age = 20 + (100 - best_composite) / 100 * 65
    tremor_adj = max(0.0, best_tremor - 10) / 5
    r.motor_age = round(base_age + tremor_adj, 1)
