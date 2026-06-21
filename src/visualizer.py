import numpy as np
import cv2
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.tracker import CELL_NAMES

_GRID_COLOR = (255, 255, 255)
_TARGET_COLOR = (0, 255, 0)
_FLASH_GREEN = (0, 200, 0)
_FLASH_RED = (0, 0, 220)
_HUD_COLOR = (255, 255, 255)

# ── Skeleton / mesh connection lists (MediaPipe Tasks API landmark indices) ───

_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

_POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),  # head
    (9, 10),                                                              # mouth
    (11, 12),                                                             # shoulders
    (11, 13), (13, 15), (15, 17), (17, 19), (19, 15), (15, 21),         # left arm
    (12, 14), (14, 16), (16, 18), (18, 20), (20, 16), (16, 22),         # right arm
    (11, 23), (12, 24), (23, 24),                                         # torso
    (23, 25), (25, 27), (27, 29), (29, 31), (31, 27),                   # left leg
    (24, 26), (26, 28), (28, 30), (30, 32), (32, 28),                   # right leg
]

_FACE_OVAL = [
    (10, 338), (338, 297), (297, 332), (332, 284), (284, 251), (251, 389),
    (389, 356), (356, 454), (454, 323), (323, 361), (361, 288), (288, 397),
    (397, 365), (365, 379), (379, 378), (378, 400), (400, 377), (377, 152),
    (152, 148), (148, 176), (176, 149), (149, 150), (150, 136), (136, 172),
    (172, 58), (58, 132), (132, 93), (93, 234), (234, 127), (127, 162),
    (162, 21), (21, 54), (54, 103), (103, 67), (67, 109), (109, 10),
]
_LEFT_EYE = [
    (263, 249), (249, 390), (390, 373), (373, 374), (374, 380), (380, 381),
    (381, 382), (382, 362), (362, 398), (398, 384), (384, 385), (385, 386),
    (386, 387), (387, 388), (388, 466), (466, 263),
]
_RIGHT_EYE = [
    (33, 7), (7, 163), (163, 144), (144, 145), (145, 153), (153, 154),
    (154, 155), (155, 133), (133, 173), (173, 157), (157, 158), (158, 159),
    (159, 160), (160, 161), (161, 246), (246, 33),
]
_LIPS = [
    (61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314),
    (314, 405), (405, 321), (321, 375), (375, 291),
    (61, 185), (185, 40), (40, 39), (39, 37), (37, 0), (0, 267),
    (267, 269), (269, 270), (270, 291),
    (78, 95), (95, 88), (88, 178), (178, 87), (87, 14), (14, 317),
    (317, 402), (402, 318), (318, 324), (324, 308),
    (78, 191), (191, 80), (80, 81), (81, 82), (82, 13), (13, 312),
    (312, 311), (311, 310), (310, 415), (415, 308),
]
_FACE_CONNECTIONS = _FACE_OVAL + _LEFT_EYE + _RIGHT_EYE + _LIPS


def _draw_connections(img, landmarks, connections, color, thickness=1):
    H, W = img.shape[:2]
    pts = [(int(lm.x * W), int(lm.y * H)) for lm in landmarks]
    for a, b in connections:
        if a < len(pts) and b < len(pts):
            cv2.line(img, pts[a], pts[b], color, thickness, cv2.LINE_AA)


def _draw_dots(img, landmarks, color, radius=3):
    H, W = img.shape[:2]
    for lm in landmarks:
        cv2.circle(img, (int(lm.x * W), int(lm.y * H)), radius, color, -1)


def draw_body_overlays(
    out: np.ndarray,
    pose_landmarks=None,
    right_hand_landmarks=None,
    left_hand_landmarks=None,
    face_landmarks=None,
) -> np.ndarray:
    """Draw pose skeleton, hand skeletons, and face mesh on a BGR frame."""
    if pose_landmarks is not None:
        _draw_connections(out, pose_landmarks, _POSE_CONNECTIONS, (0, 120, 255), 2)
        _draw_dots(out, pose_landmarks, (0, 165, 255), 4)

    for hand_lm, color in [
        (right_hand_landmarks, (0, 220, 0)),
        (left_hand_landmarks, (255, 120, 80)),
    ]:
        if hand_lm is not None:
            _draw_connections(out, hand_lm, _HAND_CONNECTIONS, color, 2)
            _draw_dots(out, hand_lm, color, 3)

    if face_landmarks is not None:
        _draw_connections(out, face_landmarks, _FACE_CONNECTIONS, (50, 150, 200), 1)

    return out


def _put_text_centered(img, text, cx, cy, font_scale, color, thickness):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x = cx - tw // 2
    y = cy + th // 2
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


def _draw_bbox_corners(img, cx0, cy0, cx1, cy1, color, length=28, thick=5):
    """Draw L-shaped corner marks — looks like a target bounding box."""
    pts = [
        ((cx0, cy0), (cx0 + length, cy0), (cx0, cy0 + length)),
        ((cx1, cy0), (cx1 - length, cy0), (cx1, cy0 + length)),
        ((cx0, cy1), (cx0 + length, cy1), (cx0, cy1 - length)),
        ((cx1, cy1), (cx1 - length, cy1), (cx1, cy1 - length)),
    ]
    for corner, h_pt, v_pt in pts:
        cv2.line(img, corner, h_pt, color, thick, cv2.LINE_AA)
        cv2.line(img, corner, v_pt, color, thick, cv2.LINE_AA)


def _draw_grid(out, x0, y0, x1, y1, target=-1, dwell=0, color=(200, 200, 200)):
    """Draw 3x3 grid with cell numbers; highlight target cell with bbox."""
    cw = max((x1 - x0) // 3, 1)
    ch = max((y1 - y0) // 3, 1)

    # Target cell fill
    if target >= 0:
        row, col = divmod(target, 3)
        cx0, cy0 = x0 + col * cw, y0 + row * ch
        cx1, cy1 = cx0 + cw, cy0 + ch
        alpha = 0.20 + min(dwell / 3, 1.0) * 0.35
        overlay = out.copy()
        cv2.rectangle(overlay, (cx0, cy0), (cx1, cy1), _TARGET_COLOR, -1)
        cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)

    # Grid lines
    for i in range(1, 3):
        cv2.line(out, (x0 + i * cw, y0), (x0 + i * cw, y1), color, 2, cv2.LINE_AA)
        cv2.line(out, (x0, y0 + i * ch), (x1, y0 + i * ch), color, 2, cv2.LINE_AA)

    # Cell numbers 1–9 (dim in non-target cells, bright in target cell)
    for cell in range(9):
        row, col = divmod(cell, 3)
        ccx = x0 + col * cw + cw // 2
        ccy = y0 + row * ch + ch // 2
        is_target = (cell == target)
        label_color = (0, 255, 0) if is_target else (180, 180, 180)
        scale = 1.8 if is_target else 0.8
        thick = 3 if is_target else 1
        _put_text_centered(out, str(cell + 1), ccx, ccy, scale, label_color, thick)

    # Bounding-box on target cell: thick border + corner marks
    if target >= 0:
        row, col = divmod(target, 3)
        cx0, cy0 = x0 + col * cw, y0 + row * ch
        cx1, cy1 = cx0 + cw, cy0 + ch
        cv2.rectangle(out, (cx0, cy0), (cx1, cy1), _TARGET_COLOR, 3)
        _draw_bbox_corners(out, cx0, cy0, cx1, cy1, (255, 255, 0), length=32, thick=5)


def draw_countdown_overlay(
    frame: np.ndarray,
    pose_landmarks=None,
    count: int = 3,
    right_hand_landmarks=None,
    left_hand_landmarks=None,
    face_landmarks=None,
) -> np.ndarray:
    """Show 3-2-1 countdown before test starts."""
    out = frame.copy()
    H, W = out.shape[:2]
    draw_body_overlays(out, pose_landmarks, right_hand_landmarks, left_hand_landmarks, face_landmarks)
    _draw_grid(out, 0, 0, W, H, target=-1, dwell=0, color=(120, 120, 120))

    # Dark vignette to make number pop
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (W, H), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.35, out, 0.65, 0, out)

    if count > 0:
        # Large countdown number
        num_text = str(count)
        scale = 8.0
        thick = 16
        (tw, th), _ = cv2.getTextSize(num_text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        nx = (W - tw) // 2
        ny = (H + th) // 2
        cv2.putText(out, num_text, (nx, ny), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 4, cv2.LINE_AA)
        cv2.putText(out, num_text, (nx, ny), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 220, 255), thick, cv2.LINE_AA)

    label = "Get Ready!" if count > 0 else "GO!"
    label_color = (0, 220, 255) if count > 0 else (0, 255, 80)
    _put_text_centered(out, label, W // 2, H // 4, 1.6, label_color, 3)
    _put_text_centered(out, "Assessment starting...", W // 2, H * 3 // 4, 0.9, (200, 200, 200), 2)
    return out


def draw_overlay(
    frame: np.ndarray,
    state,
    result,
    grid_bounds: tuple,
    right_pos: tuple | None,
    left_pos: tuple | None,
    pose_landmarks=None,
    right_hand_landmarks=None,
    left_hand_landmarks=None,
    face_landmarks=None,
) -> np.ndarray:
    out = frame.copy()
    H, W = out.shape[:2]

    x0, y0 = 0, 0
    x1, y1 = W, H

    # Flash overlay
    if state and state.flash_frames > 0 and state.flash_color != "none":
        flash_col = _FLASH_GREEN if state.flash_color == "green" else _FLASH_RED
        overlay = out.copy()
        cv2.rectangle(overlay, (0, 0), (W, H), flash_col, -1)
        cv2.addWeighted(overlay, 0.28, out, 0.72, 0, out)

    # Body overlays (pose, hands, face) drawn before grid so grid renders on top
    draw_body_overlays(out, pose_landmarks, right_hand_landmarks, left_hand_landmarks, face_landmarks)

    # 3x3 grid with cell numbers
    target = state.current_target if state else -1
    dwell = state.dwell_count if state else 0
    _draw_grid(out, x0, y0, x1, y1, target=target, dwell=dwell)

    # Hand dots with labels
    for pos, dot_color, label in [
        (right_pos, (0, 255, 100), "R"),
        (left_pos,  (100, 150, 255), "L"),
    ]:
        if pos is not None:
            px_x = int(pos[0] * W)
            px_y = int(pos[1] * H)
            cv2.circle(out, (px_x, px_y), 16, dot_color, -1)
            cv2.circle(out, (px_x, px_y), 16, (255, 255, 255), 2)
            _put_text_centered(out, label, px_x, px_y, 0.6, (0, 0, 0), 2)

    # Top bar: "Touch Zone X" + hits
    if state and target >= 0 and result:
        bar_h = 44
        overlay = out.copy()
        cv2.rectangle(overlay, (0, 0), (W, bar_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, out, 0.4, 0, out)

        zone_text = f"Touch Zone {target + 1}"
        hits_text = f"Hits: {result.hit_count}"
        _put_text_centered(out, zone_text, W // 2, bar_h // 2, 1.0, (0, 255, 0), 2)
        _put_text_centered(out, hits_text, W - 80, bar_h // 2, 0.8, (255, 255, 255), 2)

    # Bottom-left countdown timer (big, colour-coded)
    if result:
        secs = int(result.remaining_s)
        timer_color = (0, 200, 255) if secs > 10 else (0, 100, 255) if secs > 5 else (0, 0, 255)
        timer_text = f"{secs}s"
        # Dark pill background
        (tw, th), _ = cv2.getTextSize(timer_text, cv2.FONT_HERSHEY_SIMPLEX, 2.4, 4)
        pad = 12
        bx0, by0 = 10, H - th - pad * 2 - 8
        bx1, by1 = bx0 + tw + pad * 2, H - 8
        overlay2 = out.copy()
        cv2.rectangle(overlay2, (bx0, by0), (bx1, by1), (0, 0, 0), -1)
        cv2.addWeighted(overlay2, 0.6, out, 0.4, 0, out)
        cv2.putText(out, timer_text, (bx0 + pad, by1 - pad),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.4, (0, 0, 0), 7, cv2.LINE_AA)
        cv2.putText(out, timer_text, (bx0 + pad, by1 - pad),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.4, timer_color, 4, cv2.LINE_AA)

    # Progress bar at very bottom (thin strip)
    if state and result:
        total = max(state.duration_s, 1)
        progress = min(result.elapsed_s / total, 1.0)
        bar_w = int(W * progress)
        cv2.rectangle(out, (0, H - 6), (bar_w, H), (0, 165, 255), -1)
        cv2.rectangle(out, (0, H - 6), (W, H), (60, 60, 60), 1)

    return out


def draw_preview_overlay(
    frame: np.ndarray,
    grid_bounds: tuple,
    pose_detected: bool,
    pose_landmarks=None,
    right_hand_landmarks=None,
    left_hand_landmarks=None,
    face_landmarks=None,
) -> np.ndarray:
    out = frame.copy()
    H, W = out.shape[:2]

    # Body overlays first, grid on top
    draw_body_overlays(out, pose_landmarks, right_hand_landmarks, left_hand_landmarks, face_landmarks)
    _draw_grid(out, 0, 0, W, H, target=-1, dwell=0, color=(220, 220, 220))

    # Status bar at top
    bar_h = 44
    cv2.rectangle(out, (0, 0), (W, bar_h), (0, 0, 0), -1)
    cv2.addWeighted(out[0:bar_h], 0.5, out[0:bar_h], 0.5, 0, out[0:bar_h])

    status = "Ready — Click Start Assessment" if pose_detected else "Stand in front of camera..."
    color = (0, 220, 0) if pose_detected else (0, 120, 255)
    _put_text_centered(out, status, W // 2, bar_h // 2, 0.8, color, 2)

    return out


# ─── Plotly Charts ───────────────────────────────────────────────────────────


def make_speed_chart(events: list) -> go.Figure:
    right_rts = [e["reaction_time_ms"] for e in events
                 if e["hand"] == "right" and e["success"] and e["reaction_time_ms"]]
    left_rts = [e["reaction_time_ms"] for e in events
                if e["hand"] == "left" and e["success"] and e["reaction_time_ms"]]

    fig = go.Figure()

    if right_rts:
        fig.add_trace(go.Box(y=right_rts, name="Right Hand", marker_color="#2ecc71",
                             boxpoints="all", jitter=0.4, pointpos=-1.8))
    if left_rts:
        fig.add_trace(go.Box(y=left_rts, name="Left Hand", marker_color="#3498db",
                             boxpoints="all", jitter=0.4, pointpos=-1.8))

    all_rts = right_rts + left_rts
    if all_rts:
        mean_rt = float(np.mean(all_rts))
        fig.add_hline(y=mean_rt, line_dash="dash", line_color="orange",
                      annotation_text=f"Mean: {mean_rt:.0f}ms")

    fig.update_layout(
        title="Reaction Time Distribution",
        yaxis_title="Reaction Time (ms)",
        template="plotly_white",
        height=400,
    )
    return fig


def make_accuracy_chart(events: list) -> go.Figure:
    hits = {i: 0 for i in range(9)}
    totals = {i: 0 for i in range(9)}
    for e in events:
        c = e["target_cell"]
        if 0 <= c <= 8:
            totals[c] += 1
            if e["success"]:
                hits[c] += 1

    cells = [CELL_NAMES.get(i, str(i)) for i in range(9)]
    hit_rates = [hits[i] / totals[i] * 100 if totals[i] > 0 else 0 for i in range(9)]
    colors = [f"rgba({int(255*(1-r/100))}, {int(200*r/100)}, 50, 0.8)" for r in hit_rates]

    fig = go.Figure(go.Bar(x=cells, y=hit_rates, marker_color=colors,
                           text=[f"{r:.0f}%" for r in hit_rates], textposition="auto"))
    fig.update_layout(
        title="Hit Rate by Target Zone",
        yaxis_title="Hit Rate (%)",
        yaxis_range=[0, 105],
        template="plotly_white",
        height=380,
    )
    return fig


def make_quality_chart(events: list, analysis=None) -> go.Figure:
    fig = make_subplots(rows=3, cols=1, subplot_titles=[
        "Path Efficiency (higher = straighter)",
        "Movement Smoothness / Jerk Score (higher = smoother)",
        "Tremor Power (lower = less tremor)",
    ], vertical_spacing=0.12)

    if analysis:
        for prefix, color, name in [("right", "#2ecc71", "Right"), ("left", "#3498db", "Left")]:
            eff = getattr(analysis, f"efficiency_list_{prefix}", [])
            jerk = getattr(analysis, f"jerk_list_{prefix}", [])
            tremor = getattr(analysis, f"tremor_list_{prefix}", [])

            if eff:
                fig.add_trace(go.Scatter(y=eff, name=f"{name} Efficiency",
                                          line_color=color, mode="lines+markers"), row=1, col=1)
            if jerk:
                fig.add_trace(go.Scatter(y=jerk, name=f"{name} Jerk",
                                          line_color=color, mode="lines+markers", showlegend=False), row=2, col=1)
            if tremor:
                fig.add_trace(go.Scatter(y=tremor, name=f"{name} Tremor",
                                          line_color=color, mode="lines+markers", showlegend=False), row=3, col=1)

    fig.update_layout(template="plotly_white", height=600, title="Movement Quality")
    return fig


def make_dominance_chart(analysis) -> go.Figure:
    if analysis.dominant_hand == "N/A":
        fig = go.Figure()
        fig.update_layout(
            title="Hand Dominance: N/A — requires Both Hands mode",
            height=420, template="plotly_white",
        )
        return fig

    categories = ["Speed", "Accuracy", "Quality", "Composite"]

    r_vals = [
        analysis.speed_score_right,
        analysis.accuracy_score_right,
        analysis.quality_score_right,
        analysis.composite_right,
    ]
    l_vals = [
        analysis.speed_score_left,
        analysis.accuracy_score_left,
        analysis.quality_score_left,
        analysis.composite_left,
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=r_vals + [r_vals[0]], theta=categories + [categories[0]],
                                   fill="toself", name="Right Hand", line_color="#2ecc71"))
    fig.add_trace(go.Scatterpolar(r=l_vals + [l_vals[0]], theta=categories + [categories[0]],
                                   fill="toself", name="Left Hand", line_color="#3498db",
                                   opacity=0.7))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title=f"Dominant Hand (screening): {analysis.dominant_hand.upper()}",
        template="plotly_white",
        height=420,
    )
    return fig


def make_lnu_gauge(score: float | None, risk: str) -> go.Figure:
    if score is None:
        fig = go.Figure()
        fig.update_layout(
            title="Learned Non-Use Risk: N/A<br><sup>Requires Both Hands with ≥3 hits each</sup>",
            height=300, template="plotly_white",
        )
        return fig

    color = "#27ae60" if risk == "Low" else "#e67e22" if risk == "Moderate" else "#e74c3c"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        title={"text": f"Learned Non-Use Risk (screening): <b>{risk}</b>", "font": {"size": 18}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 33], "color": "#d5f5e3"},
                {"range": [33, 67], "color": "#fdebd0"},
                {"range": [67, 100], "color": "#fadbd8"},
            ],
            "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": score},
        },
        number={"suffix": " / 100"},
    ))
    fig.update_layout(height=300, template="plotly_white")
    return fig


def make_gantt_chart(events: list, total_duration_s: float) -> go.Figure:
    if not events:
        return go.Figure().update_layout(title="No events recorded", height=300)

    start0 = events[0]["target_shown_at"] if events else 0

    rows = []
    for e in events:
        t_start = e["target_shown_at"] - start0
        t_end = (e["hit_at"] - start0) if e["hit_at"] else (t_start + 5.0)
        rows.append({
            "Task": "Right Hand" if e["hand"] == "right" else "Left Hand",
            "Start": t_start,
            "Finish": t_end,
            "Status": "Hit" if e["success"] else "Miss",
            "Cell": CELL_NAMES.get(e["target_cell"], str(e["target_cell"])),
            "RT": f"{e['reaction_time_ms']:.0f}ms" if e["reaction_time_ms"] else "—",
        })

    import pandas as pd
    base = pd.Timestamp("2000-01-01")
    df = pd.DataFrame(rows)
    df["Start_dt"] = base + pd.to_timedelta(df["Start"], unit="s")
    df["Finish_dt"] = base + pd.to_timedelta(df["Finish"], unit="s")

    color_map = {"Hit": "#2ecc71", "Miss": "#e74c3c"}
    fig = px.timeline(
        df, x_start="Start_dt", x_end="Finish_dt", y="Task", color="Status",
        color_discrete_map=color_map,
        hover_data={"Cell": True, "RT": True, "Start": ":.1f", "Finish": ":.1f"},
        title="Reaching Events Timeline (Gantt)",
    )
    fig.update_layout(
        template="plotly_white", height=350,
        xaxis_title="Time from Start",
        xaxis=dict(tickformat="%Ss"),
    )
    return fig


def make_motor_age_gauge(motor_age: float | None, actual_age: int | None) -> go.Figure:
    if motor_age is None:
        return go.Figure().update_layout(title="Insufficient data for Motor Age", height=250)

    steps = [
        {"range": [20, 45], "color": "#d5f5e3"},
        {"range": [45, 65], "color": "#fef9e7"},
        {"range": [65, 85], "color": "#fdebd0"},
    ]
    delta_ref = actual_age if actual_age else motor_age

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=motor_age,
        title={"text": "Estimated Motor Age", "font": {"size": 18}},
        delta={"reference": delta_ref, "increasing": {"color": "#e74c3c"},
               "decreasing": {"color": "#27ae60"},
               "suffix": " yrs vs chronological"},
        gauge={
            "axis": {"range": [20, 85]},
            "bar": {"color": "#8e44ad"},
            "steps": steps,
        },
        number={"suffix": " yrs"},
    ))
    fig.update_layout(height=300, template="plotly_white")
    return fig


def make_summary_markdown(analysis, state) -> str:
    name = state.participant_name if state and state.participant_name else "Participant"
    duration = state.duration_s if state else 0
    total_events = len(state.events) if state else 0
    hits = sum(1 for e in state.events if e["success"]) if state else 0

    lnu_emoji = {"Low": "🟢", "Moderate": "🟡", "High": "🔴"}.get(analysis.lnu_risk, "⚪")

    if analysis.lnu_score is None:
        lnu_line = f"⚪ **{analysis.lnu_risk}** — requires Both Hands with ≥3 hits each"
    else:
        lnu_line = f"{lnu_emoji} **{analysis.lnu_risk}** — Score: {analysis.lnu_score:.1f}/100"

    if analysis.dominant_hand == "N/A":
        dom_line = "**N/A** — requires Both Hands mode"
    else:
        dom_line = f"**{analysis.dominant_hand.upper()}** (Right: {analysis.composite_right:.0f} | Left: {analysis.composite_left:.0f})"

    lines = [
        f"## Assessment Report — {name}",
        f"**Duration:** {duration}s | **Total Targets:** {total_events} | **Hits:** {hits}",
        "",
        "> **Screening only:** These are experimental estimates, not clinical diagnoses. Consult a healthcare professional for medical evaluation.",
        "",
        "### Speed",
        f"- Right: {analysis.rt_mean_right:.0f}ms mean RT (score: {analysis.speed_score_right:.0f}/100)" if analysis.rt_mean_right > 0 else "- Right: no data",
        f"- Left: {analysis.rt_mean_left:.0f}ms mean RT (score: {analysis.speed_score_left:.0f}/100)" if analysis.rt_mean_left > 0 else "- Left: no data",
        "",
        "### Accuracy",
        f"- Right: {analysis.hit_rate_right:.0f}%",
        f"- Left: {analysis.hit_rate_left:.0f}%",
        "",
        "### Quality",
        f"- Right: {analysis.quality_score_right:.0f}/100 (efficiency {analysis.path_efficiency_right:.0f}%, tremor {analysis.tremor_power_right:.0f})",
        f"- Left: {analysis.quality_score_left:.0f}/100 (efficiency {analysis.path_efficiency_left:.0f}%, tremor {analysis.tremor_power_left:.0f})",
        "",
        "### Dominant Hand",
        dom_line,
        "",
        "### Learned Non-Use Risk",
        lnu_line,
        "",
        "### Motor Age Estimate",
        f"**{analysis.motor_age} years**" + (f" (Chronological: {state.participant_age})" if state and state.participant_age else ""),
    ]
    return "\n".join(lines)
