import io
import os
import tempfile

import pandas as pd
from fpdf import FPDF


def export_csv(events: list, filepath: str | None = None) -> str:
    if not events:
        return ""

    rows = []
    for e in events:
        rows.append({
            "event_id": e["event_id"],
            "target_cell": e["target_cell"],
            "hand": e["hand"],
            "success": e["success"],
            "reaction_time_ms": e.get("reaction_time_ms"),
            "path_length": e.get("path_length"),
            "direct_distance": e.get("direct_distance"),
            "target_shown_at": e["target_shown_at"],
            "hit_at": e.get("hit_at"),
        })

    df = pd.DataFrame(rows)
    if filepath is None:
        fd, filepath = tempfile.mkstemp(suffix=".csv", prefix="dexterity_")
        os.close(fd)
    df.to_csv(filepath, index=False)
    return filepath


def export_pdf(analysis, charts: dict, state, filepath: str | None = None) -> str:
    if filepath is None:
        fd, filepath = tempfile.mkstemp(suffix=".pdf", prefix="dexterity_report_")
        os.close(fd)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    name = state.participant_name if state and state.participant_name else "Participant"
    age = state.participant_age if state else None
    duration = state.duration_s if state else 0
    total = len(state.events) if state else 0
    hits = sum(1 for e in state.events if e["success"]) if state else 0

    # Page 1: Summary
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 22)
    pdf.cell(0, 14, "Upper Limb Dexterity Assessment", ln=True, align="C")
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 8, f"Participant: {name}  |  Age: {age or 'N/A'}  |  Duration: {duration}s", ln=True, align="C")
    pdf.ln(8)

    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Performance Summary", ln=True)
    pdf.set_font("Helvetica", "", 11)

    rows = [
        ("Metric", "Right Hand", "Left Hand"),
        ("Reaction Time (mean)", f"{analysis.rt_mean_right:.0f}ms", f"{analysis.rt_mean_left:.0f}ms"),
        ("Speed Score", f"{analysis.speed_score_right:.0f}/100", f"{analysis.speed_score_left:.0f}/100"),
        ("Hit Rate", f"{analysis.hit_rate_right:.0f}%", f"{analysis.hit_rate_left:.0f}%"),
        ("Quality Score", f"{analysis.quality_score_right:.0f}/100", f"{analysis.quality_score_left:.0f}/100"),
        ("Path Efficiency", f"{analysis.path_efficiency_right:.0f}%", f"{analysis.path_efficiency_left:.0f}%"),
        ("Tremor Index", f"{analysis.tremor_power_right:.1f}", f"{analysis.tremor_power_left:.1f}"),
        ("Composite Score", f"{analysis.composite_right:.0f}/100", f"{analysis.composite_left:.0f}/100"),
    ]

    col_widths = [70, 55, 55]
    for i, row in enumerate(rows):
        if i == 0:
            pdf.set_font("Helvetica", "B", 11)
        else:
            pdf.set_font("Helvetica", "", 11)
        for j, cell in enumerate(row):
            pdf.cell(col_widths[j], 8, cell, border=1)
        pdf.ln()

    pdf.ln(8)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 10, f"Dominant Hand: {analysis.dominant_hand.upper()}", ln=True)
    pdf.cell(0, 10, f"Learn Non-Use Risk: {analysis.lnu_risk} (Score: {analysis.lnu_score:.1f}/100)", ln=True)
    pdf.cell(0, 10, f"Estimated Motor Age: {analysis.motor_age} years", ln=True)

    # Page 2: Charts (embedded as PNG if kaleido is available)
    try:
        for chart_name, fig in charts.items():
            if fig is None:
                continue
            img_bytes = _fig_to_png_bytes(fig)
            if img_bytes:
                pdf.add_page()
                pdf.set_font("Helvetica", "B", 13)
                pdf.cell(0, 10, chart_name.replace("_", " ").title(), ln=True)
                fd2, tmp_img = tempfile.mkstemp(suffix=".png")
                os.close(fd2)
                with open(tmp_img, "wb") as f:
                    f.write(img_bytes)
                try:
                    pdf.image(tmp_img, w=180)
                finally:
                    os.unlink(tmp_img)
    except Exception:
        pass

    # Methodology page
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Methodology", ln=True)
    pdf.set_font("Helvetica", "", 10)
    methodology = (
        "This assessment uses MediaPipe Holistic to track hand landmarks via webcam. "
        "Participants reach toward randomly highlighted zones in a 3x3 grid. "
        "Speed is measured as reaction time from target appearance to hand dwell (3 consecutive frames). "
        "Path Efficiency = direct distance / actual path length. "
        "Smoothness is computed via Normalized Jerk Score (NJS). "
        "Tremor Power uses FFT to measure oscillation in the 3-7 Hz band. "
        "Learn Non-Use (LNU) Risk combines use asymmetry, reaction time asymmetry, "
        "and quality asymmetry between limbs. "
        "Motor Age is estimated from composite performance relative to age-normative data."
    )
    pdf.multi_cell(0, 6, methodology)

    pdf.output(filepath)
    return filepath


def _fig_to_png_bytes(fig) -> bytes | None:
    try:
        return fig.to_image(format="png", width=800, height=400)
    except Exception:
        return None
