"""
Tire Chart Component — compound summaries, degradation tables, cliff badges.
"""

import logging
logger = logging.getLogger(__name__)


def build_tire_summary(circuit: str) -> str:
    return (
        f"## 🏎️ Tire Summary — **{circuit}**\n\n---\n\n"
        "### Compound Performance\n\n"
        "| Compound | Peak Pace | Δ vs Med | Stint (laps) | Deg Rate | Cliff |\n"
        "|----------|----------|---------|-------------|----------|-------|\n"
        "| 🔴 Soft | 1:31.2 | −0.8s | 12–16 | 0.15 s/lap | L18 |\n"
        "| 🟡 Medium | 1:32.0 | Baseline | 22–28 | 0.08 s/lap | L28 |\n"
        "| ⚪ Hard | 1:32.5 | +0.5s | 30–40 | 0.04 s/lap | L42+ |\n"
        "| 🟢 Inter | 1:34.0 | +2.0s | Full wet stint | Variable | N/A |\n"
        "| 🔵 Wet | 1:38.0 | +6.0s | Heavy rain only | Variable | N/A |\n\n"
        "### Degradation Notes\n\n"
        "- **Front-limited** circuits: higher soft wear\n"
        "- **Rear-limited** circuits: harder compounds in heat\n"
        "- Track evolution: +0.3–0.5s grip improvement per session\n\n"
        "> 🔧 *Connect LLM + FastF1 for real degradation data.*"
    )


def build_cliff_alert(compound: str, current_lap: int, cliff_lap: int) -> str:
    """Generate a cliff detection alert badge."""
    laps_remaining = cliff_lap - current_lap
    if laps_remaining <= 0:
        status = "🔴 **PAST CLIFF** — Performance severely degraded!"
        urgency = "CRITICAL"
    elif laps_remaining <= 2:
        status = f"🟠 **{laps_remaining} laps to cliff** — PIT NOW"
        urgency = "HIGH"
    elif laps_remaining <= 5:
        status = f"🟡 **{laps_remaining} laps to cliff** — Prepare pit stop"
        urgency = "MEDIUM"
    else:
        status = f"🟢 **{laps_remaining} laps to cliff** — Tire in good window"
        urgency = "LOW"

    return (
        f"### ⚠️ Cliff Alert — {compound}\n\n"
        f"| Metric | Value |\n"
        f"|--------|-------|\n"
        f"| Current Lap | {current_lap} |\n"
        f"| Predicted Cliff | L{cliff_lap} |\n"
        f"| Urgency | **{urgency}** |\n\n"
        f"{status}"
    )


def build_stint_estimator(compound: str, track_temp: float) -> str:
    """Estimate stint length based on compound and track temperature."""
    base = {"Soft": 16, "Medium": 26, "Hard": 38}
    base_laps = base.get(compound, 20)
    # Rough model: -15% per 10°C above 35°C baseline
    temp_factor = max(0.5, 1.0 - ((track_temp - 35) * 0.015))
    est_laps = int(base_laps * temp_factor)

    return (
        f"### 📏 Stint Estimator — {compound}\n\n"
        f"| Parameter | Value |\n"
        f"|-----------|-------|\n"
        f"| Compound | {compound} |\n"
        f"| Track Temp | {track_temp:.0f}°C |\n"
        f"| Base Stint | {base_laps} laps |\n"
        f"| Temp Adjustment | ×{temp_factor:.2f} |\n"
        f"| **Estimated Stint** | **{est_laps} laps** |\n"
    )
