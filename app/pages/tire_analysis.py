"""
Tire Analysis Page Handler — degradation modeling, compound comparison, cliff detection.
"""

import sys, os, logging

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logger = logging.getLogger(__name__)


def get_welcome_message() -> str:
    return (
        "## 🏎️ Tire Analysis Lab\n\n"
        "Welcome to the **Tire Degradation Center**. I model compound "
        "performance, predict cliff laps, and optimize stint lengths.\n\n---\n\n"
        "| Feature | Description |\n"
        "|---------|-------------|\n"
        "| **Degradation Curves** | Pace loss per lap per compound |\n"
        "| **Stint Estimator** | Projected optimal stint with fuel correction |\n"
        "| **Cliff Detection** | ⚠️ Alert when grip drops critically |\n"
        "| **Compound Comparison** | Side-by-side S / M / H / Inter / Wet |\n"
        "| **Track Temp Impact** | How temperature shifts compound windows |\n\n"
        "---\n\nSelect a **compound** or ask about a specific circuit. 🚀"
    )


def get_starters():
    import chainlit as cl
    return [
        cl.Starter(label="🔴 Soft Analysis", message="Analyze soft tire degradation at Monza. When does the cliff happen?", icon="/public/icon_tire.png"),
        cl.Starter(label="📊 Compound Comparison", message="Compare soft vs medium vs hard compounds at Silverstone. Show degradation curves.", icon="/public/icon_tire.png"),
        cl.Starter(label="⚠️ Cliff Detection", message="Predict when the medium tire will hit the cliff at Barcelona in hot conditions (40°C track temp).", icon="/public/icon_tire.png"),
        cl.Starter(label="🌡️ Temp Impact", message="How does track temperature affect tire degradation at Abu Dhabi between day and night sessions?", icon="/public/icon_tire.png"),
    ]


async def handle_message(user_input: str, history: list) -> str:
    lower = user_input.lower()
    if "cliff" in lower or "drop" in lower or "alert" in lower:
        return _cliff_detection(user_input)
    if "compar" in lower or ("soft" in lower and "medium" in lower) or ("soft" in lower and "hard" in lower):
        return _compound_comparison()
    if "temperature" in lower or "temp" in lower or "heat" in lower:
        return _temp_impact()
    if any(k in lower for k in ("soft", "medium", "hard", "inter", "wet", "compound", "tire", "tyre", "degradation", "deg")):
        return _degradation_analysis(user_input)
    return _fallback()


def _degradation_analysis(user_input):
    compound = "Soft"
    if "medium" in user_input.lower():
        compound = "Medium"
    elif "hard" in user_input.lower():
        compound = "Hard"
    elif "inter" in user_input.lower():
        compound = "Intermediate"

    icons = {"Soft": "🔴", "Medium": "🟡", "Hard": "⚪", "Intermediate": "🟢"}
    icon = icons.get(compound, "🔴")

    return (
        f"### {icon} {compound} Tire Degradation Analysis\n\n---\n\n"
        "#### Pace Evolution Over Stint\n\n"
        "| Lap | Pace (s) | Deg (s/lap) | Grip Level | Status |\n"
        "|-----|----------|-------------|------------|--------|\n"
        "| 1 | 1:31.2 | — | 100% | 🟢 Peak |\n"
        "| 5 | 1:31.8 | 0.12 | 96% | 🟢 Good |\n"
        "| 10 | 1:32.5 | 0.14 | 89% | 🟡 Moderate |\n"
        "| 15 | 1:33.4 | 0.18 | 78% | 🟡 Wearing |\n"
        "| 18 | 1:34.8 | 0.47 | 62% | 🟠 Near cliff |\n"
        "| 20 | 1:36.2 | 0.70 | 48% | 🔴 **CLIFF** |\n\n"
        f"⚠️ **Cliff detected at lap ~18–20** for {compound} compound.\n\n"
        f"**Recommended stint length**: 15–17 laps (pit before cliff)\n\n"
        "> 🔧 *Connect LLM + FastF1 for circuit-specific degradation data.*"
    )


def _compound_comparison():
    return (
        "### 📊 Compound Comparison\n\n---\n\n"
        "| Metric | 🔴 Soft | 🟡 Medium | ⚪ Hard |\n"
        "|--------|---------|-----------|--------|\n"
        "| Peak pace | 1:31.2 | 1:32.0 | 1:32.5 |\n"
        "| Pace delta (vs Med) | −0.8s | Baseline | +0.5s |\n"
        "| Cliff lap | ~18 | ~28 | ~40+ |\n"
        "| Optimal stint | 12–16 laps | 22–28 laps | 30–40 laps |\n"
        "| Deg rate | 0.15 s/lap | 0.08 s/lap | 0.04 s/lap |\n"
        "| Graining risk | Low | Medium | High (early) |\n"
        "| Best for | Sprint / Quali | Race primary | Race anchor |\n\n"
        "#### Degradation Curves (Pace vs Lap)\n\n"
        "```\n"
        "Pace │\n"
        "1:36 │                              ╱ Soft cliff\n"
        "1:35 │                          ╱\n"
        "1:34 │                      ╱        ╱ Medium cliff\n"
        "1:33 │        ╱ Med    ╱          ╱\n"
        "1:32 │  ╱ Soft   ╱ Hard     ╱\n"
        "1:31 │╱─────────────────────────────\n"
        "     └──────────────────────────────\n"
        "      0    5   10   15   20   25   30   35   40\n"
        "                    Stint Lap\n"
        "```\n\n"
        "> 🔧 *Connect LLM + FastF1 for real telemetry-based curves.*"
    )


def _cliff_detection(user_input):
    return (
        "### ⚠️ Cliff Detection Alert\n\n---\n\n"
        "#### Current Stint Analysis\n\n"
        "| Indicator | Value | Status |\n"
        "|-----------|-------|--------|\n"
        "| Tire age | 16 laps | 🟡 Warning zone |\n"
        "| Deg rate trend | 0.12 → 0.18 → **0.35** s/lap | 🟠 Accelerating |\n"
        "| Grip level | 68% | 🟠 Low |\n"
        "| Predicted cliff | **Lap 18–19** | 🔴 Imminent |\n"
        "| Sector 3 loss | +0.8s vs peak | 🔴 Critical |\n\n"
        "### 🚨 RECOMMENDATION: PIT WITHIN 2 LAPS\n\n"
        "The degradation rate is accelerating exponentially. "
        "Continuing past lap 19 risks:\n"
        "- **1.5+ s/lap** pace loss\n"
        "- Possible lock-ups and flat spots\n"
        "- Risk of losing positions to cars on fresh rubber\n\n"
        "> 🔧 *Connect LLM for real-time telemetry-driven cliff predictions.*"
    )


def _temp_impact():
    return (
        "### 🌡️ Track Temperature Impact\n\n---\n\n"
        "| Track Temp | Soft Stint | Medium Stint | Hard Stint | Notes |\n"
        "|-----------|-----------|-------------|-----------|-------|\n"
        "| 25°C | 18 laps | 30 laps | 42 laps | Cool — all compounds extended |\n"
        "| 35°C | 15 laps | 26 laps | 38 laps | Nominal — baseline |\n"
        "| 45°C | 11 laps | 20 laps | 32 laps | Hot — aggressive deg |\n"
        "| 55°C | 8 laps | 15 laps | 26 laps | Extreme — blistering risk |\n\n"
        "**Key insight**: Every +10°C track temp reduces stint length by ~15–20%.\n\n"
        "For **day/night races** (Abu Dhabi, Bahrain, Qatar):\n"
        "- FP1/FP2 (day): Higher deg, shorter stints\n"
        "- Race (twilight→night): Grip improves, stints extend by 3–5 laps\n"
        "- Q (night): Peak grip, soft compound is king\n\n"
        "> 🔧 *Connect LLM + Open-Meteo for real temperature correlations.*"
    )


def _fallback():
    return (
        "### 🏎️ Tire Analysis Lab\n\nI can help with:\n"
        "- **Degradation curves** — _\"Show soft tire deg at Monza\"_\n"
        "- **Compound comparison** — _\"Soft vs medium vs hard\"_\n"
        "- **Cliff detection** — _\"When will my mediums cliff?\"_\n"
        "- **Temperature impact** — _\"How does heat affect tires?\"_\n\n"
        "Tell me the **compound** and **circuit**!"
    )
