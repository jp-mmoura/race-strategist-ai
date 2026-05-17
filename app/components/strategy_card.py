"""
Strategy Card Component — renders rich strategy summary cards.
"""

import logging
logger = logging.getLogger(__name__)


def build_strategy_card(circuit: str) -> str:
    return (
        f"## 🏁 Strategy Card — **{circuit}**\n\n---\n\n"
        "### Recommended Strategy (2-stop)\n\n"
        "| Stint | Compound | Laps | Est. Pace | Fuel-Adj |\n"
        "|-------|----------|------|-----------|----------|\n"
        "| 1 | 🔴 Soft | 1–15 | 1:32.0 | 1:32.8 |\n"
        "| 2 | 🟡 Medium | 16–38 | 1:33.2 | 1:33.6 |\n"
        "| 3 | ⚪ Hard | 39–57 | 1:33.5 | 1:33.8 |\n\n"
        "### Key Metrics\n\n"
        "| Metric | Value |\n"
        "|--------|-------|\n"
        "| 🚨 SC Probability | ~35% |\n"
        "| ⏱️ Pit Loss | ~22s |\n"
        "| 🏎️ DRS Zones | Factor overtaking windows |\n"
        "| 🌧️ Rain Risk | Check weather panel |\n"
        "| 🎯 Optimal Window | L14–L18 (Stop 1) |\n\n"
        "### Alternative Strategies\n\n"
        "| Strategy | Stops | Compounds | Projected Time |\n"
        "|----------|-------|-----------|---------------|\n"
        "| Aggressive | 2 | S-M-S | 1:31:58 |\n"
        "| Baseline | 2 | S-M-H | 1:32:14 |\n"
        "| Conservative | 1 | M-H | 1:32:38 |\n"
        "| Wet contingency | 2–3 | S-Inter-M | Variable |\n\n"
        "> 🔧 *Connect LLM for data-driven strategy cards.*"
    )


def build_mini_strategy(circuit: str, stops: int = 2) -> str:
    """Build a compact strategy summary for embedding in dashboards."""
    if stops == 1:
        return (
            f"**{circuit}** — 1-Stop: 🟡 M (L1–28) → ⚪ H (L29–57) "
            "| Pit L28 | Est. 1:32:38"
        )
    return (
        f"**{circuit}** — 2-Stop: 🔴 S (L1–15) → 🟡 M (L16–38) → ⚪ H (L39–57) "
        "| Pits L15, L38 | Est. 1:32:14"
    )
