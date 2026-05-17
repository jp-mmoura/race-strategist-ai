"""
History Page Handler — past race explorer, strategy comparison, pattern insights.
"""

import sys, os, logging

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logger = logging.getLogger(__name__)


def get_welcome_message() -> str:
    return (
        "## 📊 Race History Explorer\n\n"
        "Dive into **past seasons, race results, and strategic trends**. "
        "I use FastF1 data to surface what actually happened and why.\n\n---\n\n"
        "| Feature | Description |\n"
        "|---------|-------------|\n"
        "| **Race Results** | Full classification for any GP |\n"
        "| **Strategy Comparison** | What each team did — stops, compounds, positions |\n"
        "| **Pit Stop Heatmap** | Most common pit windows by circuit |\n"
        "| **Pattern Insights** | AI-powered trend analysis |\n"
        "| **Head-to-Head** | Driver comparison across seasons |\n\n"
        "---\n\nPick a **year** and **Grand Prix** to explore. 🚀"
    )


def get_starters():
    import chainlit as cl
    return [
        cl.Starter(label="🏆 2024 Results", message="Show me the full race results from the 2024 Monaco Grand Prix.", icon="/public/icon_laptimes.png"),
        cl.Starter(label="📋 Strategy Comparison", message="Compare the pit strategies of all teams at the 2024 British Grand Prix.", icon="/public/icon_laptimes.png"),
        cl.Starter(label="🔍 Pattern Insights", message="What strategic patterns emerge at Monaco across the last 5 years?", icon="/public/icon_laptimes.png"),
        cl.Starter(label="⚔️ Head-to-Head", message="Compare Verstappen vs Hamilton race results across the 2024 season.", icon="/public/icon_laptimes.png"),
    ]


async def handle_message(user_input: str, history: list) -> str:
    lower = user_input.lower()
    if "result" in lower or "classification" in lower or "winner" in lower:
        return _race_results(user_input)
    if "strateg" in lower and ("compar" in lower or "team" in lower):
        return _strategy_comparison(user_input)
    if "pattern" in lower or "trend" in lower or "insight" in lower:
        return _pattern_insights(user_input)
    if "head" in lower or "vs" in lower or "compar" in lower:
        return _head_to_head(user_input)
    if any(k in lower for k in ("pit", "stop", "heatmap")):
        return _pit_heatmap(user_input)
    return _fallback()


def _race_results(user_input):
    return (
        "### 🏆 Race Results — 2024 Monaco Grand Prix\n\n---\n\n"
        "| Pos | Driver | Team | Time/Gap | Stops | Fastest Lap |\n"
        "|-----|--------|------|----------|-------|-------------|\n"
        "| 1 | LEC | Ferrari | 1:49:43.1 | 1 | |\n"
        "| 2 | PIA | McLaren | +7.152 | 1 | |\n"
        "| 3 | SAI | Ferrari | +13.566 | 1 | ⏱️ 1:12.3 |\n"
        "| 4 | NOR | McLaren | +15.023 | 1 | |\n"
        "| 5 | RUS | Mercedes | +18.889 | 2 | |\n"
        "| 6 | VER | Red Bull | +22.120 | 2 | |\n"
        "| 7 | HAM | Mercedes | +25.432 | 2 | |\n"
        "| 8 | TSU | RB | +31.100 | 1 | |\n"
        "| 9 | ALO | Aston Martin | +35.671 | 2 | |\n"
        "| 10 | GAS | Alpine | +42.898 | 2 | |\n\n"
        "💡 **Key takeaway**: 1-stop strategies dominated — top 4 all ran a single stop.\n\n"
        "> 🔧 *Connect LLM + FastF1 for real historical data from any GP.*"
    )


def _strategy_comparison(user_input):
    return (
        "### 📋 Strategy Comparison — 2024 British GP\n\n---\n\n"
        "| Driver | Start | Stop 1 | Stop 2 | Stop 3 | Compound Seq | Finish |\n"
        "|--------|-------|--------|--------|--------|-------------|--------|\n"
        "| HAM | P2 | L16 (S→M) | L37 (M→H) | — | S-M-H | **P1** |\n"
        "| VER | P4 | L14 (S→M) | L33 (M→H) | — | S-M-H | P2 |\n"
        "| NOR | P1 | L18 (S→M) | L39 (M→S) | — | S-M-S | P3 |\n"
        "| LEC | P3 | L12 (S→M) | L30 (M→H) | L45 (H→S) | S-M-H-S | P5 |\n"
        "| SAI | P6 | L20 (M→H) | — | — | M-H | P4 |\n\n"
        "**Winner's edge**: HAM timed stops perfectly to avoid traffic, gaining 2 positions.\n\n"
        "**Notable**: SAI's 1-stop M→H gamble paid off (P6→P4). LEC's 3-stop was too aggressive.\n\n"
        "> 🔧 *Connect LLM for automated strategy extraction from any GP.*"
    )


def _pattern_insights(user_input):
    return (
        "### 🔍 Pattern Insights — Monaco (2019–2024)\n\n---\n\n"
        "| Year | Winning Strategy | Stops | Key Factor |\n"
        "|------|-----------------|-------|------------|\n"
        "| 2024 | M→H | 1 | Track position > pace |\n"
        "| 2023 | M→H | 1 | Rain at start, dry finish |\n"
        "| 2022 | S→M→H | 2 | Late SC benefited 2-stoppers |\n"
        "| 2021 | S→H | 1 | Conservative, no SC |\n"
        "| 2019 | M→H | 1 | Tire management crucial |\n\n"
        "#### AI Pattern Summary\n\n"
        "🧠 **\"Monaco historically rewards the 1-stopper.\"**\n\n"
        "- **80%** of winners used a 1-stop strategy\n"
        "- **Track position** is the dominant factor (overtaking near-impossible)\n"
        "- **Medium start** is the most common winning starting compound\n"
        "- **Safety cars** are the only variable that makes 2-stops viable\n"
        "- Average winning gap: **7.2 seconds** (tire management wins)\n\n"
        "> 🔧 *Connect LLM for deeper multi-year pattern analysis.*"
    )


def _head_to_head(user_input):
    return (
        "### ⚔️ Head-to-Head — Verstappen vs Hamilton (2024)\n\n---\n\n"
        "| Metric | VER | HAM |\n"
        "|--------|-----|-----|\n"
        "| Wins | 9 | 2 |\n"
        "| Podiums | 14 | 8 |\n"
        "| Poles | 8 | 1 |\n"
        "| Fastest Laps | 6 | 3 |\n"
        "| Points | 393 | 211 |\n"
        "| DNFs | 1 | 2 |\n"
        "| Avg Finish | 2.1 | 4.8 |\n\n"
        "#### Direct Encounters (Same Race)\n\n"
        "| GP | VER | HAM | Who Won |\n"
        "|-----|-----|-----|--------|\n"
        "| Bahrain | P1 | P7 | VER |\n"
        "| Jeddah | P2 | P8 | VER |\n"
        "| Silverstone | P2 | P1 | **HAM** |\n"
        "| Spa | P1 | P4 | VER |\n"
        "| Monza | P6 | P3 | HAM |\n\n"
        "> 🔧 *Connect LLM + FastF1 for real season comparisons.*"
    )


def _pit_heatmap(user_input):
    return (
        "### 🗺️ Pit Stop Heatmap — Most Common Windows\n\n---\n\n"
        "| Circuit | Stop 1 (mode) | Stop 2 (mode) | Dominant Strategy |\n"
        "|---------|--------------|--------------|------------------|\n"
        "| Monaco | L18–22 | — | 1-stop |\n"
        "| Silverstone | L14–18 | L34–38 | 2-stop |\n"
        "| Monza | L16–20 | L38–42 | 2-stop |\n"
        "| Spa | L12–16 | L30–34 | 2-stop |\n"
        "| Singapore | L20–24 | — | 1-stop |\n"
        "| Abu Dhabi | L14–18 | L32–36 | 2-stop |\n\n"
        "💡 **Street circuits** favor 1-stops. **High-speed circuits** favor 2-stops.\n\n"
        "> 🔧 *Connect LLM for circuit-specific heatmap generation.*"
    )


def _fallback():
    return (
        "### 📊 History Explorer\n\nI can help with:\n"
        "- **Race results** — _\"2024 Monaco results\"_\n"
        "- **Strategy comparison** — _\"Compare team strategies at Silverstone\"_\n"
        "- **Pattern insights** — _\"Monaco strategy trends\"_\n"
        "- **Head-to-head** — _\"VER vs HAM 2024\"_\n\n"
        "Tell me a **year** and **Grand Prix**!"
    )
