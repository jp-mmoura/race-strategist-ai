"""
F1 Race Strategist AI — Chainlit Application Entry Point.

Run with:  chainlit run app/main.py -w
"""

import sys, os, logging

_PROJECT_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
# Fallback: also add CWD (Chainlit runs from project root)
_CWD = os.getcwd()
if _CWD not in sys.path:
    sys.path.insert(0, _CWD)

import chainlit as cl
from app.components.chat_interface import handle_message
from app.pages.strategy import (
    get_welcome_message as strategy_welcome,
    get_starters as strategy_starters,
)
from app.pages.tire_analysis import (
    get_welcome_message as tire_welcome,
    get_starters as tire_starters,
)
from app.pages.history import (
    get_welcome_message as history_welcome,
    get_starters as history_starters,
)
from app.pages.regulations import (
    get_welcome_message as regulations_welcome,
    get_starters as regulations_starters,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Chat Profiles — act as "pages" in the navigation
# ---------------------------------------------------------------------------
@cl.set_chat_profiles
async def chat_profiles():
    return [
        cl.ChatProfile(
            name="command_hub",
            markdown_description="**Mission Control** — Live session ticker, weather, AI briefing. The global entry point.",
            icon="/public/icon_strategy.png",
            starters=[
                cl.Starter(
                    label="🌤️ Circuit Weather",
                    message="What is the current weather at Monaco?",
                    icon="/public/icon_weather.png",
                ),
                cl.Starter(
                    label="🏁 Quick Strategy",
                    message="Suggest a pit strategy for the Silverstone Grand Prix.",
                    icon="/public/icon_strategy.png",
                ),
                cl.Starter(
                    label="📊 Latest Results",
                    message="Show me the results from the 2024 Italian Grand Prix.",
                    icon="/public/icon_laptimes.png",
                ),
                cl.Starter(
                    label="🏎️ Tire Check",
                    message="Compare soft vs medium tire degradation at Spa.",
                    icon="/public/icon_tire.png",
                ),
            ],
        ),
        cl.ChatProfile(
            name="strategy",
            markdown_description="**Race Strategy** — Pit simulation, undercut analysis, what-if scenarios.",
            icon="/public/icon_strategy.png",
            starters=strategy_starters(),
        ),
        cl.ChatProfile(
            name="tire_analysis",
            markdown_description="**Tire Analysis** — Degradation modeling, cliff detection, compound comparison.",
            icon="/public/icon_tire.png",
            starters=tire_starters(),
        ),
        cl.ChatProfile(
            name="history",
            markdown_description="**Race History** — Past results, strategy comparisons, pattern insights.",
            icon="/public/icon_laptimes.png",
            starters=history_starters(),
        ),
        cl.ChatProfile(
            name="regulations",
            markdown_description="**Regulations** — FIA rulebook Q&A, scenario checker, changelog.",
            icon="/public/icon_weather.png",
            starters=regulations_starters(),
        ),
    ]


# ---------------------------------------------------------------------------
# Chat Lifecycle — on_chat_start
# ---------------------------------------------------------------------------
@cl.on_chat_start
async def on_chat_start():
    profile = cl.user_session.get("chat_profile") or "command_hub"
    cl.user_session.set("history", [])
    logger.info("Chat started with profile: %s", profile)

    if profile == "command_hub":
        welcome = await _build_command_hub_welcome()
    elif profile == "strategy":
        welcome = strategy_welcome()
    elif profile == "tire_analysis":
        welcome = tire_welcome()
    elif profile == "history":
        welcome = history_welcome()
    elif profile == "regulations":
        welcome = regulations_welcome()
    else:
        welcome = "## 🏁 F1 Race Strategist AI\n\nSelect a profile to get started."

    await cl.Message(content=welcome).send()


async def _build_command_hub_welcome() -> str:
    """Build the Command Hub dashboard with live weather if possible."""
    # Try to fetch weather for a sample circuit
    weather_block = ""
    try:
        import importlib, types
        # Force find tools from the project root
        import sys as _sys
        _pr = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if _pr not in _sys.path:
            _sys.path.insert(0, _pr)
        from tools.weather_tool import get_current_weather
        w = get_current_weather(circuit="Monaco")
        rain_status = "🟢 DRY" if w["rain"] == 0 else "🔴 WET"
        weather_block = (
            "\n\n### 🌤️ Live Conditions — Monaco\n\n"
            f"| 🌡️ {w['temperature_2m']:.1f}°C | "
            f"💨 {w['wind_speed_10m']:.1f} km/h | "
            f"☁️ {w['cloud_cover']:.0f}% | "
            f"🏎️ {rain_status} |\n"
            "|---|---|---|---|"
        )
    except Exception as e:
        logger.warning("Could not fetch weather for hub: %s", e)

    return (
        "## 🏁 Command Hub — Mission Control\n\n"
        "Welcome to the **F1 Race Strategist AI**. This is your central "
        "briefing room — live conditions, AI insights, and quick access "
        "to every analysis tool.\n\n"
        "---"
        f"{weather_block}\n\n"
        "---\n\n"
        "### 📡 Quick Navigation\n\n"
        "| Profile | Focus Area | Switch via header dropdown |\n"
        "|---------|-----------|---------------------------|\n"
        "| 🏁 **Strategy** | Pit simulation, undercut, what-if | Select \"strategy\" |\n"
        "| 🏎️ **Tire Analysis** | Degradation, cliff detection | Select \"tire_analysis\" |\n"
        "| 📊 **History** | Past results, patterns, comparisons | Select \"history\" |\n"
        "| 📖 **Regulations** | FIA rules, scenario checker | Select \"regulations\" |\n\n"
        "---\n\n"
        "### 🤖 AI Briefing\n\n"
        "I can answer questions across **all domains** from this hub. "
        "For deep-dive analysis, switch to a specialized profile.\n\n"
        "Type a question or pick a starter below. 🚀"
    )


# ---------------------------------------------------------------------------
# Chat Lifecycle — on_message
# ---------------------------------------------------------------------------
@cl.on_message
async def on_message(message: cl.Message):
    user_input = message.content
    history = cl.user_session.get("history", [])
    profile = cl.user_session.get("chat_profile") or "command_hub"

    async with cl.Step(name="Analyzing", type="tool") as step:
        step.input = user_input
        response = await handle_message(user_input, history, profile)
        step.output = "Done"

    await cl.Message(content=response).send()

    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": response})
    cl.user_session.set("history", history)
