"""
Chat Interface — profile-aware message router.

Dispatches to the correct page handler based on the active Chainlit
chat profile. Falls back to the Command Hub for unrecognized profiles.
"""

import sys, os, logging, re

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logger = logging.getLogger(__name__)


async def handle_message(user_input: str, history: list, profile: str = "command_hub") -> str:
    """Route a user message to the correct page handler.

    Parameters
    ----------
    user_input : str
        The user's message.
    history : list
        Conversation history.
    profile : str
        The active chat profile name (from cl.user_session).
    """
    logger.info("Routing message for profile=%s", profile)

    if profile == "strategy":
        from app.pages.strategy import handle_message as handler
        return await handler(user_input, history)
    elif profile == "tire_analysis":
        from app.pages.tire_analysis import handle_message as handler
        return await handler(user_input, history)
    elif profile == "history":
        from app.pages.history import handle_message as handler
        return await handler(user_input, history)
    elif profile == "regulations":
        from app.pages.regulations import handle_message as handler
        return await handler(user_input, history)
    else:
        return await _handle_command_hub(user_input, history)


async def _handle_command_hub(user_input: str, history: list) -> str:
    """Command Hub — general-purpose handler with weather + cross-domain routing."""
    lower = user_input.lower()

    # Weather queries (live data)
    if any(k in lower for k in ("weather", "rain", "temperature", "wind", "forecast", "humidity")):
        return await _handle_weather(user_input)

    # Route to sub-domains if detected
    if any(k in lower for k in ("tire", "tyre", "compound", "degradation", "cliff")):
        from app.pages.tire_analysis import handle_message as handler
        return await handler(user_input, history)
    if any(k in lower for k in ("pit", "strategy", "undercut", "overcut", "stint")):
        from app.pages.strategy import handle_message as handler
        return await handler(user_input, history)
    if any(k in lower for k in ("result", "winner", "history", "pattern", "head")):
        from app.pages.history import handle_message as handler
        return await handler(user_input, history)
    if any(k in lower for k in ("rule", "regulation", "penalty", "parc", "flag")):
        from app.pages.regulations import handle_message as handler
        return await handler(user_input, history)

    # General fallback
    return (
        "I'm your **F1 Race Strategist AI** 🏁\n\n"
        "I can help with anything F1. Try:\n"
        "- 🌧️ _\"Weather at Monaco\"_ — live circuit conditions\n"
        "- 🏁 _\"Pit strategy for Silverstone\"_ → switch to **Strategy** profile\n"
        "- 🏎️ _\"Soft tire degradation\"_ → switch to **Tire Analysis** profile\n"
        "- 📊 _\"2024 Monaco results\"_ → switch to **History** profile\n"
        "- 📖 _\"VSC rules\"_ → switch to **Regulations** profile\n\n"
        "Or use the **profile selector** in the header to focus on a specific area."
    )


async def _handle_weather(user_input: str) -> str:
    """Handle weather queries with live Open-Meteo data."""
    try:
        from tools.weather_tool import get_current_weather, F1_CIRCUITS

        circuit_found = None
        lower = user_input.lower()
        for name in F1_CIRCUITS:
            if name.lower() in lower:
                circuit_found = name
                break

        if circuit_found:
            w = get_current_weather(circuit=circuit_found)
            # Weather code interpretation
            code = int(w.get("weather_code", 0))
            if code == 0:
                condition = "☀️ Clear sky"
            elif code <= 3:
                condition = "⛅ Partly cloudy"
            elif code <= 48:
                condition = "🌫️ Fog/haze"
            elif code <= 67:
                condition = "🌧️ Rain"
            elif code <= 77:
                condition = "🌨️ Snow"
            elif code <= 82:
                condition = "🌧️ Showers"
            elif code <= 99:
                condition = "⛈️ Thunderstorm"
            else:
                condition = "🌤️ Mixed"

            rain_risk = "🟢 Dry" if w["rain"] == 0 and w["precipitation"] == 0 else "🔴 WET"

            return (
                f"### 🌤️ Live Weather — **{circuit_found}**\n\n---\n\n"
                f"| Condition | {condition} |\n"
                f"|-----------|-----|\n"
                f"| 🌡️ Temperature | **{w['temperature_2m']:.1f}°C** |\n"
                f"| 💧 Humidity | {w['relative_humidity_2m']:.0f}% |\n"
                f"| 🌧️ Precipitation | {w['precipitation']:.1f} mm |\n"
                f"| 🌧️ Rain | {w['rain']:.1f} mm |\n"
                f"| 💨 Wind | {w['wind_speed_10m']:.1f} km/h @ {w['wind_direction_10m']:.0f}° |\n"
                f"| 💨 Gusts | {w['wind_gusts_10m']:.1f} km/h |\n"
                f"| ☁️ Cloud Cover | {w['cloud_cover']:.0f}% |\n"
                f"| 🏎️ Track Status | **{rain_risk}** |\n\n"
                f"#### Strategy Impact\n\n"
                f"{'⚠️ **WET CONDITIONS** — Consider intermediate or wet compound strategy.' if rain_risk == '🔴 WET' else '✅ **DRY CONDITIONS** — Standard dry compound strategy applies.'}\n\n"
                f"*Live data from Open-Meteo API*"
            )
        else:
            from tools.weather_tool import F1_CIRCUITS
            circuits = ", ".join(sorted(F1_CIRCUITS.keys())[:12]) + ", …"
            return (
                "I can fetch **live weather** for any F1 circuit!\n\n"
                f"Available circuits: _{circuits}_\n\n"
                "Try: _\"What's the weather at Silverstone?\"_"
            )
    except Exception as e:
        logger.error("Weather error: %s", e)
        return f"⚠️ Could not fetch weather data: `{e}`"
