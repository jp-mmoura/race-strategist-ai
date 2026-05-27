"""
Weather Agent — assesses weather impact on race strategy.

Integrates the Open-Meteo API (via tools/weather_tool.py) and,
optionally, FastF1 historical session weather to produce actionable
weather intelligence for the LangGraph orchestrator.

Main functions
--------------
get_race_forecast(circuit, race_date)
    → hourly forecast filtered to the race-day window

assess_rain_risk(circuit, race_date)
    → probability and timing of precipitation

analyze_weather_impact(circuit, race_date, year)
    → full weather analysis dict ready for RaceStrategyState

compare_historical_weather(circuit, year)
    → compare forecast vs historical conditions for same GP
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from tools.weather_tool import (
    get_current_weather,
    get_hourly_forecast,
    get_historical_weather,
    get_race_weekend_forecast,
    list_circuits,
    F1_CIRCUITS,
)

# ---------------------------------------------------------------------------
# Environment & Logging
# ---------------------------------------------------------------------------
load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# WMO weather-code descriptions (used by Open-Meteo)
# ---------------------------------------------------------------------------
_WMO_CODES: dict[int, str] = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snowfall",
    73: "Moderate snowfall",
    75: "Heavy snowfall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}

# Weather codes that indicate wet conditions
_WET_CODES = {51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82, 95, 96, 99}

# Typical F1 race window (UTC hour offsets — approximate)
_RACE_WINDOW_HOURS = (12, 18)  # 12:00–18:00 covers most race starts


# ===================================================================
# Helper: resolve circuit name to the key used by weather_tool
# ===================================================================

_CIRCUIT_ALIASES: dict[str, str] = {
    "silverstone": "Silverstone",
    "britain": "Silverstone",
    "monza": "Monza",
    "italy": "Monza",
    "spa": "Spa",
    "spa-francorchamps": "Spa",
    "belgium": "Spa",
    "interlagos": "São Paulo",
    "sao paulo": "São Paulo",
    "são paulo": "São Paulo",
    "brazil": "São Paulo",
    "montreal": "Montreal",
    "canada": "Montreal",
    "gilles villeneuve": "Montreal",
    "marina bay": "Singapore",
    "singapore": "Singapore",
    "albert park": "Melbourne",
    "melbourne": "Melbourne",
    "australia": "Melbourne",
    "sakhir": "Bahrain",
    "bahrain": "Bahrain",
    "barcelona": "Barcelona",
    "spain": "Barcelona",
    "catalunya": "Barcelona",
    "monte carlo": "Monaco",
    "monte-carlo": "Monaco",
    "monaco": "Monaco",
    "red bull ring": "Spielberg",
    "spielberg": "Spielberg",
    "austria": "Spielberg",
    "hungaroring": "Budapest",
    "budapest": "Budapest",
    "hungary": "Budapest",
    "suzuka": "Suzuka",
    "japan": "Suzuka",
    "shanghai": "Shanghai",
    "china": "Shanghai",
    "cota": "Austin",
    "austin": "Austin",
    "usa": "Austin",
    "yas marina": "Abu Dhabi",
    "abu dhabi": "Abu Dhabi",
    "abudhabi": "Abu Dhabi",
    "mexico city": "Mexico City",
    "mexico": "Mexico City",
    "hermanos rodriguez": "Mexico City",
    "jeddah": "Jeddah",
    "saudi arabia": "Jeddah",
    "las vegas": "Las Vegas",
    "zandvoort": "Zandvoort",
    "netherlands": "Zandvoort",
    "imola": "Imola",
    "baku": "Baku",
    "azerbaijan": "Baku",
    "lusail": "Lusail",
    "losail": "Lusail",
    "qatar": "Lusail",
    "miami": "Miami",
}


def _resolve_circuit_name(circuit: str) -> str:
    """Resolve a user-provided circuit name to the F1_CIRCUITS key."""
    if circuit in F1_CIRCUITS:
        return circuit

    key = circuit.lower().strip()
    resolved = _CIRCUIT_ALIASES.get(key)
    if resolved and resolved in F1_CIRCUITS:
        return resolved

    # Partial substring match
    for name in F1_CIRCUITS:
        if key in name.lower():
            return name

    raise ValueError(
        f"Unknown circuit '{circuit}'. "
        f"Available: {', '.join(sorted(F1_CIRCUITS))}"
    )


# ===================================================================
# 1. get_race_forecast
# ===================================================================

def get_race_forecast(
    circuit: str,
    race_date: str | date | None = None,
    forecast_days: int = 3,
) -> pd.DataFrame:
    """Fetch hourly forecast and filter to the race-day window.

    Parameters
    ----------
    circuit : str
        Circuit name (e.g. ``"Monaco"``).
    race_date : str | date | None
        Race date (``"YYYY-MM-DD"``). If ``None``, returns the full
        forecast without day-filtering.
    forecast_days : int
        Number of forecast days (1–16).

    Returns
    -------
    pd.DataFrame
        Hourly weather data for the race window.
    """
    resolved = _resolve_circuit_name(circuit)
    df = get_hourly_forecast(circuit=resolved, forecast_days=forecast_days)

    if race_date is not None:
        if isinstance(race_date, str):
            race_date = date.fromisoformat(race_date)

        race_start = pd.Timestamp(
            datetime.combine(race_date, datetime.min.time()), tz="UTC",
        ) + timedelta(hours=_RACE_WINDOW_HOURS[0])
        race_end = pd.Timestamp(
            datetime.combine(race_date, datetime.min.time()), tz="UTC",
        ) + timedelta(hours=_RACE_WINDOW_HOURS[1])

        df = df[(df["datetime"] >= race_start) & (df["datetime"] <= race_end)]

    logger.info("Race forecast for %s: %d hourly rows", resolved, len(df))
    return df


# ===================================================================
# 2. assess_rain_risk
# ===================================================================

def assess_rain_risk(
    circuit: str,
    race_date: str | date | None = None,
) -> dict[str, Any]:
    """Assess precipitation risk for a race.

    Returns
    -------
    dict with keys:
        risk_level, max_precip_prob, total_rain_mm, wet_hours,
        summary, rain_windows
    """
    df = get_race_forecast(circuit, race_date)

    if df.empty:
        return {
            "risk_level": "Unknown",
            "max_precip_prob": None,
            "total_rain_mm": None,
            "wet_hours": 0,
            "summary": "No forecast data available for the requested window.",
            "rain_windows": [],
        }

    precip_prob = df.get("precipitation_probability", pd.Series(dtype=float))
    rain = df.get("rain", df.get("precipitation", pd.Series(dtype=float)))

    max_prob = float(precip_prob.max()) if not precip_prob.empty else 0.0
    total_rain = float(rain.sum()) if not rain.empty else 0.0
    wet_hours = int((rain > 0).sum()) if not rain.empty else 0

    # Identify contiguous wet windows
    rain_windows: list[dict[str, str]] = []
    if not rain.empty:
        is_wet = (rain > 0).values
        start = None
        for i, wet in enumerate(is_wet):
            if wet and start is None:
                start = df.iloc[i]["datetime"]
            elif not wet and start is not None:
                rain_windows.append({"start": str(start), "end": str(df.iloc[i]["datetime"])})
                start = None
        if start is not None:
            rain_windows.append({"start": str(start), "end": str(df.iloc[-1]["datetime"])})

    # Classify risk
    if max_prob >= 70 or total_rain >= 5.0:
        risk_level = "High"
    elif max_prob >= 40 or total_rain >= 1.0:
        risk_level = "Medium"
    elif max_prob >= 15 or total_rain > 0:
        risk_level = "Low"
    else:
        risk_level = "None"

    # Build summary
    wet_codes_present = set()
    if "weather_code" in df.columns:
        codes = df["weather_code"].dropna().astype(int).unique()
        wet_codes_present = set(codes) & _WET_CODES

    parts = [f"Rain risk: {risk_level}."]
    if total_rain > 0:
        parts.append(f"{total_rain:.1f} mm expected across {wet_hours} hour(s).")
    if wet_codes_present:
        descs = [_WMO_CODES.get(c, f"Code {c}") for c in sorted(wet_codes_present)]
        parts.append(f"Conditions: {', '.join(descs)}.")
    if risk_level == "High":
        parts.append("Intermediates/wets likely required.")
    elif risk_level == "None":
        parts.append("Dry conditions expected.")

    result = {
        "risk_level": risk_level,
        "max_precip_prob": round(max_prob, 1),
        "total_rain_mm": round(total_rain, 2),
        "wet_hours": wet_hours,
        "summary": " ".join(parts),
        "rain_windows": rain_windows,
    }
    logger.info("Rain risk for %s: %s", circuit, risk_level)
    return result


# ===================================================================
# 3. analyze_weather_impact  (unified entry point for the graph)
# ===================================================================

def analyze_weather_impact(
    circuit: str,
    race_date: str | date | None = None,
    year: int | None = None,
    session_type: str = "R",
) -> dict[str, Any]:
    """Full weather-impact analysis for a circuit and race date.

    Combines forecast, rain risk, current conditions, and (optionally)
    historical session weather into a single result dict for the
    LangGraph ``RaceStrategyState``.

    Parameters
    ----------
    circuit : str
        Circuit name (e.g. ``"Silverstone"``).
    race_date : str | date | None
        Race date in ``"YYYY-MM-DD"`` format.
    year : int | None
        If provided, historical FastF1 weather is compared.
    session_type : str
        FastF1 session type for historical comparison.

    Returns
    -------
    dict  — keys: circuit, race_date, current_conditions, forecast,
            rain_risk, temperature, wind, strategy_notes,
            historical_comparison, error
    """
    result: dict[str, Any] = {
        "circuit": circuit,
        "race_date": str(race_date) if race_date else "N/A",
        "current_conditions": None,
        "forecast": None,
        "rain_risk": None,
        "temperature": None,
        "wind": None,
        "strategy_notes": [],
        "historical_comparison": None,
        "error": None,
    }

    try:
        resolved = _resolve_circuit_name(circuit)
    except ValueError as exc:
        result["error"] = str(exc)
        return result

    # ── Detect whether this is a historical race ──────────────────
    # If year is in the past, prefer FastF1 actual session weather
    # over the Open-Meteo forecast (which reflects *current* weather).
    from datetime import datetime as _dt
    _is_historical = year is not None and year < _dt.now().year

    if _is_historical:
        # ── HISTORICAL PATH: use FastF1 recorded weather ──────────
        result = _build_from_historical_weather(
            result, resolved, year, session_type,
        )
    else:
        # ── LIVE / FUTURE PATH: use Open-Meteo forecast ──────────
        # 1. Current conditions
        try:
            current = get_current_weather(circuit=resolved)
            wcode = int(current.get("weather_code", 0))
            result["current_conditions"] = {
                "temperature_c": round(current.get("temperature_2m", 0), 1),
                "humidity_pct": round(current.get("relative_humidity_2m", 0), 1),
                "wind_speed_kmh": round(current.get("wind_speed_10m", 0), 1),
                "wind_dir_deg": round(current.get("wind_direction_10m", 0)),
                "wind_gusts_kmh": round(current.get("wind_gusts_10m", 0), 1),
                "precipitation_mm": round(current.get("precipitation", 0), 2),
                "cloud_cover_pct": round(current.get("cloud_cover", 0), 1),
                "weather_code": wcode,
                "weather_desc": _WMO_CODES.get(wcode, "Unknown"),
            }
        except Exception as exc:
            logger.warning("Could not fetch current weather: %s", exc)

        # 2. Race-window forecast
        try:
            forecast_df = get_race_forecast(resolved, race_date)
            if not forecast_df.empty:
                result["forecast"] = forecast_df.to_dict(orient="records")

                # Temperature analysis
                temp = forecast_df["temperature_2m"]
                result["temperature"] = {
                    "air_temp_min_c": round(float(temp.min()), 1),
                    "air_temp_max_c": round(float(temp.max()), 1),
                    "air_temp_avg_c": round(float(temp.mean()), 1),
                    "track_temp_est_c": round(float(temp.mean()) + 15, 1),
                    "note": _temperature_note(float(temp.mean())),
                }

                # Wind analysis
                ws = forecast_df["wind_speed_10m"]
                wg = forecast_df.get("wind_gusts_10m", pd.Series(dtype=float))
                result["wind"] = {
                    "avg_speed_kmh": round(float(ws.mean()), 1),
                    "max_speed_kmh": round(float(ws.max()), 1),
                    "max_gusts_kmh": round(float(wg.max()), 1) if not wg.empty else None,
                    "note": _wind_note(float(ws.max())),
                }
        except Exception as exc:
            logger.warning("Could not fetch forecast: %s", exc)

        # 3. Rain risk
        try:
            result["rain_risk"] = assess_rain_risk(resolved, race_date)
        except Exception as exc:
            logger.warning("Could not assess rain risk: %s", exc)

    # ── 4. Historical comparison (optional) ───────────────────────
    if year is not None and not _is_historical:
        result["historical_comparison"] = _compare_with_history(
            resolved, race_date, year, session_type,
        )

    # ── 5. Strategic notes ────────────────────────────────────────
    result["strategy_notes"] = _build_strategy_notes(result)

    logger.info("Weather analysis complete for %s (%s)", resolved, race_date or "N/A")
    return result


# ===================================================================
# 4. compare_historical_weather
# ===================================================================

def compare_historical_weather(
    circuit: str,
    year: int,
    race_date: str | date | None = None,
) -> dict[str, Any]:
    """Compare current forecast with historical weather for the same GP."""
    resolved = _resolve_circuit_name(circuit)
    return _compare_with_history(resolved, race_date, year)


# ===================================================================
# Internal helpers
# ===================================================================

def _build_from_historical_weather(
    result: dict[str, Any],
    circuit: str,
    year: int,
    session_type: str = "R",
) -> dict[str, Any]:
    """Populate weather analysis from FastF1 historical session data.

    For past races, this is more accurate than the Open-Meteo forecast
    (which reflects *current* weather, not race-day conditions).
    """
    try:
        from tools.fastf1_tool import get_session, get_weather

        session = get_session(year, circuit, session_type)
        weather_df = get_weather(session)

        if weather_df is None or weather_df.empty:
            result["error"] = f"No historical weather for {circuit} {year}."
            return result

        # ── Temperature ───────────────────────────────────────────
        air_temp = weather_df["AirTemp"]
        track_temp = weather_df["TrackTemp"]
        result["temperature"] = {
            "air_temp_min_c": round(float(air_temp.min()), 1),
            "air_temp_max_c": round(float(air_temp.max()), 1),
            "air_temp_avg_c": round(float(air_temp.mean()), 1),
            "track_temp_est_c": round(float(track_temp.mean()), 1),
            "note": _temperature_note(float(air_temp.mean())),
            "source": "fastf1_historical",
        }

        # ── Wind ──────────────────────────────────────────────────
        if "WindSpeed" in weather_df.columns:
            ws = weather_df["WindSpeed"]
            result["wind"] = {
                "avg_speed_kmh": round(float(ws.mean()), 1),
                "max_speed_kmh": round(float(ws.max()), 1),
                "max_gusts_kmh": None,
                "note": _wind_note(float(ws.max())),
            }

        # ── Rain risk (from actual recorded data) ─────────────────
        rainfall_col = weather_df.get("Rainfall", pd.Series([False]))
        had_rain = bool(rainfall_col.any())
        humidity = weather_df["Humidity"]

        if had_rain:
            risk_level = "High"
            summary = (
                f"Rain risk: High. Rainfall was recorded during "
                f"{circuit} {year}. Intermediates/wets were needed."
            )
        elif float(humidity.mean()) > 80:
            risk_level = "Medium"
            summary = (
                f"Rain risk: Medium. High humidity ({humidity.mean():.0f}%) "
                f"at {circuit} {year} — damp conditions possible."
            )
        else:
            risk_level = "None"
            summary = f"Rain risk: None. Dry conditions at {circuit} {year}."

        result["rain_risk"] = {
            "risk_level": risk_level,
            "max_precip_prob": 100.0 if had_rain else 0.0,
            "total_rain_mm": None,
            "wet_hours": 0,
            "summary": summary,
            "rain_windows": [],
            "source": "fastf1_historical",
        }

        logger.info(
            "Historical weather loaded for %s %d: rain=%s, air=%.1f°C",
            circuit, year, had_rain, float(air_temp.mean()),
        )

    except Exception as exc:
        logger.warning("Historical weather failed for %s %d: %s", circuit, year, exc)
        result["error"] = f"Could not load historical weather: {exc}"

    return result


def _temperature_note(avg_temp_c: float) -> str:
    if avg_temp_c > 35:
        return (
            "Very high temperatures — expect extreme track temps (>50 °C). "
            "Soft compound life significantly reduced; consider harder compounds."
        )
    if avg_temp_c > 28:
        return (
            "Warm conditions — elevated track temperatures likely. "
            "Monitor rear tyre degradation closely."
        )
    if avg_temp_c < 15:
        return (
            "Cool conditions — tyre warm-up may be challenging. "
            "Softer compounds could be favoured for grip."
        )
    return "Moderate temperatures — standard compound performance expected."


def _wind_note(max_wind_kmh: float) -> str:
    if max_wind_kmh > 40:
        return (
            "Strong winds — significant aero impact. "
            "High-downforce setups may be compromised on straights."
        )
    if max_wind_kmh > 25:
        return (
            "Moderate winds — minor aero effect. "
            "Watch for crosswinds in exposed sections."
        )
    return "Light winds — minimal impact on car performance."


def _compare_with_history(
    circuit: str,
    race_date: str | date | None,
    year: int,
    session_type: str = "R",
) -> dict[str, Any] | None:
    """Compare forecast vs FastF1 historical weather."""
    try:
        from tools.fastf1_tool import get_session, get_weather

        session = get_session(year, circuit, session_type)
        hist_weather = get_weather(session)

        if hist_weather is None or hist_weather.empty:
            return {"note": f"No historical weather data for {circuit} {year}."}

        hist_summary = {
            "year": year,
            "air_temp_avg_c": round(float(hist_weather["AirTemp"].mean()), 1),
            "track_temp_avg_c": round(float(hist_weather["TrackTemp"].mean()), 1),
            "humidity_avg_pct": round(float(hist_weather["Humidity"].mean()), 1),
            "rainfall": bool(hist_weather.get("Rainfall", pd.Series([False])).any()),
        }

        delta = {}
        try:
            forecast_df = get_race_forecast(circuit, race_date)
            if not forecast_df.empty:
                fc_temp = float(forecast_df["temperature_2m"].mean())
                diff = fc_temp - hist_summary["air_temp_avg_c"]
                delta = {
                    "air_temp_delta_c": round(diff, 1),
                    "note": (
                        "Forecast is warmer than historical — expect higher degradation."
                        if diff > 3
                        else (
                            "Forecast is cooler than historical — grip may take longer to build."
                            if diff < -3
                            else "Forecast is similar to historical conditions."
                        )
                    ),
                }
        except Exception:
            pass

        return {"historical": hist_summary, "delta": delta}
    except Exception as exc:
        logger.warning("Historical comparison failed: %s", exc)
        return {"note": f"Could not load historical data: {exc}"}


def _build_strategy_notes(analysis: dict[str, Any]) -> list[str]:
    """Derive strategic recommendations from the analysis."""
    notes: list[str] = []

    rain = analysis.get("rain_risk") or {}
    risk = rain.get("risk_level", "Unknown")
    if risk == "High":
        notes.append(
            "🌧️ HIGH RAIN RISK — prepare intermediate and full-wet compounds. "
            "Consider a flexible strategy with an early pit window."
        )
    elif risk == "Medium":
        notes.append(
            "🌦️ MEDIUM RAIN RISK — have intermediate tyres ready. "
            "Monitor radar closely; a well-timed switch could gain positions."
        )
    elif risk == "Low":
        notes.append(
            "☁️ LOW RAIN RISK — dry strategy primary, but keep inters available."
        )
    else:
        notes.append("☀️ DRY CONDITIONS — standard dry-weather strategy applies.")

    temp = analysis.get("temperature") or {}
    avg_t = temp.get("air_temp_avg_c")
    if avg_t is not None:
        est_track = temp.get("track_temp_est_c", avg_t + 15)
        if est_track > 50:
            notes.append(
                "🔥 EXTREME TRACK TEMP — rear degradation will be high. "
                "Favour harder compounds and consider a 2-stop strategy."
            )
        elif est_track > 40:
            notes.append(
                "🌡️ HIGH TRACK TEMP — monitor tyre blistering. "
                "Offset strategy (extending stint 1) may be beneficial."
            )
        elif est_track < 25:
            notes.append(
                "❄️ COOL TRACK — tyre warm-up critical. "
                "Softer compounds may offer better initial grip."
            )

    wind = analysis.get("wind") or {}
    max_gust = wind.get("max_gusts_kmh")
    if max_gust is not None and max_gust > 40:
        notes.append(
            "💨 STRONG GUSTS — aero balance affected. "
            "Higher downforce setup recommended; braking zones may shift."
        )

    return notes


# ===================================================================
# CLI validation
# ===================================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("=" * 65)
    print("  Weather Agent — Validation")
    print("=" * 65)

    # 1. List available circuits
    circuits = list_circuits()
    print(f"\n▶ Available circuits ({len(circuits)}):")
    print(f"  {', '.join(circuits)}")

    # 2. Current weather at Monaco
    print("\n▶ Current weather at Monaco:")
    current = get_current_weather(circuit="Monaco")
    for k, v in current.items():
        print(f"  {k}: {v}")

    # 3. Rain risk
    print("\n▶ Rain risk assessment (Monaco, next 3 days):")
    rain = assess_rain_risk("Monaco")
    for k, v in rain.items():
        print(f"  {k}: {v}")

    # 4. Full weather impact analysis
    print("\n▶ Full weather impact analysis (Monaco):")
    impact = analyze_weather_impact("Monaco")
    for note in impact.get("strategy_notes", []):
        print(f"  {note}")

    temp = impact.get("temperature")
    if temp:
        print(f"\n  Temperature: {temp['air_temp_min_c']}–"
              f"{temp['air_temp_max_c']} °C (avg {temp['air_temp_avg_c']} °C)")
        print(f"  Est. track temp: {temp['track_temp_est_c']} °C")
        print(f"  Note: {temp['note']}")

    wind = impact.get("wind")
    if wind:
        print(f"\n  Wind: avg {wind['avg_speed_kmh']} km/h, "
              f"max {wind['max_speed_kmh']} km/h")
        if wind.get("max_gusts_kmh"):
            print(f"  Gusts: up to {wind['max_gusts_kmh']} km/h")
        print(f"  Note: {wind['note']}")

    print("\n" + "=" * 65)
    print("  ✅ Weather Agent validation complete!")
    print("=" * 65)
