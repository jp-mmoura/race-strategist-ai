"""
Weather Tool — Open-Meteo API wrapper for the Race Strategist AI.

Provides current conditions, hourly forecasts, and historical weather
data for any F1 circuit (or arbitrary coordinates).  All data is
returned as pandas DataFrames for easy integration with the rest of
the pipeline.

Open-Meteo is free and requires **no API key**.
"""

import os
import logging
from datetime import date, datetime

import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Client configuration — cached session with automatic retries
# ---------------------------------------------------------------------------
CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "weather_cache"
)
os.makedirs(CACHE_DIR, exist_ok=True)

_cache_session = requests_cache.CachedSession(
    os.path.join(CACHE_DIR, "openmeteo_cache"),
    expire_after=3600,  # 1-hour cache
)
_retry_session = retry(_cache_session, retries=5, backoff_factor=0.2)
openmeteo_client = openmeteo_requests.Client(session=_retry_session)

logger.info("Open-Meteo client initialised (cache at %s)", CACHE_DIR)

# ---------------------------------------------------------------------------
# F1 circuit coordinates  (lat, lon)
# ---------------------------------------------------------------------------
F1_CIRCUITS: dict[str, tuple[float, float]] = {
    "Bahrain":          (26.0325, 50.5106),
    "Jeddah":           (21.6319, 39.1044),
    "Melbourne":        (-37.8497, 144.9680),
    "Shanghai":         (31.3389, 121.2197),
    "Miami":            (25.9581, -80.2389),
    "Imola":            (44.3439, 11.7167),
    "Monaco":           (43.7347, 7.4206),
    "Montreal":         (45.5000, -73.5228),
    "Barcelona":        (41.5700, 2.2611),
    "Spielberg":        (47.2197, 14.7647),
    "Silverstone":      (52.0786, -1.0169),
    "Budapest":         (47.5789, 19.2486),
    "Spa":              (50.4372, 5.9714),
    "Zandvoort":        (52.3888, 4.5409),
    "Monza":            (45.6156, 9.2811),
    "Baku":             (40.3725, 49.8533),
    "Singapore":        (1.2914, 103.8640),
    "Austin":           (30.1328, -97.6411),
    "Mexico City":      (19.4042, -99.0907),
    "São Paulo":        (-23.7036, -46.6997),
    "Las Vegas":        (36.1162, -115.1745),
    "Lusail":           (25.4900, 51.4542),
    "Abu Dhabi":        (24.4672, 54.6031),
    "Suzuka":           (34.8431, 136.5406),
}

# ---------------------------------------------------------------------------
# Open-Meteo API base URLs
# ---------------------------------------------------------------------------
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"

# Default weather variables relevant to F1 strategy
_DEFAULT_HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "precipitation_probability",
    "rain",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "cloud_cover",
    "weather_code",
]

_DEFAULT_CURRENT_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "cloud_cover",
    "weather_code",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _resolve_coords(
    circuit: str | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
) -> tuple[float, float]:
    """Return (lat, lon) from either a circuit name or explicit coords."""
    if circuit:
        key = next(
            (k for k in F1_CIRCUITS if k.lower() == circuit.lower()), None
        )
        if key is None:
            raise ValueError(
                f"Unknown circuit '{circuit}'. "
                f"Available: {', '.join(sorted(F1_CIRCUITS))}"
            )
        return F1_CIRCUITS[key]
    if latitude is not None and longitude is not None:
        return (latitude, longitude)
    raise ValueError("Provide either 'circuit' or both 'latitude'/'longitude'.")


def _hourly_to_dataframe(hourly) -> pd.DataFrame:
    """Convert an Open-Meteo hourly response block to a DataFrame."""
    data = {
        "datetime": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        )
    }
    for i, var in enumerate(_DEFAULT_HOURLY_VARS):
        data[var] = hourly.Variables(i).ValuesAsNumpy()
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_current_weather(
    circuit: str | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
) -> dict:
    """Fetch **current** weather conditions for a circuit or coordinates.

    Parameters
    ----------
    circuit : str | None
        F1 circuit name (e.g. ``"Monaco"``).
    latitude, longitude : float | None
        Explicit coordinates (used when *circuit* is ``None``).

    Returns
    -------
    dict
        Keys include ``temperature_2m``, ``wind_speed_10m``,
        ``precipitation``, ``rain``, ``cloud_cover``, etc.
    """
    lat, lon = _resolve_coords(circuit, latitude, longitude)
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": _DEFAULT_CURRENT_VARS,
    }
    responses = openmeteo_client.weather_api(FORECAST_URL, params=params)
    current = responses[0].Current()

    result = {}
    for i, var in enumerate(_DEFAULT_CURRENT_VARS):
        result[var] = current.Variables(i).Value()

    label = circuit or f"({lat}, {lon})"
    logger.info("Current weather for %s: %s", label, result)
    return result


def get_hourly_forecast(
    circuit: str | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    forecast_days: int = 3,
) -> pd.DataFrame:
    """Fetch an **hourly forecast** for the coming days.

    Parameters
    ----------
    circuit : str | None
        F1 circuit name.
    latitude, longitude : float | None
        Explicit coordinates.
    forecast_days : int
        Number of days to forecast (1–16). Defaults to 3.

    Returns
    -------
    pd.DataFrame
        Hourly weather data with a ``datetime`` index.
    """
    lat, lon = _resolve_coords(circuit, latitude, longitude)
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": _DEFAULT_HOURLY_VARS,
        "forecast_days": forecast_days,
    }
    responses = openmeteo_client.weather_api(FORECAST_URL, params=params)
    df = _hourly_to_dataframe(responses[0].Hourly())
    logger.info(
        "Hourly forecast for %s: %d rows", circuit or f"({lat},{lon})", len(df)
    )
    return df


def get_historical_weather(
    start_date: str | date,
    end_date: str | date,
    circuit: str | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
) -> pd.DataFrame:
    """Fetch **historical** hourly weather data for a date range.

    Parameters
    ----------
    start_date, end_date : str | date
        Date range in ``"YYYY-MM-DD"`` format (or ``date`` objects).
    circuit : str | None
        F1 circuit name.
    latitude, longitude : float | None
        Explicit coordinates.

    Returns
    -------
    pd.DataFrame
        Hourly historical weather data.
    """
    lat, lon = _resolve_coords(circuit, latitude, longitude)
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": _DEFAULT_HOURLY_VARS,
        "start_date": str(start_date),
        "end_date": str(end_date),
    }
    responses = openmeteo_client.weather_api(HISTORICAL_URL, params=params)
    df = _hourly_to_dataframe(responses[0].Hourly())
    logger.info(
        "Historical weather for %s (%s → %s): %d rows",
        circuit or f"({lat},{lon})",
        start_date,
        end_date,
        len(df),
    )
    return df


def get_race_weekend_forecast(circuit: str, forecast_days: int = 3) -> dict[str, pd.DataFrame]:
    """Convenience wrapper: split a forecast into FP / Quali / Race windows.

    Returns a dict with keys ``"full"``, plus any recognised day labels
    once you integrate the event calendar.
    """
    df = get_hourly_forecast(circuit=circuit, forecast_days=forecast_days)
    return {"full": df}


def list_circuits() -> list[str]:
    """Return the names of all known F1 circuits."""
    return sorted(F1_CIRCUITS.keys())


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=== Current weather at Monaco ===")
    current = get_current_weather(circuit="Monaco")
    for k, v in current.items():
        print(f"  {k}: {v}")

    print("\n=== 1-day hourly forecast at Monaco ===")
    forecast = get_hourly_forecast(circuit="Monaco", forecast_days=1)
    print(forecast.head(10).to_string(index=False))
