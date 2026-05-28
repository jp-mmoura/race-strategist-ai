"""
FastF1 Tool — wrapper around the FastF1 library for the Race Strategist AI.

Handles cache configuration (driven by .env) and exposes helper functions
to fetch session data, lap telemetry, tire/compound info, driver stints,
race results, weather, and event schedules.
"""

import os
import logging
from pathlib import Path

import fastf1
import pandas as pd
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment & Logging
# ---------------------------------------------------------------------------
load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cache configuration (reads FASTF1_CACHE_DIR from .env, falls back to
# <project_root>/data/fastf1_cache)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CACHE = str(_PROJECT_ROOT / "data" / "fastf1_cache")
CACHE_DIR = os.getenv("FASTF1_CACHE_DIR", _DEFAULT_CACHE)

os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)
logger.info("FastF1 cache enabled at %s", CACHE_DIR)


# ===================================================================
# Session helpers
# ===================================================================

# In-memory session cache to avoid redundant loads when multiple agents
# request the same session in a single pipeline run.
_session_cache: dict[tuple, "fastf1.core.Session"] = {}


def get_session(
    year: int,
    grand_prix: str | int,
    session_type: str = "R",
) -> fastf1.core.Session:
    """Load and return a FastF1 session.

    Results are cached in-memory so that repeated calls with the same
    ``(year, grand_prix, session_type)`` return instantly without
    re-loading data from disk/network.

    Parameters
    ----------
    year : int
        Season year (e.g. 2023).
    grand_prix : str | int
        Grand Prix name (e.g. ``"Monaco"``) or round number.
    session_type : str, optional
        Session identifier — ``"FP1"``, ``"FP2"``, ``"FP3"``,
        ``"Q"`` (Qualifying), ``"S"`` (Sprint), ``"R"`` (Race).
        Defaults to ``"R"``.

    Returns
    -------
    fastf1.core.Session
        The fully-loaded session object.
    """
    # Normalise the cache key so that "silverstone" and "Silverstone"
    # hit the same entry.
    gp_key = grand_prix.lower().strip() if isinstance(grand_prix, str) else grand_prix
    cache_key = (year, gp_key, session_type)

    if cache_key in _session_cache:
        logger.info(
            "Session cache HIT: %s %s – %s", year, grand_prix, session_type,
        )
        return _session_cache[cache_key]

    session = fastf1.get_session(year, grand_prix, session_type)
    session.load()
    _session_cache[cache_key] = session
    logger.info("Loaded session: %s %s – %s (cached)", year, grand_prix, session_type)
    return session


def clear_session_cache() -> None:
    """Clear the in-memory session cache (useful between test runs)."""
    _session_cache.clear()
    logger.info("Session cache cleared.")


# ===================================================================
# Lap data
# ===================================================================

def get_laps(session) -> pd.DataFrame:
    """Return all laps for a loaded session as a DataFrame."""
    return session.laps


def get_driver_laps(session, driver: str) -> pd.DataFrame:
    """Return all laps for a specific driver.

    Parameters
    ----------
    driver : str
        Three-letter abbreviation (e.g. ``"VER"``).
    """
    return session.laps.pick_drivers(driver)


def get_fastest_lap(session, driver: str | None = None):
    """Return the fastest lap (optionally filtered by driver).

    Parameters
    ----------
    driver : str | None
        Three-letter abbreviation. If *None* the overall fastest lap
        is returned.
    """
    laps = session.laps
    if driver:
        laps = laps.pick_drivers(driver)
    return laps.pick_fastest()


# ===================================================================
# Telemetry
# ===================================================================

def get_telemetry(lap) -> pd.DataFrame:
    """Return telemetry for a given lap.

    Includes Speed, Throttle, Brake, RPM, nGear, X, Y, Z, etc.
    """
    return lap.get_telemetry()


# ===================================================================
# Tire / compound data
# ===================================================================

def get_tire_data(session, driver: str | None = None) -> pd.DataFrame:
    """Return tire-related columns for every lap in a session.

    Columns returned: Driver, LapNumber, Stint, Compound, TyreLife,
    FreshTyre, LapTime (seconds).

    Parameters
    ----------
    driver : str | None
        If provided, filter to a single driver.
    """
    laps = session.laps
    if driver:
        laps = laps.pick_drivers(driver)

    cols = [
        "Driver", "LapNumber", "Stint", "Compound",
        "TyreLife", "FreshTyre", "LapTime",
    ]
    available = [c for c in cols if c in laps.columns]
    df = laps[available].copy()

    # Convert LapTime timedelta → seconds for easy analysis
    if "LapTime" in df.columns:
        df["LapTimeSec"] = df["LapTime"].dt.total_seconds()

    return df.reset_index(drop=True)


def get_stints(session, driver: str | None = None) -> pd.DataFrame:
    """Return stint-level summary (compound, start/end lap, tyre life).

    Parameters
    ----------
    driver : str | None
        If provided, filter to a single driver.
    """
    tire = get_tire_data(session, driver)
    if tire.empty:
        return tire

    group_cols = ["Driver", "Stint"]
    if "Compound" in tire.columns:
        group_cols.append("Compound")

    stints = (
        tire.groupby(group_cols, dropna=False)
        .agg(
            StartLap=("LapNumber", "min"),
            EndLap=("LapNumber", "max"),
            MaxTyreLife=("TyreLife", "max"),
            Laps=("LapNumber", "count"),
        )
        .reset_index()
    )
    return stints


# ===================================================================
# Results & schedule
# ===================================================================

def get_race_results(session) -> pd.DataFrame:
    """Return the official results for a session."""
    return session.results


def get_event_schedule(year: int) -> pd.DataFrame:
    """Return the full event schedule for a given season."""
    schedule = fastf1.get_event_schedule(year)
    logger.info("Fetched %d-season schedule (%d events)", year, len(schedule))
    return schedule


# ===================================================================
# Weather
# ===================================================================

def get_weather(session) -> pd.DataFrame:
    """Return weather data for a loaded session."""
    return session.weather_data


# ===================================================================
# Quick validation — run directly with: python tools/fastf1_tool.py
# ===================================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("=" * 60)
    print("  FastF1 Tool — Validation: Monaco 2023 Race")
    print("=" * 60)

    # 1. Load session
    print("\n▶ Loading Monaco 2023 Race session...")
    sess = get_session(2023, "Monaco", "R")
    print(f"  ✓ Session loaded: {sess.event['EventName']} {sess.event.year}")

    # 2. Race results
    results = get_race_results(sess)
    print(f"\n▶ Race Results (top 5):")
    top5 = results.head(5)[["Abbreviation", "Position", "TeamName", "Status"]]
    print(top5.to_string(index=False))

    # 3. Tire / compound data for winner
    winner = results.iloc[0]["Abbreviation"]
    print(f"\n▶ Tire data for {winner}:")
    tires = get_tire_data(sess, driver=winner)
    print(tires[["LapNumber", "Compound", "TyreLife", "LapTimeSec"]].head(10).to_string(index=False))

    # 4. Stints
    print(f"\n▶ Stints for {winner}:")
    stints = get_stints(sess, driver=winner)
    print(stints.to_string(index=False))

    # 5. Fastest lap + telemetry
    fastest = get_fastest_lap(sess, driver=winner)
    print(f"\n▶ Fastest lap ({winner}): {fastest['LapTime']}")
    telem = get_telemetry(fastest)
    print(f"  Telemetry points: {len(telem)}")
    print(f"  Columns: {list(telem.columns)}")
    print(f"  Speed range: {telem['Speed'].min():.0f} – {telem['Speed'].max():.0f} km/h")

    # 6. Weather
    weather = get_weather(sess)
    print(f"\n▶ Weather data points: {len(weather)}")
    if not weather.empty:
        print(weather[["AirTemp", "TrackTemp", "Humidity", "Rainfall"]].describe().to_string())

    print("\n" + "=" * 60)
    print("  ✅ All checks passed — FastF1 tool is operational!")
    print("=" * 60)
