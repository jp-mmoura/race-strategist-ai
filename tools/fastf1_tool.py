"""
FastF1 Tool — wrapper around the FastF1 library for the Race Strategist AI.

Handles cache configuration and exposes helper functions to fetch
session data, lap telemetry, driver standings, and more.
"""

import os
import logging

import fastf1

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cache configuration
# ---------------------------------------------------------------------------
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "f1_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

logger.info("FastF1 cache enabled at %s", CACHE_DIR)


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------
def get_session(year: int, grand_prix: str | int, session_type: str = "R"):
    """Load and return a FastF1 session.

    Parameters
    ----------
    year : int
        Season year (e.g. 2024).
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
    session = fastf1.get_session(year, grand_prix, session_type)
    session.load()
    logger.info("Loaded session: %s %s – %s", year, grand_prix, session_type)
    return session


def get_laps(session):
    """Return all laps for a loaded session as a DataFrame."""
    return session.laps


def get_fastest_lap(session, driver: str | None = None):
    """Return the fastest lap in a session (optionally filtered by driver).

    Parameters
    ----------
    session : fastf1.core.Session
        A loaded session.
    driver : str | None
        Three-letter driver abbreviation (e.g. ``"VER"``).
        If *None*, the overall fastest lap is returned.

    Returns
    -------
    fastf1.core.Lap
    """
    laps = session.laps
    if driver:
        laps = laps.pick_driver(driver)
    return laps.pick_fastest()


def get_telemetry(lap):
    """Return telemetry data for a given lap.

    Includes speed, throttle, brake, RPM, gear, and positional data.
    """
    return lap.get_telemetry()


def get_driver_laps(session, driver: str):
    """Return all laps for a specific driver in a session.

    Parameters
    ----------
    session : fastf1.core.Session
        A loaded session.
    driver : str
        Three-letter driver abbreviation (e.g. ``"HAM"``).

    Returns
    -------
    fastf1.core.Laps
    """
    return session.laps.pick_driver(driver)


def get_race_results(session):
    """Return the official results for a session."""
    return session.results


def get_event_schedule(year: int):
    """Return the full event schedule for a given season.

    Parameters
    ----------
    year : int
        Season year.

    Returns
    -------
    fastf1.events.EventSchedule
    """
    schedule = fastf1.get_event_schedule(year)
    logger.info("Fetched %d-season schedule (%d events)", year, len(schedule))
    return schedule


def get_weather(session):
    """Return weather data for a loaded session."""
    return session.weather_data


# ---------------------------------------------------------------------------
# Quick sanity-check when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    schedule = get_event_schedule(2024)
    print(schedule[["EventName", "EventDate"]].to_string())
