"""
Tire Agent — analyses tire wear, degradation, and pit-stop strategy.

This agent combines:
  • Static track characteristics (track_features.csv, track_history.csv)
  • Live FastF1 session data (lap times, compounds, stint lengths)

to produce actionable strategy recommendations (compound selection,
pit windows, and degradation estimates) that the orchestrator can
feed into the Strategy agent.

Main functions
--------------
classify_track_tire_wear(circuit)
    → "High Tire Wear" / "Medium Tire Wear" / "Low Tire Wear"

calculate_tire_degradation(session, driver)
    → per-stint degradation rate (sec/lap) and tyre-life curves

estimate_pit_window(session, driver)
    → optimal pit-stop lap range based on degradation crossover

analyze_tire_strategy(circuit, year, session_type)
    → full analysis dict ready for the graph state
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from tools.fastf1_tool import (
    get_session,
    get_stints,
    get_tire_data,
    get_race_results,
    get_weather,
)

# ---------------------------------------------------------------------------
# Environment & Logging
# ---------------------------------------------------------------------------
load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_RAW_DIR = _PROJECT_ROOT / "data" / "raw"

_TRACK_FEATURES_PATH = _RAW_DIR / "track_features.csv"
_TRACK_HISTORY_PATH = _RAW_DIR / "track_history.csv"

# Circuit-name aliases used by _find_track() for fuzzy matching.
# Maps user-supplied names (lowercase) to the canonical track key used
# in track_features.csv.  Defined at module level so the dict is built
# once on import rather than on every call to _find_track().
_ALIASES: dict[str, str] = {
    "silverstone": "britain",
    "monza": "italy",
    "spa": "belgium",
    "spa-francorchamps": "belgium",
    "interlagos": "brazil",
    "sao paulo": "brazil",
    "são paulo": "brazil",
    "montreal": "canada",
    "gilles villeneuve": "canada",
    "marina bay": "singapore",
    "albert park": "australia",
    "melbourne": "australia",
    "sakhir": "bahrain",
    "barcelona": "spain",
    "catalunya": "spain",
    "monte carlo": "monaco",
    "monte-carlo": "monaco",
    "red bull ring": "austria",
    "spielberg": "austria",
    "hungaroring": "hungary",
    "budapest": "hungary",
    "suzuka": "japan",
    "shanghai": "china",
    "sochi": "russia",
    "cota": "usa",
    "austin": "usa",
    "yas marina": "abudhabi",
    "abu dhabi": "abudhabi",
    "mexico city": "mexico",
    "hermanos rodriguez": "mexico",
    "autodromo hermanos": "mexico",
    "jeddah": "saudi arabia",
    "las vegas": "usa",
    "zandvoort": "netherlands",
    "imola": "italy",
    "baku": "azerbaijan",
    "lusail": "qatar",
    "losail": "qatar",
}


# ===================================================================
# CSV loaders (cached at module level)
# ===================================================================

_track_features: pd.DataFrame | None = None
_track_history: pd.DataFrame | None = None


def _load_track_features() -> pd.DataFrame:
    """Load and cache track_features.csv."""
    global _track_features
    if _track_features is None:
        _track_features = pd.read_csv(_TRACK_FEATURES_PATH)
        # Normalise track names to lowercase for fuzzy matching
        _track_features["TRACK_LOWER"] = (
            _track_features["TRACK"].str.lower().str.strip()
        )
        logger.info("Loaded %d tracks from track_features.csv", len(_track_features))
    return _track_features


def _load_track_history() -> pd.DataFrame:
    """Load and cache track_history.csv."""
    global _track_history
    if _track_history is None:
        _track_history = pd.read_csv(_TRACK_HISTORY_PATH)
        _track_history["TRACK_LOWER"] = (
            _track_history["TRACK"].str.lower().str.strip()
        )
        logger.info("Loaded %d rows from track_history.csv", len(_track_history))
    return _track_history


def _find_track(circuit: str) -> pd.Series | None:
    """Fuzzy-match a circuit name against track_features.csv."""
    df = _load_track_features()
    key = circuit.lower().strip()

    # Exact match first
    match = df[df["TRACK_LOWER"] == key]
    if not match.empty:
        return match.iloc[0]

    # Substring match ("Silverstone" → "Britain", "Monza" → "Italy", etc.)
    resolved = _ALIASES.get(key, key)
    match = df[df["TRACK_LOWER"] == resolved]
    if not match.empty:
        return match.iloc[0]

    # Partial substring
    match = df[df["TRACK_LOWER"].str.contains(key, na=False)]
    if not match.empty:
        return match.iloc[0]

    return None


# ===================================================================
# 1. classify_track_tire_wear
# ===================================================================

def classify_track_tire_wear(circuit: str) -> dict[str, Any]:
    """Classify expected tire wear level for a circuit.

    Uses a weighted score from track_features.csv columns:
      TIRE_STRESS (40%), LATERAL (25%), ASPHALT_ABR (20%), ASPHALT_GRP (15%)

    Returns
    -------
    dict with keys:
        classification : str   — "High Tire Wear" / "Medium …" / "Low …"
        score          : float — 0–5 composite score
        factors        : dict  — individual factor values
        track          : str   — matched track name
    """
    track = _find_track(circuit)
    if track is None:
        return {
            "classification": "Unknown",
            "score": None,
            "factors": {},
            "track": circuit,
            "error": f"Circuit '{circuit}' not found in track_features.csv",
        }

    # Weighted composite score (0–5 scale)
    weights = {
        "TIRE_STRESS": 0.40,
        "LATERAL": 0.25,
        "ASPHALT_ABR": 0.20,
        "ASPHALT_GRP": 0.15,
    }
    score = sum(float(track.get(k, 0)) * w for k, w in weights.items())

    if score >= 3.5:
        classification = "High Tire Wear"
    elif score >= 2.0:
        classification = "Medium Tire Wear"
    else:
        classification = "Low Tire Wear"

    factors = {k: float(track.get(k, 0)) for k in weights}
    factors["DOWNFORCE"] = float(track.get("DOWNFORCE", 0))
    factors["LENGTH_km"] = float(track.get("LENGTH", 0))
    factors["LAPS"] = int(track.get("LAPS", 0))

    logger.info(
        "Tire wear classification for %s: %s (score=%.2f)",
        track["TRACK"], classification, score,
    )

    return {
        "classification": classification,
        "score": round(score, 2),
        "factors": factors,
        "track": track["TRACK"],
    }


# ===================================================================
# 2. calculate_tire_degradation
# ===================================================================

def calculate_tire_degradation(
    session,
    driver: str | None = None,
) -> list[dict[str, Any]]:
    """Estimate per-stint tire degradation from lap-time trends.

    For each stint, fits a simple linear regression on lap time vs
    tyre life to compute a degradation rate (sec/lap).  Outlier laps
    (pit in/out, safety cars — >110% of median) are excluded.

    Parameters
    ----------
    session : fastf1.core.Session
        A loaded session.
    driver : str | None
        If provided, analyse only this driver; otherwise the race
        winner is used.

    Returns
    -------
    list[dict]
        One entry per stint with keys: driver, stint, compound,
        deg_rate_sec_per_lap, avg_lap_sec, lap_count, start_lap, end_lap.
    """
    if driver is None:
        results = get_race_results(session)
        driver = results.iloc[0]["Abbreviation"]

    tire = get_tire_data(session, driver=driver)
    if tire.empty:
        return []

    stints_info = []
    for stint_num in sorted(tire["Stint"].dropna().unique()):
        stint_df = tire[tire["Stint"] == stint_num].copy()
        if "LapTimeSec" not in stint_df.columns or stint_df["LapTimeSec"].isna().all():
            continue

        # Filter outliers (pit laps, safety cars)
        valid = stint_df["LapTimeSec"].dropna()
        if len(valid) < 3:
            continue
        median_t = valid.median()
        cutoff = median_t * 1.10
        clean = stint_df[stint_df["LapTimeSec"] <= cutoff].copy()
        if len(clean) < 3:
            continue

        # Linear regression: LapTimeSec = a * TyreLife + b
        x = clean["TyreLife"].values.astype(float)
        y = clean["LapTimeSec"].values.astype(float)
        finite_mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[finite_mask], y[finite_mask]
        if len(x) < 3:
            continue
        if np.std(x) == 0:
            deg_rate = 0.0
        else:
            try:
                slope, _ = np.polyfit(x, y, 1)
                deg_rate = round(float(slope), 4)
            except np.linalg.LinAlgError:
                deg_rate = 0.0

        compound = (
            clean["Compound"].iloc[0]
            if "Compound" in clean.columns
            else "UNKNOWN"
        )

        stints_info.append({
            "driver": driver,
            "stint": int(stint_num),
            "compound": compound,
            "deg_rate_sec_per_lap": deg_rate,
            "avg_lap_sec": round(float(clean["LapTimeSec"].mean()), 3),
            "lap_count": len(clean),
            "start_lap": int(clean["LapNumber"].min()),
            "end_lap": int(clean["LapNumber"].max()),
        })

    logger.info(
        "Degradation analysis for %s: %d stints computed",
        driver, len(stints_info),
    )
    return stints_info


# ===================================================================
# 3. estimate_pit_window
# ===================================================================

def estimate_pit_window(
    session,
    driver: str | None = None,
    pit_loss_sec: float = 22.0,
) -> dict[str, Any]:
    """Estimate the optimal pit-stop window based on degradation data.

    The window is derived from:
      1. Degradation rates per stint (from calculate_tire_degradation).
      2. Historical stint lengths of the winner.
      3. A crossover model: when cumulative deg loss > pit-stop time loss.

    Parameters
    ----------
    session : fastf1.core.Session
    driver : str | None
    pit_loss_sec : float
        Estimated time lost during a pit stop (default 22 s).

    Returns
    -------
    dict with keys:
        driver, recommended_pit_laps, pit_windows (list of ranges),
        total_laps, strategy_type ("1-stop" / "2-stop" / "3-stop"),
        rationale (str).
    """
    if driver is None:
        results = get_race_results(session)
        driver = results.iloc[0]["Abbreviation"]

    deg_data = calculate_tire_degradation(session, driver)
    stints_df = get_stints(session, driver)
    total_laps = int(session.laps["LapNumber"].max())

    if not deg_data or stints_df.empty:
        return {
            "driver": driver,
            "recommended_pit_laps": [],
            "pit_windows": [],
            "total_laps": total_laps,
            "strategy_type": "Unknown",
            "rationale": "Insufficient data to compute pit window.",
        }

    # Number of pit stops = (number of stints - 1)
    num_stops = len(deg_data) - 1
    if num_stops == 1:
        strategy_type = "1-stop"
    elif num_stops == 2:
        strategy_type = "2-stop"
    elif num_stops >= 3:
        strategy_type = f"{num_stops}-stop"
    else:
        strategy_type = "0-stop"

    # For each stint, estimate when cumulative degradation exceeds pit loss
    pit_laps: list[int] = []
    pit_windows: list[dict[str, int]] = []

    for stint in deg_data:
        deg_rate = stint["deg_rate_sec_per_lap"]
        if deg_rate <= 0.01:
            # Negligible degradation — stint is tyre-limited by strategy, not wear
            continue

        # Laps until cumulative deg = pit_loss_sec
        # Cumulative deg = deg_rate * n * (n+1) / 2  (quadratic approx)
        # Simplified: crossover_lap ≈ pit_loss / deg_rate
        crossover = pit_loss_sec / deg_rate
        optimal_lap = stint["start_lap"] + int(crossover)
        # Clamp to stint boundaries
        optimal_lap = max(stint["start_lap"] + 5, min(optimal_lap, stint["end_lap"]))

        pit_laps.append(optimal_lap)
        pit_windows.append({
            "earliest": max(stint["start_lap"] + 3, optimal_lap - 5),
            "optimal": optimal_lap,
            "latest": min(optimal_lap + 5, stint["end_lap"]),
            "compound": stint["compound"],
            "deg_rate": stint["deg_rate_sec_per_lap"],
        })

    # Build rationale
    rationale_parts = [
        f"Based on {driver}'s {strategy_type} strategy across {total_laps} laps.",
    ]
    for i, win in enumerate(pit_windows, 1):
        rationale_parts.append(
            f"  Pit {i}: lap {win['earliest']}–{win['latest']} "
            f"(optimal ~{win['optimal']}), "
            f"on {win['compound']} with {win['deg_rate']:.3f} s/lap deg."
        )

    result = {
        "driver": driver,
        "recommended_pit_laps": pit_laps,
        "pit_windows": pit_windows,
        "total_laps": total_laps,
        "strategy_type": strategy_type,
        "rationale": "\n".join(rationale_parts),
    }

    logger.info(
        "Pit window estimate for %s: %s (%d windows)",
        driver, strategy_type, len(pit_windows),
    )
    return result


# ===================================================================
# 4. analyze_tire_strategy  (unified entry point for the graph)
# ===================================================================

def analyze_tire_strategy(
    circuit: str,
    year: int,
    session_type: str = "R",
    driver: str | None = None,
) -> dict[str, Any]:
    """Full tire-strategy analysis for a circuit + year.

    Combines all sub-functions into a single result dict that the
    LangGraph orchestrator can write into ``RaceStrategyState``.

    Parameters
    ----------
    circuit : str
        Circuit name (e.g. ``"Silverstone"``).
    year : int
        Season year.
    session_type : str
        FastF1 session type (default ``"R"``).
    driver : str | None
        Driver to focus on (default: race winner).

    Returns
    -------
    dict with keys:
        track_wear        – output of classify_track_tire_wear
        degradation       – output of calculate_tire_degradation
        pit_window        – output of estimate_pit_window
        stints_summary    – per-driver stint breakdown
        compound_rec      – recommended compound order
        weather_impact    – brief weather summary
        error             – None or error message
    """
    result: dict[str, Any] = {
        "track_wear": None,
        "degradation": None,
        "pit_window": None,
        "stints_summary": None,
        "compound_rec": None,
        "weather_impact": None,
        "error": None,
    }

    # ── 1. Track classification (CSV-based, no FastF1 needed) ─────
    result["track_wear"] = classify_track_tire_wear(circuit)

    # ── 2. Load FastF1 session ────────────────────────────────────
    try:
        session = get_session(year, circuit, session_type)
    except Exception as exc:
        result["error"] = f"Failed to load {circuit} {year}: {exc}"
        return result

    # Resolve driver to winner if not specified
    if driver is None:
        results_df = get_race_results(session)
        driver = results_df.iloc[0]["Abbreviation"]

    # ── 3. Degradation analysis ───────────────────────────────────
    result["degradation"] = calculate_tire_degradation(session, driver)

    # ── 4. Pit window estimation ──────────────────────────────────
    result["pit_window"] = estimate_pit_window(session, driver)

    # ── 5. Stint summary (top-5 drivers) ──────────────────────────
    results_df = get_race_results(session)
    top5 = list(results_df.head(5)["Abbreviation"])
    all_stints = get_stints(session)
    top5_stints = all_stints[all_stints["Driver"].isin(top5)]
    result["stints_summary"] = top5_stints.to_dict(orient="records")

    # ── 6. Compound recommendation ────────────────────────────────
    # Based on what the top-3 finishers actually used
    top3 = list(results_df.head(3)["Abbreviation"])
    top3_stints = all_stints[all_stints["Driver"].isin(top3)]
    if "Compound" in top3_stints.columns:
        # Most common compound order (by stint number)
        compound_orders = (
            top3_stints.sort_values(["Driver", "Stint"])
            .groupby("Driver")["Compound"]
            .apply(list)
            .tolist()
        )
        # Find the most common strategy
        from collections import Counter
        strategy_counter = Counter(
            tuple(order) for order in compound_orders
        )
        most_common = strategy_counter.most_common(1)
        if most_common:
            best_order = list(most_common[0][0])
            freq = most_common[0][1]
        else:
            best_order = []
            freq = 0

        result["compound_rec"] = {
            "recommended_order": best_order,
            "confidence": f"{freq}/{len(compound_orders)} top-3 used this",
            "all_top3_strategies": [
                {"driver": d, "compounds": c}
                for d, c in zip(top3, compound_orders)
            ],
        }

    # ── 7. Weather impact ─────────────────────────────────────────
    weather_df = get_weather(session)
    if weather_df is not None and not weather_df.empty:
        air_mean = weather_df["AirTemp"].mean()
        track_mean = weather_df["TrackTemp"].mean()
        rainfall = weather_df.get("Rainfall", pd.Series([False])).any()
        result["weather_impact"] = {
            "air_temp_avg": round(air_mean, 1),
            "track_temp_avg": round(track_mean, 1),
            "rainfall": bool(rainfall),
            "note": (
                "Wet conditions detected — intermediate/wet tyres likely needed."
                if rainfall
                else (
                    "High track temp (>45°C) — expect accelerated degradation."
                    if track_mean > 45
                    else "Normal conditions."
                )
            ),
        }

    logger.info(
        "Full tire analysis complete for %s %d (%s)",
        circuit, year, driver,
    )
    return result


# ===================================================================
# CLI validation
# ===================================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    import json

    print("=" * 65)
    print("  Tire Agent — Validation: Silverstone 2022")
    print("=" * 65)

    # 1. Track classification
    wear = classify_track_tire_wear("Silverstone")
    print(f"\n▶ Track Wear Classification:")
    print(f"  Track:          {wear['track']}")
    print(f"  Classification: {wear['classification']}")
    print(f"  Score:          {wear['score']}/5.00")
    print(f"  Factors:        {wear['factors']}")

    # 2. Full analysis
    print(f"\n▶ Running full analysis (Silverstone 2022 — Race winner)...")
    analysis = analyze_tire_strategy("Silverstone", 2022)

    if analysis["error"]:
        print(f"  ⚠ Error: {analysis['error']}")
    else:
        # Degradation
        print(f"\n▶ Degradation (winner):")
        for s in analysis["degradation"]:
            print(
                f"  Stint {s['stint']}: {s['compound']} — "
                f"{s['deg_rate_sec_per_lap']:+.4f} s/lap, "
                f"{s['lap_count']} laps (L{s['start_lap']}→L{s['end_lap']})"
            )

        # Pit window
        pw = analysis["pit_window"]
        print(f"\n▶ Pit Window ({pw['strategy_type']}):")
        print(f"  {pw['rationale']}")

        # Compound recommendation
        rec = analysis["compound_rec"]
        if rec:
            print(f"\n▶ Compound Recommendation:")
            print(f"  Best order: {' → '.join(rec['recommended_order'])}")
            print(f"  Confidence: {rec['confidence']}")
            for s in rec["all_top3_strategies"]:
                print(f"    {s['driver']}: {' → '.join(s['compounds'])}")

        # Weather
        wi = analysis["weather_impact"]
        if wi:
            print(f"\n▶ Weather Impact:")
            print(f"  Air: {wi['air_temp_avg']}°C, Track: {wi['track_temp_avg']}°C")
            print(f"  Rainfall: {wi['rainfall']}")
            print(f"  Note: {wi['note']}")

    print("\n" + "=" * 65)
    print("  ✅ Tire Agent validation complete!")
    print("=" * 65)
