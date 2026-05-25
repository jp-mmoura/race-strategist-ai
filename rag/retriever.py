"""
RAG Retriever — semantic search + FastF1 enrichment for the Race Strategist.

Three retrieval modes:
  1. **retrieve_circuits** — pure vector search against ChromaDB f1_circuits.
  2. **retrieve_regulations** — filtered vector search against f1_regulations,
     with optional year and section constraints.
  3. **retrieve_race_context** — combines circuit retrieval with live
     FastF1 data (stints, results, weather) to build a rich context
     string that an LLM can use to answer strategy questions.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from rag.embedder import get_collection
from tools.fastf1_tool import (
    get_race_results,
    get_session,
    get_stints,
    get_tire_data,
    get_weather,
)

# ---------------------------------------------------------------------------
# Environment & Logging
# ---------------------------------------------------------------------------
load_dotenv()
logger = logging.getLogger(__name__)


# ===================================================================
# 1. Circuit retrieval (vector search only)
# ===================================================================

def retrieve_circuits(
    query: str,
    n_results: int = 3,
    where_filter: dict | None = None,
) -> list[dict[str, Any]]:
    """Search the f1_circuits collection by semantic similarity.

    Parameters
    ----------
    query : str
        Free-text query (e.g. ``"high-speed circuit in Italy"``).
    n_results : int
        Maximum number of results to return.
    where_filter : dict | None
        Optional ChromaDB ``where`` filter on metadata fields,
        e.g. ``{"country": "UK"}``.

    Returns
    -------
    list[dict]
        Each dict contains ``document``, ``metadata``, and ``distance``.
    """
    collection = get_collection("f1_circuits")

    kwargs: dict[str, Any] = {
        "query_texts": [query],
        "n_results": n_results,
    }
    if where_filter:
        kwargs["where"] = where_filter

    raw = collection.query(**kwargs)

    results = []
    for doc, meta, dist in zip(
        raw["documents"][0],
        raw["metadatas"][0],
        raw["distances"][0],
    ):
        results.append({
            "document": doc,
            "metadata": meta,
            "distance": dist,
        })

    logger.info(
        "Circuit query '%s' → %d results (best=%.4f)",
        query, len(results),
        results[0]["distance"] if results else float("inf"),
    )
    return results


# ===================================================================
# 2. Regulation retrieval (vector search with metadata filters)
# ===================================================================

def retrieve_regulations(
    query: str,
    year: int | None = None,
    section: str | None = None,
    n_results: int = 5,
) -> list[dict[str, Any]]:
    """Search the f1_regulations collection by semantic similarity.

    Supports optional metadata filters so callers can restrict results
    to a specific regulation year or section letter.

    Parameters
    ----------
    query : str
        Free-text question (e.g. ``"pit stop minimum time"``).
    year : int | None
        If provided, only chunks with ``metadata.year == year`` are
        returned.  Pass ``None`` to search across all available years —
        ChromaDB will naturally return the most semantically relevant
        chunks regardless of year.
    section : str | None
        Single uppercase letter (``"A"``–``"F"``) to restrict to one
        regulation section.  Pass ``None`` to search all sections.
    n_results : int
        Maximum number of chunks to return.

    Returns
    -------
    list[dict]
        Each dict contains ``document``, ``metadata``, and ``distance``.

    Raises
    ------
    ValueError
        If the ``f1_regulations`` collection does not exist (run the
        ingestor first with ``python -m rag.ingestor --regulations``).
    """
    collection = get_collection("f1_regulations")

    # Build where-filter
    where: dict[str, Any] | None = None
    if year is not None and section is not None:
        where = {"$and": [{"year": {"$eq": year}}, {"section": {"$eq": section.upper()}}]}
    elif year is not None:
        where = {"year": {"$eq": year}}
    elif section is not None:
        where = {"section": {"$eq": section.upper()}}

    kwargs: dict[str, Any] = {
        "query_texts": [query],
        "n_results": n_results,
    }
    if where:
        kwargs["where"] = where

    raw = collection.query(**kwargs)

    results = []
    for doc, meta, dist in zip(
        raw["documents"][0],
        raw["metadatas"][0],
        raw["distances"][0],
    ):
        results.append({
            "document": doc,
            "metadata": meta,
            "distance": dist,
        })

    logger.info(
        "Regulation query '%s' (year=%s, section=%s) → %d results",
        query, year, section, len(results),
    )
    return results


# ===================================================================
# 3. Race-context retrieval (vector search + FastF1)
# ===================================================================

def _stints_to_text(stints_df: pd.DataFrame) -> str:
    """Format a stints DataFrame as readable text."""
    lines = []
    for _, row in stints_df.iterrows():
        compound = row.get("Compound", "?")
        lines.append(
            f"  - {row['Driver']} stint {int(row['Stint'])}: "
            f"{compound} tyres, laps {int(row['StartLap'])}–{int(row['EndLap'])} "
            f"({int(row['Laps'])} laps, max tyre life {int(row['MaxTyreLife'])})"
        )
    return "\n".join(lines)


def _results_to_text(results_df: pd.DataFrame, top_n: int = 10) -> str:
    """Format race results as readable text."""
    lines = []
    for _, row in results_df.head(top_n).iterrows():
        pos = int(row["Position"]) if pd.notna(row["Position"]) else "?"
        lines.append(
            f"  P{pos} {row['Abbreviation']} — "
            f"{row['TeamName']} ({row['Status']})"
        )
    return "\n".join(lines)


def _weather_to_text(weather_df: pd.DataFrame) -> str:
    """Summarise weather data as readable text."""
    if weather_df.empty:
        return "  No weather data available."
    summary = weather_df[["AirTemp", "TrackTemp", "Humidity"]].describe()
    air = summary.loc["mean", "AirTemp"]
    track = summary.loc["mean", "TrackTemp"]
    hum = summary.loc["mean", "Humidity"]
    rain = "Yes" if weather_df.get("Rainfall", pd.Series([False])).any() else "No"
    return (
        f"  Air temp: {air:.1f}°C (avg), "
        f"Track temp: {track:.1f}°C (avg), "
        f"Humidity: {hum:.0f}%, "
        f"Rainfall: {rain}"
    )


def retrieve_race_context(
    query: str,
    year: int | None = None,
    circuit: str | None = None,
    session_type: str = "R",
    n_circuit_results: int = 1,
) -> dict[str, Any]:
    """Build a rich context dict combining ChromaDB + FastF1.

    The function:
      1. Searches ChromaDB for the most relevant circuit.
      2. Loads the corresponding FastF1 session.
      3. Extracts results, stint strategies, and weather.
      4. Assembles everything into a ``context_text`` string ready
         for LLM consumption.

    Parameters
    ----------
    query : str
        Natural-language question (e.g. "what strategy won at
        Silverstone 2022?").
    year : int | None
        If provided, used directly; otherwise extracted from the query
        is left to the caller.
    circuit : str | None
        If provided, skips ChromaDB lookup and uses this circuit name.
    session_type : str
        FastF1 session type (default ``"R"``).
    n_circuit_results : int
        How many circuits to retrieve from ChromaDB.

    Returns
    -------
    dict with keys:
        ``circuit_info``  – ChromaDB result(s)
        ``race_results``  – top-10 finishing order
        ``winner_stints`` – stint breakdown for the race winner
        ``all_stints``    – stint breakdown for all drivers
        ``weather``       – weather summary
        ``context_text``  – assembled string for LLM prompt
        ``error``         – error message if something went wrong
    """
    context: dict[str, Any] = {
        "circuit_info": None,
        "race_results": None,
        "winner_stints": None,
        "all_stints": None,
        "weather": None,
        "context_text": "",
        "error": None,
    }

    # ── 1. Circuit lookup via ChromaDB ────────────────────────────
    circuit_name = circuit
    if not circuit_name:
        hits = retrieve_circuits(query, n_results=n_circuit_results)
        if hits:
            context["circuit_info"] = hits
            circuit_name = hits[0]["metadata"]["name"]
            # Also try the circuitRef for FastF1 compatibility
            circuit_name_for_f1 = hits[0]["metadata"].get(
                "circuitRef", circuit_name
            )
        else:
            context["error"] = "No matching circuit found in ChromaDB."
            return context
    else:
        circuit_name_for_f1 = circuit_name

    if not year:
        context["error"] = (
            f"Found circuit '{circuit_name}' but no year was provided. "
            "Please specify the year."
        )
        context["context_text"] = (
            f"Circuit: {circuit_name}. Year not specified."
        )
        return context

    # ── 2. Load FastF1 session ────────────────────────────────────
    try:
        session = get_session(year, circuit_name_for_f1, session_type)
    except Exception:
        # Retry with full circuit name if circuitRef failed
        try:
            session = get_session(year, circuit_name, session_type)
        except Exception as exc:
            context["error"] = (
                f"FastF1 could not load {circuit_name} {year}: {exc}"
            )
            return context

    # ── 3. Extract data ───────────────────────────────────────────
    # Race results
    results_df = get_race_results(session)
    context["race_results"] = results_df

    # Winner stints
    winner = results_df.iloc[0]["Abbreviation"]
    winner_stints_df = get_stints(session, driver=winner)
    context["winner_stints"] = winner_stints_df

    # All stints (top 5 for brevity)
    top5_drivers = list(results_df.head(5)["Abbreviation"])
    all_stints_df = get_stints(session)
    top5_stints = all_stints_df[all_stints_df["Driver"].isin(top5_drivers)]
    context["all_stints"] = top5_stints

    # Weather
    weather_df = get_weather(session)
    context["weather"] = weather_df

    # ── 4. Assemble context text ──────────────────────────────────
    event_name = session.event["EventName"]
    parts = [
        f"=== Race Context: {event_name} {year} ===\n",
        f"Circuit: {circuit_name}",
        f"Session: {session_type}\n",
        f"--- Race Results (top 10) ---",
        _results_to_text(results_df),
        f"\n--- Winning Strategy ({winner}) ---",
        _stints_to_text(winner_stints_df),
        f"\n--- Top-5 Strategies ---",
        _stints_to_text(top5_stints),
        f"\n--- Weather ---",
        _weather_to_text(weather_df),
    ]
    context["context_text"] = "\n".join(parts)

    logger.info(
        "Built race context for %s %d (%d chars)",
        circuit_name, year, len(context["context_text"]),
    )
    return context


# ===================================================================
# Quick sanity-check
# ===================================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Test 1: Pure circuit search
    print("=" * 60)
    print("  Test 1: Circuit vector search")
    print("=" * 60)
    hits = retrieve_circuits("fast track in Great Britain")
    for h in hits:
        print(f"  [{h['distance']:.4f}] {h['metadata']['name']}")

    # Test 2: Regulation search (year + section filters)
    print("\n" + "=" * 60)
    print("  Test 2: Regulation search (safety car — 2026 Sporting)")
    print("=" * 60)
    try:
        reg_hits = retrieve_regulations(
            "safety car deployment procedure",
            year=2026,
            section="B",
            n_results=3,
        )
        for rh in reg_hits:
            preview = rh["document"][:120].replace("\n", " ")
            m = rh["metadata"]
            print(
                f"  [{rh['distance']:.4f}] {m['year']} Section {m['section']}"
                f" chunk {m['chunk_index']}/{m['total_chunks']}: \"{preview}…\""
            )
    except ValueError as e:
        print(f"  ⚠ Regulations not yet ingested: {e}")
        print("  Run: python -m rag.ingestor --regulations")

    # Test 3: Full race context
    print("\n" + "=" * 60)
    print("  Test 3: Full race context — Silverstone 2022")
    print("=" * 60)
    ctx = retrieve_race_context(
        query="what strategy won at Silverstone 2022?",
        year=2022,
        circuit="Silverstone",
    )
    if ctx["error"]:
        print(f"  ⚠ Error: {ctx['error']}")
    else:
        print(ctx["context_text"])
        print(f"\n  ✅ Context length: {len(ctx['context_text'])} chars")
