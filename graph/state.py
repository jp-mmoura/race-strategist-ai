"""
Shared state definition for the F1 Race Strategist AI graph.

This TypedDict is the single source of truth that flows through every
LangGraph node.  Each agent reads what it needs and writes back its
results, so downstream nodes always have the latest context.
"""

from __future__ import annotations

from typing import Any, TypedDict

import pandas as pd


class RaceStrategyState(TypedDict, total=False):
    """State shared across all LangGraph nodes.

    Attributes
    ----------
    circuit : str
        Circuit / Grand Prix name (e.g. ``"Monaco"``).
    year : int
        Season year (e.g. ``2023``).
    session_type : str
        FastF1 session identifier (``"FP1"``, ``"FP2"``, ``"FP3"``,
        ``"Q"``, ``"S"``, ``"R"``).  Defaults to ``"R"``.
    lap_data : pd.DataFrame | None
        All laps returned by FastF1 for the loaded session.
    tire_data : pd.DataFrame | None
        Per-lap tire/compound information (compound, tyre life,
        stint number, fresh-tyre flag, lap time).
    weather_data : pd.DataFrame | None
        Weather readings throughout the session (air temp, track
        temp, humidity, rainfall, wind speed / direction).
    strategy_recommendation : dict[str, Any] | None
        Output of the Strategy agent — suggested pit windows,
        compound choices, and reasoning.
    evaluation_result : dict[str, Any] | None
        Output of the Evaluation agent — risk assessment, expected
        delta, and confidence score for the recommended strategy.
    messages : list[dict[str, str]]
        Conversation history (role / content pairs) that flows
        through the graph for LLM context.
    error : str | None
        If any node fails, it writes a human-readable error here
        so the UI can surface it gracefully.
    """

    # ── Core identifiers ──────────────────────────────────────────
    circuit: str
    year: int
    session_type: str

    # ── Data payloads (populated by the data-collection node) ─────
    lap_data: pd.DataFrame | None
    tire_data: pd.DataFrame | None
    weather_data: pd.DataFrame | None

    # ── Agent outputs ─────────────────────────────────────────────
    strategy_recommendation: dict[str, Any] | None
    evaluation_result: dict[str, Any] | None

    # ── Conversation & error handling ─────────────────────────────
    messages: list[dict[str, str]]
    error: str | None
