"""
Strategy Agent — generates the final race-strategy recommendation.

Consumes the outputs of:
  • **Tire Agent**    → wear classification, degradation, pit windows
  • **Weather Agent** → forecast, rain risk, temperature/wind impact
  • **RAG Retriever** → historical race context from ChromaDB + FastF1

and synthesises a unified strategic recommendation with a structured
justification that can be written into ``RaceStrategyState``.

Main functions
--------------
build_strategy_context(circuit, year, race_date, driver)
    → gathers all upstream data into a single context dict

generate_strategy(circuit, year, race_date, driver)
    → LLM-powered strategic recommendation (uses OpenAI via LangChain)

generate_strategy_offline(circuit, year, race_date, driver)
    → rule-based fallback (no LLM required)

run_strategy_node(state)
    → LangGraph node entry-point — reads/writes RaceStrategyState
"""

from __future__ import annotations

import json
import logging
import os
import textwrap
from datetime import date
from typing import Any

from dotenv import load_dotenv

from agents.tire_agent import (
    analyze_tire_strategy,
    classify_track_tire_wear,
)
from agents.weather_agent import (
    analyze_weather_impact,
    assess_rain_risk,
)
from rag.retriever import retrieve_race_context

# ---------------------------------------------------------------------------
# Environment & Logging
# ---------------------------------------------------------------------------
load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM configuration
# ---------------------------------------------------------------------------
_LLM_PROVIDER = os.getenv("STRATEGY_LLM_PROVIDER", "google")  # "google" or "openai"
_LLM_MODEL = os.getenv("STRATEGY_LLM_MODEL", "gemini-2.0-flash")
_LLM_TEMPERATURE = float(os.getenv("STRATEGY_LLM_TEMPERATURE", "0.3"))
_LLM_TIMEOUT = int(os.getenv("STRATEGY_LLM_TIMEOUT", "10"))  # seconds
_LLM_MAX_RETRIES = int(os.getenv("STRATEGY_LLM_MAX_RETRIES", "0"))


def _get_llm():
    """Instantiate the configured LLM (Google Gemini or OpenAI)."""
    if _LLM_PROVIDER == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=_LLM_MODEL,
            temperature=_LLM_TEMPERATURE,
            timeout=_LLM_TIMEOUT,
            max_retries=_LLM_MAX_RETRIES,
        )
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=_LLM_MODEL,
            temperature=_LLM_TEMPERATURE,
            request_timeout=_LLM_TIMEOUT,
            max_retries=_LLM_MAX_RETRIES,
        )


# ===================================================================
# System prompt for the strategy LLM
# Prompt Version: 1.1.0
# Tracked in prompts/strategist_system_prompt.md
# ===================================================================

_SYSTEM_PROMPT = textwrap.dedent("""\
    You are the Chief Race Strategist for an F1 team.  Your role is to
    synthesise data from the Tire Engineer, Weather Analyst, and
    Historical Database into a single, actionable race strategy.

    IMPORTANT: Your audience includes engineers AND non-technical
    stakeholders (team principals, sponsors, media).  Every section
    must be understandable by someone who knows F1 but has no data
    science background.

    Always structure your response with the following sections:

    ## Recommended Strategy
    State the strategy type (1-stop, 2-stop, etc.), the compound order,
    and the target pit-stop laps.

    ## Compound Selection
    Explain which compounds to use in each stint and why.  For each
    compound choice, briefly state WHY it is the best option for that
    stint (e.g., "Mediums in stint 1 because degradation on softs
    exceeds +0.09 s/lap at this circuit, meaning they would lose grip
    by lap 12").

    ## Pit Windows
    Specify the optimal, earliest, and latest pit-stop laps for each stop.

    ## Weather Contingency
    Describe the plan if weather changes (rain, temperature shift).

    ## Risk Assessment
    Identify the top risks and how to mitigate them.

    ## Factors Considered
    Explicitly list EVERY data source you used and how much it
    influenced your recommendation.  Use this format:
    - **Tire degradation data** (weight: high/medium/low) — what it told you
    - **Weather forecast** (weight: high/medium/low) — what it told you
    - **Historical race data** (weight: high/medium/low) — what it told you
    - **Track classification** (weight: high/medium/low) — what it told you
    If any data source was unavailable or incomplete, say so.

    ## Alternatives Considered
    List at least 2 alternative strategies you evaluated and explain
    WHY you rejected each one.  For example:
    - "1-stop (Medium → Hard): rejected because degradation data shows
       the hard compound loses 0.06 s/lap here, making a 30-lap hard
       stint too slow by ~1.8 s."
    Be specific — cite numbers from the data.

    ## Confidence Assessment
    Rate your overall confidence: **High**, **Medium**, or **Low**.
    Then explain WHY in 1-2 sentences.  Consider:
    - How complete was the data? (all 3 sources available?)
    - Do the data sources agree with each other?
    - Does your recommendation match the historical winner's strategy?
    - Are there unusual conditions (rain, extreme heat) adding uncertainty?

    ## Justification Summary
    In plain language (no jargon), explain in 2-3 sentences why this
    is the best strategy.  Write it so that a fan watching on TV could
    understand.  Example: "We start on mediums because they last longer
    on this track, switch to hards at lap 25 when grip drops, and this
    matches what the race winner actually did last year."

    Be precise with lap numbers.  Use bullet points and keep the total
    response under 800 words.
""")


# ===================================================================
# 1. build_strategy_context — gather all upstream data
# ===================================================================

def build_strategy_context(
    circuit: str,
    year: int,
    race_date: str | date | None = None,
    driver: str | None = None,
    session_type: str = "R",
) -> dict[str, Any]:
    """Collect tire, weather, and RAG outputs into a single context dict.

    Parameters
    ----------
    circuit : str
        Circuit name (e.g. ``"Silverstone"``).
    year : int
        Season year.
    race_date : str | date | None
        Race date for weather forecast filtering.
    driver : str | None
        Focus driver (defaults to race winner inside sub-agents).
    session_type : str
        FastF1 session type.

    Returns
    -------
    dict with keys: tire_analysis, weather_analysis, rag_context,
                    context_text (assembled prompt), error.
    """
    context: dict[str, Any] = {
        "tire_analysis": None,
        "weather_analysis": None,
        "rag_context": None,
        "context_text": "",
        "error": None,
    }

    errors: list[str] = []

    # ── Tire Agent ────────────────────────────────────────────────
    try:
        tire = analyze_tire_strategy(circuit, year, session_type, driver)
        context["tire_analysis"] = tire
        if tire.get("error"):
            errors.append(f"Tire: {tire['error']}")
    except Exception as exc:
        logger.warning("Tire analysis failed: %s", exc)
        errors.append(f"Tire: {exc}")

    # ── Weather Agent ─────────────────────────────────────────────
    try:
        weather = analyze_weather_impact(circuit, race_date, year, session_type)
        context["weather_analysis"] = weather
        if weather.get("error"):
            errors.append(f"Weather: {weather['error']}")
    except Exception as exc:
        logger.warning("Weather analysis failed: %s", exc)
        errors.append(f"Weather: {exc}")

    # ── RAG Retriever ─────────────────────────────────────────────
    try:
        rag = retrieve_race_context(
            query=f"race strategy {circuit} {year}",
            year=year,
            circuit=circuit,
            session_type=session_type,
        )
        context["rag_context"] = rag
        if rag.get("error"):
            errors.append(f"RAG: {rag['error']}")
    except Exception as exc:
        logger.warning("RAG retrieval failed: %s", exc)
        errors.append(f"RAG: {exc}")

    if errors:
        context["error"] = "; ".join(errors)

    # ── Assemble human-readable context for LLM ───────────────────
    context["context_text"] = _assemble_context_text(context)

    logger.info(
        "Strategy context built for %s %d (%d chars)",
        circuit, year, len(context["context_text"]),
    )
    return context


def _assemble_context_text(ctx: dict[str, Any]) -> str:
    """Convert structured context into a text block for the LLM prompt."""
    sections: list[str] = []

    # ── Tire data ─────────────────────────────────────────────────
    tire = ctx.get("tire_analysis")
    if tire:
        tw = tire.get("track_wear") or {}
        sections.append(
            f"### Tire Analysis\n"
            f"- Track wear: {tw.get('classification', 'N/A')} "
            f"(score {tw.get('score', '?')}/5)\n"
            f"- Track: {tw.get('track', 'N/A')}"
        )

        deg = tire.get("degradation") or []
        if deg:
            lines = ["- Degradation per stint:"]
            for s in deg:
                lines.append(
                    f"  • Stint {s['stint']} ({s['compound']}): "
                    f"{s['deg_rate_sec_per_lap']:+.4f} s/lap, "
                    f"{s['lap_count']} laps (L{s['start_lap']}–L{s['end_lap']})"
                )
            sections.append("\n".join(lines))

        pw = tire.get("pit_window") or {}
        if pw.get("pit_windows"):
            lines = [f"- Strategy type: {pw.get('strategy_type', '?')}"]
            for i, w in enumerate(pw["pit_windows"], 1):
                lines.append(
                    f"  • Pit {i}: lap {w['earliest']}–{w['latest']} "
                    f"(optimal ~{w['optimal']}), {w['compound']}"
                )
            sections.append("\n".join(lines))

        rec = tire.get("compound_rec") or {}
        if rec.get("recommended_order"):
            sections.append(
                f"- Recommended compound order: "
                f"{' → '.join(rec['recommended_order'])} "
                f"(confidence: {rec.get('confidence', '?')})"
            )

    # ── Weather data ──────────────────────────────────────────────
    wx = ctx.get("weather_analysis")
    if wx:
        parts = ["### Weather Analysis"]

        rain = wx.get("rain_risk") or {}
        parts.append(f"- Rain risk: {rain.get('risk_level', 'N/A')}")
        parts.append(f"- {rain.get('summary', '')}")

        temp = wx.get("temperature") or {}
        if temp:
            parts.append(
                f"- Air temp: {temp.get('air_temp_min_c', '?')}–"
                f"{temp.get('air_temp_max_c', '?')} °C "
                f"(avg {temp.get('air_temp_avg_c', '?')} °C)"
            )
            parts.append(
                f"- Est. track temp: {temp.get('track_temp_est_c', '?')} °C"
            )
            if temp.get("note"):
                parts.append(f"- {temp['note']}")

        wind = wx.get("wind") or {}
        if wind:
            parts.append(
                f"- Wind: avg {wind.get('avg_speed_kmh', '?')} km/h, "
                f"max {wind.get('max_speed_kmh', '?')} km/h"
            )
            if wind.get("note"):
                parts.append(f"- {wind['note']}")

        notes = wx.get("strategy_notes") or []
        for n in notes:
            parts.append(f"- {n}")

        sections.append("\n".join(parts))

    # ── RAG context ───────────────────────────────────────────────
    rag = ctx.get("rag_context")
    if rag and rag.get("context_text"):
        sections.append(
            f"### Historical Race Data\n{rag['context_text']}"
        )

    return "\n\n".join(sections) if sections else "No context data available."


# ===================================================================
# 2. generate_strategy — LLM-powered recommendation
# ===================================================================

def generate_strategy(
    circuit: str,
    year: int,
    race_date: str | date | None = None,
    driver: str | None = None,
    session_type: str = "R",
    *,
    precomputed_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate a full race-strategy recommendation using an LLM.

    Parameters
    ----------
    circuit, year, race_date, driver, session_type
        Passed through to ``build_strategy_context``.
    precomputed_context : dict | None
        If provided, skip ``build_strategy_context`` and use this dict
        directly. Expected keys: tire_analysis, weather_analysis,
        rag_context, context_text. Useful when the graph has already
        populated these from upstream nodes.

    Returns
    -------
    dict with keys:
        circuit, year, driver, strategy_type, compounds, pit_laps,
        recommendation_text, context_summary, confidence, error.
    """
    result: dict[str, Any] = {
        "circuit": circuit,
        "year": year,
        "driver": driver,
        "strategy_type": None,
        "compounds": [],
        "pit_laps": [],
        "recommendation_text": "",
        "context_summary": "",
        "confidence": None,
        "error": None,
    }

    # ── 1. Build context (or reuse pre-computed) ──────────────────
    if precomputed_context is not None:
        ctx = precomputed_context
        # Ensure context_text is assembled if not present
        if not ctx.get("context_text"):
            ctx["context_text"] = _assemble_context_text(ctx)
    else:
        ctx = build_strategy_context(
            circuit, year, race_date, driver, session_type,
        )
    result["context_summary"] = ctx["context_text"]

    # Resolve driver from tire analysis if not provided
    if driver is None:
        tire = ctx.get("tire_analysis") or {}
        pw = tire.get("pit_window") or {}
        driver = pw.get("driver")
        result["driver"] = driver

    # Copy structured data from tire analysis for easy access
    tire = ctx.get("tire_analysis") or {}
    pw = tire.get("pit_window") or {}
    result["strategy_type"] = pw.get("strategy_type")
    result["pit_laps"] = pw.get("recommended_pit_laps", [])
    rec = tire.get("compound_rec") or {}
    result["compounds"] = rec.get("recommended_order", [])

    # ── 2. Call LLM ───────────────────────────────────────────────
    try:
        from langchain_core.messages import SystemMessage, HumanMessage

        llm = _get_llm()

        user_prompt = (
            f"Generate a race strategy recommendation for the "
            f"{circuit} Grand Prix ({year}).\n"
            f"Focus driver: {driver or 'race winner'}.\n\n"
            f"Here is the data from the Tire Engineer, Weather Analyst, "
            f"and Historical Database:\n\n"
            f"{ctx['context_text']}"
        )

        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        response = llm.invoke(messages)
        result["recommendation_text"] = response.content

        # Extract confidence from the LLM response if present
        resp_lower = response.content.lower()
        if "confidence" in resp_lower and "**high**" in resp_lower:
            result["confidence"] = "high"
        elif "confidence" in resp_lower and "**low**" in resp_lower:
            result["confidence"] = "low"
        elif "confidence" in resp_lower and "**medium**" in resp_lower:
            result["confidence"] = "medium"
        else:
            # Fallback: data completeness heuristic
            result["confidence"] = "high" if not ctx.get("error") else "medium"

        logger.info(
            "LLM strategy generated for %s %d (%d chars)",
            circuit, year, len(response.content),
        )

    except Exception as exc:
        logger.warning(
            "LLM generation failed (%s), falling back to offline strategy: %s",
            type(exc).__name__, exc,
        )
        # Fallback to rule-based (reuse already-built context)
        offline = generate_strategy_offline(
            circuit, year, race_date, driver, session_type,
            precomputed_context=ctx,
        )
        result["recommendation_text"] = offline["recommendation_text"]
        result["confidence"] = "low"
        result["error"] = f"LLM unavailable ({exc}); used rule-based fallback."

    return result


# ===================================================================
# Explainability helpers
# ===================================================================

def _generate_offline_alternatives(
    strategy_type: str,
    compounds: list[str],
    deg: list[dict],
    wear_class: str,
    risk_level: str,
    total_laps: int,
    rag_winner_compounds: list[str],
    rag_winner_strategy_type: str,
) -> list[str]:
    """Generate reasoning about discarded alternative strategies.

    Produces at least 2 alternative strategies with data-driven
    explanations for why each was rejected.
    """
    alt: list[str] = []
    num_stops = max(0, len(compounds) - 1) if compounds else 0

    # ── Alternative 1: fewer stops ────────────────────────────────
    if num_stops >= 2:
        fewer = f"{num_stops - 1}-stop"
        if deg:
            avg_deg = sum(d["deg_rate_sec_per_lap"] for d in deg) / len(deg)
            laps_per_stint = total_laps // num_stops if num_stops else total_laps
            time_loss = abs(avg_deg) * laps_per_stint
            alt.append(
                f"- **{fewer} strategy**: rejected because average "
                f"degradation is {avg_deg:+.4f} s/lap. Over a "
                f"{laps_per_stint}-lap stint, tires would lose ~{time_loss:.1f}s "
                f"— too much to stay competitive without an extra stop."
            )
        else:
            alt.append(
                f"- **{fewer} strategy**: rejected — {wear_class} track "
                f"classification makes longer stints risky."
            )
    elif num_stops == 1:
        alt.append(
            "- **0-stop (no pit) strategy**: rejected — regulations require "
            "using at least two different dry-weather compounds during a race."
        )
    else:
        alt.append(
            "- **1-stop strategy**: rejected — insufficient data to "
            "validate a 1-stop would be competitive."
        )

    # ── Alternative 2: more stops ─────────────────────────────────
    if num_stops <= 1:
        more = f"{num_stops + 1}-stop"
        alt.append(
            f"- **{more} strategy**: rejected because the extra pit stop "
            f"costs ~22-25 seconds of track time, and the degradation data "
            f"does not show enough tire wear to justify it."
        )
    else:
        more = f"{num_stops + 1}-stop"
        alt.append(
            f"- **{more} strategy**: rejected — the additional pit stop "
            f"would cost ~22-25s of track time with no significant tire "
            f"life benefit based on degradation data."
        )

    # ── Alternative 3: different compound order ───────────────────
    if compounds and len(compounds) >= 2:
        reversed_order = list(reversed(compounds))
        if reversed_order != compounds:
            alt.append(
                f"- **Reversed compound order "
                f"({' → '.join(reversed_order)})**: rejected because "
                f"starting on a {'harder' if compounds[0] in ('SOFT', 'MEDIUM') else 'softer'} "
                f"compound at this {wear_class.lower()} circuit would "
                f"{'sacrifice early pace' if compounds[0] in ('HARD',) else 'risk excessive early wear'}."
            )

    # ── Alternative 4: historical mismatch note ───────────────────
    if rag_winner_compounds and compounds != rag_winner_compounds:
        alt.append(
            f"- **Historical winner's strategy "
            f"({' → '.join(rag_winner_compounds)}, "
            f"{rag_winner_strategy_type})**: was considered but not "
            f"adopted because current conditions (weather, tire allocation) "
            f"may differ from the historical race."
        )

    # ── Weather-based alternative ─────────────────────────────────
    if risk_level in ("High", "Medium"):
        # If recommending dry compounds under rain risk
        wet_compounds = [c for c in compounds if c in ("INTERMEDIATE", "WET")]
        if not wet_compounds:
            alt.append(
                f"- **Starting on intermediates**: considered due to "
                f"{risk_level.lower()} rain risk, but rejected in favor of "
                f"the dry primary strategy with intermediates on standby. "
                f"If rain arrives, we will reactively switch."
            )

    return alt if alt else [
        "- No clear alternative strategies identified with available data."
    ]


def _generate_plain_justification(
    strategy_type: str,
    compounds: list[str],
    pit_laps: list[int],
    wear_class: str,
    risk_level: str,
    rag_winner: str | None,
    rag_winner_compounds: list[str],
) -> str:
    """Generate a plain-language justification a TV fan could understand.

    Returns 2-3 sentences, no jargon.
    """
    parts: list[str] = []

    # Opening — what we're doing
    if compounds:
        first = compounds[0]
        parts.append(
            f"We start the race on {first.lower()} tires because "
            f"they offer the best balance of pace and durability for "
            f"this {'demanding' if 'High' in wear_class else 'moderate wear'} "
            f"circuit."
        )
    else:
        parts.append("Strategy based on available data.")

    # Middle — pit stops
    if pit_laps:
        pit_str = " and ".join(f"lap {l}" for l in pit_laps)
        parts.append(
            f"We plan to pit on {pit_str} to switch to fresher tires "
            f"before grip drops off."
        )

    # Closing — confidence anchor
    if rag_winner and rag_winner_compounds == compounds:
        parts.append(
            f"This matches exactly what {rag_winner} did to win "
            f"this race, giving us strong confidence in the plan."
        )
    elif rag_winner:
        parts.append(
            f"The race winner ({rag_winner}) used a slightly different "
            f"approach, but current conditions support our choice."
        )

    if risk_level in ("High", "Medium"):
        parts.append(
            f"We have rain tires ready as a backup given the "
            f"{risk_level.lower()} chance of rain."
        )

    return " ".join(parts)


# ===================================================================
# 3. generate_strategy_offline — rule-based fallback
# ===================================================================

def generate_strategy_offline(
    circuit: str,
    year: int,
    race_date: str | date | None = None,
    driver: str | None = None,
    session_type: str = "R",
    *,
    precomputed_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate a strategy recommendation without an LLM.

    Uses deterministic rules based on the tire and weather data.
    Useful as a fallback when no API key is configured.

    Parameters
    ----------
    precomputed_context : dict | None
        If provided, skip ``build_strategy_context`` and use this dict
        directly. Avoids redundant data fetching when called from the
        LangGraph workflow.

    Returns
    -------
    dict with the same keys as ``generate_strategy``.
    """
    if precomputed_context is not None:
        ctx = precomputed_context
        if not ctx.get("context_text"):
            ctx["context_text"] = _assemble_context_text(ctx)
    else:
        ctx = build_strategy_context(
            circuit, year, race_date, driver, session_type,
        )

    tire = ctx.get("tire_analysis") or {}
    weather = ctx.get("weather_analysis") or {}
    rain = weather.get("rain_risk") or {}
    temp = weather.get("temperature") or {}
    pw = tire.get("pit_window") or {}
    tw = tire.get("track_wear") or {}
    rec = tire.get("compound_rec") or {}
    deg = tire.get("degradation") or []

    # ── Resolve key values ────────────────────────────────────────
    strategy_type = pw.get("strategy_type", "Unknown")
    compounds = rec.get("recommended_order", [])
    pit_laps = pw.get("recommended_pit_laps", [])
    pit_windows = pw.get("pit_windows", [])
    total_laps = pw.get("total_laps", 0)
    risk_level = rain.get("risk_level", "Unknown")
    wear_class = tw.get("classification", "Unknown")
    avg_temp = temp.get("air_temp_avg_c")
    track_temp_est = temp.get("track_temp_est_c")

    # ── RAG Cross-Validation ──────────────────────────────────────
    rag = ctx.get("rag_context") or {}
    rag_winner = None
    rag_winner_compounds = []
    rag_winner_stops = 0
    rag_winner_strategy_type = "Unknown"

    rag_results = rag.get("race_results")
    if rag_results is not None and not rag_results.empty:
        rag_winner = rag_results.iloc[0]["Abbreviation"]

    rag_winner_stints = rag.get("winner_stints")
    if rag_winner_stints is not None and not rag_winner_stints.empty:
        # Get sequence of compounds used by actual winner
        rag_winner_compounds = list(
            rag_winner_stints.sort_values("Stint")["Compound"].dropna()
        )
        rag_winner_stops = max(0, len(rag_winner_compounds) - 1)
        rag_winner_strategy_type = f"{rag_winner_stops}-stop" if rag_winner_stops > 0 else "0-stop"

    # ── Build recommendation text ─────────────────────────────────
    lines: list[str] = []

    lines.append("## Recommended Strategy")
    lines.append(f"**{strategy_type}** over {total_laps} laps.")
    if compounds:
        lines.append(f"Compound order: **{' → '.join(compounds)}**.")
    if pit_laps:
        pit_str = ", ".join(f"lap {l}" for l in pit_laps)
        lines.append(f"Target pit stops: {pit_str}.")

    lines.append("\n## Compound Selection")
    if compounds:
        for i, c in enumerate(compounds, 1):
            stint_deg = deg[i - 1] if i - 1 < len(deg) else None
            deg_info = (
                f" (deg: {stint_deg['deg_rate_sec_per_lap']:+.4f} s/lap)"
                if stint_deg
                else ""
            )
            lines.append(f"- **Stint {i}**: {c}{deg_info}")
    else:
        lines.append("- Insufficient data to recommend compounds.")

    lines.append("\n## Pit Windows")
    if pit_windows:
        for i, w in enumerate(pit_windows, 1):
            lines.append(
                f"- **Pit {i}**: laps {w['earliest']}–{w['latest']} "
                f"(optimal ~{w['optimal']}), currently on {w['compound']}"
            )
    else:
        lines.append("- No pit window data available.")

    lines.append("\n## Weather Contingency")
    if risk_level == "High":
        lines.append(
            "- **HIGH RAIN RISK** — prepare intermediates. "
            "If rain hits mid-stint, pit immediately to inters. "
            "Pre-plan a wet-weather pit stop lap range."
        )
    elif risk_level == "Medium":
        lines.append(
            "- **MEDIUM RAIN RISK** — have intermediates on standby. "
            "Monitor radar; an early switch could yield track position."
        )
    elif risk_level == "Low":
        lines.append(
            "- **LOW RAIN RISK** — primary dry strategy. "
            "Intermediates available as a precaution."
        )
    else:
        lines.append(
            "- **DRY CONDITIONS** — no weather contingency needed."
        )

    if track_temp_est and track_temp_est > 45:
        lines.append(
            f"- Track temp ~{track_temp_est:.0f} °C — "
            "consider extending first stint on harder compound."
        )

    lines.append("\n## Risk Assessment")
    risks: list[str] = []
    if wear_class == "High Tire Wear":
        risks.append(
            "High tire wear circuit — degradation may force an additional stop."
        )
    if risk_level in ("High", "Medium"):
        risks.append(
            f"{risk_level} rain risk — wrong tyre choice at the wrong "
            "time could cost 20+ seconds."
        )
    if avg_temp and avg_temp > 35:
        risks.append(
            "Extreme heat — rear blistering risk on softer compounds."
        )
    if not risks:
        risks.append("No major risks identified.")
    for r in risks:
        lines.append(f"- {r}")

    lines.append("\n## Historical Cross-Verification (RAG)")
    if rag_winner:
        lines.append(f"- **Historical Winner**: {rag_winner}")
        if rag_winner_compounds:
            compounds_str = " → ".join(rag_winner_compounds)
            lines.append(f"- **Winner's Actual Strategy**: {rag_winner_strategy_type} ({compounds_str})")
            
            # Check alignment
            if compounds == rag_winner_compounds:
                lines.append("- **Status**: ✅ Recommendation matches historical winner strategy exactly.")
            else:
                lines.append("- **Status**: ⚠️ Recommendation differs from historical winner strategy.")
                lines.append(f"  - Recommended: {' → '.join(compounds)} ({strategy_type})")
                lines.append(f"  - Actual: {compounds_str} ({rag_winner_strategy_type})")
        else:
            lines.append("- Winner strategy stints not available in RAG context.")
    else:
        lines.append("- No historical winner details found in RAG context.")

    # ── Factors Considered ─────────────────────────────────────────
    lines.append("\n## Factors Considered")

    tire_weight = "high" if deg else "low"
    lines.append(
        f"- **Tire degradation data** (weight: {tire_weight}) — "
    )
    if deg:
        avg_deg = sum(d["deg_rate_sec_per_lap"] for d in deg) / len(deg)
        lines.append(
            f"  Average degradation rate: {avg_deg:+.4f} s/lap "
            f"across {len(deg)} stint(s). "
            f"Track classification: {wear_class} (score {tw.get('score', '?')}/5)."
        )
    else:
        lines.append("  No degradation data available.")

    wx_weight = "high" if risk_level in ("High", "Medium") else "medium"
    lines.append(
        f"- **Weather forecast** (weight: {wx_weight}) — "
        f"Rain risk: {risk_level}."
    )
    if avg_temp is not None:
        lines.append(
            f"  Air temp: {avg_temp:.1f} °C "
            f"(est. track: {track_temp_est:.1f} °C)."
        )

    rag_weight = "high" if rag_winner else "low"
    lines.append(
        f"- **Historical race data** (weight: {rag_weight}) — "
    )
    if rag_winner:
        lines.append(
            f"  Winner {rag_winner} used "
            f"{' → '.join(rag_winner_compounds)} ({rag_winner_strategy_type})."
        )
    else:
        lines.append("  No historical winner data available.")

    lines.append(
        f"- **Track classification** (weight: medium) — "
        f"{wear_class} (score {tw.get('score', '?')}/5)."
    )

    # ── Alternatives Considered ────────────────────────────────────
    lines.append("\n## Alternatives Considered")

    alt_lines = _generate_offline_alternatives(
        strategy_type=strategy_type,
        compounds=compounds,
        deg=deg,
        wear_class=wear_class,
        risk_level=risk_level,
        total_laps=total_laps,
        rag_winner_compounds=rag_winner_compounds,
        rag_winner_strategy_type=rag_winner_strategy_type,
    )
    for a in alt_lines:
        lines.append(a)

    # ── Confidence Assessment ─────────────────────────────────────
    lines.append("\n## Confidence Assessment")

    data_sources_ok = sum([
        bool(deg),
        risk_level != "Unknown",
        bool(rag_winner),
    ])
    sources_agree = (
        compounds == rag_winner_compounds if rag_winner_compounds else True
    )

    if data_sources_ok == 3 and sources_agree:
        conf_level = "**High**"
        conf_reason = (
            "All three data sources (tire, weather, historical) are available "
            "and the recommendation aligns with the historical winner's strategy."
        )
    elif data_sources_ok >= 2:
        conf_level = "**Medium**"
        reasons = []
        if not deg:
            reasons.append("tire degradation data is missing")
        if risk_level == "Unknown":
            reasons.append("weather data is unavailable")
        if not rag_winner:
            reasons.append("no historical winner data")
        if not sources_agree:
            reasons.append(
                "recommendation differs from historical winner's strategy"
            )
        conf_reason = (
            f"Some uncertainty because: {'; '.join(reasons)}."
            if reasons
            else "Most data sources agree but some minor gaps exist."
        )
    else:
        conf_level = "**Low**"
        conf_reason = (
            "Limited data availability — fewer than 2 of 3 data sources "
            "returned usable results."
        )

    lines.append(f"- Confidence: {conf_level}")
    lines.append(f"- Reason: {conf_reason}")

    # ── Justification Summary (plain language) ────────────────────
    lines.append("\n## Justification Summary")
    lines.append(
        _generate_plain_justification(
            strategy_type=strategy_type,
            compounds=compounds,
            pit_laps=pit_laps,
            wear_class=wear_class,
            risk_level=risk_level,
            rag_winner=rag_winner,
            rag_winner_compounds=rag_winner_compounds,
        )
    )

    recommendation_text = "\n".join(lines)

    result = {
        "circuit": circuit,
        "year": year,
        "driver": driver or pw.get("driver"),
        "strategy_type": strategy_type,
        "compounds": compounds,
        "pit_laps": pit_laps,
        "recommendation_text": recommendation_text,
        "context_summary": ctx["context_text"],
        "confidence": conf_level.strip("*").lower() if not ctx.get("error") else "low",
        "error": ctx.get("error"),
    }

    logger.info(
        "Offline strategy generated for %s %d (%d chars)",
        circuit, year, len(recommendation_text),
    )
    return result


# ===================================================================
# 4. run_strategy_node — LangGraph node entry-point
# ===================================================================

def run_strategy_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node: read state, run strategy, write result back.

    Reads ``circuit``, ``year``, ``session_type`` from the state and
    writes ``strategy_recommendation`` back.

    Parameters
    ----------
    state : dict
        Current ``RaceStrategyState``.

    Returns
    -------
    dict
        Updated state fields (``strategy_recommendation``, possibly
        ``error``).
    """
    circuit = state.get("circuit", "")
    year = state.get("year", 2024)
    session_type = state.get("session_type", "R")

    if not circuit:
        return {
            "strategy_recommendation": None,
            "error": "No circuit specified in state.",
        }

    try:
        result = generate_strategy(
            circuit=circuit,
            year=year,
            session_type=session_type,
        )
        return {
            "strategy_recommendation": result,
            "error": result.get("error"),
        }
    except Exception as exc:
        logger.error("Strategy node failed: %s", exc, exc_info=True)
        return {
            "strategy_recommendation": None,
            "error": f"Strategy agent failed: {exc}",
        }


# ===================================================================
# CLI validation
# ===================================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("=" * 65)
    print("  Strategy Agent — Validation (offline / rule-based)")
    print("=" * 65)

    # Use offline mode so no API key is needed for validation
    strategy = generate_strategy_offline("Silverstone", 2023)

    if strategy["error"]:
        print(f"\n⚠ Errors encountered: {strategy['error']}")

    print(f"\n▶ Driver:   {strategy['driver']}")
    print(f"▶ Strategy: {strategy['strategy_type']}")
    print(f"▶ Compounds: {' → '.join(strategy['compounds']) if strategy['compounds'] else 'N/A'}")
    print(f"▶ Pit laps:  {strategy['pit_laps']}")
    print(f"▶ Confidence: {strategy['confidence']}")

    print(f"\n{'─' * 65}")
    print(strategy["recommendation_text"])
    print(f"{'─' * 65}")

    print(f"\n▶ Context summary length: {len(strategy['context_summary'])} chars")
    print("\n" + "=" * 65)
    print("  ✅ Strategy Agent validation complete!")
    print("=" * 65)
