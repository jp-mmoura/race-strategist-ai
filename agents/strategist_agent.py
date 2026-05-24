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

generate_strategy_from_context(tire_analysis, weather_analysis, rag_context, ...)
    → primary LLM entry point used by strategy_node and revision_node;
      accepts pre-computed state data to avoid duplicate API calls (Bug B fix)

generate_strategy(circuit, year, race_date, driver)
    → standalone LLM recommendation (fetches its own context via
      build_strategy_context; use generate_strategy_from_context inside the graph)

generate_strategy_offline(circuit, year, race_date, driver)
    → rule-based fallback (no LLM required)
"""

from __future__ import annotations

import logging
import os
import textwrap
from datetime import date
from typing import Any

from dotenv import load_dotenv

from agents.tire_agent import analyze_tire_strategy
from agents.weather_agent import analyze_weather_impact
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

_VALID_PROVIDERS = ("google", "openai")


def _get_llm():
    """Instantiate the configured LLM (Google Gemini or OpenAI)."""
    if _LLM_PROVIDER not in _VALID_PROVIDERS:
        raise ValueError(
            f"Unsupported STRATEGY_LLM_PROVIDER={_LLM_PROVIDER!r}. "
            f"Valid options: {_VALID_PROVIDERS}."
        )
    if _LLM_PROVIDER == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=_LLM_MODEL,
            temperature=_LLM_TEMPERATURE,
        )
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=_LLM_MODEL,
        temperature=_LLM_TEMPERATURE,
    )


# ===================================================================
# System prompt for the strategy LLM
# ===================================================================

_SYSTEM_PROMPT = textwrap.dedent("""\
    You are the Chief Race Strategist for an F1 team.  Your role is to
    synthesise data from the Tire Engineer, Weather Analyst, and
    Historical Database into a single, actionable race strategy.

    Always structure your response with the following sections:

    ## Recommended Strategy
    State the strategy type (1-stop, 2-stop, etc.), the compound order,
    and the target pit-stop laps.

    ## Compound Selection
    Explain which compounds to use in each stint and why.

    ## Pit Windows
    Specify the optimal, earliest, and latest pit-stop laps for each stop.

    ## Weather Contingency
    Describe the plan if weather changes (rain, temperature shift).

    ## Risk Assessment
    Identify the top risks and how to mitigate them.

    ## Justification
    Summarise the key data points that support this strategy (tire
    degradation rates, historical precedent, forecast conditions).

    Be precise with lap numbers.  Use bullet points and keep the total
    response under 600 words.
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
) -> dict[str, Any]:
    """Generate a full race-strategy recommendation using an LLM.

    Parameters
    ----------
    circuit, year, race_date, driver, session_type
        Passed through to ``build_strategy_context``.

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

    # ── 1. Build context ──────────────────────────────────────────
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
        # Fallback to rule-based
        offline = generate_strategy_offline(
            circuit, year, race_date, driver, session_type,
        )
        result["recommendation_text"] = offline["recommendation_text"]
        result["confidence"] = "low"
        result["error"] = f"LLM unavailable ({exc}); used rule-based fallback."

    return result


# ===================================================================
# 3. _build_offline_from_ctx — shared rule-based text builder
# ===================================================================

def _build_offline_from_ctx(
    ctx: dict[str, Any],
    circuit: str,
    year: int,
    driver: str | None = None,
) -> dict[str, Any]:
    """Build a rule-based strategy result from a pre-assembled context dict.

    Contains all offline recommendation logic. Shared by
    ``generate_strategy_offline`` and the LLM fallback inside
    ``generate_strategy_from_context`` so the logic is never duplicated.
    """
    tire = ctx.get("tire_analysis") or {}
    weather = ctx.get("weather_analysis") or {}
    rain = weather.get("rain_risk") or {}
    temp = weather.get("temperature") or {}
    pw = tire.get("pit_window") or {}
    tw = tire.get("track_wear") or {}
    rec = tire.get("compound_rec") or {}
    deg = tire.get("degradation") or []

    strategy_type = pw.get("strategy_type", "Unknown")
    compounds = rec.get("recommended_order", [])
    pit_laps = pw.get("recommended_pit_laps", [])
    pit_windows = pw.get("pit_windows", [])
    total_laps = pw.get("total_laps", 0)
    risk_level = rain.get("risk_level", "Unknown")
    wear_class = tw.get("classification", "Unknown")
    avg_temp = temp.get("air_temp_avg_c")
    track_temp_est = temp.get("track_temp_est_c")

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

    lines.append("\n## Justification")
    justification: list[str] = []
    justification.append(
        f"Track classification: {wear_class} "
        f"(score {tw.get('score', '?')}/5)."
    )
    if deg:
        avg_deg = sum(d["deg_rate_sec_per_lap"] for d in deg) / len(deg)
        justification.append(
            f"Average degradation rate: {avg_deg:+.4f} s/lap "
            f"across {len(deg)} stint(s)."
        )
    if avg_temp is not None:
        justification.append(
            f"Forecast air temp: {avg_temp:.1f} °C "
            f"(est. track: {track_temp_est:.1f} °C)."
        )
    justification.append(f"Rain risk: {risk_level}.")
    if rec.get("confidence"):
        justification.append(
            f"Compound order confidence: {rec['confidence']}."
        )
    for j in justification:
        lines.append(f"- {j}")

    recommendation_text = "\n".join(lines)

    return {
        "circuit": circuit,
        "year": year,
        "driver": driver or pw.get("driver"),
        "strategy_type": strategy_type,
        "compounds": compounds,
        "pit_laps": pit_laps,
        "recommendation_text": recommendation_text,
        "context_summary": ctx.get("context_text", ""),
        "confidence": "medium" if not ctx.get("error") else "low",
        "error": ctx.get("error"),
    }


# ===================================================================
# 4. generate_strategy_offline — rule-based fallback
# ===================================================================

def generate_strategy_offline(
    circuit: str,
    year: int,
    race_date: str | date | None = None,
    driver: str | None = None,
    session_type: str = "R",
) -> dict[str, Any]:
    """Generate a strategy recommendation without an LLM.

    Uses deterministic rules based on the tire and weather data.
    Useful as a fallback when no API key is configured.

    Returns
    -------
    dict with the same keys as ``generate_strategy``.
    """
    ctx = build_strategy_context(circuit, year, race_date, driver, session_type)
    result = _build_offline_from_ctx(ctx, circuit, year, driver)
    logger.info(
        "Offline strategy generated for %s %d (%d chars)",
        circuit, year, len(result["recommendation_text"]),
    )
    return result


# ===================================================================
# 5. generate_strategy_from_context — uses pre-built upstream data
# ===================================================================

def generate_strategy_from_context(
    tire_analysis: dict[str, Any] | None,
    weather_analysis: dict[str, Any] | None,
    rag_context: dict[str, Any] | None,
    circuit: str,
    year: int,
    session_type: str = "R",
    driver: str | None = None,
    revision_feedback: str | None = None,
) -> dict[str, Any]:
    """Generate a strategy from data already computed by upstream nodes.

    Unlike ``generate_strategy()``, this function does **not** call
    ``analyze_tire_strategy``, ``analyze_weather_impact``, or
    ``retrieve_race_context``.  It assembles the LLM prompt directly
    from the dicts produced by the tire/weather/rag graph nodes,
    eliminating the duplicate API calls that ``generate_strategy``
    would otherwise trigger.

    Parameters
    ----------
    tire_analysis : dict | None
        Output of ``analyze_tire_strategy`` (from tire_node in state).
    weather_analysis : dict | None
        Output of ``analyze_weather_impact`` (from weather_node in state).
    rag_context : dict | None
        Output of ``retrieve_race_context`` (from rag_node in state).
    circuit, year, session_type, driver
        Race identifiers forwarded to the LLM prompt.
    revision_feedback : str | None
        Evaluator findings from a previous rejected strategy.  When
        present, injected as a second system message so the LLM knows
        exactly which issues to fix in the revised output.

    Returns
    -------
    dict with the same keys as ``generate_strategy``.
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

    # Collect partial errors from upstream agents without discarding them
    upstream_errors: list[str] = []
    if tire_analysis and tire_analysis.get("error"):
        upstream_errors.append(f"Tire: {tire_analysis['error']}")
    if weather_analysis and weather_analysis.get("error"):
        upstream_errors.append(f"Weather: {weather_analysis['error']}")
    if rag_context and rag_context.get("error"):
        upstream_errors.append(f"RAG: {rag_context['error']}")

    # Build context dict from pre-computed data — no re-fetching
    ctx: dict[str, Any] = {
        "tire_analysis": tire_analysis,
        "weather_analysis": weather_analysis,
        "rag_context": rag_context,
        "error": "; ".join(upstream_errors) or None,
    }
    ctx["context_text"] = _assemble_context_text(ctx)
    result["context_summary"] = ctx["context_text"]

    # Extract structured fields from tire analysis
    tire = tire_analysis or {}
    pw = tire.get("pit_window") or {}
    rec = tire.get("compound_rec") or {}

    if driver is None:
        driver = pw.get("driver")
        result["driver"] = driver

    result["strategy_type"] = pw.get("strategy_type")
    result["pit_laps"] = pw.get("recommended_pit_laps", [])
    result["compounds"] = rec.get("recommended_order", [])

    # Call LLM
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

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
        ]
        if revision_feedback:
            messages.append(
                SystemMessage(
                    content=(
                        "IMPORTANT — the previous strategy was REJECTED by the "
                        "evaluator.  You MUST address every issue listed below "
                        "in your revised strategy:\n\n"
                        f"{revision_feedback}"
                    )
                )
            )
        messages.append(HumanMessage(content=user_prompt))

        response = llm.invoke(messages)
        result["recommendation_text"] = response.content
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
        # Fallback uses the same pre-built ctx — still no re-fetching
        offline = _build_offline_from_ctx(ctx, circuit, year, driver)
        result["recommendation_text"] = offline["recommendation_text"]
        result["confidence"] = "low"
        result["error"] = f"LLM unavailable ({exc}); used rule-based fallback."

    return result


# ===================================================================
# 6. run_strategy_node — LangGraph node entry-point
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
