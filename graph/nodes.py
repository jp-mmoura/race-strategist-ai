"""
Node wrappers for the LangGraph workflow.

Each function takes the shared ``RaceStrategyState`` dict and returns a
**partial** state update (only the keys it writes).  This is the standard
LangGraph convention — the framework merges the returned dict back into
the state automatically.

Important
---------
These wrappers *read* upstream results from state rather than re-invoking
the agents internally, which avoids duplicate data fetching (e.g. the
strategy node no longer re-runs tire + weather analysis).
"""

from __future__ import annotations

import logging
import re
from typing import Any

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment & Logging
# ---------------------------------------------------------------------------
load_dotenv()
logger = logging.getLogger(__name__)

# Maximum number of strategy revisions before giving up
MAX_REVISIONS: int = 2


# ===================================================================
# 1. supervisor_node — parse user query
# ===================================================================

def supervisor_node(state: dict[str, Any]) -> dict[str, Any]:
    """Parse the user query to extract circuit, year, and session type.

    Uses simple regex/keyword extraction.  Falls back to sensible
    defaults when a field cannot be resolved.
    """
    messages = state.get("messages", [])
    user_text = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            user_text = msg.get("content", "")
            break

    if not user_text:
        return {"error": "No user message found in state."}

    circuit, year, session_type = _parse_query(user_text)

    updates: dict[str, Any] = {}
    if circuit:
        updates["circuit"] = circuit
    if year:
        updates["year"] = year
    updates["session_type"] = session_type or "R"

    # Initialise revision counter
    updates["revision_count"] = 0

    if not circuit:
        updates["error"] = (
            f"Could not extract a circuit name from: '{user_text}'. "
            "Please specify a circuit (e.g. 'Silverstone 2023')."
        )

    logger.info(
        "Supervisor parsed → circuit=%s, year=%s, session=%s",
        circuit, year, session_type,
    )
    return updates


# -- Query parser (regex-based, no LLM needed) ---------------------------

# Known GP / circuit names (lowercase → canonical)
_CIRCUIT_MAP: dict[str, str] = {
    "silverstone": "Silverstone", "britain": "Silverstone",
    "monza": "Monza", "italy": "Monza", "italian": "Monza",
    "spa": "Spa", "belgium": "Spa", "belgian": "Spa",
    "monaco": "Monaco",
    "bahrain": "Bahrain", "sakhir": "Bahrain",
    "melbourne": "Melbourne", "australia": "Melbourne", "australian": "Melbourne",
    "barcelona": "Barcelona", "spain": "Barcelona", "spanish": "Barcelona",
    "montreal": "Montreal", "canada": "Montreal", "canadian": "Montreal",
    "spielberg": "Spielberg", "austria": "Spielberg", "austrian": "Spielberg",
    "budapest": "Budapest", "hungary": "Budapest", "hungarian": "Budapest",
    "zandvoort": "Zandvoort", "netherlands": "Zandvoort", "dutch": "Zandvoort",
    "singapore": "Singapore",
    "suzuka": "Suzuka", "japan": "Suzuka", "japanese": "Suzuka",
    "austin": "Austin", "cota": "Austin", "usa": "Austin",
    "mexico": "Mexico City", "mexico city": "Mexico City",
    "interlagos": "São Paulo", "sao paulo": "São Paulo", "brazil": "São Paulo",
    "las vegas": "Las Vegas",
    "abu dhabi": "Abu Dhabi", "yas marina": "Abu Dhabi",
    "jeddah": "Jeddah", "saudi": "Jeddah", "saudi arabia": "Jeddah",
    "imola": "Imola",
    "baku": "Baku", "azerbaijan": "Baku",
    "miami": "Miami",
    "shanghai": "Shanghai", "china": "Shanghai", "chinese": "Shanghai",
    "lusail": "Lusail", "qatar": "Lusail",
    # Portuguese / Spanish aliases
    "brasil": "São Paulo", "canadá": "Montreal",
    "espanha": "Barcelona", "hungria": "Budapest",
    "holanda": "Zandvoort", "japão": "Suzuka",
    "cingapura": "Singapore", "singapura": "Singapore",
    "méxico": "Mexico City", "mônaco": "Monaco",
    "itália": "Monza", "bélgica": "Spa",
    "austrália": "Melbourne", "áustria": "Spielberg",
}

_SESSION_MAP: dict[str, str] = {
    "race": "R", "corrida": "R",
    "qualifying": "Q", "quali": "Q", "classificação": "Q",
    "sprint": "S",
    "fp1": "FP1", "practice 1": "FP1", "treino 1": "FP1",
    "fp2": "FP2", "practice 2": "FP2", "treino 2": "FP2",
    "fp3": "FP3", "practice 3": "FP3", "treino 3": "FP3",
}


def _parse_query(text: str) -> tuple[str | None, int | None, str | None]:
    """Extract (circuit, year, session_type) from free text."""
    lower = text.lower().strip()

    # --- Year ---
    year_match = re.search(r"\b(20[0-2]\d)\b", lower)
    year = int(year_match.group(1)) if year_match else None

    # --- Session type ---
    session_type: str | None = None
    for keyword, stype in sorted(_SESSION_MAP.items(), key=lambda x: -len(x[0])):
        if keyword in lower:
            session_type = stype
            break

    # --- Circuit (longest match first to handle multi-word names) ---
    circuit: str | None = None
    for keyword in sorted(_CIRCUIT_MAP, key=len, reverse=True):
        if keyword in lower:
            circuit = _CIRCUIT_MAP[keyword]
            break

    return circuit, year, session_type


# ===================================================================
# 2. tire_node
# ===================================================================

def tire_node(state: dict[str, Any]) -> dict[str, Any]:
    """Run the tire analysis agent."""
    from agents.tire_agent import analyze_tire_strategy

    circuit = state.get("circuit", "")
    year = state.get("year", 2024)
    session_type = state.get("session_type", "R")

    if not circuit:
        return {"tire_analysis": None, "error": "Tire node: no circuit in state."}

    try:
        result = analyze_tire_strategy(circuit, year, session_type)
        logger.info("Tire analysis complete for %s %d", circuit, year)
        return {"tire_analysis": result}
    except Exception as exc:
        logger.error("Tire node failed: %s", exc, exc_info=True)
        return {"tire_analysis": None, "error": f"Tire agent failed: {exc}"}


# ===================================================================
# 3. weather_node
# ===================================================================

def weather_node(state: dict[str, Any]) -> dict[str, Any]:
    """Run the weather analysis agent."""
    from agents.weather_agent import analyze_weather_impact

    circuit = state.get("circuit", "")
    year = state.get("year", 2024)
    session_type = state.get("session_type", "R")

    if not circuit:
        return {"weather_analysis": None, "error": "Weather node: no circuit in state."}

    try:
        result = analyze_weather_impact(circuit, year=year, session_type=session_type)
        logger.info("Weather analysis complete for %s", circuit)
        return {"weather_analysis": result}
    except Exception as exc:
        logger.error("Weather node failed: %s", exc, exc_info=True)
        return {"weather_analysis": None, "error": f"Weather agent failed: {exc}"}


# ===================================================================
# 4. rag_node
# ===================================================================

def rag_node(state: dict[str, Any]) -> dict[str, Any]:
    """Retrieve historical race context via RAG (ChromaDB + FastF1)."""
    from rag.retriever import retrieve_race_context

    circuit = state.get("circuit", "")
    year = state.get("year")
    session_type = state.get("session_type", "R")

    if not circuit:
        return {"rag_context": None, "error": "RAG node: no circuit in state."}

    try:
        result = retrieve_race_context(
            query=f"race strategy {circuit} {year}",
            year=year,
            circuit=circuit,
            session_type=session_type,
        )
        logger.info("RAG retrieval complete for %s %s", circuit, year)
        return {"rag_context": result}
    except Exception as exc:
        logger.error("RAG node failed: %s", exc, exc_info=True)
        return {"rag_context": None, "error": f"RAG retrieval failed: {exc}"}


# ===================================================================
# 5. strategy_node
# ===================================================================

def strategy_node(state: dict[str, Any]) -> dict[str, Any]:
    """Generate a strategy recommendation.

    Reads tire_analysis, weather_analysis, and rag_context from state
    (already populated by upstream nodes) and calls the strategy agent.

    On the first pass the agent generates from scratch; on revision
    passes it receives ``revision_feedback`` from the evaluator.
    """
    from agents.strategist_agent import generate_strategy, generate_strategy_offline

    circuit = state.get("circuit", "")
    year = state.get("year", 2024)
    session_type = state.get("session_type", "R")

    if not circuit:
        return {
            "strategy_recommendation": None,
            "error": "Strategy node: no circuit in state.",
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
# 6. evaluator_node
# ===================================================================

def evaluator_node(state: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the strategy recommendation for coherence."""
    from agents.evaluator_agent import evaluate_strategy

    strategy = state.get("strategy_recommendation")
    if not strategy:
        return {
            "evaluation_result": {
                "score": 0,
                "verdict": "❌ Rejected",
                "findings": [],
                "summary": "No strategy to evaluate.",
                "checks_passed": 0,
                "checks_failed": 0,
                "total_penalty": 0,
            },
            "error": "Evaluator: no strategy_recommendation in state.",
        }

    tire = state.get("tire_analysis")
    weather = state.get("weather_analysis")

    try:
        evaluation = evaluate_strategy(strategy, tire, weather)

        # Build revision feedback from findings for the reflection loop
        feedback_lines: list[str] = []
        for f in evaluation.get("findings", []):
            if f.get("severity") in ("critical", "major"):
                feedback_lines.append(
                    f"[{f['severity'].upper()}] {f['rule']}: {f['message']} "
                    f"— {f.get('detail', '')}"
                )

        revision_feedback = "\n".join(feedback_lines) if feedback_lines else None

        return {
            "evaluation_result": evaluation,
            "revision_feedback": revision_feedback,
        }
    except Exception as exc:
        logger.error("Evaluator node failed: %s", exc, exc_info=True)
        return {
            "evaluation_result": None,
            "error": f"Evaluator agent failed: {exc}",
        }


# ===================================================================
# 7. revision_node  (Reflection loop)
# ===================================================================

def revision_node(state: dict[str, Any]) -> dict[str, Any]:
    """Re-generate strategy incorporating evaluator feedback.

    Increments ``revision_count`` and injects the evaluator's findings
    into the messages so the strategy agent can correct its output.
    """
    from agents.strategist_agent import generate_strategy, generate_strategy_offline

    circuit = state.get("circuit", "")
    year = state.get("year", 2024)
    session_type = state.get("session_type", "R")
    revision_count = state.get("revision_count", 0)
    feedback = state.get("revision_feedback", "")

    logger.info(
        "Revision #%d for %s %d — feedback: %s",
        revision_count + 1, circuit, year,
        feedback[:120] if feedback else "(none)",
    )

    # Inject feedback as a system-level hint
    if feedback:
        extra_messages = state.get("messages", []).copy()
        extra_messages.append({
            "role": "system",
            "content": (
                "The previous strategy was REJECTED by the evaluator. "
                "Address the following issues in your revised strategy:\n\n"
                f"{feedback}"
            ),
        })
    else:
        extra_messages = state.get("messages", [])

    try:
        result = generate_strategy(
            circuit=circuit,
            year=year,
            session_type=session_type,
        )
        return {
            "strategy_recommendation": result,
            "revision_count": revision_count + 1,
            "messages": extra_messages,
            "error": result.get("error"),
        }
    except Exception as exc:
        logger.error("Revision node failed: %s", exc, exc_info=True)
        # Fallback to offline
        try:
            result = generate_strategy_offline(
                circuit=circuit,
                year=year,
                session_type=session_type,
            )
            return {
                "strategy_recommendation": result,
                "revision_count": revision_count + 1,
                "error": f"Revision LLM failed ({exc}); used offline fallback.",
            }
        except Exception as exc2:
            return {
                "strategy_recommendation": None,
                "revision_count": revision_count + 1,
                "error": f"Revision failed completely: {exc2}",
            }
