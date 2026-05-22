"""
Supervisor Agent — orchestration entry point for the F1 Race Strategist.

This module serves two purposes:

1. **As a LangGraph node** (``supervisor_node``) — imported and used by
   ``graph/nodes.py`` for the query-parsing step.

2. **As a convenience entry point** (``run_graph``) — accepts a
   natural-language query string, builds the initial state, invokes the
   compiled LangGraph workflow, and returns the final state.

Usage
-----
>>> from agents.supervisor import run_graph
>>> result = run_graph("Silverstone 2023 race strategy")
>>> print(result["evaluation_result"]["verdict"])
'✅ Approved'
"""

from __future__ import annotations

import logging
from typing import Any

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment & Logging
# ---------------------------------------------------------------------------
load_dotenv()
logger = logging.getLogger(__name__)


# ===================================================================
# run_graph — high-level convenience function
# ===================================================================

def run_graph(
    user_query: str,
    *,
    circuit: str | None = None,
    year: int | None = None,
    session_type: str | None = None,
) -> dict[str, Any]:
    """Run the full LangGraph workflow from a user query.

    Parameters
    ----------
    user_query : str
        Natural-language query (e.g. ``"Silverstone 2023 race strategy"``).
    circuit : str | None
        Override circuit (skips parsing from ``user_query``).
    year : int | None
        Override year.
    session_type : str | None
        Override session type (``"R"``, ``"Q"``, ``"FP1"``, etc.).

    Returns
    -------
    dict
        The final ``RaceStrategyState`` after the workflow completes.
    """
    from graph.workflow import app

    # Build initial state with the user message
    initial_state: dict[str, Any] = {
        "messages": [{"role": "user", "content": user_query}],
        "revision_count": 0,
    }

    # Allow explicit overrides (bypass supervisor parsing)
    if circuit:
        initial_state["circuit"] = circuit
    if year:
        initial_state["year"] = year
    if session_type:
        initial_state["session_type"] = session_type

    logger.info("Starting workflow for query: %s", user_query[:80])

    result = app.invoke(initial_state)

    # Log final outcome
    evaluation = result.get("evaluation_result") or {}
    logger.info(
        "Workflow complete — score=%s, verdict=%s, revisions=%d",
        evaluation.get("score", "?"),
        evaluation.get("verdict", "?"),
        result.get("revision_count", 0),
    )

    return result


# ===================================================================
# Convenience: print a summary of the workflow result
# ===================================================================

def print_result_summary(result: dict[str, Any]) -> None:
    """Pretty-print the key outputs from a workflow run."""
    print("=" * 65)
    print("  F1 Race Strategist AI — Result Summary")
    print("=" * 65)

    print(f"\n  Circuit:  {result.get('circuit', '?')}")
    print(f"  Year:     {result.get('year', '?')}")
    print(f"  Session:  {result.get('session_type', '?')}")
    print(f"  Revisions: {result.get('revision_count', 0)}")

    strategy = result.get("strategy_recommendation") or {}
    print(f"\n▶ Strategy: {strategy.get('strategy_type', 'N/A')}")
    compounds = strategy.get("compounds", [])
    if compounds:
        print(f"  Compounds: {' → '.join(compounds)}")
    pit_laps = strategy.get("pit_laps", [])
    if pit_laps:
        print(f"  Pit laps:  {pit_laps}")
    print(f"  Confidence: {strategy.get('confidence', 'N/A')}")

    evaluation = result.get("evaluation_result") or {}
    print(f"\n▶ Evaluation:")
    print(f"  Score:   {evaluation.get('score', '?')}/100")
    print(f"  Verdict: {evaluation.get('verdict', '?')}")
    print(f"  Passed:  {evaluation.get('checks_passed', '?')} | "
          f"Failed: {evaluation.get('checks_failed', '?')}")

    summary = evaluation.get("summary", "")
    if summary:
        print(f"\n{summary}")

    if result.get("error"):
        print(f"\n⚠ Errors: {result['error']}")

    print("\n" + "=" * 65)


# ===================================================================
# CLI entry point
# ===================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("=" * 65)
    print("  F1 Race Strategist AI — Supervisor")
    print("=" * 65)

    query = "Silverstone 2023 race strategy"
    print(f"\n▶ Running workflow for: '{query}'")
    print("  (This may take a minute — loading FastF1 data...)\n")

    result = run_graph(query)
    print_result_summary(result)
