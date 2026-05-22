"""
LangGraph Workflow — F1 Race Strategist AI.

Defines the ``StateGraph`` with all nodes, conditional edges, and the
**reflection loop** (strategy → evaluate → revise if rejected).

Graph Topology
--------------
::

    START → supervisor → ┬─ tire_node    ─┐
                         ├─ weather_node ─┤→ strategy → evaluator ──┬─→ END
                         └─ rag_node     ─┘       ↑                 │
                                                   │   (score < 45   │
                                                   │    & retries    │
                                                   │    < MAX)       │
                                                   └── revision ←───┘

The evaluator decides whether the strategy is acceptable:
  • score ≥ 45  → END  (Approved or Review — acceptable)
  • score < 45  AND revision_count < MAX_REVISIONS → revision_node
  • score < 45  AND revision_count ≥ MAX_REVISIONS → END (give up)

Usage
-----
>>> from graph.workflow import app
>>> result = app.invoke(initial_state)
"""

from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph

from graph.nodes import (
    MAX_REVISIONS,
    evaluator_node,
    rag_node,
    revision_node,
    strategy_node,
    supervisor_node,
    tire_node,
    weather_node,
)
from graph.state import RaceStrategyState

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ===================================================================
# Conditional edge: should we revise or finish?
# ===================================================================

def _should_revise(state: dict) -> str:
    """Decide the next step after evaluation.

    Returns
    -------
    str
        ``"revision"`` to loop back, or ``"end"`` to finish.
    """
    evaluation = state.get("evaluation_result") or {}
    score = evaluation.get("score", 0)
    revision_count = state.get("revision_count", 0)

    if score >= 45:
        logger.info(
            "Strategy accepted (score=%d, verdict=%s) — finishing.",
            score, evaluation.get("verdict", "?"),
        )
        return "end"

    if revision_count >= MAX_REVISIONS:
        logger.warning(
            "Strategy rejected (score=%d) but max revisions (%d) reached "
            "— finishing with best effort.",
            score, MAX_REVISIONS,
        )
        return "end"

    logger.info(
        "Strategy rejected (score=%d, revision %d/%d) — revising.",
        score, revision_count + 1, MAX_REVISIONS,
    )
    return "revision"


# ===================================================================
# Fan-in gate: wait for all parallel data nodes
# ===================================================================

def _fan_in_gate(state: dict) -> str:
    """After parallel data collection, always proceed to strategy."""
    return "strategy"


# ===================================================================
# Build the StateGraph
# ===================================================================

def build_workflow() -> StateGraph:
    """Construct and return the (uncompiled) StateGraph.

    Call ``.compile()`` on the result to get a runnable graph.
    """
    workflow = StateGraph(RaceStrategyState)

    # ── Register nodes ────────────────────────────────────────────
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("tire", tire_node)
    workflow.add_node("weather", weather_node)
    workflow.add_node("rag", rag_node)
    workflow.add_node("strategy", strategy_node)
    workflow.add_node("evaluator", evaluator_node)
    workflow.add_node("revision", revision_node)

    # ── Edges ─────────────────────────────────────────────────────

    # START → supervisor
    workflow.add_edge(START, "supervisor")

    # supervisor → fan-out to parallel data-collection nodes
    workflow.add_edge("supervisor", "tire")
    workflow.add_edge("supervisor", "weather")
    workflow.add_edge("supervisor", "rag")

    # Data nodes → strategy (fan-in)
    workflow.add_edge("tire", "strategy")
    workflow.add_edge("weather", "strategy")
    workflow.add_edge("rag", "strategy")

    # strategy → evaluator
    workflow.add_edge("strategy", "evaluator")

    # evaluator → conditional: end or revision
    workflow.add_conditional_edges(
        "evaluator",
        _should_revise,
        {
            "end": END,
            "revision": "revision",
        },
    )

    # revision → evaluator (loop back)
    workflow.add_edge("revision", "evaluator")

    return workflow


# ===================================================================
# Compiled graph — ready to use
# ===================================================================

_workflow = build_workflow()
app = _workflow.compile()

logger.info("LangGraph workflow compiled successfully.")
