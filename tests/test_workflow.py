"""
Integration tests for the LangGraph workflow.

All external agents (FastF1, Open-Meteo, ChromaDB, LLM) are mocked with
fixed responses.  The tests verify:

    TestWorkflowHappyPath          — full graph runs and produces expected outputs
    TestBugBStrategyReadsFromState — strategy_node reads state, not re-fetches data
    TestBugCRevisionPassesFeedback — revision_node passes feedback to the LLM
    TestBugAFanInAndStrategyCounts — fan_in fires once; strategy fires once

Run with:
    pytest tests/test_workflow.py -v
"""
from __future__ import annotations

import sys
from contextlib import ExitStack
from unittest.mock import MagicMock

# ===================================================================
# Fixed test data
# ===================================================================

FIXED_TIRE: dict = {
    "track_wear": {"classification": "Low Tire Wear", "score": 2.0, "track": "Monaco"},
    "degradation": [
        {
            "deg_rate_sec_per_lap": 0.025, "lap_count": 30, "stint": 1,
            "compound": "MEDIUM", "start_lap": 1, "end_lap": 30,
        },
        {
            "deg_rate_sec_per_lap": 0.018, "lap_count": 48, "stint": 2,
            "compound": "HARD", "start_lap": 31, "end_lap": 78,
        },
    ],
    "pit_window": {
        "total_laps": 78,
        "recommended_pit_laps": [30],
        "strategy_type": "1-stop",
        "driver": "LEC",
        "pit_windows": [
            {"earliest": 25, "optimal": 30, "latest": 36,
             "compound": "MEDIUM", "deg_rate": 0.025},
        ],
    },
    "compound_rec": {
        "recommended_order": ["MEDIUM", "HARD"],
        "confidence": "2/3 top-3 used this",
    },
    "stints_summary": [],
    "weather_impact": None,
    "error": None,
}

FIXED_WEATHER: dict = {
    "circuit": "Monaco",
    "race_date": "N/A",
    "current_conditions": None,
    "forecast": None,
    "rain_risk": {
        "risk_level": "None",
        "max_precip_prob": 5.0,
        "total_rain_mm": 0.0,
        "wet_hours": 0,
        "summary": "Rain risk: None. Dry conditions expected.",
        "rain_windows": [],
    },
    "temperature": {
        "air_temp_min_c": 20.0, "air_temp_max_c": 24.0,
        "air_temp_avg_c": 22.0, "track_temp_est_c": 37.0,
        "note": "Moderate temperatures.",
    },
    "wind": {
        "avg_speed_kmh": 10.0, "max_speed_kmh": 18.0,
        "max_gusts_kmh": 22.0, "note": "Light winds.",
    },
    "strategy_notes": ["DRY CONDITIONS — standard dry-weather strategy applies."],
    "historical_comparison": None,
    "error": None,
}

FIXED_RAG: dict = {
    "context_text": (
        "Monaco 2023: Verstappen 1-stop MEDIUM->HARD, pit lap 29.\n"
        "Monaco 2022: Perez 1-stop MEDIUM->HARD, pit lap 32."
    ),
    "sources": ["fastf1://monaco/2023/race", "fastf1://monaco/2022/race"],
    "num_results": 2,
    "error": None,
}

FIXED_STRATEGY: dict = {
    "circuit": "Monaco",
    "year": 2024,
    "driver": "LEC",
    "strategy_type": "1-stop",
    "compounds": ["MEDIUM", "HARD"],
    "pit_laps": [30],
    "recommendation_text": "## Recommended Strategy\n1-stop over 78 laps. MEDIUM->HARD, pit lap 30.",
    "context_summary": "Tire: Low wear. Weather: dry. History: 1-stop favoured.",
    "confidence": "high",
    "error": None,
}

EVAL_APPROVED: dict = {
    "score": 85,
    "verdict": "Approved",
    "findings": [],
    "summary": "Score: 85/100. All checks passed.",
    "checks_passed": 8,
    "checks_failed": 0,
    "total_penalty": 0,
}

EVAL_REJECTED: dict = {
    "score": 20,
    "verdict": "Rejected",
    "findings": [
        {
            "rule": "DRY_UNDER_RAIN",
            "severity": "critical",
            "message": "All-dry compound strategy under HIGH rain risk.",
            "detail": "No wet contingency — add intermediate/wet to the compound plan.",
            "penalty": 20,
        }
    ],
    "summary": "Score: 20/100. Critical: DRY_UNDER_RAIN.",
    "checks_passed": 7,
    "checks_failed": 1,
    "total_penalty": 20,
}

INITIAL_STATE: dict = {
    "messages": [{"role": "user", "content": "Estrategia para Monaco 2024"}]
}


# ===================================================================
# Mock helper
# ===================================================================

def _enter_mocks(stack: ExitStack, *, eval_return=None, eval_side_effect=None):
    """Pre-populate sys.modules with lightweight mock agent modules.

    Python 3.13 changed unittest.mock.patch to use pkgutil.resolve_name,
    which does getattr() at each dotted step without falling back to
    importlib.  Since agents/__init__.py is empty and the real submodules
    have unavailable dependencies (fastf1, chromadb, openmeteo_requests),
    direct patching fails with AttributeError.

    Installing MagicMock instances directly into sys.modules bypasses the
    resolution issue: when node functions do
        from agents.tire_agent import analyze_tire_strategy
    Python finds the mock module in sys.modules and returns the mock
    attribute, without ever touching the real file.

    Returns (m_tire, m_weather, m_rag, m_gen, m_eval) as MagicMock
    function objects whose .call_count and .call_args_list are inspectable
    after the ExitStack closes.
    """
    # Build mock modules
    tire_mod = MagicMock()
    tire_mod.analyze_tire_strategy.return_value = FIXED_TIRE

    weather_mod = MagicMock()
    weather_mod.analyze_weather_impact.return_value = FIXED_WEATHER

    rag_mod = MagicMock()
    rag_mod.retrieve_race_context.return_value = FIXED_RAG

    gen_mod = MagicMock()
    gen_mod.generate_strategy_from_context.return_value = FIXED_STRATEGY

    eval_mod = MagicMock()
    if eval_side_effect is not None:
        eval_mod.evaluate_strategy.side_effect = eval_side_effect
    else:
        eval_mod.evaluate_strategy.return_value = (
            eval_return if eval_return is not None else EVAL_APPROVED
        )

    mapping = [
        ("agents.tire_agent", tire_mod),
        ("agents.weather_agent", weather_mod),
        ("rag.retriever", rag_mod),
        ("agents.strategist_agent", gen_mod),
        ("agents.evaluator_agent", eval_mod),
    ]

    # Save originals and install mocks
    saved = {name: sys.modules.get(name) for name, _ in mapping}
    for name, mod in mapping:
        sys.modules[name] = mod

    def _restore():
        for name, original in saved.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original

    stack.callback(_restore)

    return (
        tire_mod.analyze_tire_strategy,
        weather_mod.analyze_weather_impact,
        rag_mod.retrieve_race_context,
        gen_mod.generate_strategy_from_context,
        eval_mod.evaluate_strategy,
    )


# ===================================================================
# 1. Happy-path smoke test
# ===================================================================

class TestWorkflowHappyPath:
    """Full graph runs end-to-end with a strategy approved on the first attempt."""

    def test_graph_completes_and_returns_state(self):
        from graph.workflow import app

        with ExitStack() as stack:
            _enter_mocks(stack)
            result = app.invoke(INITIAL_STATE)

        assert result is not None
        assert result.get("strategy_recommendation") is not None

    def test_circuit_and_year_parsed_from_message(self):
        from graph.workflow import app

        with ExitStack() as stack:
            _enter_mocks(stack)
            result = app.invoke(INITIAL_STATE)

        assert result.get("circuit") == "Monaco"
        assert result.get("year") == 2024

    def test_all_analysis_fields_present_in_final_state(self):
        from graph.workflow import app

        with ExitStack() as stack:
            _enter_mocks(stack)
            result = app.invoke(INITIAL_STATE)

        assert result.get("tire_analysis") == FIXED_TIRE
        assert result.get("weather_analysis") == FIXED_WEATHER
        assert result.get("rag_context") == FIXED_RAG

    def test_evaluation_result_present_in_final_state(self):
        from graph.workflow import app

        with ExitStack() as stack:
            _enter_mocks(stack)
            result = app.invoke(INITIAL_STATE)

        evaluation = result.get("evaluation_result") or {}
        assert evaluation.get("score") == 85


# ===================================================================
# 2. Bug B — strategy_node must not re-fetch data
# ===================================================================

class TestBugBStrategyReadsFromState:
    """
    Bug B: the old strategy_node called generate_strategy() which internally
    re-ran build_strategy_context(), duplicating FastF1 / Open-Meteo requests.

    The fix uses generate_strategy_from_context() and reads pre-computed
    tire_analysis, weather_analysis, and rag_context directly from state.

    Detection: after app.invoke(), verify that each data-collection agent
    was called exactly once and that generate_strategy_from_context received
    the pre-computed dicts — not None or re-fetched values.
    """

    def _run(self):
        from graph.workflow import app

        with ExitStack() as stack:
            m_tire, m_weather, m_rag, m_gen, m_eval = _enter_mocks(stack)
            result = app.invoke(INITIAL_STATE)

        return result, m_tire, m_weather, m_rag, m_gen, m_eval

    def test_tire_agent_called_exactly_once(self):
        _, m_tire, *_ = self._run()
        assert m_tire.call_count == 1, (
            f"analyze_tire_strategy called {m_tire.call_count} times; "
            "strategy_node must not re-invoke it."
        )

    def test_weather_agent_called_exactly_once(self):
        _, _, m_weather, *_ = self._run()
        assert m_weather.call_count == 1, (
            f"analyze_weather_impact called {m_weather.call_count} times."
        )

    def test_rag_called_exactly_once(self):
        _, _, _, m_rag, *_ = self._run()
        assert m_rag.call_count == 1, (
            f"retrieve_race_context called {m_rag.call_count} times."
        )

    def test_generate_strategy_received_tire_analysis_from_state(self):
        _, _, _, _, m_gen, _ = self._run()
        kwargs = m_gen.call_args.kwargs
        assert kwargs.get("tire_analysis") == FIXED_TIRE, (
            "strategy_node did not pass tire_analysis from state to "
            "generate_strategy_from_context."
        )

    def test_generate_strategy_received_weather_analysis_from_state(self):
        _, _, _, _, m_gen, _ = self._run()
        kwargs = m_gen.call_args.kwargs
        assert kwargs.get("weather_analysis") == FIXED_WEATHER, (
            "strategy_node did not pass weather_analysis from state to "
            "generate_strategy_from_context."
        )


# ===================================================================
# 3. Bug C — revision_node must pass feedback to the LLM
# ===================================================================

class TestBugCRevisionPassesFeedback:
    """
    Bug C: the old revision_node built extra_messages but never forwarded
    them to the LLM call, making the reflection loop ineffective.

    The fix passes revision_feedback= to generate_strategy_from_context,
    which injects it as a second SystemMessage so the LLM knows what to fix.

    Detection: configure the evaluator to reject on the first call and
    approve on the second, then verify:
      - generate_strategy_from_context is called twice (initial + revision)
      - the first call has revision_feedback=None
      - the second call has revision_feedback containing the rejected rule
      - revision_count is incremented to 1
    """

    def _run_with_rejection(self):
        from graph.workflow import app

        with ExitStack() as stack:
            m_tire, m_weather, m_rag, m_gen, m_eval = _enter_mocks(
                stack, eval_side_effect=[EVAL_REJECTED, EVAL_APPROVED]
            )
            result = app.invoke(INITIAL_STATE)

        return result, m_gen, m_eval

    def test_generate_strategy_called_twice_on_rejection(self):
        _, m_gen, _ = self._run_with_rejection()
        assert m_gen.call_count == 2, (
            f"Expected 2 calls (initial + revision), got {m_gen.call_count}."
        )

    def test_first_call_has_no_revision_feedback(self):
        _, m_gen, _ = self._run_with_rejection()
        first_kwargs = m_gen.call_args_list[0].kwargs
        assert first_kwargs.get("revision_feedback") is None, (
            "Initial strategy_node call must not carry revision_feedback."
        )

    def test_revision_call_includes_feedback_text(self):
        _, m_gen, _ = self._run_with_rejection()
        second_kwargs = m_gen.call_args_list[1].kwargs
        feedback = second_kwargs.get("revision_feedback")
        assert feedback is not None, (
            "revision_node did not pass revision_feedback to "
            "generate_strategy_from_context — Bug C not fixed."
        )
        assert "DRY_UNDER_RAIN" in feedback, (
            f"Expected rejected rule in feedback string, got: {feedback!r}"
        )

    def test_revision_count_incremented_after_rejection(self):
        result, _, _ = self._run_with_rejection()
        assert result.get("revision_count") == 1


# ===================================================================
# 4. Bug A — fan_in fires once; strategy fires once
# ===================================================================

class TestBugAFanInAndStrategyCounts:
    """
    Bug A: without fan_in_node the three parallel data edges (tire/weather/rag)
    each independently triggered strategy_node, running it three times with a
    partially-populated state.

    The fix inserts fan_in as a synchronisation barrier so strategy fires
    exactly once with a fully-populated state.

    Detection: generate_strategy_from_context must be called exactly once
    in the happy path (not 3x, which would indicate the missing barrier).
    """

    def test_strategy_node_fires_exactly_once_happy_path(self):
        from graph.workflow import app

        with ExitStack() as stack:
            *_, m_gen, _ = _enter_mocks(stack)
            app.invoke(INITIAL_STATE)

        assert m_gen.call_count == 1, (
            f"generate_strategy_from_context called {m_gen.call_count} times; "
            "fan_in_node must ensure it fires exactly once per query."
        )

    def test_evaluator_fires_exactly_once_happy_path(self):
        from graph.workflow import app

        with ExitStack() as stack:
            *_, m_eval = _enter_mocks(stack)
            app.invoke(INITIAL_STATE)

        assert m_eval.call_count == 1

    def test_strategy_fires_once_before_revision_not_three_times(self):
        """Even in the revision path the initial strategy fires once, not 3x."""
        from graph.workflow import app

        with ExitStack() as stack:
            *_, m_gen, _ = _enter_mocks(
                stack, eval_side_effect=[EVAL_REJECTED, EVAL_APPROVED]
            )
            app.invoke(INITIAL_STATE)

        # 1 initial + 1 revision = 2; if Bug A were present it would be 3 + 1 = 4
        assert m_gen.call_count == 2

    def test_parallel_data_nodes_each_fire_once(self):
        """tire, weather, and rag nodes each run exactly once — collected by fan_in."""
        from graph.workflow import app

        with ExitStack() as stack:
            m_tire, m_weather, m_rag, *_ = _enter_mocks(stack)
            app.invoke(INITIAL_STATE)

        assert m_tire.call_count == 1
        assert m_weather.call_count == 1
        assert m_rag.call_count == 1
