"""
Unit tests for agent business logic.

Run with:
    pytest tests/test_agents.py -v
"""
from __future__ import annotations

import math

import pandas as pd
import pytest
from unittest.mock import MagicMock, patch


# ===================================================================
# 1. evaluate_strategy — score always in [0, 100]
# ===================================================================

class TestEvaluateStrategyScoreBounds:
    """
    evaluate_strategy must always return 0 <= score <= 100 regardless
    of how many penalty-generating findings are triggered.
    No FastF1 or network calls — pure data-dict evaluation.
    """

    @pytest.fixture
    def good_strategy(self):
        return {
            "strategy_type": "1-stop",
            "compounds": ["HARD", "MEDIUM"],
            "pit_laps": [28],
        }

    @pytest.fixture
    def good_tire(self):
        return {
            "track_wear": {"classification": "Low Tire Wear", "score": 1.5},
            "degradation": [
                {
                    "deg_rate_sec_per_lap": 0.03,
                    "lap_count": 27,
                    "stint": 1,
                    "compound": "HARD",
                    "start_lap": 1,
                    "end_lap": 27,
                },
                {
                    "deg_rate_sec_per_lap": 0.02,
                    "lap_count": 25,
                    "stint": 2,
                    "compound": "MEDIUM",
                    "start_lap": 28,
                    "end_lap": 52,
                },
            ],
            "pit_window": {
                "total_laps": 52,
                "pit_windows": [
                    {"earliest": 23, "optimal": 27, "latest": 33, "compound": "HARD"},
                ],
            },
        }

    @pytest.fixture
    def good_weather(self):
        return {
            "rain_risk": {
                "risk_level": "None",
                "max_precip_prob": 0.0,
                "total_rain_mm": 0.0,
            },
            "temperature": {"track_temp_est_c": 35},
        }

    @pytest.fixture
    def bad_strategy(self):
        # 2× SOFT on high-wear circuit under heavy rain, pit laps far outside windows.
        # Accumulates exactly 100 penalty points so score floors at 0:
        #   SOFT_HIGH_WEAR (critical=20) + DRY_UNDER_RAIN (critical=20)
        #   + 2× PIT_WINDOW_MISS (major=12 each) + DEG_VS_STOPS (major=12)
        #   + SOFT_START (major=12) + TEMP_COMPOUND (major=12) = 100 pts
        return {
            "strategy_type": "1-stop",
            "compounds": ["SOFT", "SOFT"],
            "pit_laps": [5, 8],
        }

    @pytest.fixture
    def bad_tire(self):
        return {
            "track_wear": {"classification": "High Tire Wear", "score": 4.5},
            "degradation": [
                {
                    "deg_rate_sec_per_lap": 0.12,
                    "lap_count": 25,
                    "stint": 1,
                    "compound": "SOFT",
                    "start_lap": 1,
                    "end_lap": 25,
                },
                {
                    "deg_rate_sec_per_lap": 0.14,
                    "lap_count": 27,
                    "stint": 2,
                    "compound": "SOFT",
                    "start_lap": 26,
                    "end_lap": 52,
                },
            ],
            "pit_window": {
                "total_laps": 52,
                "pit_windows": [
                    {"earliest": 18, "optimal": 22, "latest": 28, "compound": "SOFT"},
                    {"earliest": 35, "optimal": 40, "latest": 45, "compound": "SOFT"},
                ],
            },
        }

    @pytest.fixture
    def bad_weather(self):
        return {
            "rain_risk": {
                "risk_level": "High",
                "max_precip_prob": 85.0,
                "total_rain_mm": 8.5,
            },
            "temperature": {"track_temp_est_c": 55},
        }

    def test_good_strategy_score_is_high_and_approved(
        self, good_strategy, good_tire, good_weather
    ):
        from agents.evaluator_agent import evaluate_strategy

        result = evaluate_strategy(good_strategy, good_tire, good_weather)
        score = result["score"]

        assert 0 <= score <= 100
        assert score >= 75, f"Good strategy should be Approved; got score={score}"
        assert result["verdict"] == "✅ Approved"

    def test_bad_strategy_score_not_negative(
        self, bad_strategy, bad_tire, bad_weather
    ):
        from agents.evaluator_agent import evaluate_strategy

        result = evaluate_strategy(bad_strategy, bad_tire, bad_weather)
        score = result["score"]

        assert score >= 0, f"score must never be negative; got {score}"
        assert score <= 100

    def test_bad_strategy_is_not_approved(
        self, bad_strategy, bad_tire, bad_weather
    ):
        from agents.evaluator_agent import evaluate_strategy

        result = evaluate_strategy(bad_strategy, bad_tire, bad_weather)
        assert result["score"] < 75, (
            "Strategy with multiple critical violations should not be Approved; "
            f"got score={result['score']}"
        )

    def test_maximally_bad_strategy_floors_at_zero(
        self, bad_strategy, bad_tire, bad_weather
    ):
        from agents.evaluator_agent import evaluate_strategy

        result = evaluate_strategy(bad_strategy, bad_tire, bad_weather)
        assert result["score"] == 0, (
            f"Expected score=0 (total penalty = 100 pts); got {result['score']}. "
            f"total_penalty={result['total_penalty']}"
        )

    def test_empty_inputs_yield_perfect_score(self):
        from agents.evaluator_agent import evaluate_strategy

        result = evaluate_strategy(
            {"strategy_type": "1-stop", "compounds": ["HARD"], "pit_laps": []},
            None,
            None,
        )
        assert result["score"] == 100
        assert result["checks_passed"] == 8
        assert result["checks_failed"] == 0

    def test_verdict_approved_threshold(self):
        from agents.evaluator_agent import evaluate_strategy

        result = evaluate_strategy(
            {"strategy_type": "1-stop", "compounds": ["HARD"], "pit_laps": []},
            None,
            None,
        )
        assert result["verdict"] == "✅ Approved"

    def test_verdict_rejected_threshold(
        self, bad_strategy, bad_tire, bad_weather
    ):
        from agents.evaluator_agent import evaluate_strategy

        result = evaluate_strategy(bad_strategy, bad_tire, bad_weather)
        assert result["verdict"] == "❌ Rejected", (
            f"Score {result['score']} should map to Rejected; got {result['verdict']}"
        )


# ===================================================================
# 2. checks_passed never negative — regression for Bug 3.6
# ===================================================================

class TestChecksPassedNeverNegative:
    """
    Bug 3.6: checks_passed = 8 - checks_failed becomes negative when
    more than 8 findings are raised.  With 6 pit-window misses plus 5
    other rule violations the total is 11 findings → 8 - 11 = -3
    without the max(0, …) guard in the result dict.
    """

    @pytest.fixture
    def overloaded_strategy(self):
        # 6 pit laps at laps 1-6, all far outside their respective windows
        return {
            "strategy_type": "1-stop",
            "compounds": ["SOFT", "SOFT"],
            "pit_laps": [1, 2, 3, 4, 5, 6],
        }

    @pytest.fixture
    def overloaded_tire(self):
        return {
            "track_wear": {"classification": "High Tire Wear", "score": 4.8},
            "degradation": [
                {
                    "deg_rate_sec_per_lap": 0.15,
                    "lap_count": 20,
                    "stint": 1,
                    "compound": "SOFT",
                    "start_lap": 1,
                    "end_lap": 20,
                },
                {
                    "deg_rate_sec_per_lap": 0.16,
                    "lap_count": 40,
                    "stint": 2,
                    "compound": "SOFT",
                    "start_lap": 21,
                    "end_lap": 60,
                },
            ],
            "pit_window": {
                "total_laps": 60,
                "pit_windows": [
                    {"earliest": 20, "optimal": 25, "latest": 30, "compound": "SOFT"},
                    {"earliest": 35, "optimal": 40, "latest": 45, "compound": "SOFT"},
                    {"earliest": 20, "optimal": 25, "latest": 30, "compound": "SOFT"},
                    {"earliest": 35, "optimal": 40, "latest": 45, "compound": "SOFT"},
                    {"earliest": 20, "optimal": 25, "latest": 30, "compound": "SOFT"},
                    {"earliest": 35, "optimal": 40, "latest": 45, "compound": "SOFT"},
                ],
            },
        }

    @pytest.fixture
    def overloaded_weather(self):
        return {
            "rain_risk": {
                "risk_level": "High",
                "max_precip_prob": 90.0,
                "total_rain_mm": 10.0,
            },
            "temperature": {"track_temp_est_c": 56},
        }

    def test_fixture_produces_more_than_eight_findings(
        self, overloaded_strategy, overloaded_tire, overloaded_weather
    ):
        from agents.evaluator_agent import evaluate_strategy

        result = evaluate_strategy(
            overloaded_strategy, overloaded_tire, overloaded_weather
        )
        assert result["checks_failed"] > 8, (
            "Fixture must produce >8 findings to exercise the guard; "
            f"got checks_failed={result['checks_failed']}"
        )

    def test_checks_passed_is_not_negative(
        self, overloaded_strategy, overloaded_tire, overloaded_weather
    ):
        from agents.evaluator_agent import evaluate_strategy

        result = evaluate_strategy(
            overloaded_strategy, overloaded_tire, overloaded_weather
        )
        assert result["checks_passed"] >= 0, (
            f"checks_passed must never be negative; got {result['checks_passed']}. "
            f"checks_failed={result['checks_failed']}, "
            f"total findings={len(result['findings'])}"
        )

    def test_checks_passed_is_clamped_to_zero(
        self, overloaded_strategy, overloaded_tire, overloaded_weather
    ):
        from agents.evaluator_agent import evaluate_strategy

        result = evaluate_strategy(
            overloaded_strategy, overloaded_tire, overloaded_weather
        )
        assert result["checks_passed"] == 0, (
            f"With 11 findings vs 8 rule categories, checks_passed should be 0; "
            f"got {result['checks_passed']}"
        )

    def test_score_non_negative_when_overloaded(
        self, overloaded_strategy, overloaded_tire, overloaded_weather
    ):
        from agents.evaluator_agent import evaluate_strategy

        result = evaluate_strategy(
            overloaded_strategy, overloaded_tire, overloaded_weather
        )
        assert result["score"] >= 0


# ===================================================================
# 3. assess_rain_risk — probability and rainfall thresholds
# ===================================================================

def _rain_df(max_prob: float, total_rain: float) -> pd.DataFrame:
    """Minimal forecast DataFrame consumed by assess_rain_risk."""
    return pd.DataFrame({
        "datetime": [
            pd.Timestamp("2023-06-04 13:00:00", tz="UTC"),
            pd.Timestamp("2023-06-04 14:00:00", tz="UTC"),
        ],
        "precipitation_probability": [max_prob, max_prob],
        "rain": [total_rain, 0.0],
    })


class TestAssessRainRiskThresholds:
    """
    Exact boundary tests for the three-level classification:
      max_prob < 15  AND  total_rain == 0  →  None
      max_prob >= 15 OR   total_rain > 0   →  Low
      max_prob >= 40 OR   total_rain >= 1  →  Medium
      max_prob >= 70 OR   total_rain >= 5  →  High
    get_race_forecast is mocked — no Open-Meteo requests are made.
    """

    def _risk(self, max_prob: float, total_rain: float) -> str:
        from agents.weather_agent import assess_rain_risk

        with patch(
            "agents.weather_agent.get_race_forecast",
            return_value=_rain_df(max_prob, total_rain),
        ):
            return assess_rain_risk("Monaco")["risk_level"]

    # -- None band (prob < 15, rain == 0) --------------------------------

    def test_none_at_zero_prob_and_zero_rain(self):
        assert self._risk(0.0, 0.0) == "None"

    def test_none_just_below_low_threshold(self):
        assert self._risk(14.9, 0.0) == "None"

    # -- Low band (15 ≤ prob < 40, OR 0 < rain < 1) ----------------------

    def test_low_at_prob_exactly_15(self):
        assert self._risk(15.0, 0.0) == "Low"

    def test_low_at_prob_39(self):
        assert self._risk(39.0, 0.0) == "Low"

    def test_low_when_trace_rain_and_low_prob(self):
        assert self._risk(0.0, 0.1) == "Low"

    # -- Medium band (40 ≤ prob < 70, OR 1 ≤ rain < 5) ------------------

    def test_medium_at_prob_exactly_40(self):
        assert self._risk(40.0, 0.0) == "Medium"

    def test_medium_at_prob_69(self):
        assert self._risk(69.0, 0.0) == "Medium"

    def test_medium_at_rain_exactly_1mm(self):
        assert self._risk(0.0, 1.0) == "Medium"

    def test_medium_at_rain_4_99mm(self):
        assert self._risk(0.0, 4.99) == "Medium"

    # -- High band (prob ≥ 70, OR rain ≥ 5) -----------------------------

    def test_high_at_prob_exactly_70(self):
        assert self._risk(70.0, 0.0) == "High"

    def test_high_at_prob_85(self):
        assert self._risk(85.0, 0.0) == "High"

    def test_high_at_rain_exactly_5mm(self):
        assert self._risk(0.0, 5.0) == "High"

    def test_high_at_rain_above_5mm(self):
        assert self._risk(0.0, 10.0) == "High"

    # -- Both dimensions can independently reach the same level ----------

    def test_prob_70_with_zero_rain_is_high(self):
        assert self._risk(70.0, 0.0) == "High"

    def test_prob_40_with_zero_rain_is_medium(self):
        assert self._risk(40.0, 0.0) == "Medium"


# ===================================================================
# 4. calculate_tire_degradation — NaN input handling
# ===================================================================

class TestCalculateTireDegradationNaN:
    """
    np.polyfit propagates NaN silently when x (TyreLife) contains NaN,
    producing deg_rate_sec_per_lap=NaN without raising any exception.
    Tests assert that every returned rate must be a finite number.
    get_tire_data is mocked — no FastF1 or network calls are made.
    """

    def _session(self) -> MagicMock:
        return MagicMock()

    def test_nan_in_tyre_life_must_not_produce_nan_rate(self):
        """One NaN in TyreLife propagates through np.polyfit — the function must guard it."""
        from agents.tire_agent import calculate_tire_degradation

        df = pd.DataFrame({
            "Stint":      [1, 1, 1, 1, 1, 1, 1],
            "LapNumber":  [1, 2, 3, 4, 5, 6, 7],
            "LapTimeSec": [90.0, 91.2, 90.8, 91.5, 91.0, 92.1, 91.7],
            "TyreLife":   [1.0, 2.0, float("nan"), 4.0, 5.0, 6.0, 7.0],
            "Compound":   ["SOFT"] * 7,
        })

        with patch("agents.tire_agent.get_tire_data", return_value=df):
            result = calculate_tire_degradation(self._session(), driver="VER")

        assert len(result) > 0, "Expected at least one computed stint"
        for stint in result:
            rate = stint["deg_rate_sec_per_lap"]
            assert math.isfinite(rate), (
                f"deg_rate_sec_per_lap is not finite: {rate!r}. "
                "NaN from np.polyfit(x_with_nan) must be guarded explicitly."
            )

    def test_all_nan_tyre_life_must_not_produce_nan_rate(self):
        """All-NaN TyreLife: if a stint is returned its rate must be finite."""
        from agents.tire_agent import calculate_tire_degradation

        df = pd.DataFrame({
            "Stint":      [1, 1, 1, 1, 1],
            "LapNumber":  [1, 2, 3, 4, 5],
            "LapTimeSec": [90.0, 91.0, 92.0, 91.5, 91.8],
            "TyreLife":   [float("nan")] * 5,
            "Compound":   ["SOFT"] * 5,
        })

        with patch("agents.tire_agent.get_tire_data", return_value=df):
            result = calculate_tire_degradation(self._session(), driver="VER")

        for stint in result:
            assert math.isfinite(stint["deg_rate_sec_per_lap"]), (
                f"All-NaN TyreLife returned non-finite deg_rate: "
                f"{stint['deg_rate_sec_per_lap']!r}"
            )

    def test_clean_data_returns_finite_rate(self):
        """Baseline: no NaN in inputs — rate must be finite and compound preserved."""
        from agents.tire_agent import calculate_tire_degradation

        df = pd.DataFrame({
            "Stint":      [1, 1, 1, 1, 1, 1, 1],
            "LapNumber":  [1, 2, 3, 4, 5, 6, 7],
            "LapTimeSec": [90.0, 90.5, 91.0, 91.5, 92.0, 92.5, 93.0],
            "TyreLife":   [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "Compound":   ["MEDIUM"] * 7,
        })

        with patch("agents.tire_agent.get_tire_data", return_value=df):
            result = calculate_tire_degradation(self._session(), driver="VER")

        assert len(result) == 1
        assert math.isfinite(result[0]["deg_rate_sec_per_lap"])
        assert math.isfinite(result[0]["avg_lap_sec"])
        assert result[0]["compound"] == "MEDIUM"

    def test_nan_in_lap_time_is_excluded_by_cutoff_filter(self):
        """NaN in LapTimeSec is filtered out by the <= cutoff comparison, leaving TyreLife clean."""
        from agents.tire_agent import calculate_tire_degradation

        df = pd.DataFrame({
            "Stint":      [1, 1, 1, 1, 1, 1, 1],
            "LapNumber":  [1, 2, 3, 4, 5, 6, 7],
            "LapTimeSec": [90.0, float("nan"), 91.0, float("nan"), 92.0, 91.5, 91.2],
            "TyreLife":   [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "Compound":   ["HARD"] * 7,
        })

        with patch("agents.tire_agent.get_tire_data", return_value=df):
            result = calculate_tire_degradation(self._session(), driver="VER")

        for stint in result:
            assert math.isfinite(stint["deg_rate_sec_per_lap"])

    def test_empty_tire_data_returns_empty_list(self):
        """Empty DataFrame from get_tire_data must produce an empty result."""
        from agents.tire_agent import calculate_tire_degradation

        with patch(
            "agents.tire_agent.get_tire_data",
            return_value=pd.DataFrame(),
        ):
            result = calculate_tire_degradation(self._session(), driver="VER")

        assert result == []
