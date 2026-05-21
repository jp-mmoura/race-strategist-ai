"""
Evaluator Agent — verifies coherence of the strategy recommendation.

Applies a battery of rule-based checks against the tire, weather, and
strategy data to produce a coherence score (0–100) with itemised
findings.  The score penalises inconsistencies such as:

  • Recommending SOFT on a high-wear circuit
  • Recommending dry compounds under high rain risk
  • Pit laps outside the computed optimal window
  • Strategy type that conflicts with degradation data
  • Compound order that contradicts historical precedent
  • Ignoring extreme temperature conditions

Main functions
--------------
evaluate_strategy(strategy, tire_analysis, weather_analysis)
    → coherence evaluation dict with score, findings, verdict

run_evaluator_node(state)
    → LangGraph node entry-point — reads/writes RaceStrategyState
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

# ---------------------------------------------------------------------------
# Compound hardness hierarchy (softer → harder)
# ---------------------------------------------------------------------------
_COMPOUND_HARDNESS: dict[str, int] = {
    "SOFT": 1,
    "MEDIUM": 2,
    "HARD": 3,
    "INTERMEDIATE": 4,
    "WET": 5,
}

# Severity weights for the scoring system
_SEVERITY_WEIGHTS: dict[str, int] = {
    "critical": 20,
    "major": 12,
    "minor": 5,
    "info": 0,
}


# ===================================================================
# Data classes for findings
# ===================================================================

class Finding:
    """A single evaluation finding."""

    def __init__(
        self,
        rule: str,
        severity: str,
        message: str,
        detail: str = "",
    ):
        self.rule = rule
        self.severity = severity  # critical / major / minor / info
        self.message = message
        self.detail = detail
        self.penalty = _SEVERITY_WEIGHTS.get(severity, 0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule": self.rule,
            "severity": self.severity,
            "message": self.message,
            "detail": self.detail,
            "penalty": self.penalty,
        }


# ===================================================================
# Individual rule checks
# ===================================================================

def _check_soft_on_high_wear(
    compounds: list[str],
    track_wear: dict[str, Any],
) -> list[Finding]:
    """SOFT compound on a high-wear circuit is risky."""
    findings: list[Finding] = []
    classification = track_wear.get("classification", "")
    score = track_wear.get("score", 0) or 0

    if classification == "High Tire Wear" and "SOFT" in compounds:
        soft_count = compounds.count("SOFT")
        if soft_count >= 2:
            findings.append(Finding(
                rule="SOFT_HIGH_WEAR",
                severity="critical",
                message=(
                    f"SOFT compound used {soft_count}x on a high-wear "
                    f"circuit (score {score}/5)."
                ),
                detail=(
                    "High tire stress and abrasion will cause rapid "
                    "degradation on SOFT tyres, risking a forced "
                    "extra pit stop."
                ),
            ))
        else:
            # Single SOFT stint (e.g. final short stint) is acceptable
            findings.append(Finding(
                rule="SOFT_HIGH_WEAR",
                severity="minor",
                message=(
                    f"SOFT compound on a high-wear circuit (score {score}/5). "
                    "Acceptable only for a short final stint."
                ),
                detail="Monitor degradation closely if stint > 10 laps.",
            ))

    return findings


def _check_dry_under_rain(
    compounds: list[str],
    rain_risk: dict[str, Any],
) -> list[Finding]:
    """Dry-only strategy under high rain risk."""
    findings: list[Finding] = []
    risk_level = rain_risk.get("risk_level", "None")
    dry_compounds = {"SOFT", "MEDIUM", "HARD"}
    all_dry = all(c in dry_compounds for c in compounds)

    if risk_level == "High" and all_dry:
        findings.append(Finding(
            rule="DRY_UNDER_RAIN",
            severity="critical",
            message="All-dry compound strategy under HIGH rain risk.",
            detail=(
                f"Rain probability: {rain_risk.get('max_precip_prob', '?')}%, "
                f"expected: {rain_risk.get('total_rain_mm', '?')} mm. "
                "No intermediate/wet contingency in the compound plan."
            ),
        ))
    elif risk_level == "Medium" and all_dry:
        findings.append(Finding(
            rule="DRY_UNDER_RAIN",
            severity="major",
            message="All-dry compound strategy under MEDIUM rain risk.",
            detail=(
                "Should have a contingency plan mentioning intermediate "
                "tyres even if the primary strategy is dry."
            ),
        ))

    return findings


def _check_pit_window_alignment(
    pit_laps: list[int],
    pit_windows: list[dict[str, Any]],
) -> list[Finding]:
    """Pit laps should fall within the computed optimal windows."""
    findings: list[Finding] = []

    if not pit_laps or not pit_windows:
        return findings

    for i, lap in enumerate(pit_laps):
        if i >= len(pit_windows):
            break
        window = pit_windows[i]
        earliest = window.get("earliest", 0)
        latest = window.get("latest", 999)
        optimal = window.get("optimal", 0)

        if lap < earliest or lap > latest:
            findings.append(Finding(
                rule="PIT_WINDOW_MISS",
                severity="major",
                message=(
                    f"Pit {i + 1} at lap {lap} is outside the optimal "
                    f"window (laps {earliest}–{latest})."
                ),
                detail=(
                    f"Optimal pit lap: ~{optimal}. "
                    f"Delta: {abs(lap - optimal)} laps off target."
                ),
            ))
        elif abs(lap - optimal) > 3:
            findings.append(Finding(
                rule="PIT_WINDOW_OFFSET",
                severity="minor",
                message=(
                    f"Pit {i + 1} at lap {lap} is within the window "
                    f"but {abs(lap - optimal)} laps from optimal (~{optimal})."
                ),
                detail="Small offset — acceptable if reacting to track position.",
            ))

    return findings


def _check_strategy_vs_degradation(
    strategy_type: str | None,
    degradation: list[dict[str, Any]],
    total_laps: int,
) -> list[Finding]:
    """Strategy type should match degradation severity."""
    findings: list[Finding] = []

    if not degradation or not strategy_type:
        return findings

    avg_deg = sum(d["deg_rate_sec_per_lap"] for d in degradation) / len(degradation)

    # High degradation (>0.08 s/lap avg) but only 1-stop
    if avg_deg > 0.08 and strategy_type == "1-stop" and total_laps > 40:
        findings.append(Finding(
            rule="DEG_VS_STOPS",
            severity="major",
            message=(
                f"1-stop strategy with high average degradation "
                f"({avg_deg:+.4f} s/lap)."
            ),
            detail=(
                "At this degradation rate, a 2-stop strategy may be "
                f"faster over {total_laps} laps. Cumulative time loss: "
                f"~{avg_deg * total_laps / 2:.1f}s by mid-race."
            ),
        ))

    # Very low degradation but multi-stop
    if avg_deg < 0.02 and strategy_type in ("2-stop", "3-stop"):
        findings.append(Finding(
            rule="DEG_VS_STOPS",
            severity="minor",
            message=(
                f"{strategy_type} strategy with low degradation "
                f"({avg_deg:+.4f} s/lap)."
            ),
            detail=(
                "Low degradation suggests tyres can last longer. "
                "A fewer-stop strategy may save pit-stop time loss."
            ),
        ))

    return findings


def _check_compound_order(
    compounds: list[str],
    track_wear: dict[str, Any],
) -> list[Finding]:
    """Check if compound order is logical (generally harder → softer
    or strategic inverse)."""
    findings: list[Finding] = []

    if len(compounds) < 2:
        return findings

    # Check for starting on SOFT at a high/medium wear track
    classification = track_wear.get("classification", "")
    if compounds[0] == "SOFT" and classification in (
        "High Tire Wear", "Medium Tire Wear"
    ):
        findings.append(Finding(
            rule="SOFT_START",
            severity="major",
            message=(
                f"Starting on SOFT at a {classification.lower()} circuit."
            ),
            detail=(
                "Starting on SOFT typically forces an early first stop. "
                "MEDIUM or HARD is usually preferred for the opening stint "
                "on high/medium-wear tracks."
            ),
        ))

    return findings


def _check_temperature_compound_match(
    compounds: list[str],
    temperature: dict[str, Any],
) -> list[Finding]:
    """Extreme temperatures should influence compound choice."""
    findings: list[Finding] = []

    if not temperature:
        return findings

    track_temp_est = temperature.get("track_temp_est_c")
    if track_temp_est is None:
        return findings

    # Very high track temp + multiple SOFT stints
    if track_temp_est > 50 and compounds.count("SOFT") >= 2:
        findings.append(Finding(
            rule="TEMP_COMPOUND",
            severity="major",
            message=(
                f"Multiple SOFT stints with extreme track temp "
                f"(~{track_temp_est:.0f} °C)."
            ),
            detail=(
                "Track temperatures above 50 °C cause accelerated "
                "thermal degradation on SOFT compounds. "
                "Consider replacing one SOFT stint with MEDIUM."
            ),
        ))

    # Very low track temp + only HARD compounds
    if track_temp_est < 20 and all(c == "HARD" for c in compounds):
        findings.append(Finding(
            rule="TEMP_COMPOUND",
            severity="major",
            message=(
                f"All-HARD strategy with low track temp "
                f"(~{track_temp_est:.0f} °C)."
            ),
            detail=(
                "Cold track temperatures make it difficult to bring "
                "HARD tyres into their operating window. "
                "MEDIUM or SOFT may provide better grip."
            ),
        ))

    return findings


def _check_total_laps_coverage(
    compounds: list[str],
    degradation: list[dict[str, Any]],
    total_laps: int,
) -> list[Finding]:
    """Check that the planned stints can cover the race distance."""
    findings: list[Finding] = []

    if not degradation or total_laps == 0:
        return findings

    planned_laps = sum(d.get("lap_count", 0) for d in degradation)

    if planned_laps > 0 and abs(planned_laps - total_laps) > 5:
        findings.append(Finding(
            rule="STINT_COVERAGE",
            severity="minor",
            message=(
                f"Planned stint laps ({planned_laps}) differ from "
                f"race distance ({total_laps}) by "
                f"{abs(planned_laps - total_laps)} laps."
            ),
            detail="Small discrepancy expected from pit-in/out laps.",
        ))

    return findings


def _check_wet_without_rain(
    compounds: list[str],
    rain_risk: dict[str, Any],
) -> list[Finding]:
    """Wet/intermediate compounds without rain expected."""
    findings: list[Finding] = []
    risk_level = rain_risk.get("risk_level", "None")
    wet_compounds = {"INTERMEDIATE", "WET"}

    has_wet = any(c in wet_compounds for c in compounds)
    if has_wet and risk_level == "None":
        findings.append(Finding(
            rule="WET_NO_RAIN",
            severity="major",
            message="Wet/intermediate tyres planned but no rain expected.",
            detail=(
                "Rain risk is None — using intermediate or wet compounds "
                "will be significantly slower than dry options."
            ),
        ))

    return findings


# ===================================================================
# Main evaluation function
# ===================================================================

def evaluate_strategy(
    strategy: dict[str, Any],
    tire_analysis: dict[str, Any] | None = None,
    weather_analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate the coherence of a strategy recommendation.

    Parameters
    ----------
    strategy : dict
        Output of ``generate_strategy`` or ``generate_strategy_offline``
        from the strategy agent.
    tire_analysis : dict | None
        Output of ``analyze_tire_strategy`` from the tire agent.
    weather_analysis : dict | None
        Output of ``analyze_weather_impact`` from the weather agent.

    Returns
    -------
    dict with keys:
        score          : int    — coherence score (0–100, higher = better)
        verdict        : str    — "Approved" / "Review" / "Rejected"
        findings       : list   — individual rule-check results
        summary        : str    — human-readable evaluation summary
        checks_passed  : int    — number of rules with no issues
        checks_failed  : int    — number of rules with findings
        total_penalty  : int    — raw penalty points before capping
    """
    tire = tire_analysis or {}
    weather = weather_analysis or {}

    # Extract sub-structures
    compounds = strategy.get("compounds", [])
    pit_laps = strategy.get("pit_laps", [])
    strategy_type = strategy.get("strategy_type")
    track_wear = tire.get("track_wear") or {}
    degradation = tire.get("degradation") or []
    pit_window = tire.get("pit_window") or {}
    pit_windows = pit_window.get("pit_windows", [])
    total_laps = pit_window.get("total_laps", 0)
    rain_risk = weather.get("rain_risk") or {}
    temperature = weather.get("temperature") or {}

    # ── Run all checks ────────────────────────────────────────────
    all_findings: list[Finding] = []

    all_findings.extend(_check_soft_on_high_wear(compounds, track_wear))
    all_findings.extend(_check_dry_under_rain(compounds, rain_risk))
    all_findings.extend(_check_pit_window_alignment(pit_laps, pit_windows))
    all_findings.extend(
        _check_strategy_vs_degradation(strategy_type, degradation, total_laps)
    )
    all_findings.extend(_check_compound_order(compounds, track_wear))
    all_findings.extend(
        _check_temperature_compound_match(compounds, temperature)
    )
    all_findings.extend(
        _check_total_laps_coverage(compounds, degradation, total_laps)
    )
    all_findings.extend(_check_wet_without_rain(compounds, rain_risk))

    # ── Compute score ─────────────────────────────────────────────
    total_penalty = sum(f.penalty for f in all_findings)
    score = max(0, 100 - total_penalty)

    checks_failed = len([f for f in all_findings if f.severity != "info"])
    checks_passed = 8 - checks_failed  # 8 total rule categories

    # ── Verdict ───────────────────────────────────────────────────
    if score >= 75:
        verdict = "✅ Approved"
    elif score >= 45:
        verdict = "⚠️ Review"
    else:
        verdict = "❌ Rejected"

    # ── Summary ───────────────────────────────────────────────────
    summary = _build_summary(
        strategy, score, verdict, all_findings,
        track_wear, rain_risk, temperature,
    )

    result = {
        "score": score,
        "verdict": verdict,
        "findings": [f.to_dict() for f in all_findings],
        "summary": summary,
        "checks_passed": max(0, checks_passed),
        "checks_failed": checks_failed,
        "total_penalty": total_penalty,
    }

    logger.info(
        "Strategy evaluation: score=%d, verdict=%s, findings=%d",
        score, verdict, len(all_findings),
    )
    return result


def _build_summary(
    strategy: dict[str, Any],
    score: int,
    verdict: str,
    findings: list[Finding],
    track_wear: dict[str, Any],
    rain_risk: dict[str, Any],
    temperature: dict[str, Any],
) -> str:
    """Build a human-readable evaluation summary."""
    lines: list[str] = []

    lines.append(f"## Strategy Evaluation — {verdict}")
    lines.append(f"**Coherence score: {score}/100**\n")

    # Strategy overview
    lines.append(
        f"Strategy: {strategy.get('strategy_type', '?')} | "
        f"Compounds: {' → '.join(strategy.get('compounds', [])) or 'N/A'} | "
        f"Pit laps: {strategy.get('pit_laps', [])}"
    )
    lines.append(
        f"Circuit wear: {track_wear.get('classification', '?')} | "
        f"Rain risk: {rain_risk.get('risk_level', '?')} | "
        f"Track temp: ~{temperature.get('track_temp_est_c', '?')} °C"
    )

    # Findings by severity
    critical = [f for f in findings if f.severity == "critical"]
    major = [f for f in findings if f.severity == "major"]
    minor = [f for f in findings if f.severity == "minor"]

    if critical:
        lines.append("\n### 🔴 Critical Issues")
        for f in critical:
            lines.append(f"- **{f.rule}**: {f.message}")
            if f.detail:
                lines.append(f"  _{f.detail}_")

    if major:
        lines.append("\n### 🟡 Major Concerns")
        for f in major:
            lines.append(f"- **{f.rule}**: {f.message}")
            if f.detail:
                lines.append(f"  _{f.detail}_")

    if minor:
        lines.append("\n### 🟢 Minor Notes")
        for f in minor:
            lines.append(f"- **{f.rule}**: {f.message}")

    if not findings:
        lines.append("\n### ✅ All Checks Passed")
        lines.append("No coherence issues detected.")

    return "\n".join(lines)


# ===================================================================
# Convenience: evaluate from raw inputs (circuit + year)
# ===================================================================

def evaluate_full_pipeline(
    circuit: str,
    year: int,
    race_date: str | None = None,
    driver: str | None = None,
    session_type: str = "R",
) -> dict[str, Any]:
    """Run the full pipeline (tire → weather → strategy → evaluation).

    Useful for standalone testing. In the graph, each agent runs
    as a separate node instead.

    Returns
    -------
    dict with keys: strategy, evaluation, tire_analysis,
                    weather_analysis, error.
    """
    from agents.strategist_agent import generate_strategy_offline
    from agents.tire_agent import analyze_tire_strategy
    from agents.weather_agent import analyze_weather_impact

    result: dict[str, Any] = {
        "strategy": None,
        "evaluation": None,
        "tire_analysis": None,
        "weather_analysis": None,
        "error": None,
    }

    errors: list[str] = []

    # ── 1. Tire analysis ──────────────────────────────────────────
    try:
        tire = analyze_tire_strategy(circuit, year, session_type, driver)
        result["tire_analysis"] = tire
    except Exception as exc:
        errors.append(f"Tire: {exc}")
        tire = {}

    # ── 2. Weather analysis ───────────────────────────────────────
    try:
        weather = analyze_weather_impact(circuit, race_date, year, session_type)
        result["weather_analysis"] = weather
    except Exception as exc:
        errors.append(f"Weather: {exc}")
        weather = {}

    # ── 3. Strategy generation ────────────────────────────────────
    try:
        strategy = generate_strategy_offline(
            circuit, year, race_date, driver, session_type,
        )
        result["strategy"] = strategy
    except Exception as exc:
        errors.append(f"Strategy: {exc}")
        strategy = {}

    # ── 4. Evaluation ─────────────────────────────────────────────
    if strategy:
        result["evaluation"] = evaluate_strategy(strategy, tire, weather)
    else:
        result["evaluation"] = {
            "score": 0,
            "verdict": "❌ Rejected",
            "findings": [],
            "summary": "Cannot evaluate — no strategy was generated.",
            "checks_passed": 0,
            "checks_failed": 0,
            "total_penalty": 0,
        }

    if errors:
        result["error"] = "; ".join(errors)

    return result


# ===================================================================
# LangGraph node entry-point
# ===================================================================

def run_evaluator_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node: evaluate the strategy already in state.

    Reads ``strategy_recommendation``, ``tire_analysis``, and
    ``weather_analysis`` from the state and writes
    ``evaluation_result`` back.

    Parameters
    ----------
    state : dict
        Current ``RaceStrategyState``.

    Returns
    -------
    dict — updated state fields.
    """
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
        return {
            "evaluation_result": evaluation,
        }
    except Exception as exc:
        logger.error("Evaluator node failed: %s", exc, exc_info=True)
        return {
            "evaluation_result": None,
            "error": f"Evaluator agent failed: {exc}",
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
    print("  Evaluator Agent — Validation")
    print("=" * 65)

    # ── Test 1: Full pipeline (Silverstone 2023) ──────────────────
    print("\n▶ Test 1: Full pipeline — Silverstone 2023")
    result = evaluate_full_pipeline("Silverstone", 2023)

    if result["error"]:
        print(f"  ⚠ Errors: {result['error']}")

    ev = result["evaluation"]
    print(f"\n  Score:   {ev['score']}/100")
    print(f"  Verdict: {ev['verdict']}")
    print(f"  Passed:  {ev['checks_passed']} | Failed: {ev['checks_failed']}")
    print(f"\n{ev['summary']}")

    # ── Test 2: Deliberately bad strategy ─────────────────────────
    print(f"\n{'─' * 65}")
    print("▶ Test 2: Deliberately incoherent strategy")

    bad_strategy = {
        "strategy_type": "1-stop",
        "compounds": ["SOFT", "SOFT"],
        "pit_laps": [10],
    }
    bad_tire = {
        "track_wear": {
            "classification": "High Tire Wear",
            "score": 4.5,
        },
        "degradation": [
            {"deg_rate_sec_per_lap": 0.12, "lap_count": 25,
             "stint": 1, "compound": "SOFT", "start_lap": 1, "end_lap": 25},
            {"deg_rate_sec_per_lap": 0.15, "lap_count": 27,
             "stint": 2, "compound": "SOFT", "start_lap": 26, "end_lap": 52},
        ],
        "pit_window": {
            "total_laps": 52,
            "pit_windows": [
                {"earliest": 18, "optimal": 22, "latest": 28, "compound": "SOFT"},
            ],
        },
    }
    bad_weather = {
        "rain_risk": {
            "risk_level": "High",
            "max_precip_prob": 85.0,
            "total_rain_mm": 8.5,
        },
        "temperature": {
            "track_temp_est_c": 55,
        },
    }

    ev2 = evaluate_strategy(bad_strategy, bad_tire, bad_weather)
    print(f"\n  Score:   {ev2['score']}/100")
    print(f"  Verdict: {ev2['verdict']}")
    print(f"\n{ev2['summary']}")

    print("\n" + "=" * 65)
    print("  ✅ Evaluator Agent validation complete!")
    print("=" * 65)
