"""
Run the full Race Strategist pipeline against 5 ground-truth scenarios
and log results to a CSV for manual scoring.

Usage:
    python -m tests.run_eval_scenarios          # from project root
    python tests/run_eval_scenarios.py          # direct

Output:
    tests/eval_results.csv   — one row per scenario
    tests/eval_log.txt       — detailed per-scenario log
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
import textwrap
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any

# ── Ensure project root is importable ──────────────────────────────
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _PROJECT_ROOT)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ===================================================================
# Ground Truth definitions (same as the notebook)
# ===================================================================

@dataclass
class GroundTruth:
    name: str
    circuit: str
    year: int
    winner: str
    winner_team: str
    podium: list[str]
    strategy_type: str
    winner_compounds: list[str]
    total_laps: int
    rainfall: bool
    safety_car: bool
    red_flag: bool
    key_facts: list[str] = field(default_factory=list)
    notes: str = ""


SCENARIOS: list[GroundTruth] = [
    GroundTruth(
        name="British GP 2022",
        circuit="Silverstone",
        year=2022,
        winner="SAI",
        winner_team="Ferrari",
        podium=["SAI", "PER", "HAM"],
        strategy_type="multi-stop",
        winner_compounds=["SOFT", "HARD", "MEDIUM", "SOFT"],
        total_laps=52,
        rainfall=False,
        safety_car=True,
        red_flag=True,
        key_facts=[
            "Lap-1 red flag (Zhou crash)",
            "Late safety car decisive for Sainz win",
        ],
    ),
    GroundTruth(
        name="Monaco GP 2023",
        circuit="Monaco",
        year=2023,
        winner="VER",
        winner_team="Red Bull Racing",
        podium=["VER", "ALO", "OCO"],
        strategy_type="1-stop",
        winner_compounds=["MEDIUM", "INTERMEDIATE"],
        total_laps=78,
        rainfall=True,
        safety_car=False,
        red_flag=False,
        key_facts=[
            "55 laps on mediums before rain",
            "Graining management was key",
        ],
    ),
    GroundTruth(
        name="Bahrain GP 2023",
        circuit="Bahrain",
        year=2023,
        winner="VER",
        winner_team="Red Bull Racing",
        podium=["VER", "PER", "ALO"],
        strategy_type="2-stop",
        winner_compounds=["SOFT", "SOFT", "HARD"],
        total_laps=57,
        rainfall=False,
        safety_car=False,
        red_flag=False,
        key_facts=[
            "Season opener 2023",
            "Double-soft opening stints by Red Bull",
        ],
    ),
    GroundTruth(
        name="Hungarian GP 2023",
        circuit="Budapest",
        year=2023,
        winner="VER",
        winner_team="Red Bull Racing",
        podium=["VER", "NOR", "PER"],
        strategy_type="1-stop",
        winner_compounds=["MEDIUM", "HARD"],
        total_laps=70,
        rainfall=False,
        safety_car=False,
        red_flag=False,
        key_facts=[
            "Pitted lap 23 for hards",
            "Classic 1-stop at twisty circuit",
        ],
    ),
    GroundTruth(
        name="São Paulo GP 2022",
        circuit="São Paulo",
        year=2022,
        winner="RUS",
        winner_team="Mercedes",
        podium=["RUS", "HAM", "SAI"],
        strategy_type="2-stop",
        winner_compounds=["SOFT", "MEDIUM", "SOFT"],
        total_laps=71,
        rainfall=False,
        safety_car=True,
        red_flag=False,
        key_facts=[
            "Russell first career victory",
            "Mercedes 1-2 (Russell-Hamilton)",
        ],
    ),
]


# ===================================================================
# Helper: format expected answer as a compact string
# ===================================================================

def format_expected(gt: GroundTruth) -> str:
    return (
        f"Winner: {gt.winner} ({gt.winner_team}) | "
        f"Strategy: {gt.strategy_type} | "
        f"Compounds: {' → '.join(gt.winner_compounds)} | "
        f"Laps: {gt.total_laps} | "
        f"Rain: {gt.rainfall} | SC: {gt.safety_car} | "
        f"Red flag: {gt.red_flag}"
    )


# ===================================================================
# Helper: format generated answer from pipeline result
# ===================================================================

def format_generated(result: dict[str, Any]) -> str:
    strategy = result.get("strategy") or {}
    evaluation = result.get("evaluation") or {}

    strat_type = strategy.get("strategy_type", "N/A")
    compounds = strategy.get("compounds", [])
    pit_laps = strategy.get("pit_laps", [])
    driver = strategy.get("driver", "N/A")
    confidence = strategy.get("confidence", "N/A")
    score = evaluation.get("score", "N/A")
    verdict = evaluation.get("verdict", "N/A")

    return (
        f"Driver: {driver} | "
        f"Strategy: {strat_type} | "
        f"Compounds: {' → '.join(compounds) if compounds else 'N/A'} | "
        f"Pit laps: {pit_laps} | "
        f"Eval score: {score}/100 | "
        f"Verdict: {verdict} | "
        f"Confidence: {confidence}"
    )


# ===================================================================
# Main runner
# ===================================================================

def run_all_scenarios() -> list[dict[str, Any]]:
    """Run every scenario through evaluate_full_pipeline and collect results."""
    from agents.evaluator_agent import evaluate_full_pipeline

    rows: list[dict[str, Any]] = []
    log_lines: list[str] = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_lines.append("=" * 70)
    log_lines.append(f"  F1 RACE STRATEGIST — FULL PIPELINE EVALUATION")
    log_lines.append(f"  Run at: {timestamp}")
    log_lines.append(f"  Scenarios: {len(SCENARIOS)}")
    log_lines.append("=" * 70)

    for i, gt in enumerate(SCENARIOS, 1):
        log_lines.append("")
        log_lines.append(f"{'─' * 70}")
        log_lines.append(f"  SCENARIO {i}/{len(SCENARIOS)}: {gt.name}")
        log_lines.append(f"  Circuit: {gt.circuit} | Year: {gt.year}")
        log_lines.append(f"{'─' * 70}")

        print(f"\n[{i}/{len(SCENARIOS)}] Running: {gt.name} ...", end=" ", flush=True)
        t0 = time.time()

        try:
            result = evaluate_full_pipeline(
                circuit=gt.circuit,
                year=gt.year,
                session_type="R",
            )
            elapsed = time.time() - t0
            print(f"done ({elapsed:.1f}s)")

            strategy = result.get("strategy") or {}
            evaluation = result.get("evaluation") or {}
            tire = result.get("tire_analysis") or {}
            weather = result.get("weather_analysis") or {}

            generated = format_generated(result)
            expected = format_expected(gt)

            # ── Auto-score (basic heuristic) ──────────────────────
            auto_score = 0

            # +1 if strategy type matches or is reasonable
            gen_type = (strategy.get("strategy_type") or "").lower()
            exp_type = gt.strategy_type.lower()
            if exp_type in gen_type or gen_type in exp_type:
                auto_score += 1

            # +1 if compounds are non-empty
            gen_compounds = strategy.get("compounds", [])
            if gen_compounds:
                auto_score += 1

            # +1 if evaluator score >= 45
            if evaluation.get("score", 0) >= 45:
                auto_score += 1

            # +1 if pit laps are populated
            if strategy.get("pit_laps"):
                auto_score += 1

            # +1 if recommendation text is substantial (>200 chars)
            rec_text = strategy.get("recommendation_text", "")
            if len(rec_text) > 200:
                auto_score += 1

            # ── Log details ───────────────────────────────────────
            log_lines.append(f"  Generated: {generated}")
            log_lines.append(f"  Expected:  {expected}")
            log_lines.append(f"  Auto-score: {auto_score}/5")
            log_lines.append(f"  Elapsed: {elapsed:.1f}s")

            if result.get("error"):
                log_lines.append(f"  ⚠ Errors: {result['error']}")

            # Strategy recommendation (truncated)
            if rec_text:
                truncated = rec_text[:500].replace("\n", "\n    ")
                log_lines.append(f"  Strategy text (first 500 chars):")
                log_lines.append(f"    {truncated}")

            # Evaluation findings
            findings = evaluation.get("findings", [])
            if findings:
                log_lines.append(f"  Evaluator findings ({len(findings)}):")
                for f in findings:
                    log_lines.append(
                        f"    [{f.get('severity', '?').upper()}] "
                        f"{f.get('rule', '?')}: {f.get('message', '')}"
                    )

            row = {
                "scenario": gt.name,
                "circuit": gt.circuit,
                "year": gt.year,
                "generated_response": generated,
                "expected_response": expected,
                "auto_score": auto_score,
                "manual_score": "",  # placeholder for manual review
                "eval_score": evaluation.get("score", ""),
                "eval_verdict": evaluation.get("verdict", ""),
                "strategy_type_gen": strategy.get("strategy_type", ""),
                "strategy_type_exp": gt.strategy_type,
                "compounds_gen": " → ".join(gen_compounds) if gen_compounds else "",
                "compounds_exp": " → ".join(gt.winner_compounds),
                "error": result.get("error") or "",
                "elapsed_sec": f"{elapsed:.1f}",
            }

        except Exception as exc:
            elapsed = time.time() - t0
            print(f"FAILED ({elapsed:.1f}s): {exc}")
            log_lines.append(f"  💥 EXCEPTION: {exc}")

            row = {
                "scenario": gt.name,
                "circuit": gt.circuit,
                "year": gt.year,
                "generated_response": f"ERROR: {exc}",
                "expected_response": format_expected(gt),
                "auto_score": 0,
                "manual_score": "",
                "eval_score": "",
                "eval_verdict": "",
                "strategy_type_gen": "",
                "strategy_type_exp": gt.strategy_type,
                "compounds_gen": "",
                "compounds_exp": " → ".join(gt.winner_compounds),
                "error": str(exc),
                "elapsed_sec": f"{elapsed:.1f}",
            }

        rows.append(row)

    # ── Summary ───────────────────────────────────────────────────
    log_lines.append("")
    log_lines.append("=" * 70)
    log_lines.append("  SUMMARY")
    log_lines.append("=" * 70)

    total = len(rows)
    avg_auto = sum(r["auto_score"] for r in rows) / total if total else 0
    errors = sum(1 for r in rows if r["error"])
    log_lines.append(f"  Total scenarios:    {total}")
    log_lines.append(f"  Avg auto-score:     {avg_auto:.1f}/5")
    log_lines.append(f"  Scenarios w/errors: {errors}")
    log_lines.append("")

    for r in rows:
        log_lines.append(
            f"  {r['scenario']:.<30} auto={r['auto_score']}/5  "
            f"eval={r['eval_score']}/100  {r['eval_verdict']}"
        )

    log_lines.append("")
    log_lines.append("=" * 70)

    # ── Write outputs ─────────────────────────────────────────────
    output_dir = os.path.join(_PROJECT_ROOT, "tests")
    os.makedirs(output_dir, exist_ok=True)

    # CSV
    csv_path = os.path.join(output_dir, "eval_results.csv")
    fieldnames = list(rows[0].keys()) if rows else []
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Log
    log_path = os.path.join(output_dir, "eval_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    print(f"\n{'=' * 60}")
    print(f"📊 Results saved:")
    print(f"   CSV: {csv_path}")
    print(f"   Log: {log_path}")
    print(f"   Avg auto-score: {avg_auto:.1f}/5")
    print(f"   Errors: {errors}/{total}")
    print(f"{'=' * 60}")
    print(f"\n💡 Open eval_results.csv and fill in the 'manual_score' column (1-5).")

    return rows


if __name__ == "__main__":
    run_all_scenarios()
