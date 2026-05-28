"""
Task Success Rate Evaluation — verify that the pipeline generates
COMPLETE and COHERENT strategies for each ground-truth scenario.

A strategy is considered "successful" if it produces ALL of:
  1. Compound selection   — ≥1 compound recommended
  2. Pit-stop windows     — at least 1 pit window with earliest/latest/optimal
  3. Justification        — ≥100 chars of recommendation text
  4. Strategy type        — defined (1-stop, 2-stop, etc.)
  5. No fatal errors      — pipeline completes without exceptions

Coherence sub-checks (each can downgrade from PASS to PARTIAL):
  A. Compounds match degradation data  (stints = compounds)
  B. Pit laps fall within windows
  C. Strategy type aligns with number of stops
  D. Weather contingency is present when rain risk > 0
  E. Historical cross-verification section present

Target: ≥70% Task Success Rate

Output:
  tests/task_success_report.txt   — detailed findings
  tests/task_success_summary.csv  — per-scenario verdicts
"""

from __future__ import annotations

import csv
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _PROJECT_ROOT)

logging.basicConfig(level=logging.WARNING)


# ===================================================================
# Ground Truth (same 5 races used across all evaluations)
# ===================================================================

@dataclass
class GroundTruth:
    name: str
    circuit: str
    year: int
    winner: str
    strategy_type: str
    winner_compounds: list[str]
    total_laps: int
    rainfall: bool


SCENARIOS: list[GroundTruth] = [
    GroundTruth(
        name="British GP 2022",
        circuit="Silverstone",
        year=2022,
        winner="SAI",
        strategy_type="multi-stop",
        winner_compounds=["SOFT", "HARD", "MEDIUM", "SOFT"],
        total_laps=52,
        rainfall=False,
    ),
    GroundTruth(
        name="Monaco GP 2023",
        circuit="Monaco",
        year=2023,
        winner="VER",
        strategy_type="1-stop",
        winner_compounds=["MEDIUM", "INTERMEDIATE"],
        total_laps=78,
        rainfall=True,
    ),
    GroundTruth(
        name="Bahrain GP 2023",
        circuit="Bahrain",
        year=2023,
        winner="VER",
        strategy_type="2-stop",
        winner_compounds=["SOFT", "SOFT", "HARD"],
        total_laps=57,
        rainfall=False,
    ),
    GroundTruth(
        name="Hungarian GP 2023",
        circuit="Budapest",
        year=2023,
        winner="VER",
        strategy_type="1-stop",
        winner_compounds=["MEDIUM", "HARD"],
        total_laps=70,
        rainfall=False,
    ),
    GroundTruth(
        name="São Paulo GP 2022",
        circuit="São Paulo",
        year=2022,
        winner="RUS",
        strategy_type="2-stop",
        winner_compounds=["SOFT", "MEDIUM", "SOFT"],
        total_laps=71,
        rainfall=False,
    ),
]


# ===================================================================
# Result model
# ===================================================================

@dataclass
class CompletionCheck:
    """One completeness/coherence check for a scenario."""
    name: str
    description: str
    passed: bool = False
    detail: str = ""

    @property
    def icon(self) -> str:
        return "✅" if self.passed else "❌"


@dataclass
class ScenarioResult:
    """Full evaluation result for a single scenario."""
    name: str
    circuit: str
    year: int
    completeness_checks: list[CompletionCheck] = field(default_factory=list)
    coherence_checks: list[CompletionCheck] = field(default_factory=list)
    pipeline_error: str | None = None
    elapsed_sec: float = 0.0

    @property
    def completeness_passed(self) -> int:
        return sum(1 for c in self.completeness_checks if c.passed)

    @property
    def completeness_total(self) -> int:
        return len(self.completeness_checks)

    @property
    def coherence_passed(self) -> int:
        return sum(1 for c in self.coherence_checks if c.passed)

    @property
    def coherence_total(self) -> int:
        return len(self.coherence_checks)

    @property
    def is_complete(self) -> bool:
        """A strategy is 'complete' if ALL 5 completeness checks pass."""
        return all(c.passed for c in self.completeness_checks)

    @property
    def is_coherent(self) -> bool:
        """A strategy is 'coherent' if ≥3/5 coherence checks pass."""
        return self.coherence_passed >= 3

    @property
    def task_success(self) -> bool:
        """Task is successful if complete AND coherent."""
        return self.is_complete and self.is_coherent

    @property
    def verdict(self) -> str:
        if self.pipeline_error:
            return "💥 PIPELINE ERROR"
        if self.task_success:
            return "✅ SUCCESS"
        if self.is_complete:
            return "⚠️ COMPLETE BUT INCOHERENT"
        return "❌ INCOMPLETE"


# ===================================================================
# Core evaluation logic
# ===================================================================

def evaluate_task_success(gt: GroundTruth) -> ScenarioResult:
    """Run the full pipeline for a scenario and evaluate completeness
    and coherence of the generated strategy.
    """
    from agents.evaluator_agent import evaluate_full_pipeline

    result = ScenarioResult(
        name=gt.name, circuit=gt.circuit, year=gt.year,
    )

    try:
        pipeline = evaluate_full_pipeline(
            circuit=gt.circuit, year=gt.year, session_type="R",
        )
    except Exception as exc:
        result.pipeline_error = str(exc)
        return result

    strategy = pipeline.get("strategy") or {}
    evaluation = pipeline.get("evaluation") or {}
    tire = pipeline.get("tire_analysis") or {}
    weather = pipeline.get("weather_analysis") or {}

    # Extract key fields
    compounds = strategy.get("compounds", [])
    pit_laps = strategy.get("pit_laps", [])
    strategy_type = strategy.get("strategy_type", "")
    rec_text = strategy.get("recommendation_text", "")
    confidence = strategy.get("confidence", "")
    error = pipeline.get("error")

    pit_window_data = tire.get("pit_window") or {}
    pit_windows = pit_window_data.get("pit_windows", [])
    deg_data = tire.get("degradation") or []
    rain_risk = (weather.get("rain_risk") or {}).get("risk_level", "None")

    # =================================================================
    # COMPLETENESS CHECKS (all 5 must pass for "complete")
    # =================================================================

    # C1: Compound selection present
    c1 = CompletionCheck(
        name="compounds",
        description="At least 1 compound recommended",
    )
    if compounds and len(compounds) >= 1:
        c1.passed = True
        c1.detail = f"{len(compounds)} compounds: {' → '.join(compounds)}"
    else:
        c1.detail = "No compounds in strategy output"
    result.completeness_checks.append(c1)

    # C2: Pit-stop windows present
    c2 = CompletionCheck(
        name="pit_windows",
        description="At least 1 pit window with earliest/latest/optimal",
    )
    if pit_windows:
        valid_windows = [
            w for w in pit_windows
            if all(k in w for k in ("earliest", "latest", "optimal"))
        ]
        if valid_windows:
            c2.passed = True
            c2.detail = (
                f"{len(valid_windows)} window(s): "
                + ", ".join(
                    f"L{w['earliest']}-{w['latest']} (opt ~{w['optimal']})"
                    for w in valid_windows
                )
            )
        else:
            c2.detail = f"{len(pit_windows)} window(s) found but missing fields"
    else:
        # Monaco-type races with no pit data — check if rec_text mentions windows
        if "Pit Windows" in rec_text:
            # Section exists but says "No pit window data available"
            c2.detail = "Pit Windows section exists but no structured data"
        else:
            c2.detail = "No pit window data generated"
    result.completeness_checks.append(c2)

    # C3: Justification text is substantive
    c3 = CompletionCheck(
        name="justification",
        description="Recommendation text ≥ 100 characters",
    )
    if len(rec_text) >= 100:
        c3.passed = True
        c3.detail = f"{len(rec_text)} chars of recommendation text"
    else:
        c3.detail = f"Only {len(rec_text)} chars (minimum: 100)"
    result.completeness_checks.append(c3)

    # C4: Strategy type defined
    c4 = CompletionCheck(
        name="strategy_type",
        description="Strategy type is defined (e.g. 1-stop, 2-stop)",
    )
    if strategy_type and strategy_type != "Unknown":
        c4.passed = True
        c4.detail = f"Type: {strategy_type}"
    else:
        c4.detail = f"Type is '{strategy_type}' — undefined or unknown"
    result.completeness_checks.append(c4)

    # C5: No fatal pipeline errors
    c5 = CompletionCheck(
        name="no_fatal_errors",
        description="Pipeline completed without fatal exceptions",
    )
    # Partial errors (e.g., weather API timeout) are tolerable
    # Fatal = strategy dict is empty or pipeline returned None
    if strategy and rec_text:
        c5.passed = True
        if error:
            c5.detail = f"Completed with non-fatal warnings: {error}"
        else:
            c5.detail = "No errors"
    else:
        c5.detail = f"Fatal error: {error or 'strategy output is empty'}"
    result.completeness_checks.append(c5)

    # =================================================================
    # COHERENCE CHECKS (≥3/5 must pass for "coherent")
    # =================================================================

    # A: Compounds match degradation data
    ca = CompletionCheck(
        name="compounds_vs_deg",
        description="Number of compounds matches degradation stints",
    )
    if deg_data and compounds:
        deg_compounds = [d.get("compound", "") for d in deg_data]
        if len(compounds) == len(deg_compounds):
            ca.passed = True
            ca.detail = f"Both have {len(compounds)} stints"
        elif abs(len(compounds) - len(deg_compounds)) <= 1:
            # Off by 1 is common (e.g., final short stint has no deg data)
            ca.passed = True
            ca.detail = (
                f"Close match: {len(compounds)} compounds vs "
                f"{len(deg_compounds)} deg stints (±1 tolerance)"
            )
        else:
            ca.detail = (
                f"Mismatch: {len(compounds)} compounds vs "
                f"{len(deg_compounds)} degradation stints"
            )
    elif not deg_data and not compounds:
        ca.passed = True
        ca.detail = "Both empty — consistent"
    else:
        ca.detail = (
            f"Data gap: compounds={'yes' if compounds else 'no'}, "
            f"degradation={'yes' if deg_data else 'no'}"
        )
    result.coherence_checks.append(ca)

    # B: Pit laps within windows
    cb = CompletionCheck(
        name="pit_laps_in_windows",
        description="Recommended pit laps fall within computed windows",
    )
    if pit_laps and pit_windows:
        all_in = True
        details = []
        for i, lap in enumerate(pit_laps):
            if i < len(pit_windows):
                w = pit_windows[i]
                earliest = w.get("earliest", 0)
                latest = w.get("latest", 999)
                if earliest <= lap <= latest:
                    details.append(f"Pit {i+1}: L{lap} ∈ [{earliest},{latest}] ✓")
                else:
                    all_in = False
                    details.append(f"Pit {i+1}: L{lap} ∉ [{earliest},{latest}] ✗")
        cb.passed = all_in
        cb.detail = "; ".join(details)
    elif not pit_laps and not pit_windows:
        cb.passed = True
        cb.detail = "No pits or windows — consistent (e.g., Monaco wet)"
    elif pit_laps and not pit_windows:
        cb.detail = "Pit laps recommended but no windows to validate against"
    else:
        cb.passed = True
        cb.detail = "Windows exist, pit laps are flexible"
    result.coherence_checks.append(cb)

    # C: Strategy type vs number of stops
    cc = CompletionCheck(
        name="type_vs_stops",
        description="Strategy type matches number of pit stops",
    )
    if strategy_type and compounds:
        num_stops = max(0, len(compounds) - 1)
        expected_type = f"{num_stops}-stop"

        if strategy_type == expected_type:
            cc.passed = True
            cc.detail = f"{strategy_type} matches {num_stops} stops"
        elif strategy_type == "multi-stop" and num_stops >= 2:
            cc.passed = True
            cc.detail = f"multi-stop covers {num_stops} stops"
        elif strategy_type in ("1-stop", "2-stop", "3-stop"):
            declared_stops = int(strategy_type.split("-")[0])
            if declared_stops == num_stops:
                cc.passed = True
                cc.detail = f"{strategy_type} = {num_stops} stops ✓"
            else:
                cc.detail = (
                    f"Type says {strategy_type} but {num_stops} stops "
                    f"implied by {len(compounds)} compounds"
                )
        else:
            cc.detail = f"Unrecognized type: {strategy_type}"
    else:
        cc.detail = "Cannot verify — missing type or compounds"
    result.coherence_checks.append(cc)

    # D: Weather contingency when rain risk > None
    cd = CompletionCheck(
        name="weather_contingency",
        description="Weather contingency plan present when rain risk exists",
    )
    has_contingency = "Weather Contingency" in rec_text
    if rain_risk in ("High", "Medium"):
        if has_contingency and "intermediate" in rec_text.lower():
            cd.passed = True
            cd.detail = f"Rain risk={rain_risk}, contingency mentions intermediates"
        elif has_contingency:
            cd.passed = True
            cd.detail = f"Rain risk={rain_risk}, contingency section present"
        else:
            cd.detail = f"Rain risk={rain_risk} but NO contingency plan"
    else:
        # Dry conditions — contingency section is optional
        cd.passed = True
        cd.detail = f"Rain risk={rain_risk} — dry, contingency optional"
    result.coherence_checks.append(cd)

    # E: Historical cross-verification present
    ce = CompletionCheck(
        name="historical_verification",
        description="Historical winner data referenced in strategy",
    )
    if "Historical Cross-Verification" in rec_text or "Historical Winner" in rec_text:
        ce.passed = True
        # Extract winner abbreviation
        match = re.search(r"Historical Winner\*?\*?:?\s*(\w{3})", rec_text)
        if match:
            ce.detail = f"References winner: {match.group(1)}"
        else:
            ce.detail = "Historical section present"
    else:
        ce.detail = "No historical cross-verification in output"
    result.coherence_checks.append(ce)

    return result


# ===================================================================
# Ground truth accuracy bonus checks
# ===================================================================

def _check_accuracy(result: ScenarioResult, gt: GroundTruth,
                    strategy: dict) -> list[CompletionCheck]:
    """Optional accuracy checks against ground truth (not part of
    success rate but useful for reporting)."""
    checks: list[CompletionCheck] = []

    compounds = strategy.get("compounds", [])

    # Driver match
    ck_driver = CompletionCheck(
        name="driver_match",
        description=f"Focal driver matches GT winner ({gt.winner})",
    )
    driver = strategy.get("driver", "")
    if driver == gt.winner:
        ck_driver.passed = True
        ck_driver.detail = f"Both: {driver}"
    else:
        ck_driver.detail = f"Generated: {driver}, Expected: {gt.winner}"
    checks.append(ck_driver)

    # Compounds match
    ck_comp = CompletionCheck(
        name="compounds_match",
        description="Generated compounds match GT winner's compounds",
    )
    if compounds == gt.winner_compounds:
        ck_comp.passed = True
        ck_comp.detail = f"Exact match: {' → '.join(compounds)}"
    else:
        ck_comp.passed = False
        ck_comp.detail = (
            f"Gen: {' → '.join(compounds) if compounds else 'N/A'} | "
            f"GT: {' → '.join(gt.winner_compounds)}"
        )
    checks.append(ck_comp)

    # Strategy type match
    ck_type = CompletionCheck(
        name="strategy_type_match",
        description=f"Strategy type matches GT ({gt.strategy_type})",
    )
    gen_type = strategy.get("strategy_type", "")
    gt_type = gt.strategy_type
    if gen_type == gt_type:
        ck_type.passed = True
        ck_type.detail = f"Both: {gen_type}"
    elif gt_type == "multi-stop" and gen_type in ("2-stop", "3-stop"):
        ck_type.passed = True
        ck_type.detail = f"{gen_type} qualifies as multi-stop"
    else:
        ck_type.detail = f"Gen: {gen_type}, GT: {gt_type}"
    checks.append(ck_type)

    return checks


# ===================================================================
# Main runner
# ===================================================================

def run_task_success_eval():
    lines: list[str] = []
    csv_rows: list[dict] = []
    results: list[ScenarioResult] = []

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines.append("=" * 70)
    lines.append("  TASK SUCCESS RATE EVALUATION — F1 Race Strategist AI")
    lines.append(f"  Run at: {timestamp}")
    lines.append(f"  Target: ≥70% Task Success Rate")
    lines.append("=" * 70)
    lines.append("")
    lines.append("  DEFINITION: A task is 'successful' when the pipeline produces a")
    lines.append("  COMPLETE strategy (all 5 completeness checks pass) that is also")
    lines.append("  COHERENT (≥3 of 5 coherence checks pass).")
    lines.append("")
    lines.append("  COMPLETENESS CHECKS (all must pass):")
    lines.append("    C1. Compounds     — ≥1 compound recommended")
    lines.append("    C2. Pit Windows   — structured window data present")
    lines.append("    C3. Justification — recommendation text ≥100 chars")
    lines.append("    C4. Strategy Type — defined (1-stop, 2-stop, etc.)")
    lines.append("    C5. No Fatal Err  — pipeline completed successfully")
    lines.append("")
    lines.append("  COHERENCE CHECKS (≥3/5 must pass):")
    lines.append("    A. Compounds vs degradation stints")
    lines.append("    B. Pit laps within computed windows")
    lines.append("    C. Strategy type vs actual number of stops")
    lines.append("    D. Weather contingency when rain risk exists")
    lines.append("    E. Historical cross-verification present")
    lines.append("")

    for i, gt in enumerate(SCENARIOS, 1):
        print(f"\n[{i}/{len(SCENARIOS)}] Evaluating: {gt.name} ...", end=" ", flush=True)
        t0 = time.time()

        sr = evaluate_task_success(gt)
        sr.elapsed_sec = time.time() - t0
        results.append(sr)

        print(f"done ({sr.elapsed_sec:.1f}s) — {sr.verdict}")

        # Also get accuracy data
        from agents.evaluator_agent import evaluate_full_pipeline
        try:
            pipeline = evaluate_full_pipeline(gt.circuit, gt.year, session_type="R")
            strategy = pipeline.get("strategy") or {}
            accuracy_checks = _check_accuracy(sr, gt, strategy)
        except Exception:
            accuracy_checks = []

        # ── Report section ────────────────────────────────────────
        lines.append(f"{'━' * 70}")
        lines.append(f"  {gt.name} ({gt.circuit} {gt.year})")
        lines.append(f"  Verdict: {sr.verdict}")
        lines.append(f"  Completeness: {sr.completeness_passed}/{sr.completeness_total}")
        lines.append(f"  Coherence:    {sr.coherence_passed}/{sr.coherence_total}")
        lines.append(f"  Time: {sr.elapsed_sec:.1f}s")
        lines.append(f"{'━' * 70}")

        if sr.pipeline_error:
            lines.append(f"  💥 Pipeline error: {sr.pipeline_error}")
            csv_rows.append({
                "scenario": gt.name,
                "circuit": gt.circuit,
                "year": gt.year,
                "verdict": sr.verdict,
                "task_success": "NO",
                "completeness": f"{sr.completeness_passed}/{sr.completeness_total}",
                "coherence": f"{sr.coherence_passed}/{sr.coherence_total}",
                "error": sr.pipeline_error,
            })
            continue

        lines.append("")
        lines.append("  Completeness:")
        for c in sr.completeness_checks:
            lines.append(f"    {c.icon} [{c.name}] {c.description}")
            lines.append(f"       {c.detail}")

        lines.append("")
        lines.append("  Coherence:")
        for c in sr.coherence_checks:
            lines.append(f"    {c.icon} [{c.name}] {c.description}")
            lines.append(f"       {c.detail}")

        if accuracy_checks:
            lines.append("")
            lines.append("  Accuracy (vs Ground Truth):")
            for c in accuracy_checks:
                lines.append(f"    {c.icon} [{c.name}] {c.detail}")

        lines.append("")

        # CSV row
        csv_row = {
            "scenario": gt.name,
            "circuit": gt.circuit,
            "year": gt.year,
            "verdict": sr.verdict,
            "task_success": "YES" if sr.task_success else "NO",
            "completeness": f"{sr.completeness_passed}/{sr.completeness_total}",
            "coherence": f"{sr.coherence_passed}/{sr.coherence_total}",
        }
        for c in sr.completeness_checks:
            csv_row[f"comp_{c.name}"] = "PASS" if c.passed else "FAIL"
        for c in sr.coherence_checks:
            csv_row[f"cohr_{c.name}"] = "PASS" if c.passed else "FAIL"
        for c in accuracy_checks:
            csv_row[f"acc_{c.name}"] = "PASS" if c.passed else "FAIL"
        csv_row["elapsed_sec"] = f"{sr.elapsed_sec:.1f}"
        csv_row["error"] = ""

        csv_rows.append(csv_row)

    # =================================================================
    # AGGREGATE RESULTS
    # =================================================================
    lines.append("")
    lines.append("=" * 70)
    lines.append("  AGGREGATE RESULTS")
    lines.append("=" * 70)

    total = len(results)
    successful = sum(1 for r in results if r.task_success)
    complete = sum(1 for r in results if r.is_complete)
    coherent = sum(1 for r in results if r.is_coherent)
    errored = sum(1 for r in results if r.pipeline_error)

    success_rate = (successful / total * 100) if total else 0
    target_met = success_rate >= 70

    lines.append("")
    lines.append(f"  Scenarios tested:    {total}")
    lines.append(f"  Pipeline errors:     {errored}")
    lines.append(f"  Complete strategies:  {complete}/{total} ({complete/total*100:.0f}%)")
    lines.append(f"  Coherent strategies:  {coherent}/{total} ({coherent/total*100:.0f}%)")
    lines.append(f"  Successful (both):   {successful}/{total}")
    lines.append("")
    lines.append(f"  ╔══════════════════════════════════════════╗")
    lines.append(f"  ║  TASK SUCCESS RATE: {success_rate:5.1f}%               ║")
    lines.append(f"  ║  TARGET (≥70%):     {'✅ MET' if target_met else '❌ NOT MET':21s} ║")
    lines.append(f"  ╚══════════════════════════════════════════╝")

    lines.append("")
    lines.append("  Per-scenario breakdown:")
    for r in results:
        lines.append(
            f"    {r.name:.<30s} "
            f"comp={r.completeness_passed}/{r.completeness_total}  "
            f"cohr={r.coherence_passed}/{r.coherence_total}  "
            f"{r.verdict}"
        )

    # Identify weakest checks across scenarios
    lines.append("")
    lines.append("  Weakest completeness checks:")
    comp_names = {c.name for r in results for c in r.completeness_checks}
    for name in sorted(comp_names):
        pass_count = sum(
            1 for r in results
            for c in r.completeness_checks
            if c.name == name and c.passed
        )
        lines.append(f"    {name:.<25s} {pass_count}/{total}")

    lines.append("")
    lines.append("  Weakest coherence checks:")
    cohr_names = {c.name for r in results for c in r.coherence_checks}
    for name in sorted(cohr_names):
        pass_count = sum(
            1 for r in results
            for c in r.coherence_checks
            if c.name == name and c.passed
        )
        lines.append(f"    {name:.<25s} {pass_count}/{total}")

    lines.append("")
    lines.append("=" * 70)

    # ── Write outputs ─────────────────────────────────────────────
    output_dir = os.path.join(_PROJECT_ROOT, "tests")
    os.makedirs(output_dir, exist_ok=True)

    report_path = os.path.join(output_dir, "task_success_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    csv_path = os.path.join(output_dir, "task_success_summary.csv")
    if csv_rows:
        # Ensure all rows have the same keys
        all_keys = set()
        for row in csv_rows:
            all_keys.update(row.keys())
        fieldnames = sorted(all_keys)
        # Put important columns first
        priority = [
            "scenario", "circuit", "year", "verdict", "task_success",
            "completeness", "coherence", "elapsed_sec", "error",
        ]
        fieldnames = [k for k in priority if k in all_keys] + [
            k for k in fieldnames if k not in priority
        ]

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(csv_rows)

    print(f"\n{'=' * 60}")
    print(f"📊 Task Success Rate evaluation complete:")
    print(f"   Report:  {report_path}")
    print(f"   CSV:     {csv_path}")
    print(f"   Rate:    {success_rate:.0f}% ({successful}/{total})")
    print(f"   Target:  {'✅ MET (≥70%)' if target_met else '❌ NOT MET (<70%)'}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run_task_success_eval()
