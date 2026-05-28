"""
Explainability Score Evaluation — verify that strategy recommendations
are understandable by non-technical stakeholders.

Checks for the presence and quality of:
  1. Factors Considered       — explicit data source attribution
  2. Alternatives Considered  — rejected options with reasoning
  3. Confidence Assessment    — self-assessed confidence with rationale
  4. Justification Summary    — plain-language explanation

Output:
  tests/explainability_report.txt   — detailed findings
  tests/explainability_summary.csv  — per-scenario scores
"""

from __future__ import annotations

import csv
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _PROJECT_ROOT)

logging.basicConfig(level=logging.WARNING)

# ===================================================================
# Scenarios
# ===================================================================

SCENARIOS = [
    {"name": "British GP 2022",     "circuit": "Silverstone", "year": 2022},
    {"name": "Monaco GP 2023",      "circuit": "Monaco",      "year": 2023},
    {"name": "Bahrain GP 2023",     "circuit": "Bahrain",     "year": 2023},
    {"name": "Hungarian GP 2023",   "circuit": "Budapest",    "year": 2023},
    {"name": "São Paulo GP 2022",   "circuit": "São Paulo",   "year": 2022},
]


# ===================================================================
# Explainability criteria
# ===================================================================

@dataclass
class ExplainabilityCriterion:
    """A single explainability check."""
    name: str
    description: str
    max_score: int
    score: int = 0
    findings: list[str] = field(default_factory=list)


def _check_section_exists(text: str, heading: str) -> bool:
    """Check if a markdown section heading exists in the text."""
    return bool(re.search(rf"^##\s+{re.escape(heading)}", text, re.MULTILINE))


def _count_bullet_points(text: str, heading: str) -> int:
    """Count bullet points under a specific section heading."""
    pattern = rf"^##\s+{re.escape(heading)}(.*?)(?=^##|\Z)"
    match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
    if not match:
        return 0
    section = match.group(1)
    return len(re.findall(r"^\s*[-•]\s+", section, re.MULTILINE))


def _extract_section(text: str, heading: str) -> str:
    """Extract text under a specific section heading."""
    pattern = rf"^##\s+{re.escape(heading)}(.*?)(?=^##|\Z)"
    match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
    return match.group(1).strip() if match else ""


def _has_specific_numbers(text: str) -> bool:
    """Check if text contains specific numerical data (not just generic)."""
    # Look for patterns like +0.04, 0.06, lap 25, 22s, 3/5, etc.
    return bool(re.search(r"[\d]+\.[\d]+|lap\s+\d+|\d+\s*s\b|\d+/\d+", text, re.IGNORECASE))


def _has_reasoning_words(text: str) -> bool:
    """Check if text contains reasoning/causal language."""
    reasoning_words = [
        "because", "therefore", "rejected", "since", "due to",
        "means", "making", "would", "risk", "too much", "not enough",
        "so that", "in order to", "as a result", "consequently",
    ]
    text_lower = text.lower()
    return any(w in text_lower for w in reasoning_words)


def _jargon_density(text: str) -> float:
    """Estimate jargon density (lower = more accessible).

    Returns fraction of 'jargon words' in the text.
    """
    jargon = {
        "degradation", "graining", "blistering", "understeer", "oversteer",
        "aero", "downforce", "undercut", "overcut", "stint", "outlap",
        "inlap", "delta", "offset", "s/lap", "regression", "coefficient",
        "heuristic", "baseline", "extrapolate", "interpolate",
    }
    words = text.lower().split()
    if not words:
        return 0.0
    jargon_count = sum(1 for w in words if w.strip(".,;:!?") in jargon)
    return jargon_count / len(words)


# ===================================================================
# Evaluation per scenario
# ===================================================================

def evaluate_explainability(circuit: str, year: int) -> dict[str, Any]:
    """Evaluate explainability of the offline strategy for a scenario.

    Returns a dict with criteria, total score, and detailed findings.
    """
    from agents.strategist_agent import generate_strategy_offline

    result = generate_strategy_offline(circuit, year)
    text = result.get("recommendation_text", "")

    criteria: list[ExplainabilityCriterion] = []

    # ── C1: Factors Considered section present and populated ──────
    c1 = ExplainabilityCriterion(
        name="factors_considered",
        description="Explicit listing of data sources with weights",
        max_score=20,
    )

    if _check_section_exists(text, "Factors Considered"):
        c1.findings.append("✅ 'Factors Considered' section exists")
        c1.score += 5

        section = _extract_section(text, "Factors Considered")
        bullets = _count_bullet_points(text, "Factors Considered")
        c1.findings.append(f"   {bullets} bullet points found")

        # Check for weight annotations
        weights = re.findall(r"weight:\s*(high|medium|low)", section, re.IGNORECASE)
        c1.findings.append(f"   {len(weights)} weight annotations found")
        if len(weights) >= 3:
            c1.score += 10
            c1.findings.append("   ✅ At least 3 data sources weighted")
        elif len(weights) >= 1:
            c1.score += 5
            c1.findings.append("   ⚠️ Only some data sources weighted")

        if _has_specific_numbers(section):
            c1.score += 5
            c1.findings.append("   ✅ Contains specific numerical data")
        else:
            c1.findings.append("   ⚠️ No specific numbers cited")
    else:
        c1.findings.append("❌ 'Factors Considered' section MISSING")

    criteria.append(c1)

    # ── C2: Alternatives Considered section ───────────────────────
    c2 = ExplainabilityCriterion(
        name="alternatives_considered",
        description="Rejected alternatives with data-driven reasoning",
        max_score=25,
    )

    if _check_section_exists(text, "Alternatives Considered"):
        c2.findings.append("✅ 'Alternatives Considered' section exists")
        c2.score += 5

        section = _extract_section(text, "Alternatives Considered")
        bullets = _count_bullet_points(text, "Alternatives Considered")
        c2.findings.append(f"   {bullets} alternative(s) listed")

        if bullets >= 2:
            c2.score += 5
            c2.findings.append("   ✅ At least 2 alternatives listed")
        else:
            c2.findings.append("   ⚠️ Fewer than 2 alternatives")

        if _has_reasoning_words(section):
            c2.score += 5
            c2.findings.append("   ✅ Contains causal/reasoning language")
        else:
            c2.findings.append("   ⚠️ Missing reasoning language (because, rejected, etc.)")

        if _has_specific_numbers(section):
            c2.score += 5
            c2.findings.append("   ✅ Alternatives reference specific data")
        else:
            c2.findings.append("   ⚠️ Alternatives are generic (no numbers)")

        # Check for 'rejected' keyword specifically
        if "rejected" in section.lower():
            c2.score += 5
            c2.findings.append("   ✅ Explicitly states 'rejected' for each alternative")
        else:
            c2.findings.append("   ⚠️ Doesn't explicitly say 'rejected'")
    else:
        c2.findings.append("❌ 'Alternatives Considered' section MISSING")

    criteria.append(c2)

    # ── C3: Confidence Assessment ─────────────────────────────────
    c3 = ExplainabilityCriterion(
        name="confidence_assessment",
        description="Self-assessed confidence level with justification",
        max_score=20,
    )

    if _check_section_exists(text, "Confidence Assessment"):
        c3.findings.append("✅ 'Confidence Assessment' section exists")
        c3.score += 5

        section = _extract_section(text, "Confidence Assessment")

        # Check for explicit confidence level
        confidence_levels = re.findall(
            r"\*\*(High|Medium|Low)\*\*", section, re.IGNORECASE,
        )
        if confidence_levels:
            c3.score += 5
            c3.findings.append(f"   ✅ Confidence level: {confidence_levels[0]}")
        else:
            c3.findings.append("   ⚠️ No explicit confidence level found")

        # Check for reasoning
        if "reason" in section.lower() or "because" in section.lower():
            c3.score += 5
            c3.findings.append("   ✅ Confidence is justified with reasoning")
        else:
            c3.findings.append("   ⚠️ No reasoning provided for confidence level")

        # Check if it references data completeness
        if any(w in section.lower() for w in ["data source", "available", "missing", "agree"]):
            c3.score += 5
            c3.findings.append("   ✅ References data completeness")
        else:
            c3.findings.append("   ⚠️ Doesn't reference data quality/completeness")
    else:
        c3.findings.append("❌ 'Confidence Assessment' section MISSING")

    criteria.append(c3)

    # ── C4: Justification Summary (plain language) ────────────────
    c4 = ExplainabilityCriterion(
        name="justification_summary",
        description="Plain-language summary understandable by non-technical audience",
        max_score=20,
    )

    if _check_section_exists(text, "Justification Summary"):
        c4.findings.append("✅ 'Justification Summary' section exists")
        c4.score += 5

        section = _extract_section(text, "Justification Summary")

        # Check jargon density
        jargon = _jargon_density(section)
        c4.findings.append(f"   Jargon density: {jargon:.1%}")
        if jargon < 0.03:
            c4.score += 5
            c4.findings.append("   ✅ Low jargon — accessible to non-technical audience")
        elif jargon < 0.08:
            c4.score += 3
            c4.findings.append("   ⚠️ Moderate jargon — some terms may confuse non-experts")
        else:
            c4.findings.append("   ❌ High jargon density — not accessible")

        # Check for conversational tone
        if any(w in section.lower() for w in ["we ", "our ", "the race"]):
            c4.score += 5
            c4.findings.append("   ✅ Uses conversational/inclusive language")
        else:
            c4.findings.append("   ⚠️ Tone is too formal/technical")

        # Check minimum length (too short = too vague)
        word_count = len(section.split())
        c4.findings.append(f"   Word count: {word_count}")
        if word_count >= 20:
            c4.score += 5
            c4.findings.append("   ✅ Sufficient detail")
        elif word_count >= 10:
            c4.score += 3
            c4.findings.append("   ⚠️ A bit brief — could be more informative")
        else:
            c4.findings.append("   ❌ Too short — lacks meaningful explanation")
    else:
        c4.findings.append("❌ 'Justification Summary' section MISSING")

    criteria.append(c4)

    # ── C5: Historical Cross-Verification presence ────────────────
    c5 = ExplainabilityCriterion(
        name="historical_cross_verification",
        description="References historical race data for validation",
        max_score=15,
    )

    if _check_section_exists(text, "Historical Cross-Verification"):
        c5.findings.append("✅ 'Historical Cross-Verification' section exists")
        c5.score += 5

        section = _extract_section(text, "Historical Cross-Verification")

        # Check for winner reference
        if re.search(r"[A-Z]{3}", section):
            c5.score += 5
            c5.findings.append("   ✅ References specific driver abbreviation")
        else:
            c5.findings.append("   ⚠️ No driver reference found")

        # Check for strategy comparison (✅ or ⚠️)
        if "✅" in section or "⚠️" in section:
            c5.score += 5
            c5.findings.append("   ✅ Includes match/mismatch indicator")
        else:
            c5.findings.append("   ⚠️ No alignment status indicator")
    else:
        c5.findings.append("⚠️ 'Historical Cross-Verification' section missing")

    criteria.append(c5)

    # ── Aggregate ─────────────────────────────────────────────────
    total_score = sum(c.score for c in criteria)
    max_score = sum(c.max_score for c in criteria)
    pct = (total_score / max_score * 100) if max_score else 0

    if pct >= 80:
        grade = "🟢 EXCELLENT"
    elif pct >= 60:
        grade = "🟡 GOOD"
    elif pct >= 40:
        grade = "🟠 NEEDS IMPROVEMENT"
    else:
        grade = "🔴 POOR"

    return {
        "criteria": criteria,
        "total_score": total_score,
        "max_score": max_score,
        "percentage": pct,
        "grade": grade,
        "recommendation_text": text,
        "confidence": result.get("confidence"),
        "strategy_type": result.get("strategy_type"),
    }


# ===================================================================
# Main runner
# ===================================================================

def run_explainability_eval():
    from datetime import datetime

    lines: list[str] = []
    csv_rows: list[dict] = []

    lines.append("=" * 70)
    lines.append("  EXPLAINABILITY SCORE EVALUATION — F1 Race Strategist AI")
    lines.append(f"  Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)
    lines.append("")
    lines.append("  PURPOSE: Verify that strategy justifications are clear enough")
    lines.append("  for a non-technical human to understand the reasoning.")
    lines.append("")
    lines.append("  CHECKS:")
    lines.append("    C1. Factors Considered — data sources explicitly listed")
    lines.append("    C2. Alternatives Considered — rejected options with reasons")
    lines.append("    C3. Confidence Assessment — confidence level + justification")
    lines.append("    C4. Justification Summary — plain-language explanation")
    lines.append("    C5. Historical Cross-Verification — data validation")
    lines.append("")

    all_pcts: list[float] = []

    for i, sc in enumerate(SCENARIOS, 1):
        print(f"\n[{i}/{len(SCENARIOS)}] Evaluating: {sc['name']} ...", end=" ", flush=True)
        t0 = time.time()

        try:
            result = evaluate_explainability(sc["circuit"], sc["year"])
            elapsed = time.time() - t0
            print(f"done ({elapsed:.1f}s)")

            lines.append(f"{'━' * 70}")
            lines.append(f"  {sc['name']} ({sc['circuit']} {sc['year']})")
            lines.append(f"  Score: {result['total_score']}/{result['max_score']} "
                         f"({result['percentage']:.0f}%) — {result['grade']}")
            lines.append(f"{'━' * 70}")

            for c in result["criteria"]:
                lines.append(f"\n  [{c.name}] ({c.score}/{c.max_score})")
                lines.append(f"  {c.description}")
                for f in c.findings:
                    lines.append(f"    {f}")

            all_pcts.append(result["percentage"])

            csv_rows.append({
                "scenario": sc["name"],
                "circuit": sc["circuit"],
                "year": sc["year"],
                "total_score": result["total_score"],
                "max_score": result["max_score"],
                "percentage": f"{result['percentage']:.0f}%",
                "grade": result["grade"],
                "factors_considered": next(
                    (c.score for c in result["criteria"] if c.name == "factors_considered"), 0
                ),
                "alternatives_considered": next(
                    (c.score for c in result["criteria"] if c.name == "alternatives_considered"), 0
                ),
                "confidence_assessment": next(
                    (c.score for c in result["criteria"] if c.name == "confidence_assessment"), 0
                ),
                "justification_summary": next(
                    (c.score for c in result["criteria"] if c.name == "justification_summary"), 0
                ),
                "historical_cross_verification": next(
                    (c.score for c in result["criteria"] if c.name == "historical_cross_verification"), 0
                ),
            })

        except Exception as exc:
            elapsed = time.time() - t0
            print(f"FAILED ({elapsed:.1f}s): {exc}")
            lines.append(f"\n  💥 {sc['name']}: {exc}")
            csv_rows.append({
                "scenario": sc["name"],
                "circuit": sc["circuit"],
                "year": sc["year"],
                "error": str(exc),
            })

    # ── Summary ───────────────────────────────────────────────────
    lines.append("")
    lines.append("=" * 70)
    lines.append("  AGGREGATE RESULTS")
    lines.append("=" * 70)

    if all_pcts:
        avg_pct = sum(all_pcts) / len(all_pcts)
        min_pct = min(all_pcts)
        max_pct = max(all_pcts)
        lines.append(f"  Average explainability: {avg_pct:.0f}%")
        lines.append(f"  Range: {min_pct:.0f}% – {max_pct:.0f}%")
        lines.append(f"  Scenarios evaluated: {len(all_pcts)}/{len(SCENARIOS)}")
        lines.append("")

        if avg_pct >= 80:
            lines.append("  🟢 VERDICT: Strategies are well-explained and accessible.")
            lines.append("  Non-technical stakeholders should be able to understand")
            lines.append("  the reasoning behind each recommendation.")
        elif avg_pct >= 60:
            lines.append("  🟡 VERDICT: Strategies are reasonably explained but could")
            lines.append("  be improved in areas like alternatives or plain-language summary.")
        elif avg_pct >= 40:
            lines.append("  🟠 VERDICT: Strategies need significant improvement in")
            lines.append("  explainability. Key sections are missing or too vague.")
        else:
            lines.append("  🔴 VERDICT: Strategies are NOT explainable to non-technical")
            lines.append("  audience. Major overhaul of system prompt needed.")

    lines.append("")
    lines.append("=" * 70)

    # ── Write outputs ─────────────────────────────────────────────
    output_dir = os.path.join(_PROJECT_ROOT, "tests")
    os.makedirs(output_dir, exist_ok=True)

    report_path = os.path.join(output_dir, "explainability_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    csv_path = os.path.join(output_dir, "explainability_summary.csv")
    if csv_rows:
        fieldnames = list(csv_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

    print(f"\n{'=' * 60}")
    print(f"📊 Explainability evaluation complete:")
    print(f"   Report: {report_path}")
    print(f"   CSV:    {csv_path}")
    if all_pcts:
        print(f"   Average: {sum(all_pcts)/len(all_pcts):.0f}%")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run_explainability_eval()
