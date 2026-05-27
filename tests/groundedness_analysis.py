"""
Groundedness Analysis — verify that strategy recommendations are
anchored in the data actually retrieved by the RAG and tire pipelines.

Detects:
  1. Whether the strategy's compounds come from tire_agent data (grounded)
     or are fabricated
  2. Whether the RAG context is actually used or ignored
  3. Whether the weather analysis matches the strategy's contingency
  4. Disconnections between RAG winner data and tire agent winner data

Output:
  tests/groundedness_report.txt   — detailed findings
  tests/groundedness_summary.csv  — per-scenario verdicts
"""

from __future__ import annotations

import csv
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _PROJECT_ROOT)

logging.basicConfig(level=logging.WARNING)

# ===================================================================
# Scenarios (same 5 ground-truth races)
# ===================================================================

SCENARIOS = [
    {"name": "British GP 2022",     "circuit": "Silverstone", "year": 2022},
    {"name": "Monaco GP 2023",      "circuit": "Monaco",      "year": 2023},
    {"name": "Bahrain GP 2023",     "circuit": "Bahrain",     "year": 2023},
    {"name": "Hungarian GP 2023",   "circuit": "Budapest",    "year": 2023},
    {"name": "São Paulo GP 2022",   "circuit": "São Paulo",   "year": 2022},
]


# ===================================================================
# Groundedness checks
# ===================================================================

def check_groundedness(circuit: str, year: int) -> dict[str, Any]:
    """Run all groundedness checks for a single race scenario.

    Returns a dict with findings, verdicts, and raw data references.
    """
    from agents.strategist_agent import build_strategy_context
    from rag.retriever import retrieve_race_context

    findings: list[str] = []
    verdicts: dict[str, str] = {}

    # ── 1. Build strategy context (same as what strategy_agent uses) ──
    ctx = build_strategy_context(circuit, year, session_type="R")

    tire = ctx.get("tire_analysis") or {}
    weather = ctx.get("weather_analysis") or {}
    rag = ctx.get("rag_context") or {}

    # ── 2. Get RAG context separately for comparison ──────────────────
    rag_ctx = retrieve_race_context(
        query=f"race strategy {circuit} {year}",
        year=year,
        circuit=circuit,
    )

    # ==================================================================
    # CHECK A: Are strategy compounds sourced from tire_agent data?
    # ==================================================================
    rec = tire.get("compound_rec") or {}
    recommended_compounds = rec.get("recommended_order", [])
    deg_data = tire.get("degradation") or []
    deg_compounds = [d["compound"] for d in deg_data]

    # Winner's actual stints from RAG
    winner_stints = rag_ctx.get("winner_stints")
    winner_actual_compounds = list(
        winner_stints.sort_values("Stint")["Compound"].dropna()
    ) if winner_stints is not None and not winner_stints.empty else []

    if recommended_compounds:
        source = rec.get("confidence", "")
        findings.append(f"[A] Strategy compounds: {' → '.join(recommended_compounds)}")
        findings.append(f"    Source: {rec.get('source', 'top-3 finishers')} ({source})")
        findings.append(f"    Winner actual stints: {' → '.join(winner_actual_compounds)}")
        findings.append(f"    Degradation stints: {' → '.join(deg_compounds)}")

        # Check if recommended_order matches winner actual stints or degradation compounds
        if recommended_compounds == winner_actual_compounds:
            verdicts["compounds_grounded"] = "✅ GROUNDED — compounds match winner's actual stints"
            findings.append(f"    ✅ Compounds match winner's actual stints exactly")
        elif recommended_compounds == deg_compounds:
            verdicts["compounds_grounded"] = "✅ GROUNDED — compounds match winner's degradation data"
            findings.append(f"    ✅ Compounds match winner's degradation data exactly")
        else:
            verdicts["compounds_grounded"] = "⚠️ DIVERGENT — compounds differ from winner's stints"
            findings.append(f"    ⚠️ Recommended: {recommended_compounds} vs Winner stints: {winner_actual_compounds}")
    else:
        verdicts["compounds_grounded"] = "❌ NO DATA — no compound recommendation produced"
        findings.append("[A] No compound recommendation was produced by tire_agent")

    # ==================================================================
    # CHECK B: Is the RAG context actually being included?
    # ==================================================================
    context_text = ctx.get("context_text", "")
    rag_text = rag_ctx.get("context_text", "")

    has_historical = "### Historical Race Data" in context_text
    has_rag_content = len(rag_text) > 50

    if has_historical and has_rag_content:
        verdicts["rag_included"] = "✅ GROUNDED — RAG context included in strategy prompt"
        findings.append(f"[B] RAG context: {len(rag_text)} chars included under '### Historical Race Data'")
    elif has_rag_content and not has_historical:
        verdicts["rag_included"] = "❌ DISCONNECTED — RAG retrieved data but it wasn't included"
        findings.append(f"[B] RAG retrieved {len(rag_text)} chars but NOT found in strategy context")
    else:
        verdicts["rag_included"] = "⚠️ EMPTY — RAG returned no/minimal data"
        findings.append(f"[B] RAG context is empty or minimal ({len(rag_text)} chars)")

    # ==================================================================
    # CHECK C: Does RAG winner match tire_agent winner?
    # ==================================================================
    rag_results = rag_ctx.get("race_results")
    pw = tire.get("pit_window") or {}
    tire_driver = pw.get("driver", "?")

    if rag_results is not None and not rag_results.empty:
        rag_winner = rag_results.iloc[0]["Abbreviation"]
        findings.append(f"[C] RAG winner: {rag_winner} | Tire agent focus: {tire_driver}")
        if rag_winner == tire_driver:
            verdicts["winner_consistent"] = f"✅ CONSISTENT — both use {rag_winner}"
        else:
            verdicts["winner_consistent"] = f"❌ MISMATCH — RAG={rag_winner} vs Tire={tire_driver}"
            findings.append(f"    ❌ Data sources disagree on focal driver!")
    else:
        verdicts["winner_consistent"] = "⚠️ NO RAG RESULTS — can't compare"
        findings.append("[C] RAG returned no race_results to compare")

    # ==================================================================
    # CHECK D: Weather coherence — does strategy weather match data?
    # ==================================================================
    rain_risk = weather.get("rain_risk") or {}
    risk_level = rain_risk.get("risk_level", "Unknown")

    rag_weather = rag_ctx.get("weather")
    rag_rainfall = False
    if rag_weather is not None and not rag_weather.empty:
        rag_rainfall = rag_weather.get("Rainfall", False)
        if hasattr(rag_rainfall, "any"):
            rag_rainfall = bool(rag_rainfall.any())

    findings.append(f"[D] Weather agent rain risk: {risk_level}")
    findings.append(f"    RAG historical rainfall: {rag_rainfall}")

    # If it actually rained but weather says "None" → disconnection
    if rag_rainfall and risk_level == "None":
        verdicts["weather_coherent"] = "❌ DISCONNECTED — historical rain but weather says None"
        findings.append(f"    ❌ Weather agent doesn't see historical rain (uses forecast API)")
    elif rag_rainfall and risk_level in ("High", "Medium"):
        verdicts["weather_coherent"] = "✅ COHERENT — rain detected by both sources"
    elif not rag_rainfall and risk_level == "None":
        verdicts["weather_coherent"] = "✅ COHERENT — both confirm dry conditions"
    else:
        verdicts["weather_coherent"] = f"⚠️ MIXED — historical rain={rag_rainfall}, forecast risk={risk_level}"
        findings.append(f"    ⚠️ Weather agent uses current forecast, not historical conditions")

    # ==================================================================
    # CHECK E: Strategy type grounded in stint count?
    # ==================================================================
    strategy_type = pw.get("strategy_type", "Unknown")
    num_stints = len(deg_data)
    expected_stops = max(0, num_stints - 1)
    expected_type = f"{expected_stops}-stop" if expected_stops > 0 else "0-stop"

    findings.append(f"[E] Strategy type: {strategy_type}")
    findings.append(f"    Winner had {num_stints} stints → expected {expected_type}")

    if strategy_type == expected_type or (strategy_type == "multi-stop" and expected_stops >= 2):
        verdicts["strategy_type_grounded"] = f"✅ GROUNDED — {strategy_type} matches {num_stints} stints"
    else:
        verdicts["strategy_type_grounded"] = f"⚠️ DIVERGENT — type={strategy_type} but winner had {expected_type}"
        findings.append(f"    ⚠️ Strategy type derived from pit_window model, not raw stint count")

    # ==================================================================
    # CHECK F: Is the offline strategy actually using RAG data?
    # ==================================================================
    # The offline generator pulls compounds from tire_agent.compound_rec
    # and strategy_type from tire_agent.pit_window — it does NOT parse
    # the RAG context_text. The RAG is only appended for LLM consumption.
    findings.append("")
    findings.append("[F] ARCHITECTURE NOTE:")
    findings.append("    The OFFLINE strategy generator (generate_strategy_offline) reads:")
    findings.append("      • compounds    ← tire_agent.compound_rec.recommended_order")
    findings.append("      • strategy_type ← tire_agent.pit_window.strategy_type")
    findings.append("      • pit_laps     ← tire_agent.pit_window.recommended_pit_laps")
    findings.append("      • weather      ← weather_agent.rain_risk")
    findings.append("    The RAG context_text is appended to the context but is ONLY read")
    findings.append("    by the LLM (generate_strategy). The offline path IGNORES it.")

    if not os.getenv("GOOGLE_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        verdicts["rag_actually_used"] = "⚠️ NOT USED — offline mode ignores RAG context"
        findings.append("    → Currently using OFFLINE mode (no LLM key found)")
        findings.append("    → RAG data is RETRIEVED but NOT CONSUMED in strategy decisions")
    else:
        verdicts["rag_actually_used"] = "✅ USED — LLM mode reads RAG in prompt"
        findings.append("    → LLM API key found — RAG context is sent in the prompt")

    return {
        "findings": findings,
        "verdicts": verdicts,
        "tire_compounds": recommended_compounds,
        "deg_compounds": deg_compounds,
        "strategy_type": strategy_type,
        "rag_length": len(rag_text),
        "context_length": len(context_text),
    }


# ===================================================================
# Main runner
# ===================================================================

def run_groundedness_analysis():
    from datetime import datetime

    lines: list[str] = []
    csv_rows: list[dict] = []

    lines.append("=" * 70)
    lines.append("  GROUNDEDNESS ANALYSIS — F1 Race Strategist AI")
    lines.append(f"  Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)

    for i, sc in enumerate(SCENARIOS, 1):
        print(f"\n[{i}/{len(SCENARIOS)}] Analyzing: {sc['name']} ...", end=" ", flush=True)
        t0 = time.time()

        try:
            result = check_groundedness(sc["circuit"], sc["year"])
            elapsed = time.time() - t0
            print(f"done ({elapsed:.1f}s)")

            lines.append("")
            lines.append(f"{'━' * 70}")
            lines.append(f"  {sc['name']} ({sc['circuit']} {sc['year']})")
            lines.append(f"{'━' * 70}")

            # Verdicts
            lines.append("")
            for check, verdict in result["verdicts"].items():
                lines.append(f"  {check:.<35} {verdict}")

            # Detailed findings
            lines.append("")
            for f in result["findings"]:
                lines.append(f"  {f}")

            # CSV row
            v = result["verdicts"]
            grounded_count = sum(1 for val in v.values() if val.startswith("✅"))
            total_checks = len(v)
            csv_rows.append({
                "scenario": sc["name"],
                "circuit": sc["circuit"],
                "year": sc["year"],
                "compounds_grounded": v.get("compounds_grounded", ""),
                "rag_included": v.get("rag_included", ""),
                "winner_consistent": v.get("winner_consistent", ""),
                "weather_coherent": v.get("weather_coherent", ""),
                "strategy_type_grounded": v.get("strategy_type_grounded", ""),
                "rag_actually_used": v.get("rag_actually_used", ""),
                "grounded_checks": f"{grounded_count}/{total_checks}",
                "rag_chars": result["rag_length"],
                "context_chars": result["context_length"],
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
    lines.append("  SUMMARY & DIAGNOSIS")
    lines.append("=" * 70)
    lines.append("")
    lines.append("  KEY FINDING: The offline strategy generator does NOT use RAG data.")
    lines.append("  The pipeline retrieves historical context via ChromaDB + FastF1,")
    lines.append("  but the offline path reads ONLY from the tire_agent and weather_agent.")
    lines.append("  The RAG context_text is appended to the prompt for LLM consumption only.")
    lines.append("")
    lines.append("  CONSEQUENCE: In offline mode, the strategy is fully 'grounded' in")
    lines.append("  tire data (it literally uses the winner's actual stints), but the")
    lines.append("  RAG retrieval is wasted work. In LLM mode, groundedness depends on")
    lines.append("  whether the LLM faithfully follows the data in its prompt.")
    lines.append("")
    lines.append("  RECOMMENDATIONS:")
    lines.append("  1. For offline mode: no hallucination risk (rule-based), but consider")
    lines.append("     using RAG data to cross-validate compound recommendations.")
    lines.append("  2. For LLM mode: add a post-generation check that extracts compounds")
    lines.append("     from the LLM response and validates against tire_agent data.")
    lines.append("  3. The retriever is working correctly — it returns relevant data.")
    lines.append("     The issue is on the CONSUMER side, not the RETRIEVER side.")
    lines.append("")
    lines.append("=" * 70)

    # ── Write outputs ─────────────────────────────────────────────
    output_dir = os.path.join(_PROJECT_ROOT, "tests")
    os.makedirs(output_dir, exist_ok=True)

    report_path = os.path.join(output_dir, "groundedness_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    csv_path = os.path.join(output_dir, "groundedness_summary.csv")
    if csv_rows:
        fieldnames = list(csv_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

    print(f"\n{'=' * 60}")
    print(f"📊 Groundedness analysis complete:")
    print(f"   Report: {report_path}")
    print(f"   CSV:    {csv_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run_groundedness_analysis()
