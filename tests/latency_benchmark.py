"""
Latency Benchmark — profile each pipeline node and the full flow.

Measures:
  1. Per-node wall-clock time (tire, weather, RAG, strategy, evaluator)
  2. Total sequential pipeline time
  3. Total with parallel tire + weather + RAG (ThreadPoolExecutor)
  4. Speedup factor from parallelization

Identifies the bottleneck node and evaluates whether parallelizing
independent agents (tire, weather, RAG) provides meaningful speedup.

Output:
  tests/latency_report.txt       — detailed findings
  tests/latency_summary.csv      — per-scenario timings
"""

from __future__ import annotations

import csv
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

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

LATENCY_THRESHOLD = 15.0  # seconds — above this, we flag for optimization


# ===================================================================
# Timing helpers
# ===================================================================

@dataclass
class NodeTiming:
    """Timing result for a single node."""
    name: str
    elapsed_sec: float = 0.0
    success: bool = True
    error: str = ""

    @property
    def icon(self) -> str:
        if not self.success:
            return "💥"
        if self.elapsed_sec > 5.0:
            return "🔴"
        if self.elapsed_sec > 2.0:
            return "🟡"
        return "🟢"


@dataclass
class ScenarioBenchmark:
    """Full benchmark result for one scenario."""
    name: str
    circuit: str
    year: int
    node_timings: list[NodeTiming] = field(default_factory=list)
    total_sequential_sec: float = 0.0
    total_parallel_sec: float = 0.0
    parallel_data_sec: float = 0.0  # just the parallel tire+weather+rag
    speedup_factor: float = 1.0


def _time_call(name: str, fn: Callable, *args, **kwargs) -> tuple[NodeTiming, Any]:
    """Time a function call and return (timing, result)."""
    t0 = time.perf_counter()
    try:
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        return NodeTiming(name=name, elapsed_sec=elapsed), result
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return NodeTiming(name=name, elapsed_sec=elapsed, success=False, error=str(exc)), None


# ===================================================================
# Sequential pipeline profiling
# ===================================================================

def profile_sequential(circuit: str, year: int) -> ScenarioBenchmark:
    """Profile each pipeline node sequentially."""
    from agents.evaluator_agent import evaluate_strategy
    from agents.strategist_agent import generate_strategy_offline
    from agents.tire_agent import analyze_tire_strategy
    from agents.weather_agent import analyze_weather_impact
    from rag.retriever import retrieve_race_context

    bench = ScenarioBenchmark(name="", circuit=circuit, year=year)
    t_total = time.perf_counter()

    # 1. Tire Agent
    t_tire, tire = _time_call("tire_agent", analyze_tire_strategy, circuit, year, "R")
    bench.node_timings.append(t_tire)

    # 2. Weather Agent
    t_wx, weather = _time_call("weather_agent", analyze_weather_impact, circuit, None, year, "R")
    bench.node_timings.append(t_wx)

    # 3. RAG Retriever
    t_rag, rag = _time_call(
        "rag_retriever", retrieve_race_context,
        query=f"race strategy {circuit} {year}",
        year=year, circuit=circuit, session_type="R",
    )
    bench.node_timings.append(t_rag)

    # 4. Strategy (offline — rule-based, no LLM latency)
    t_strat, strategy = _time_call(
        "strategy_offline", generate_strategy_offline, circuit, year,
    )
    bench.node_timings.append(t_strat)

    # 5. Evaluator
    if strategy:
        t_eval, evaluation = _time_call(
            "evaluator", evaluate_strategy, strategy, tire, weather,
        )
    else:
        t_eval = NodeTiming(name="evaluator", elapsed_sec=0, success=False, error="No strategy")
    bench.node_timings.append(t_eval)

    bench.total_sequential_sec = time.perf_counter() - t_total
    return bench


# ===================================================================
# Parallel pipeline profiling
# ===================================================================

def profile_parallel(circuit: str, year: int) -> ScenarioBenchmark:
    """Profile the pipeline with tire + weather + RAG running in parallel."""
    from agents.evaluator_agent import evaluate_strategy
    from agents.strategist_agent import generate_strategy_offline
    from agents.tire_agent import analyze_tire_strategy
    from agents.weather_agent import analyze_weather_impact
    from rag.retriever import retrieve_race_context

    bench = ScenarioBenchmark(name="", circuit=circuit, year=year)
    t_total = time.perf_counter()

    # ── Parallel phase: tire + weather + RAG ──────────────────────
    tire = weather = rag = None
    timings: dict[str, NodeTiming] = {}

    t_parallel = time.perf_counter()
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(
                _time_call, "tire_agent", analyze_tire_strategy, circuit, year, "R"
            ): "tire",
            executor.submit(
                _time_call, "weather_agent", analyze_weather_impact, circuit, None, year, "R"
            ): "weather",
            executor.submit(
                _time_call, "rag_retriever", retrieve_race_context,
                f"race strategy {circuit} {year}", year, circuit, "R",
            ): "rag",
        }

        for future in as_completed(futures):
            key = futures[future]
            timing, result = future.result()
            timings[key] = timing
            if key == "tire":
                tire = result
            elif key == "weather":
                weather = result
            elif key == "rag":
                rag = result

    bench.parallel_data_sec = time.perf_counter() - t_parallel

    bench.node_timings.append(timings.get("tire", NodeTiming("tire_agent")))
    bench.node_timings.append(timings.get("weather", NodeTiming("weather_agent")))
    bench.node_timings.append(timings.get("rag", NodeTiming("rag_retriever")))

    # ── Sequential phase: strategy + evaluator ────────────────────
    t_strat, strategy = _time_call(
        "strategy_offline", generate_strategy_offline, circuit, year,
    )
    bench.node_timings.append(t_strat)

    if strategy:
        t_eval, evaluation = _time_call(
            "evaluator", evaluate_strategy, strategy, tire, weather,
        )
    else:
        t_eval = NodeTiming(name="evaluator", elapsed_sec=0, success=False, error="No strategy")
    bench.node_timings.append(t_eval)

    bench.total_parallel_sec = time.perf_counter() - t_total
    return bench


# ===================================================================
# Main runner
# ===================================================================

def run_latency_benchmark():
    lines: list[str] = []
    csv_rows: list[dict] = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines.append("=" * 70)
    lines.append("  LATENCY BENCHMARK — F1 Race Strategist AI")
    lines.append(f"  Run at: {timestamp}")
    lines.append(f"  Threshold: {LATENCY_THRESHOLD}s per full pipeline")
    lines.append("=" * 70)
    lines.append("")
    lines.append("  Pipeline: tire_agent → weather_agent → rag_retriever")
    lines.append("            → strategy_offline → evaluator")
    lines.append("")
    lines.append("  NOTE: strategy_offline internally re-runs tire + weather + RAG")
    lines.append("  (build_strategy_context), so total time ≈ 2× the data phase.")
    lines.append("  The 'parallel' benchmark parallelizes the STANDALONE data phase.")
    lines.append("")

    all_seq: list[float] = []
    all_par: list[float] = []
    all_speedups: list[float] = []

    for i, sc in enumerate(SCENARIOS, 1):
        print(f"\n[{i}/{len(SCENARIOS)}] Benchmarking: {sc['name']}")

        # ── Sequential ────────────────────────────────────────────
        print(f"  Sequential ...", end=" ", flush=True)
        seq = profile_sequential(sc["circuit"], sc["year"])
        seq.name = sc["name"]
        print(f"{seq.total_sequential_sec:.2f}s")

        # ── Parallel ──────────────────────────────────────────────
        print(f"  Parallel   ...", end=" ", flush=True)
        par = profile_parallel(sc["circuit"], sc["year"])
        par.name = sc["name"]
        print(f"{par.total_parallel_sec:.2f}s")

        # Speedup
        if par.total_parallel_sec > 0:
            speedup = seq.total_sequential_sec / par.total_parallel_sec
        else:
            speedup = 1.0
        par.speedup_factor = speedup

        all_seq.append(seq.total_sequential_sec)
        all_par.append(par.total_parallel_sec)
        all_speedups.append(speedup)

        # ── Report ────────────────────────────────────────────────
        over_threshold = seq.total_sequential_sec > LATENCY_THRESHOLD
        flag = " ⚠️ OVER THRESHOLD" if over_threshold else ""

        lines.append(f"{'━' * 70}")
        lines.append(f"  {sc['name']} ({sc['circuit']} {sc['year']}){flag}")
        lines.append(f"{'━' * 70}")
        lines.append("")

        # Sequential node breakdown
        lines.append("  SEQUENTIAL breakdown:")
        seq_data_total = 0.0
        for t in seq.node_timings:
            pct = (t.elapsed_sec / seq.total_sequential_sec * 100) if seq.total_sequential_sec else 0
            bar_len = int(pct / 2)
            bar = "█" * bar_len + "░" * (25 - bar_len)
            lines.append(
                f"    {t.icon} {t.name:.<22s} "
                f"{t.elapsed_sec:6.2f}s  ({pct:4.1f}%)  {bar}"
            )
            if t.name in ("tire_agent", "weather_agent", "rag_retriever"):
                seq_data_total += t.elapsed_sec
            if t.error:
                lines.append(f"       Error: {t.error}")

        lines.append(f"    {'─' * 52}")
        lines.append(f"    Total sequential:     {seq.total_sequential_sec:6.2f}s")
        lines.append(f"    Data phase (seq):     {seq_data_total:6.2f}s")
        lines.append("")

        # Parallel results
        lines.append("  PARALLEL breakdown:")
        for t in par.node_timings:
            lines.append(f"    {t.icon} {t.name:.<22s} {t.elapsed_sec:6.2f}s")
        lines.append(f"    {'─' * 52}")
        lines.append(f"    Total parallel:       {par.total_parallel_sec:6.2f}s")
        lines.append(f"    Data phase (par):     {par.parallel_data_sec:6.2f}s")
        lines.append(f"    Speedup:              {speedup:.2f}×")
        lines.append("")

        # Bottleneck analysis
        bottleneck = max(seq.node_timings, key=lambda t: t.elapsed_sec)
        lines.append(f"  🔍 Bottleneck: {bottleneck.name} ({bottleneck.elapsed_sec:.2f}s)")

        if over_threshold:
            lines.append(f"  ⚠️ Total ({seq.total_sequential_sec:.1f}s) exceeds "
                         f"{LATENCY_THRESHOLD}s threshold")
        else:
            lines.append(f"  ✅ Total ({seq.total_sequential_sec:.1f}s) within "
                         f"{LATENCY_THRESHOLD}s threshold")
        lines.append("")

        # CSV row
        csv_row = {
            "scenario": sc["name"],
            "circuit": sc["circuit"],
            "year": sc["year"],
        }
        for t in seq.node_timings:
            csv_row[f"seq_{t.name}"] = f"{t.elapsed_sec:.2f}"
        csv_row["seq_total"] = f"{seq.total_sequential_sec:.2f}"
        csv_row["par_data_phase"] = f"{par.parallel_data_sec:.2f}"
        csv_row["par_total"] = f"{par.total_parallel_sec:.2f}"
        csv_row["speedup"] = f"{speedup:.2f}"
        csv_row["bottleneck"] = bottleneck.name
        csv_row["over_threshold"] = "YES" if over_threshold else "NO"
        csv_rows.append(csv_row)

    # =================================================================
    # AGGREGATE RESULTS
    # =================================================================
    lines.append("")
    lines.append("=" * 70)
    lines.append("  AGGREGATE RESULTS")
    lines.append("=" * 70)
    lines.append("")

    if all_seq:
        avg_seq = sum(all_seq) / len(all_seq)
        avg_par = sum(all_par) / len(all_par)
        avg_speedup = sum(all_speedups) / len(all_speedups)
        max_seq = max(all_seq)
        min_seq = min(all_seq)
        over_count = sum(1 for s in all_seq if s > LATENCY_THRESHOLD)

        lines.append(f"  Sequential latency:")
        lines.append(f"    Average:  {avg_seq:.2f}s")
        lines.append(f"    Range:    {min_seq:.2f}s – {max_seq:.2f}s")
        lines.append(f"    Over {LATENCY_THRESHOLD}s: {over_count}/{len(all_seq)}")
        lines.append("")
        lines.append(f"  Parallel latency:")
        lines.append(f"    Average:  {avg_par:.2f}s")
        lines.append(f"    Avg speedup: {avg_speedup:.2f}×")
        lines.append(f"    Time saved:  {avg_seq - avg_par:.2f}s avg")
        lines.append("")

        # Per-node average
        node_totals: dict[str, list[float]] = {}
        for sc_bench in [profile_sequential(s["circuit"], s["year"]) for s in SCENARIOS[:1]]:
            pass  # skip re-profiling, use CSV data instead

        # Architecture diagnosis
        lines.append("  ╔══════════════════════════════════════════════════════╗")
        if avg_seq <= LATENCY_THRESHOLD:
            lines.append(f"  ║  VERDICT: ✅ WITHIN THRESHOLD ({avg_seq:.1f}s avg)           ║")
            lines.append(f"  ║  Parallelization saves ~{avg_seq - avg_par:.1f}s but is optional.   ║")
        else:
            lines.append(f"  ║  VERDICT: ⚠️ OVER THRESHOLD ({avg_seq:.1f}s avg)            ║")
            lines.append(f"  ║  Parallelization recommended (saves ~{avg_seq - avg_par:.1f}s).     ║")
        lines.append(f"  ╚══════════════════════════════════════════════════════╝")
        lines.append("")

        # Architecture note about double data fetching
        lines.append("  ARCHITECTURE NOTE:")
        lines.append("  ─────────────────")
        lines.append("  The strategy_offline node internally calls build_strategy_context(),")
        lines.append("  which re-invokes tire_agent, weather_agent, and RAG sequentially.")
        lines.append("  This means data is fetched TWICE in the LangGraph flow:")
        lines.append("    1st: by the standalone tire/weather/rag nodes (parallel in graph)")
        lines.append("    2nd: by strategy_node → build_strategy_context() (sequential)")
        lines.append("")
        lines.append("  FIX: Modify strategy_node to read pre-computed results from state")
        lines.append("  instead of re-running build_strategy_context(). This would halve")
        lines.append("  the data-fetching time. FastF1 caching mitigates this for cached")
        lines.append("  sessions, but the first run for a new session would benefit greatly.")

    lines.append("")
    lines.append("=" * 70)

    # ── Write outputs ─────────────────────────────────────────────
    output_dir = os.path.join(_PROJECT_ROOT, "tests")
    os.makedirs(output_dir, exist_ok=True)

    report_path = os.path.join(output_dir, "latency_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    csv_path = os.path.join(output_dir, "latency_summary.csv")
    if csv_rows:
        fieldnames = list(csv_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

    print(f"\n{'=' * 60}")
    print(f"📊 Latency benchmark complete:")
    print(f"   Report:  {report_path}")
    print(f"   CSV:     {csv_path}")
    if all_seq:
        print(f"   Avg sequential: {sum(all_seq)/len(all_seq):.2f}s")
        print(f"   Avg parallel:   {sum(all_par)/len(all_par):.2f}s")
        print(f"   Avg speedup:    {sum(all_speedups)/len(all_speedups):.2f}×")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run_latency_benchmark()
