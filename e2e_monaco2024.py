"""
End-to-end smoke test -- Monaco 2024 Race Strategy.

Verifies:
  1. Each graph node runs exactly once (fan_in barrier)
  2. strategy_node reads upstream data from state instead of re-fetching
  3. revision_node receives and forwards evaluator feedback (if triggered)
  4. evaluation_result.score is between 0 and 100 inclusive

Run with:
    python e2e_monaco2024.py
"""
from __future__ import annotations

import logging
import sys
import time
from collections import defaultdict
from functools import wraps
from typing import Any

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Step 1: import the workflow module so its top-level names are bound
# ---------------------------------------------------------------------------

import graph.workflow as _wf_mod

# ---------------------------------------------------------------------------
# Step 2: instrument by patching graph.workflow's local name bindings
#         (the module does `from graph.nodes import X`, so X lives in
#         graph.workflow's namespace and that's what build_workflow() uses)
# ---------------------------------------------------------------------------

_call_counts: dict[str, int] = defaultdict(int)
_revision_feedback_seen: list[str | None] = []


def _wrap(name: str, fn):
    @wraps(fn)
    def _counted(state: dict[str, Any]) -> dict[str, Any]:
        _call_counts[name] += 1
        logger.info("[INSTRUMENT] node=%s invocation=%d", name, _call_counts[name])
        result = fn(state)
        if name == "revision_node":
            _revision_feedback_seen.append(state.get("revision_feedback"))
        return result
    return _counted


# Patch the names that build_workflow() will pick up
for _node_name in [
    "supervisor_node", "tire_node", "weather_node", "rag_node",
    "strategy_node", "evaluator_node", "revision_node",
]:
    setattr(_wf_mod, _node_name, _wrap(_node_name, getattr(_wf_mod, _node_name)))

# Rebuild + recompile so the patched references are used
_wf_mod.app = _wf_mod.build_workflow().compile()
app = _wf_mod.app

# ---------------------------------------------------------------------------
# Step 3: track whether strategy_node re-fetches upstream data (Bug B)
#
# The tracking works differently here: we wrap the module-level functions on
# agents.tire_agent etc., and count total calls.  Each dedicated data node
# is responsible for exactly 1 call.  Any extra call = strategy re-fetched.
# ---------------------------------------------------------------------------

_upstream_direct_calls: dict[str, int] = defaultdict(int)


def _track(agent_name: str, original_fn):
    @wraps(original_fn)
    def _tracked(*args, **kwargs):
        _upstream_direct_calls[agent_name] += 1
        return original_fn(*args, **kwargs)
    return _tracked


from agents import tire_agent, weather_agent
from rag import retriever as rag_retriever

tire_agent.analyze_tire_strategy     = _track("analyze_tire_strategy",  tire_agent.analyze_tire_strategy)
weather_agent.analyze_weather_impact = _track("analyze_weather_impact", weather_agent.analyze_weather_impact)
rag_retriever.retrieve_race_context  = _track("retrieve_race_context",  rag_retriever.retrieve_race_context)

# ---------------------------------------------------------------------------
# Run the workflow
# ---------------------------------------------------------------------------

INITIAL_STATE = {
    "messages": [{"role": "user", "content": "Monaco 2024 race strategy"}]
}

print("\n" + "=" * 70)
print("  F1 Race Strategist AI -- Monaco 2024 End-to-End Test")
print("=" * 70)
print("  (FastF1 will use cached data if already downloaded)")
print()

t0 = time.perf_counter()
try:
    result = app.invoke(INITIAL_STATE, config={"recursion_limit": 20})
except Exception as exc:
    print(f"\nFATAL: workflow raised an exception: {exc}")
    logger.exception("Workflow failed")
    sys.exit(1)
elapsed = time.perf_counter() - t0

print(f"\n  Workflow completed in {elapsed:.1f}s")

# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

PASS = "[PASS]"
FAIL = "[FAIL]"
issues: list[str] = []

print("\n" + "-" * 70)
print("  Verification Results")
print("-" * 70)


# 1. Each node ran exactly once
print("\n1. Node invocation counts:")
expected_once = ["supervisor_node", "tire_node", "weather_node", "rag_node",
                 "strategy_node", "evaluator_node"]
for name in expected_once:
    count = _call_counts[name]
    ok = count == 1
    tag = PASS if ok else FAIL
    print(f"   {tag} {name}: {count} call(s) (expected 1)")
    if not ok:
        issues.append(f"{name} ran {count}x (expected 1)")

rev_count_state = result.get("revision_count", 0)
rev_node_calls  = _call_counts["revision_node"]
print(f"   [INFO]  revision_node: {rev_node_calls} call(s), "
      f"state.revision_count={rev_count_state}")


# 2. Bug B -- strategy_node must not re-fetch upstream data
print("\n2. Bug B -- strategy_node must not re-fetch upstream data:")
# Each dedicated data node calls its sub-agent exactly once.
# strategy_node reads from state and must NOT call them again.
# Total calls per sub-agent should be <= 1.
if _upstream_direct_calls:
    for agent, total in _upstream_direct_calls.items():
        ok = total <= 1
        tag = PASS if ok else FAIL
        print(f"   {tag} {agent}: {total} total call(s) (expected <=1)")
        if not ok:
            issues.append(f"Bug B: {agent} called {total}x -- strategy re-fetched data!")
else:
    print("   [INFO] No upstream agent calls recorded (data nodes may have used fallbacks)")


# 3. Bug C -- revision_node receives evaluator feedback
print("\n3. Bug C -- revision_node receives evaluator feedback:")
if rev_node_calls == 0:
    print("   [INFO] revision_node did not run (strategy accepted on first attempt)")
else:
    for i, fb in enumerate(_revision_feedback_seen, 1):
        ok = fb is not None
        tag = PASS if ok else FAIL
        preview = fb[:80] + "..." if fb and len(fb) > 80 else fb
        print(f"   {tag} Revision #{i}: feedback={'<present>' if fb else '<None>'}")
        if fb:
            print(f"         preview: {repr(preview)}")
        if not ok:
            issues.append(f"Bug C: revision #{i} received no feedback")


# 4. Score in [0, 100]
print("\n4. Evaluation score bounds:")
evaluation = result.get("evaluation_result") or {}
score      = evaluation.get("score")
verdict    = evaluation.get("verdict", "?")
summary    = evaluation.get("summary", "")

if score is None:
    tag = FAIL
    issues.append("evaluation_result.score is None")
elif 0 <= score <= 100:
    tag = PASS
else:
    tag = FAIL
    issues.append(f"score={score} is outside [0, 100]")

safe_verdict = verdict.encode("ascii", errors="replace").decode("ascii")
safe_summary = summary[:100].encode("ascii", errors="replace").decode("ascii")
print(f"   {tag} score={score}, verdict={safe_verdict}")
print(f"        summary: {safe_summary}")


# 5. State key presence
print("\n5. Key state values populated:")
keys = ["circuit", "year", "session_type", "tire_analysis",
        "weather_analysis", "rag_context", "strategy_recommendation"]
for k in keys:
    v = result.get(k)
    present = v is not None
    tag = PASS if present else "[WARN]"
    display = str(v)[:60] if present else "None"
    print(f"   {tag} {k}: {display}")


# Overall result
print("\n" + "-" * 70)
if issues:
    print(f"  RESULT: FAILED -- {len(issues)} issue(s):")
    for iss in issues:
        print(f"    * {iss}")
    sys.exit(1)
else:
    print("  RESULT: ALL CHECKS PASSED")

# Show strategy preview
strategy   = result.get("strategy_recommendation") or {}
rec_text   = strategy.get("recommendation_text", "")
confidence = strategy.get("confidence", "?")
print(f"\n  Strategy confidence: {confidence}")
print(f"  Recommendation preview (first 400 chars):\n")
# Encode-safe output for Windows CP1252
safe_text = rec_text[:400].encode("ascii", errors="replace").decode("ascii")
print(safe_text)
if len(rec_text) > 400:
    print(f"  ... [{len(rec_text) - 400} chars truncated]")

print("\n" + "=" * 70)
print("  End-to-end test complete.")
print("=" * 70)
