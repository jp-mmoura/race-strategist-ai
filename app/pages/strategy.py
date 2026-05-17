"""
Strategy Page Handler — pit-stop simulation, undercut analysis, AI advisor.
"""

import sys, os, logging

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logger = logging.getLogger(__name__)


def get_welcome_message() -> str:
    return (
        "## 🏁 Race Strategy Advisor\n\n"
        "Welcome to the **Strategy Control Room**. I simulate pit-stop "
        "scenarios and recommend optimal windows.\n\n---\n\n"
        "| Feature | Description |\n"
        "|---------|-------------|\n"
        "| **Pit Window Optimizer** | Find ideal lap range for each stop |\n"
        "| **Undercut / Overcut** | Compare gain vs. traffic cost |\n"
        "| **What-If Simulation** | \"What if VER pits lap 18?\" |\n"
        "| **Safety Car Model** | SC probability per lap window |\n"
        "| **Compound Sequencing** | Best S→M→H vs M→H ordering |\n\n"
        "---\n\nTell me the **Grand Prix**, **session**, and **driver** — or use a starter below. 🚀"
    )


def get_starters():
    import chainlit as cl
    return [
        cl.Starter(label="🏁 Pit Strategy", message="Simulate the optimal pit-stop strategy for the Monaco Grand Prix. Compare 2-stop vs 1-stop.", icon="/public/icon_strategy.png"),
        cl.Starter(label="⚔️ Undercut Analysis", message="Analyze the undercut window at Silverstone. When should a driver pit to gain position?", icon="/public/icon_strategy.png"),
        cl.Starter(label="🔮 What-If Scenario", message="What if Verstappen pits on lap 18 at Monza? How does this affect Hamilton?", icon="/public/icon_strategy.png"),
        cl.Starter(label="🚨 Safety Car Impact", message="How should a team adjust strategy if a safety car comes out on lap 25 at Spa?", icon="/public/icon_strategy.png"),
    ]


async def handle_message(user_input: str, history: list) -> str:
    lower = user_input.lower()
    if "what if" in lower or "simulate" in lower:
        return _whatif(user_input)
    if "undercut" in lower or "overcut" in lower:
        return _undercut()
    if "safety car" in lower or "vsc" in lower:
        return _safety_car()
    if any(k in lower for k in ("pit", "stop", "strategy", "window", "stint")):
        return _pit_strategy(user_input)
    return _fallback()


def _extract_circuit(text):
    try:
        from tools.weather_tool import F1_CIRCUITS
        for n in F1_CIRCUITS:
            if n.lower() in text.lower():
                return n
    except ImportError:
        pass
    return None


def _pit_strategy(user_input):
    c = _extract_circuit(user_input) or "Selected Circuit"
    return (
        f"### 🏁 Pit Strategy — {c}\n\n---\n\n"
        "#### Optimal 2-Stop\n\n"
        "| Stint | Compound | Laps | Avg Pace | Notes |\n"
        "|-------|----------|------|----------|-------|\n"
        "| 1 | 🔴 Soft | L1–16 | 1:32.4 | Aggressive start |\n"
        "| 2 | 🟡 Medium | L17–38 | 1:33.1 | Longest stint |\n"
        "| 3 | ⚪ Hard | L39–57 | 1:33.8 | Push to finish |\n\n"
        "#### Alternative 1-Stop\n\n"
        "| Stint | Compound | Laps | Avg Pace | Notes |\n"
        "|-------|----------|------|----------|-------|\n"
        "| 1 | 🟡 Medium | L1–28 | 1:33.5 | Conservative |\n"
        "| 2 | ⚪ Hard | L29–57 | 1:34.2 | Manage deg |\n\n"
        "| Strategy | Total Time | Pit Loss | Net |\n"
        "|----------|-----------|----------|-----|\n"
        "| 2-stop | 1:32:14 | ~44s | **Fastest** |\n"
        "| 1-stop | 1:32:38 | ~22s | +24s |\n\n"
        "💡 **2-stop** is ~24s faster on pace. On high-traffic circuits, 1-stop avoids pit-lane position loss.\n\n"
        "> 🔧 *Connect LLM agent + FastF1 for live projections.*"
    )


def _undercut():
    return (
        "### ⚔️ Undercut / Overcut Analysis\n\n---\n\n"
        "| Lap Window | Tire Delta | Pit Loss | Undercut Gain | Viable? |\n"
        "|------------|-----------|----------|---------------|---------|\n"
        "| L12–15 | 1.8s/lap | 22s | **~3.2s** | ✅ Yes |\n"
        "| L16–18 | 1.2s/lap | 22s | ~1.4s | ⚠️ Marginal |\n"
        "| L19+ | 0.6s/lap | 22s | −0.8s | ❌ Too late |\n\n"
        "**Key factors**: tire delta, out-lap performance, traffic, pit crew speed.\n\n"
        "**Overcut** works when track is rubbering in and rival has a slow out-lap on cold tires.\n\n"
        "> 🔧 *Connect LLM for driver-specific undercut modeling.*"
    )


def _safety_car():
    return (
        "### 🚨 Safety Car Strategy\n\n---\n\n"
        "| Action | SC (Full) | VSC |\n"
        "|--------|-----------|-----|\n"
        "| Pit time loss | ~12s (vs ~22s) | ~8s |\n"
        "| Change tires? | ✅ Yes | ✅ Yes |\n"
        "| Should you pit? | **Yes** if >40% race left | **Yes** if tire age >15 laps |\n\n"
        "#### SC Probability Model\n\n"
        "| Lap Window | Probability | Action |\n"
        "|------------|------------|--------|\n"
        "| L1–5 | 45% | Stay out, pit under SC |\n"
        "| L10–20 | 30% | Plan primary stop here |\n"
        "| L25–40 | 25% | Extend if possible |\n"
        "| L40+ | 15% | Committed |\n\n"
        "> 🔧 *Connect LLM for real-time SC modeling.*"
    )


def _whatif(user_input):
    return (
        "### 🔮 What-If Simulation\n\n---\n\n"
        f"**Scenario**: _{user_input}_\n\n"
        "| Driver | Action | Proj. Position | Delta |\n"
        "|--------|--------|---------------|-------|\n"
        "| VER | Pits L18 (S→H) | P1 | — |\n"
        "| HAM | Stays out to L25 | P2 | +4.2s |\n"
        "| NOR | Covers VER, pits L19 | P3 | +6.8s |\n"
        "| LEC | Extends to L28 | P2↔P4 | High variance |\n\n"
        "**Impact**: VER emerges P3 on fresh Hards, gains ~0.8s/lap, back to P1 by L30.\n\n"
        "| Risk | Probability | Impact |\n"
        "|------|------------|--------|\n"
        "| SC neutralizes gap | 25% | High |\n"
        "| Traffic on out-lap | 15% | Medium |\n"
        "| Hard tire graining | 10% | Low |\n\n"
        "> 🔧 *Connect LLM for dynamic scenario modeling.*"
    )


def _fallback():
    return (
        "### 🏁 Strategy Advisor\n\nI can help with:\n"
        "- **Pit windows** — _\"Best pit window for Monza\"_\n"
        "- **Undercut/overcut** — _\"Can HAM undercut VER at Spa?\"_\n"
        "- **Safety car** — _\"What if SC on lap 25?\"_\n"
        "- **What-if** — _\"What if Verstappen pits lap 18?\"_\n\n"
        "Tell me the **Grand Prix** and **scenario**!"
    )
