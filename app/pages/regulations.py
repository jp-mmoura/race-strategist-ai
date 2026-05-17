"""
Regulations Page Handler — FIA rulebook Q&A, scenario checker, changelog.
"""

import sys, os, logging

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logger = logging.getLogger(__name__)


def get_welcome_message() -> str:
    return (
        "## 📖 FIA Regulations Assistant\n\n"
        "Your **AI-powered rulebook**. Ask about sporting regulations, "
        "technical rules, or check if a scenario is compliant.\n\n---\n\n"
        "| Feature | Description |\n"
        "|---------|-------------|\n"
        "| **Regulation Q&A** | Ask about any FIA rule in plain language |\n"
        "| **Scenario Checker** | Input an edge case, get compliance assessment |\n"
        "| **Article Viewer** | Raw regulation text + AI interpretation |\n"
        "| **Changelog** | Track regulation updates across seasons |\n"
        "| **Comparison** | Compare rules between seasons |\n\n"
        "---\n\nAsk a question like _\"Can a team change tires under VSC?\"_ 🚀"
    )


def get_starters():
    import chainlit as cl
    return [
        cl.Starter(label="📋 VSC Rules", message="Can a team change a front wing under Virtual Safety Car conditions? What are the restrictions?", icon="/public/icon_weather.png"),
        cl.Starter(label="⚖️ Scenario Check", message="A driver exceeds track limits 4 times at turn 9. What penalties apply under current regulations?", icon="/public/icon_weather.png"),
        cl.Starter(label="🔄 2025 Changes", message="What are the major regulation changes for the 2025 F1 season compared to 2024?", icon="/public/icon_weather.png"),
        cl.Starter(label="🏎️ Parc Fermé", message="Explain the parc fermé rules. What modifications can a team make between qualifying and the race?", icon="/public/icon_weather.png"),
    ]


async def handle_message(user_input: str, history: list) -> str:
    lower = user_input.lower()
    if "scenario" in lower or "check" in lower or "compli" in lower or "penalty" in lower or "penalt" in lower:
        return _scenario_checker(user_input)
    if "change" in lower and ("202" in lower or "season" in lower or "new" in lower):
        return _changelog(user_input)
    if "parc" in lower or "fermé" in lower or "ferme" in lower:
        return _parc_ferme()
    if any(k in lower for k in ("vsc", "safety car", "red flag", "flag")):
        return _safety_car_rules()
    if any(k in lower for k in ("tire", "tyre", "wing", "front wing", "drs")):
        return _technical_rules(user_input)
    return _general_qa(user_input)


def _safety_car_rules():
    return (
        "### 🚩 Safety Car & VSC Regulations\n\n---\n\n"
        "#### Full Safety Car (SC)\n\n"
        "| Rule | Detail |\n"
        "|------|--------|\n"
        "| Pit lane | **Open** — teams may pit freely |\n"
        "| Tire changes | ✅ Allowed |\n"
        "| Front wing change | ✅ Allowed |\n"
        "| Overtaking | ❌ Prohibited until SC line |\n"
        "| Delta time | Must stay within SC delta |\n"
        "| Lapped cars | May unlap (Race Director discretion) |\n\n"
        "#### Virtual Safety Car (VSC)\n\n"
        "| Rule | Detail |\n"
        "|------|--------|\n"
        "| Pit lane | **Open** — but pit-lane time is monitored |\n"
        "| Tire changes | ✅ Allowed |\n"
        "| Front wing change | ⚠️ Allowed but risky (strict time delta) |\n"
        "| Speed | Must maintain VSC delta (±5%) |\n"
        "| DRS | ❌ Disabled |\n\n"
        "#### Red Flag\n\n"
        "| Rule | Detail |\n"
        "|------|--------|\n"
        "| All cars | Return to pit lane / grid |\n"
        "| Repairs | ✅ Any repairs allowed (inc. tire changes) |\n"
        "| Restart | Standing or rolling start |\n\n"
        "📖 *Ref: FIA Sporting Regulations Art. 55–57*\n\n"
        "> 🔧 *Connect LLM + regulation RAG for full article text.*"
    )


def _scenario_checker(user_input):
    return (
        "### ⚖️ Regulation Scenario Assessment\n\n---\n\n"
        f"**Scenario**: _{user_input}_\n\n"
        "#### Compliance Assessment\n\n"
        "| Check | Result | Reference |\n"
        "|-------|--------|----------|\n"
        "| Track limits (3 strikes) | ⚠️ Warning issued | Art. 33.3 |\n"
        "| Track limits (4th offense) | 🔴 **Penalty triggered** | Art. 33.3 |\n"
        "| Penalty type | 5-second time penalty | Art. 54.3(c) |\n"
        "| Escalation | 10s if repeat offender | Art. 54.3(d) |\n\n"
        "#### Ruling\n\n"
        "🔴 **PENALTY APPLIES** — The driver has exceeded the 3-strike threshold "
        "for track limits at the designated corner.\n\n"
        "**Standard penalty**: 5-second time penalty added at next pit stop or to race time.\n\n"
        "**Precedent**: Verstappen, 2024 Austrian GP — received 5s penalty for "
        "4× track limits at Turn 9/10.\n\n"
        "> 🔧 *Connect LLM + regulation RAG for binding rule text.*"
    )


def _changelog(user_input):
    return (
        "### 🔄 Major Regulation Changes — 2025 Season\n\n---\n\n"
        "#### Sporting Regulations\n\n"
        "| Change | 2024 | 2025 | Impact |\n"
        "|--------|------|------|--------|\n"
        "| Sprint format | Sprint Shootout + Sprint | Revised Sprint Qualifying | Medium |\n"
        "| Penalty points | 12-point threshold | Adjusted scale | Low |\n"
        "| Track limits | 3-strike system | Automated monitoring | High |\n"
        "| Budget cap | $135M | $135M (adjusted inflation) | Medium |\n\n"
        "#### Technical Regulations\n\n"
        "| Change | Detail | Impact |\n"
        "|--------|--------|--------|\n"
        "| Floor edge | Modified floor edge design rules | High — downforce affected |\n"
        "| Weight limit | +3kg (798kg) | Low |\n"
        "| Power unit | No changes (pre-2026 freeze) | None |\n"
        "| Crash structures | Revised side impact requirements | Medium |\n\n"
        "#### 2026 Preview (Upcoming)\n\n"
        "- **New PU regulations** — increased electrical power\n"
        "- **Active aero** — adjustable front/rear wing elements\n"
        "- **Chassis redesign** — smaller, lighter cars\n\n"
        "> 🔧 *Connect LLM + regulation RAG for full change details.*"
    )


def _parc_ferme():
    return (
        "### 🔒 Parc Fermé Rules\n\n---\n\n"
        "Parc fermé conditions apply from **the start of qualifying** until "
        "**the start of the race**.\n\n"
        "#### Allowed Modifications\n\n"
        "| Modification | Allowed? | Notes |\n"
        "|-------------|----------|-------|\n"
        "| Front wing angle | ✅ Yes | Within defined range |\n"
        "| Tire pressures | ✅ Yes | Must meet minimums |\n"
        "| Brake ducts | ✅ Yes | Cooling config only |\n"
        "| Fuel load | ✅ Yes | Can add fuel |\n"
        "| Weight ballast | ❌ No | Fixed from quali |\n"
        "| Suspension setup | ❌ No | Cannot change geometry |\n"
        "| Engine mode | ✅ Yes | Software changes allowed |\n"
        "| Broken parts | ✅ Yes | Like-for-like replacement only |\n\n"
        "#### Penalty for Breach\n\n"
        "**Pit lane start** — if a team makes unauthorized changes, the car "
        "must start from the pit lane.\n\n"
        "📖 *Ref: FIA Sporting Regulations Art. 40*\n\n"
        "> 🔧 *Connect LLM + regulation RAG for binding article text.*"
    )


def _technical_rules(user_input):
    return (
        "### 🏎️ Technical Regulation Reference\n\n---\n\n"
        f"**Query**: _{user_input}_\n\n"
        "I can look up specific technical regulations covering:\n\n"
        "| Category | Key Rules |\n"
        "|----------|----------|\n"
        "| Aerodynamics | Floor, front/rear wing, DRS rules |\n"
        "| Power Unit | Engine, ERS, fuel flow limits |\n"
        "| Tires | Compound allocations, blanket temps |\n"
        "| Chassis | Weight, dimensions, crash structures |\n"
        "| Safety | Halo, fire suppression, driver equipment |\n\n"
        "> 🔧 *Connect LLM + regulation RAG for full article retrieval.*"
    )


def _general_qa(user_input):
    return (
        "### 📖 Regulations Assistant\n\nI can help with:\n"
        "- **Safety car rules** — _\"What can you do under VSC?\"_\n"
        "- **Scenario checks** — _\"4 track limits at turn 9, what penalty?\"_\n"
        "- **Rule changes** — _\"What's new in the 2025 regulations?\"_\n"
        "- **Parc fermé** — _\"What can teams change after qualifying?\"_\n"
        "- **Technical rules** — _\"DRS activation rules\"_\n\n"
        "Ask any regulation question!"
    )
