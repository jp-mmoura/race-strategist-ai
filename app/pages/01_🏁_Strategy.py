"""
01 — Strategy Hub Page.

Interactive chat with the LangGraph orchestrator. Renders:
  - Chat history and interactive user input
  - Recommendations with evaluation loop results
  - Visual Pirelli stint timeline
  - Context parameters in the sidebar (weather, safety car, laps remaining)
"""

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st
from app.styles.theme import inject_css, RED_ACCENT
from app.components.chat_interface import (
    init_chat,
    render_chat_history,
    render_chat_input,
    add_message,
)
from app.components.tire_chart import render_strategy_timeline

st.set_page_config(
    page_title="Race Strategy AI — F1 Strategist",
    page_icon="🏁",
    layout="wide",
)
inject_css()

# ── Constants ─────────────────────────────────────────────────────────
CIRCUITS = [
    "Australia", "Bahrain", "China", "Spain", "Monaco", "Canada",
    "Austria", "Britain", "Hungary", "Belgium", "Italy", "Singapore",
    "Japan", "USA", "Mexico", "Brazil", "AbuDhabi", "Netherlands",
    "Azerbaijan", "Miami", "Qatar", "Las Vegas", "Saudi Arabia",
]
YEARS = list(range(2025, 2017, -1))

# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Race Control Context")
    selected_circuit = st.selectbox(
        "Current Circuit", CIRCUITS,
        index=CIRCUITS.index("Britain"),
        key="strat_circuit",
    )
    selected_year = st.selectbox(
        "Reference Season", YEARS,
        index=0,
        key="strat_year",
    )

    st.markdown("---")
    st.markdown("### 🌦️ Environment")
    weather_cond = st.selectbox(
        "Weather Condition",
        ["Dry", "Light Rain", "Heavy Rain", "Damp Track"],
        index=0,
    )
    track_temp = st.slider("Track Temp (°C)", 10, 60, 35)

    st.markdown("---")
    st.markdown("### 🚨 Live Incidents")
    safety_car = st.selectbox(
        "Safety Car / VSC Status",
        ["Normal Racing", "Virtual Safety Car", "Safety Car", "Red Flag"],
        index=0,
    )
    laps_remaining = st.slider("Laps Remaining", 1, 80, 52)

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state["chat_messages"] = []
        st.rerun()

# ── Header ────────────────────────────────────────────────────────────
st.markdown("# 🏁 Race Strategy Control Hub")
st.markdown(
    f'<p style="color:#A0A0A0; margin-top:-0.5rem;">'
    f"LangGraph Orchestrator · <b>{selected_circuit} GP {selected_year}</b> · "
    f"{weather_cond} · {laps_remaining} laps remaining"
    f"</p>",
    unsafe_allow_html=True,
)

# ── Chat ──────────────────────────────────────────────────────────────
init_chat()

messages = st.session_state.get("chat_messages", [])

# Empty state — example shortcuts
if not messages:
    st.markdown(
        f"""
        <div style="text-align:center; padding:2.5rem 0 1.5rem; color:#A0A0A0;">
            <div style="font-size:3rem; margin-bottom:0.8rem;">🏁</div>
            <div style="font-size:1.1rem; color:#fff; margin-bottom:0.4rem;">
                Ready to analyse <b style="color:{RED_ACCENT};">{selected_circuit} GP {selected_year}</b>
            </div>
            <div style="font-size:0.88rem;">Describe a scenario or use one of the shortcuts below.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    q_col1, q_col2, q_col3 = st.columns(3)
    with q_col1:
        if st.button("📊 Full Strategy", use_container_width=True):
            add_message(
                "user",
                f"Generate a full race strategy for the {selected_circuit} GP {selected_year}.",
            )
            st.rerun()
    with q_col2:
        if st.button("🔄 Safety Car Scenario", use_container_width=True):
            add_message(
                "user",
                f"With {safety_car} and {laps_remaining} laps remaining, "
                f"what is the best strategy for the {selected_circuit} GP?",
            )
            st.rerun()
    with q_col3:
        if st.button("🌧️ Rain Contingency", use_container_width=True):
            add_message(
                "user",
                f"What is the rain contingency plan for the {selected_circuit} GP {selected_year}?",
            )
            st.rerun()

# Render history
render_chat_history()

# User input
user_message = render_chat_input(placeholder="Describe your race scenario...")
if user_message:
    add_message("user", user_message)
    st.rerun()

# ── Run workflow if last message is from user ─────────────────────────
messages = st.session_state.get("chat_messages", [])
if messages and messages[-1]["role"] == "user":
    user_query = messages[-1]["content"]

    with st.chat_message("assistant", avatar="🏁"):
        status_ph = st.empty()
        status_ph.markdown(
            "⏳ *Analysing strategy… Building tire, weather and race history context…*"
        )

        try:
            from agents.supervisor import run_graph

            context_str = (
                f"Weather: {weather_cond}, Track Temp: {track_temp}°C, "
                f"Incident: {safety_car}, Laps remaining: {laps_remaining}."
            )
            full_query = f"{user_query}. [Context: {context_str}]"

            result = run_graph(
                user_query=full_query,
                circuit=selected_circuit,
                year=selected_year,
                session_type="R",
            )

            status_ph.empty()

            rec      = result.get("strategy_recommendation") or {}
            rec_text = rec.get("recommendation_text", "") or result.get("error", "No recommendation generated.")
            eval_res = result.get("evaluation_result") or {}

            response_md = (
                f"### 📊 Race Strategy Analysis for {selected_circuit} ({selected_year})\n\n"
                f"{rec_text}\n\n"
            )

            if eval_res:
                score   = eval_res.get("score", 0)
                verdict = eval_res.get("verdict", "N/A")
                s_emoji = "✅" if score >= 75 else "⚠️" if score >= 45 else "❌"
                response_md += (
                    f"#### ⚖️ Coherence Evaluator Verdict\n"
                    f"- **Score:** {s_emoji} {score}/100\n"
                    f"- **Verdict:** {verdict}\n"
                )
                findings = eval_res.get("findings", [])
                if findings:
                    response_md += "- **Checks & Warnings:**\n"
                    for f in findings:
                        sev = f.get("severity", "")
                        sev_emoji = "🔴" if sev == "critical" else "🟡" if sev == "major" else "ℹ️"
                        response_md += (
                            f"  - {sev_emoji} *{f.get('rule', '')}*: {f.get('message', '')}\n"
                        )

            revisions = result.get("revision_count", 0)
            if revisions:
                response_md += (
                    f"\n🔄 *Strategy went through {revisions} feedback revision loop(s) to improve coherence.*"
                )

            st.markdown(response_md)

            compounds  = rec.get("compounds", [])
            pit_laps   = rec.get("pit_laps", [])
            if compounds:
                st.markdown("#### 🛠️ Visual Pit Strategy Timeline")
                render_strategy_timeline(compounds, pit_laps, laps_remaining)

            add_message("assistant", response_md)

        except Exception as exc:
            status_ph.empty()
            err_msg = f"❌ Error invoking LangGraph workflow: {exc}"
            st.error(err_msg)
            add_message("assistant", err_msg)
