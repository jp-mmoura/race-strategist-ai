"""
01 — General Strategy Page.

Allows users to chat with the Strategy Agent. Renders:
  - Chat history and interactive user input
  - Explanable recommendations and feedback loop results
  - Visual strategy timelines with Pirelli stint segments
  - Context parameters in the sidebar (Weather, Safety Car, Laps Remaining)
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path for imports
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st
from app.styles.theme import inject_css
from app.components.chat_interface import init_chat, render_chat_history, render_chat_input, add_message
from app.components.strategy_card import render_strategy_card
from app.components.tire_chart import render_strategy_timeline

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Race Strategy AI — F1 Strategist",
    page_icon="🏁",
    layout="wide",
)
inject_css()

# ── Available circuits & years ────────────────────────────────────
CIRCUITS = [
    "Australia", "Bahrain", "China", "Spain", "Monaco", "Canada",
    "Austria", "Britain", "Hungary", "Belgium", "Italy", "Singapore",
    "Japan", "USA", "Mexico", "Brazil", "AbuDhabi", "Netherlands",
    "Azerbaijan", "Miami", "Qatar", "Las Vegas", "Saudi Arabia",
]
YEARS = list(range(2025, 2017, -1))

# ── Sidebar Configuration ─────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Race Control Context")
    selected_circuit = st.selectbox("Current Circuit", CIRCUITS, index=CIRCUITS.index("Britain"), key="strat_circuit")
    selected_year = st.selectbox("Reference Season", YEARS, index=0, key="strat_year")
    
    st.markdown("---")
    st.markdown("### 🌦️ Environment")
    weather_cond = st.selectbox("Weather Condition", ["Dry", "Light Rain", "Heavy Rain", "Damp Track"], index=0)
    track_temp = st.slider("Track Temp (°C)", 10, 60, 35)
    
    st.markdown("---")
    st.markdown("### 🚨 Live Incidents")
    safety_car = st.selectbox("Safety Car / VSC Status", ["Normal Racing", "Virtual Safety Car", "Safety Car", "Red Flag"], index=0)
    laps_remaining = st.slider("Laps Remaining", 1, 80, 52)

# ── Header ────────────────────────────────────────────────────────
st.markdown("# 🏁 Race Strategy Control Hub")
st.markdown(
    '<p style="color: #A0A0A0; margin-top: -0.5rem;">'
    "Interact with the LangGraph Strategy orchestrator to plan, evaluate and refine your race scenarios."
    "</p>",
    unsafe_allow_html=True,
)

# Initialize Chat
init_chat()

# Render Chat History
render_chat_history()

# Render Chat Input
user_message = render_chat_input()

if user_message:
    # Append user message
    add_message("user", user_message)
    
    # Rerender chat history with the new user message immediately
    st.rerun()

# If last message was from user, run the workflow
messages = st.session_state.get("chat_messages", [])
if messages and messages[-1]["role"] == "user":
    user_query = messages[-1]["content"]
    
    with st.chat_message("assistant", avatar="🏁"):
        status_placeholder = st.empty()
        status_placeholder.markdown("⏳ *Analisando estratégia... Montando contexto dos pneus, clima e histórico de corrida...*")
        
        try:
            # We import and call run_graph from the supervisor agent
            from agents.supervisor import run_graph
            
            # Map Live incident/sidebar values to overrides or system prompts
            # Construct a comprehensive prompt or context overrides
            incidents_context = f"Weather: {weather_cond}, Track Temp: {track_temp}C, Incidents: {safety_car}, Laps remaining: {laps_remaining}."
            full_query = f"{user_query}. [Current Incident/Weather Context: {incidents_context}]"
            
            # Call the LangGraph workflow
            result = run_graph(
                user_query=full_query,
                circuit=selected_circuit,
                year=selected_year,
                session_type="R"
            )
            
            # Clear status spinner placeholder
            status_placeholder.empty()
            
            # Parse recommendation
            rec = result.get("strategy_recommendation") or {}
            rec_text = rec.get("recommendation_text", "") or result.get("error", "No recommendation text generated.")
            eval_res = result.get("evaluation_result") or {}
            
            # Build assistant response markdown
            response_md = f"### 📊 Race Strategy Analysis for {selected_circuit} ({selected_year})\n\n"
            
            # Add timeline if we have strategy parameters
            compounds = rec.get("compounds", [])
            pit_laps = rec.get("pit_laps", [])
            
            # Display text recommendation
            response_md += f"{rec_text}\n\n"
            
            if eval_res:
                score = eval_res.get("score", 0)
                verdict = eval_res.get("verdict", "N/A")
                response_md += f"#### ⚖️ Coherence Evaluator Verdict\n"
                response_md += f"- **Score:** {score}/100\n"
                response_md += f"- **Verdict:** {verdict}\n"
                
                findings = eval_res.get("findings", [])
                if findings:
                    response_md += "- **Checks & Warnings:**\n"
                    for f in findings:
                        sev_emoji = "🔴" if f.get("severity") == "critical" else "🟡" if f.get("severity") == "major" else "ℹ️"
                        response_md += f"  - {sev_emoji} *{f.get('rule', 'Rule')}*: {f.get('message', '')}\n"
            
            if result.get("revision_count", 0) > 0:
                response_md += f"\n🔄 *Strategy went through {result['revision_count']} feedback revision loops to improve coherence.*"
                
            st.markdown(response_md)
            
            # Draw visual timeline if stint compounds are available
            if compounds:
                st.markdown("#### 🛠️ Visual Pit Strategy Timeline")
                render_strategy_timeline(compounds, pit_laps, laps_remaining)
            
            # Save assistant message to history
            add_message("assistant", response_md)
            
        except Exception as exc:
            status_placeholder.empty()
            st.error(f"❌ Error invoking LangGraph workflow: {exc}")
            add_message("assistant", f"❌ Error invoking LangGraph workflow: {exc}")
