"""
03 — Historical Race Data Page.

Displays past race strategies and allows comparison. Renders:
  - Selectors for circuit, season, and driver
  - Highlight card showing the winning strategy of the historic race
  - Table of official race results and tire stint logs
  - Bar chart comparing total race times (or median lap times) across different strategy types
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path for imports
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st
import pandas as pd
import plotly.express as px
from app.styles.theme import inject_css, BG_CARD, BG_PRIMARY, GREY_BORDER, GREY_TEXT, WHITE, RED_ACCENT
from app.components.strategy_card import render_strategy_card

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Race History — F1 Strategist",
    page_icon="📊",
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
YEARS = list(range(2024, 2017, -1)) # History is from past seasons

# ── Header ────────────────────────────────────────────────────────
st.markdown("# 📊 Race History Database")
st.markdown(
    '<p style="color: #A0A0A0; margin-top: -0.5rem;">'
    "Analyze historical race strategies, stint selections, and race results."
    "</p>",
    unsafe_allow_html=True,
)

# ── Selectors ─────────────────────────────────────────────────────
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    circuit = st.selectbox("Circuit", CIRCUITS, index=CIRCUITS.index("Britain"))
with col2:
    year = st.selectbox("Season", YEARS, index=0)
with col3:
    st.markdown("<div style='height: 1.7rem'></div>", unsafe_allow_html=True)
    fetch = st.button("📊 Fetch History", use_container_width=True)

# ── Fetch Data ────────────────────────────────────────────────────
if fetch or st.session_state.get("history_last_fetch"):
    if fetch:
        with st.spinner("⏳ Retrieving session data from FastF1..."):
            try:
                from tools.fastf1_tool import get_session, get_race_results, get_stints
                
                # Fetch session
                session = get_session(year, circuit, "R")
                results_df = get_race_results(session)
                
                # Fetch winner strategy
                winner_driver = results_df.iloc[0]["Abbreviation"] if not results_df.empty else "VER"
                winner_stints = get_stints(session, driver=winner_driver)
                
                # Sample comparison data: build stats for top drivers
                comparison_list = []
                drivers_to_compare = results_df.head(8)["Abbreviation"].tolist()
                for d in drivers_to_compare:
                    st_df = get_stints(session, driver=d)
                    if not st_df.empty:
                        # Estimate total laps, average tyres, number of stops
                        stops = len(st_df) - 1
                        compounds_str = " → ".join(st_df["Compound"].dropna().tolist())
                        
                        # Fetch total race time if available
                        driver_res = results_df[results_df["Abbreviation"] == d]
                        status = driver_res.iloc[0]["Status"] if not driver_res.empty else "Finished"
                        
                        # Get fastest/median lap time to plot instead of total time (since total time might be delta-based)
                        from tools.fastf1_tool import get_tire_data
                        td = get_tire_data(session, driver=d)
                        median_lap = td["LapTimeSec"].median() if not td.empty else 90.0
                        
                        comparison_list.append({
                            "Driver": d,
                            "Stops": stops,
                            "Strategy": compounds_str,
                            "Median Lap Time (s)": median_lap,
                            "Status": status
                        })
                
                st.session_state["history_last_fetch"] = True
                st.session_state["history_results_df"] = results_df
                st.session_state["history_winner_driver"] = winner_driver
                st.session_state["history_winner_stints"] = winner_stints
                st.session_state["history_comparison_df"] = pd.DataFrame(comparison_list)
                st.session_state["history_circuit"] = circuit
                st.session_state["history_year"] = year
                
            except Exception as exc:
                st.error(f"❌ Failed to fetch historical data: {exc}")
                st.stop()
                
    results_df = st.session_state.get("history_results_df", pd.DataFrame())
    winner_driver = st.session_state.get("history_winner_driver", "Winner")
    winner_stints = st.session_state.get("history_winner_stints", pd.DataFrame())
    comp_df = st.session_state.get("history_comparison_df", pd.DataFrame())
    h_circuit = st.session_state.get("history_circuit", circuit)
    h_year = st.session_state.get("history_year", year)
    
    st.divider()
    
    # ── Winning Strategy Card & Stint Graph ───────────────────────
    left_col, right_col = st.columns([1.5, 2.5])
    
    with left_col:
        st.markdown("### 🏆 Reference Strategy")
        if not winner_stints.empty:
            comp_list = winner_stints["Compound"].dropna().tolist()
            laps_list = winner_stints["Laps"].dropna().tolist()
            strategy_desc = " → ".join(comp_list)
            stints_desc = "<br>".join([f"Stint {int(row['Stint'])}: {row['Compound']} ({int(row['Laps'])} laps)" for _, row in winner_stints.iterrows()])
            
            render_strategy_card(
                title=f"{winner_driver} — Winning Strategy",
                subtitle=f"{strategy_desc} ({len(comp_list) - 1} stops)",
                body=f"<b>Stint Details:</b><br>{stints_desc}",
                risk_level="low"
            )
        else:
            st.info("Winner stint info not available.")
            
    with right_col:
        st.markdown("### ⏱️ Strategy Performance Comparison")
        if not comp_df.empty:
            # Render horizontal bar chart of median lap times
            fig = px.bar(
                comp_df,
                x="Median Lap Time (s)",
                y="Driver",
                color="Strategy",
                text="Strategy",
                orientation="h",
                title=f"Median Lap Time by Driver Strategy · {h_circuit} {h_year}",
                template="plotly_dark",
            )
            fig.update_layout(
                plot_bgcolor=BG_PRIMARY,
                paper_bgcolor=BG_CARD,
                xaxis=dict(gridcolor=GREY_BORDER, zeroline=False),
                yaxis=dict(categoryorder="total descending"),
                font=dict(family="Inter", color=WHITE),
                margin=dict(l=50, r=20, t=50, b=40),
                height=320,
            )
            st.plotly_chart(fig, use_container_width=True)
            
    st.divider()
    
    # ── Race Results Table ────────────────────────────────────────
    st.markdown("### 📋 Official Race Classification")
    if not results_df.empty:
        display_results = results_df[["ClassifiedPosition", "Abbreviation", "TeamName", "GridPosition", "Points", "Status"]].copy()
        display_results = display_results.rename(columns={
            "ClassifiedPosition": "Pos",
            "Abbreviation": "Driver",
            "TeamName": "Constructor",
            "GridPosition": "Grid",
            "Points": "Points",
            "Status": "Status"
        })
        st.dataframe(display_results, use_container_width=True, hide_index=True)
    else:
        st.info("No race results found.")

else:
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; padding: 3rem 0; color: #A0A0A0;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">📊</div>
            <div style="font-size: 1.1rem;">
                Select a <b style="color: #E8002D;">circuit</b> and <b style="color: #E8002D;">season</b>, then click
                <b>Fetch History</b> to view historical results.
            </div>
            <div style="font-size: 0.85rem; margin-top: 0.5rem;">
                Data is loaded live from the FastF1 database
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
