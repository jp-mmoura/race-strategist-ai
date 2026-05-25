"""
03 — Historical Race Data Page.

Displays past race strategies and allows comparison. Renders:
  - Circuit and season selectors (sidebar)
  - Winning strategy card
  - Strategy performance comparison chart (median lap times)
  - Compound distribution stacked bar chart per driver
  - Strategy breakdown pie chart by stop count
  - Official race results table
"""

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st
import pandas as pd
import plotly.express as px

from app.styles.theme import (
    inject_css,
    BG_CARD, BG_PRIMARY, GREY_BORDER, WHITE, RED_ACCENT,
    COMPOUND_COLORS,
)
from app.components.strategy_card import render_strategy_card

st.set_page_config(
    page_title="Race History — F1 Strategist",
    page_icon="📊",
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
YEARS = list(range(2024, 2017, -1))

# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Race History")
    circuit = st.selectbox("Circuit", CIRCUITS, index=CIRCUITS.index("Britain"), key="hist_circuit_sel")
    year    = st.selectbox("Season", YEARS, index=0, key="hist_year_sel")

    st.markdown("---")
    fetch = st.button("📊 Fetch History", use_container_width=True)

    if st.session_state.get("history_last_fetch"):
        st.markdown("---")
        if st.button("🗑️ Clear", use_container_width=True):
            for k in [
                "history_last_fetch", "history_results_df",
                "history_winner_driver", "history_winner_stints",
                "history_comparison_df", "history_circuit", "history_year",
            ]:
                st.session_state.pop(k, None)
            st.rerun()

# ── Header ────────────────────────────────────────────────────────────
st.markdown("# 📊 Race History Database")
st.markdown(
    '<p style="color:#A0A0A0; margin-top:-0.5rem;">'
    "Analyze historical race strategies, stint selections, and race results."
    "</p>",
    unsafe_allow_html=True,
)

# ── Fetch data ────────────────────────────────────────────────────────
if fetch or st.session_state.get("history_last_fetch"):

    if fetch:
        with st.spinner("⏳ Retrieving session data from FastF1…"):
            try:
                from tools.fastf1_tool import (
                    get_session, get_race_results, get_stints, get_tire_data,
                )

                session    = get_session(year, circuit, "R")
                results_df = get_race_results(session)

                winner_driver = results_df.iloc[0]["Abbreviation"] if not results_df.empty else "VER"
                winner_stints = get_stints(session, driver=winner_driver)

                comparison_list = []
                for d in results_df.head(8)["Abbreviation"].tolist():
                    st_df = get_stints(session, driver=d)
                    if not st_df.empty:
                        stops         = len(st_df) - 1
                        compounds_str = " → ".join(st_df["Compound"].dropna().tolist())
                        driver_res    = results_df[results_df["Abbreviation"] == d]
                        status        = driver_res.iloc[0]["Status"] if not driver_res.empty else "Finished"
                        td            = get_tire_data(session, driver=d)
                        median_lap    = round(td["LapTimeSec"].median(), 3) if not td.empty else 90.0
                        comparison_list.append({
                            "Driver":             d,
                            "Stops":              stops,
                            "Strategy":           compounds_str,
                            "Median Lap Time (s)": median_lap,
                            "Status":             status,
                        })

                st.session_state["history_last_fetch"]    = True
                st.session_state["history_results_df"]    = results_df
                st.session_state["history_winner_driver"] = winner_driver
                st.session_state["history_winner_stints"] = winner_stints
                st.session_state["history_comparison_df"] = pd.DataFrame(comparison_list)
                st.session_state["history_circuit"]       = circuit
                st.session_state["history_year"]          = year

            except Exception as exc:
                st.error(f"❌ Failed to fetch historical data: {exc}")
                st.stop()

    results_df    = st.session_state.get("history_results_df",    pd.DataFrame())
    winner_driver = st.session_state.get("history_winner_driver", "Winner")
    winner_stints = st.session_state.get("history_winner_stints", pd.DataFrame())
    comp_df       = st.session_state.get("history_comparison_df", pd.DataFrame())
    h_circuit     = st.session_state.get("history_circuit", circuit)
    h_year        = st.session_state.get("history_year",    year)

    st.markdown(f"### {h_circuit} Grand Prix {h_year}")
    st.divider()

    # ── Winning strategy card + Performance chart ─────────────────
    left_col, right_col = st.columns([1.5, 2.5])

    with left_col:
        st.markdown("### 🏆 Winning Strategy")
        if not winner_stints.empty:
            comp_list     = winner_stints["Compound"].dropna().tolist()
            strategy_desc = " → ".join(comp_list)
            stints_desc   = "<br>".join([
                f"Stint {int(r['Stint'])}: {r['Compound']} ({int(r['Laps'])} laps)"
                for _, r in winner_stints.iterrows()
            ])
            render_strategy_card(
                title=f"{winner_driver} — Winning Strategy",
                subtitle=f"{strategy_desc} · {len(comp_list) - 1} stop(s)",
                body=f"<b>Stint Details:</b><br>{stints_desc}",
                risk_level="low",
            )
        else:
            st.info("Winner stint info not available.")

    with right_col:
        st.markdown("### ⏱️ Strategy Performance Comparison")
        if not comp_df.empty:
            fig = px.bar(
                comp_df,
                x="Median Lap Time (s)",
                y="Driver",
                color="Stops",
                text="Strategy",
                orientation="h",
                title=f"Median Lap Time by Driver Strategy · {h_circuit} {h_year}",
                template="plotly_dark",
                color_continuous_scale=[[0, "#4CAF50"], [0.5, "#FFC107"], [1, RED_ACCENT]],
            )
            fig.update_layout(
                plot_bgcolor=BG_PRIMARY,
                paper_bgcolor=BG_CARD,
                xaxis=dict(gridcolor=GREY_BORDER, zeroline=False),
                yaxis=dict(categoryorder="total ascending"),
                font=dict(family="Inter", color=WHITE),
                margin=dict(l=50, r=20, t=50, b=40),
                height=320,
            )
            fig.update_traces(textposition="outside", cliponaxis=False)
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Compound distribution per driver ──────────────────────────
    if not comp_df.empty:
        st.markdown("### 🔄 Compound Distribution by Driver")

        compound_rows = []
        for _, row in comp_df.iterrows():
            for compound in row["Strategy"].split(" → "):
                compound = compound.strip()
                if compound:
                    compound_rows.append({"Driver": row["Driver"], "Compound": compound})

        if compound_rows:
            cd_df = (
                pd.DataFrame(compound_rows)
                .groupby(["Driver", "Compound"])
                .size()
                .reset_index(name="Stints")
            )
            fig2 = px.bar(
                cd_df,
                x="Driver",
                y="Stints",
                color="Compound",
                title=f"Stints by Compound · {h_circuit} {h_year}",
                template="plotly_dark",
                color_discrete_map=COMPOUND_COLORS,
                barmode="stack",
            )
            fig2.update_layout(
                plot_bgcolor=BG_PRIMARY,
                paper_bgcolor=BG_CARD,
                xaxis=dict(gridcolor=GREY_BORDER),
                yaxis=dict(gridcolor=GREY_BORDER, title="Number of Stints"),
                font=dict(family="Inter", color=WHITE),
                margin=dict(l=50, r=20, t=50, b=40),
                height=300,
            )
            st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # ── Official race results ─────────────────────────────────────
    st.markdown("### 📋 Official Race Classification")
    if not results_df.empty:
        display_cols = [
            "ClassifiedPosition", "Abbreviation", "TeamName",
            "GridPosition", "Points", "Status",
        ]
        available  = [c for c in display_cols if c in results_df.columns]
        display_df = results_df[available].copy().rename(columns={
            "ClassifiedPosition": "Pos",
            "Abbreviation":       "Driver",
            "TeamName":           "Constructor",
            "GridPosition":       "Grid",
            "Points":             "Points",
            "Status":             "Status",
        })
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info("No race results found.")

    # ── Strategy stop count summary ───────────────────────────────
    if not comp_df.empty:
        st.divider()
        st.markdown("### 🔢 Strategy Summary")

        stops_df = (
            comp_df.groupby("Stops")
            .size()
            .reset_index(name="Drivers")
        )
        stops_df["Label"] = stops_df["Stops"].apply(
            lambda x: f"{x} stop" if x == 1 else f"{x} stops"
        )

        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.dataframe(
                stops_df[["Label", "Drivers"]].rename(columns={"Label": "Strategy"}),
                use_container_width=True,
                hide_index=True,
            )
        with col_b:
            fig3 = px.pie(
                stops_df,
                values="Drivers",
                names="Label",
                title=f"Strategy Distribution · {h_circuit} {h_year}",
                template="plotly_dark",
                color_discrete_sequence=[RED_ACCENT, "#FFC107", "#4CAF50", "#2196F3"],
            )
            fig3.update_layout(
                paper_bgcolor=BG_CARD,
                font=dict(family="Inter", color=WHITE),
                margin=dict(l=20, r=20, t=50, b=20),
                height=260,
            )
            st.plotly_chart(fig3, use_container_width=True)

else:
    # ── Initial state ─────────────────────────────────────────────
    st.markdown(
        f"""
        <div style="text-align:center; padding:3rem 0; color:#A0A0A0;">
            <div style="font-size:3rem; margin-bottom:1rem;">📊</div>
            <div style="font-size:1.1rem;">
                Select a <b style="color:{RED_ACCENT};">circuit</b> and a
                <b style="color:{RED_ACCENT};">season</b> in the sidebar,
                then click <b>Fetch History</b>.
            </div>
            <div style="font-size:0.85rem; margin-top:0.5rem;">
                Data is loaded live from the FastF1 database
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
