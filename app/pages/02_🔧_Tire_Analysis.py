"""
02 — Tire Analysis Page.

The core analysis page of the F1 Race Strategist AI. Provides:
  - Circuit / year / driver selectors
  - Track wear classification card
  - Pit window recommendation card with risk indicator
  - Degradation chart (Plotly line chart per compound)
  - Stint summary table
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path for imports
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st
import pandas as pd

from app.styles.theme import inject_css, COMPOUND_COLORS
from app.components.strategy_card import render_strategy_card, render_metric_row
from app.components.tire_chart import (
    render_degradation_chart,
    render_stint_table,
    render_strategy_timeline,
)

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Tire Analysis — F1 Strategist",
    page_icon="🔧",
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

# ── Header ────────────────────────────────────────────────────────
st.markdown("# 🔧 Tire Analysis")
st.markdown(
    '<p style="color: #A0A0A0; margin-top: -0.5rem;">'
    "Degradation, pit windows, and compound strategy powered by FastF1 data"
    "</p>",
    unsafe_allow_html=True,
)

# ── Selectors ─────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns([2, 1, 1.5, 1])

with col1:
    circuit = st.selectbox("Circuit", CIRCUITS, index=CIRCUITS.index("Britain"))
with col2:
    year = st.selectbox("Year", YEARS, index=0)
with col3:
    driver_input = st.text_input(
        "Driver (3-letter code)",
        value="",
        placeholder="e.g. VER, HAM, LEC (blank = winner)",
    )
with col4:
    st.markdown("<div style='height: 1.7rem'></div>", unsafe_allow_html=True)
    analyze = st.button("🔍 Analyze", use_container_width=True)

driver = driver_input.strip().upper() if driver_input.strip() else None

# ── Run analysis ──────────────────────────────────────────────────
if analyze or st.session_state.get("tire_last_analysis"):
    # Cache the analysis result in session state
    if analyze:
        with st.spinner("⏳ Loading FastF1 data and running tire analysis..."):
            try:
                from agents.tire_agent import analyze_tire_strategy
                from tools.fastf1_tool import get_session, get_tire_data, get_stints

                analysis = analyze_tire_strategy(circuit, year, "R", driver)
                st.session_state["tire_last_analysis"] = analysis
                st.session_state["tire_circuit"] = circuit
                st.session_state["tire_year"] = year
                st.session_state["tire_driver"] = driver

                # Also load raw tire data for the chart
                session = get_session(year, circuit, "R")
                resolved_driver = driver
                if not resolved_driver:
                    from tools.fastf1_tool import get_race_results
                    results = get_race_results(session)
                    resolved_driver = results.iloc[0]["Abbreviation"]

                tire_df = get_tire_data(session, driver=resolved_driver)
                stints_df = get_stints(session, driver=resolved_driver)
                st.session_state["tire_df"] = tire_df
                st.session_state["tire_stints_df"] = stints_df
                st.session_state["tire_resolved_driver"] = resolved_driver

            except Exception as exc:
                st.error(f"❌ Analysis failed: {exc}")
                st.stop()

    analysis = st.session_state.get("tire_last_analysis")
    if not analysis:
        st.stop()

    tire_df = st.session_state.get("tire_df", pd.DataFrame())
    stints_df = st.session_state.get("tire_stints_df", pd.DataFrame())
    resolved_driver = st.session_state.get("tire_resolved_driver", "N/A")

    # Show errors if any
    if analysis.get("error"):
        st.warning(f"⚠️ {analysis['error']}")

    st.divider()

    # ── Track classification + Pit window ─────────────────────────
    row1_left, row1_right = st.columns(2)

    with row1_left:
        tw = analysis.get("track_wear") or {}
        classification = tw.get("classification", "Unknown")
        score = tw.get("score", "?")
        track = tw.get("track", circuit)
        factors = tw.get("factors") or {}

        # Determine risk from classification
        risk = "low"
        if "High" in classification:
            risk = "high"
        elif "Medium" in classification:
            risk = "medium"

        factors_html = ""
        if factors:
            factor_items = [
                f"Tire Stress: <b>{factors.get('TIRE_STRESS', '?')}</b>/5",
                f"Lateral Load: <b>{factors.get('LATERAL', '?')}</b>/5",
                f"Asphalt Abrasion: <b>{factors.get('ASPHALT_ABR', '?')}</b>/5",
                f"Asphalt Grip: <b>{factors.get('ASPHALT_GRP', '?')}</b>/5",
            ]
            factors_html = " · ".join(factor_items)
            if factors.get("LAPS"):
                factors_html += f"<br>Track: {factors.get('LENGTH_km', '?')} km · {int(factors.get('LAPS', 0))} laps"

        render_strategy_card(
            title=f"🏁 {track} — {classification}",
            subtitle="TRACK WEAR CLASSIFICATION",
            body=f"<b>Score: {score}/5.00</b><br>{factors_html}",
            risk_level=risk,
        )

    with row1_right:
        pw = analysis.get("pit_window") or {}
        strategy_type = pw.get("strategy_type", "Unknown")
        pit_windows = pw.get("pit_windows", [])
        rec_laps = pw.get("recommended_pit_laps", [])
        total_laps = pw.get("total_laps", 0)

        pit_body = f"<b>{strategy_type}</b> over {total_laps} laps<br>"
        if pit_windows:
            for i, w in enumerate(pit_windows, 1):
                compound_color = COMPOUND_COLORS.get(w.get("compound", ""), "#888")
                pit_body += (
                    f'Pit {i}: Lap {w["earliest"]}–{w["latest"]} '
                    f'(optimal <b>~{w["optimal"]}</b>) on '
                    f'<span style="color:{compound_color};font-weight:700;">'
                    f'{w.get("compound", "?")}</span> '
                    f'({w.get("deg_rate", 0):.3f} s/lap deg)<br>'
                )
        else:
            pit_body += "No pit window data available."

        # Risk based on degradation rate
        deg = analysis.get("degradation") or []
        if deg:
            avg_deg = sum(d["deg_rate_sec_per_lap"] for d in deg) / len(deg)
            pit_risk = "high" if avg_deg > 0.08 else "medium" if avg_deg > 0.04 else "low"
        else:
            pit_risk = None

        render_strategy_card(
            title="🔧 Pit Stop Window",
            subtitle="OPTIMAL PIT STRATEGY",
            body=pit_body,
            risk_level=pit_risk,
        )

    # ── Compound recommendation + metrics row ─────────────────────
    rec = analysis.get("compound_rec") or {}
    recommended_order = rec.get("recommended_order", [])
    if recommended_order:
        st.markdown("### Recommended Compound Order")
        render_strategy_timeline(recommended_order, rec_laps, total_laps)

        if rec.get("all_top3_strategies"):
            with st.expander("📊 Top-3 finishers' strategies"):
                for s in rec["all_top3_strategies"]:
                    compounds_str = " → ".join(s.get("compounds", []))
                    st.markdown(f"**{s['driver']}**: {compounds_str}")

    st.divider()

    # ── Degradation chart ─────────────────────────────────────────
    st.markdown(f"### Degradation — {resolved_driver}")
    render_degradation_chart(
        tire_df,
        title=f"Lap Time Degradation — {resolved_driver} · {circuit} {year}",
    )

    # ── Degradation metrics ───────────────────────────────────────
    if deg:
        metrics = []
        for s in deg:
            compound = s.get("compound", "?")
            rate = s.get("deg_rate_sec_per_lap", 0)
            metrics.append({
                "label": f"Stint {s['stint']} ({compound})",
                "value": f"{rate:+.4f} s/lap",
                "delta": f"{s['lap_count']} laps",
            })
        render_metric_row(metrics[:4])  # Max 4 metrics in a row

    st.divider()

    # ── Stint table ───────────────────────────────────────────────
    st.markdown(f"### Stint Summary — {resolved_driver}")
    render_stint_table(stints_df)

    # ── Weather impact note ───────────────────────────────────────
    wi = analysis.get("weather_impact")
    if wi:
        st.divider()
        st.markdown("### Weather Impact")
        w_cols = st.columns(3)
        with w_cols[0]:
            st.metric("Air Temp", f"{wi.get('air_temp_avg', '?')} °C")
        with w_cols[1]:
            st.metric("Track Temp", f"{wi.get('track_temp_avg', '?')} °C")
        with w_cols[2]:
            st.metric("Rainfall", "Yes ⛈️" if wi.get("rainfall") else "No ☀️")
        if wi.get("note"):
            st.info(wi["note"])

else:
    # Default state: show instructions
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; padding: 3rem 0; color: #A0A0A0;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🔧</div>
            <div style="font-size: 1.1rem;">
                Select a <b style="color: #E8002D;">circuit</b>,
                <b style="color: #E8002D;">year</b>, and optionally a
                <b style="color: #E8002D;">driver</b>, then click
                <b>Analyze</b> to view tire degradation data.
            </div>
            <div style="font-size: 0.85rem; margin-top: 0.5rem;">
                Data powered by FastF1 · Leave driver blank to analyze the race winner
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
