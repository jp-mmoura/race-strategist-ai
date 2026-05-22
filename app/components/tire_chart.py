"""
Tire Chart — Plotly degradation chart and stint table components.

Provides:
  - render_degradation_chart(tire_df) — line chart of lap times per compound
  - render_stint_table(stints_df) — styled dataframe of stint data
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.styles.theme import (
    BG_CARD,
    BG_PRIMARY,
    COMPOUND_COLORS,
    GREY_BORDER,
    GREY_TEXT,
    RED_ACCENT,
    WHITE,
)


def render_degradation_chart(
    tire_df: pd.DataFrame,
    title: str = "Tire Degradation — Lap Time by Compound",
    height: int = 450,
) -> None:
    """Render a Plotly line chart of lap times per compound/stint.

    Parameters
    ----------
    tire_df : pd.DataFrame
        Output of ``get_tire_data(session, driver)`` — must have
        columns: LapNumber, LapTimeSec, Compound, Stint.
    title : str
        Chart title.
    height : int
        Chart height in pixels.
    """
    if tire_df.empty or "LapTimeSec" not in tire_df.columns:
        st.info("No tire data available for this selection.")
        return

    df = tire_df.dropna(subset=["LapTimeSec"]).copy()

    # Filter outliers (pit laps / safety cars)
    median_t = df["LapTimeSec"].median()
    df = df[df["LapTimeSec"] <= median_t * 1.12]

    fig = go.Figure()

    # Group by stint to draw separate traces with compound colour
    for stint_num in sorted(df["Stint"].dropna().unique()):
        stint_df = df[df["Stint"] == stint_num].sort_values("LapNumber")
        compound = stint_df["Compound"].iloc[0] if "Compound" in stint_df.columns else "UNKNOWN"
        color = COMPOUND_COLORS.get(compound, COMPOUND_COLORS["UNKNOWN"])

        fig.add_trace(
            go.Scatter(
                x=stint_df["LapNumber"],
                y=stint_df["LapTimeSec"],
                mode="lines+markers",
                name=f"S{int(stint_num)} — {compound}",
                line=dict(color=color, width=2.5),
                marker=dict(color=color, size=4),
                hovertemplate=(
                    f"<b>{compound}</b> (Stint {int(stint_num)})<br>"
                    "Lap %{x}<br>"
                    "Time: %{y:.3f}s<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(color=WHITE, size=16, family="Inter"),
            x=0,
        ),
        xaxis=dict(
            title="Lap Number",
            color=GREY_TEXT,
            gridcolor=GREY_BORDER,
            linecolor=GREY_BORDER,
            zeroline=False,
        ),
        yaxis=dict(
            title="Lap Time (seconds)",
            color=GREY_TEXT,
            gridcolor=GREY_BORDER,
            linecolor=GREY_BORDER,
            zeroline=False,
        ),
        plot_bgcolor=BG_PRIMARY,
        paper_bgcolor=BG_CARD,
        font=dict(family="Inter", color=WHITE),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=WHITE, size=11),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        margin=dict(l=60, r=20, t=60, b=50),
        height=height,
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)


def render_stint_table(stints_df: pd.DataFrame) -> None:
    """Render a styled stint summary table.

    Parameters
    ----------
    stints_df : pd.DataFrame
        Output of ``get_stints(session, driver)`` — columns:
        Driver, Stint, Compound, StartLap, EndLap, MaxTyreLife, Laps.
    """
    if stints_df.empty:
        st.info("No stint data available.")
        return

    display_cols = ["Stint", "Compound", "StartLap", "EndLap", "Laps", "MaxTyreLife"]
    available = [c for c in display_cols if c in stints_df.columns]
    df = stints_df[available].copy()

    # Rename for display
    rename = {
        "StartLap": "Start Lap",
        "EndLap": "End Lap",
        "MaxTyreLife": "Max Tyre Life",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Compound": st.column_config.TextColumn("Compound", width="medium"),
            "Stint": st.column_config.NumberColumn("Stint", format="%d"),
        },
    )


def render_strategy_timeline(
    compounds: list[str],
    pit_laps: list[int],
    total_laps: int,
) -> None:
    """Render a horizontal strategy timeline with compound colours.

    Parameters
    ----------
    compounds : list[str]
        Ordered list of compounds per stint.
    pit_laps : list[int]
        Lap numbers for each pit stop.
    total_laps : int
        Total race laps.
    """
    if not compounds or total_laps == 0:
        return

    # Build stint boundaries
    boundaries = [1] + [p + 1 for p in pit_laps] + [total_laps]
    stints = []
    for i, compound in enumerate(compounds):
        start = boundaries[i] if i < len(boundaries) else 1
        end = boundaries[i + 1] if i + 1 < len(boundaries) else total_laps
        width_pct = max(5, int((end - start) / total_laps * 100))
        stints.append((compound, start, end, width_pct))

    timeline_html = '<div class="timeline-container">'
    for i, (compound, start, end, width_pct) in enumerate(stints):
        color = COMPOUND_COLORS.get(compound, "#888")
        timeline_html += f"""
            <div class="timeline-stint" style="
                background-color: {color};
                flex: {width_pct};
                min-width: 60px;
            ">
                {compound}<br>
                <span style="font-size:0.65rem; opacity:0.8;">L{start}–L{end}</span>
            </div>
        """
        # Add pit marker between stints
        if i < len(pit_laps):
            pit = pit_laps[i]
            timeline_html += f"""
                <div class="timeline-pit">
                    <div class="timeline-pit-label">PIT L{pit}</div>
                </div>
            """

    timeline_html += "</div>"
    st.markdown(timeline_html, unsafe_allow_html=True)
