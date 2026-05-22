"""
Strategy Card — reusable F1-styled card component.

Renders a card with a red left border, title, subtitle,
body text, and optional risk badge.
"""

from __future__ import annotations

import streamlit as st

from app.styles.theme import RED_ACCENT, BG_CARD, GREY_BORDER, WHITE, GREY_TEXT


def render_strategy_card(
    title: str,
    subtitle: str = "",
    body: str = "",
    risk_level: str | None = None,
    accent_color: str = RED_ACCENT,
) -> None:
    """Render a styled strategy card.

    Parameters
    ----------
    title : str
        Card title (bold, white).
    subtitle : str
        Small uppercase subtitle (grey).
    body : str
        Card body content (supports HTML).
    risk_level : str | None
        If provided, shows a coloured risk badge.
        Values: "low", "medium", "high".
    accent_color : str
        Left border colour (default: F1 red).
    """
    risk_html = ""
    if risk_level:
        risk_map = {
            "low": ("🟢 LOW RISK", "risk-low"),
            "medium": ("🟡 MEDIUM RISK", "risk-medium"),
            "high": ("🔴 HIGH RISK", "risk-high"),
        }
        label, cls = risk_map.get(risk_level.lower(), ("⚪ UNKNOWN", "risk-low"))
        risk_html = f'<span class="risk-badge {cls}">{label}</span>'

    subtitle_html = (
        f'<div class="f1-card-subtitle">{subtitle}</div>' if subtitle else ""
    )

    st.markdown(
        f"""
        <div style="
            background-color: {BG_CARD};
            border: 1px solid {GREY_BORDER};
            border-left: 3px solid {accent_color};
            border-radius: 8px;
            padding: 1.2rem 1.5rem;
            margin-bottom: 1rem;
        ">
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div>
                    <div class="f1-card-title">{title}</div>
                    {subtitle_html}
                </div>
                {risk_html}
            </div>
            <div class="f1-card-body">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_row(metrics: list[dict]) -> None:
    """Render a row of metric cards.

    Parameters
    ----------
    metrics : list[dict]
        Each dict has keys: label, value, delta (optional).
    """
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        with col:
            st.metric(
                label=m.get("label", ""),
                value=m.get("value", ""),
                delta=m.get("delta"),
            )
