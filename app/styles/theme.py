"""
F1 Race Strategist — Visual Theme.

Injects custom CSS to achieve the telemetry-panel aesthetic:
  - Background #0D0D0D, cards #1A1A1A, accent #E8002D
  - Inter font via Google Fonts
  - Custom metric cards, tables, sidebar styling
"""

import streamlit as st

# ── Colour palette ────────────────────────────────────────────────────
BG_PRIMARY = "#0D0D0D"
BG_CARD = "#1A1A1A"
BG_CARD_HOVER = "#242424"
RED_ACCENT = "#E8002D"
RED_DARK = "#B80023"
WHITE = "#FFFFFF"
GREY_TEXT = "#A0A0A0"
GREY_BORDER = "#333333"

# Pirelli compound colours
COMPOUND_COLORS = {
    "SOFT": "#E8002D",
    "MEDIUM": "#FFC107",
    "HARD": "#CCCCCC",
    "INTERMEDIATE": "#4CAF50",
    "WET": "#2196F3",
    "UNKNOWN": "#888888",
}

COMPOUND_COLORS_RGBA = {
    "SOFT": "rgba(232, 0, 45, 0.15)",
    "MEDIUM": "rgba(255, 193, 7, 0.15)",
    "HARD": "rgba(204, 204, 204, 0.15)",
    "INTERMEDIATE": "rgba(76, 175, 80, 0.15)",
    "WET": "rgba(33, 150, 243, 0.15)",
}


def inject_css():
    """Inject the full F1 theme CSS into the Streamlit app."""
    st.markdown(
        f"""
        <style>
        /* ── Google Fonts ─────────────────────────────────── */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

        /* ── Global ───────────────────────────────────────── */
        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
        }}

        .stApp {{
            background-color: {BG_PRIMARY};
        }}

        /* ── Sidebar ──────────────────────────────────────── */
        section[data-testid="stSidebar"] {{
            background-color: {BG_CARD};
            border-right: 1px solid {GREY_BORDER};
        }}

        section[data-testid="stSidebar"] .stMarkdown h1,
        section[data-testid="stSidebar"] .stMarkdown h2,
        section[data-testid="stSidebar"] .stMarkdown h3 {{
            color: {RED_ACCENT};
        }}

        /* ── Headers ──────────────────────────────────────── */
        h1, h2, h3 {{
            font-family: 'Inter', sans-serif !important;
            font-weight: 700 !important;
        }}

        h1 {{
            color: {WHITE} !important;
            font-size: 2rem !important;
            letter-spacing: -0.02em;
        }}

        h2 {{
            color: {RED_ACCENT} !important;
            font-size: 1.4rem !important;
            letter-spacing: -0.01em;
        }}

        h3 {{
            color: {GREY_TEXT} !important;
            font-size: 1.1rem !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        /* ── Metrics ──────────────────────────────────────── */
        [data-testid="stMetric"] {{
            background-color: {BG_CARD};
            border: 1px solid {GREY_BORDER};
            border-radius: 8px;
            padding: 1rem 1.2rem;
            border-left: 3px solid {RED_ACCENT};
        }}

        [data-testid="stMetricLabel"] {{
            color: {GREY_TEXT} !important;
            font-size: 0.75rem !important;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }}

        [data-testid="stMetricValue"] {{
            color: {WHITE} !important;
            font-size: 1.8rem !important;
            font-weight: 700 !important;
        }}

        /* ── Buttons ──────────────────────────────────────── */
        .stButton > button {{
            background-color: {RED_ACCENT};
            color: {WHITE};
            border: none;
            border-radius: 6px;
            font-weight: 600;
            letter-spacing: 0.02em;
            padding: 0.5rem 1.5rem;
            transition: all 0.2s ease;
        }}

        .stButton > button:hover {{
            background-color: {RED_DARK};
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(232, 0, 45, 0.3);
        }}

        /* ── Select boxes ─────────────────────────────────── */
        .stSelectbox > div > div {{
            background-color: {BG_CARD};
            border-color: {GREY_BORDER};
            color: {WHITE};
        }}

        /* ── Dataframes / Tables ──────────────────────────── */
        .stDataFrame {{
            border-radius: 8px;
            overflow: hidden;
        }}

        /* ── Dividers ─────────────────────────────────────── */
        hr {{
            border-color: {GREY_BORDER};
        }}

        /* ── Chat messages ────────────────────────────────── */
        [data-testid="stChatMessage"] {{
            background-color: {BG_CARD};
            border: 1px solid {GREY_BORDER};
            border-radius: 8px;
        }}

        /* ── Spinner ──────────────────────────────────────── */
        .stSpinner > div {{
            border-top-color: {RED_ACCENT} !important;
        }}

        /* ── Expander ─────────────────────────────────────── */
        .streamlit-expanderHeader {{
            background-color: {BG_CARD};
            border-radius: 6px;
            color: {WHITE};
        }}

        /* ── Tabs ─────────────────────────────────────────── */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 0;
            border-bottom: 1px solid {GREY_BORDER};
        }}

        .stTabs [data-baseweb="tab"] {{
            color: {GREY_TEXT};
            font-weight: 500;
            padding: 0.75rem 1.5rem;
        }}

        .stTabs [aria-selected="true"] {{
            color: {WHITE} !important;
            border-bottom: 2px solid {RED_ACCENT};
        }}

        /* ── Custom card class ────────────────────────────── */
        .f1-card {{
            background-color: {BG_CARD};
            border: 1px solid {GREY_BORDER};
            border-left: 3px solid {RED_ACCENT};
            border-radius: 8px;
            padding: 1.2rem 1.5rem;
            margin-bottom: 1rem;
        }}

        .f1-card-title {{
            color: {WHITE};
            font-size: 1rem;
            font-weight: 700;
            margin-bottom: 0.3rem;
        }}

        .f1-card-subtitle {{
            color: {GREY_TEXT};
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.8rem;
        }}

        .f1-card-body {{
            color: {WHITE};
            font-size: 0.9rem;
            line-height: 1.6;
        }}

        /* ── Risk badges ──────────────────────────────────── */
        .risk-badge {{
            display: inline-block;
            padding: 0.2rem 0.7rem;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.03em;
        }}
        .risk-low {{ background-color: rgba(76, 175, 80, 0.2); color: #4CAF50; }}
        .risk-medium {{ background-color: rgba(255, 193, 7, 0.2); color: #FFC107; }}
        .risk-high {{ background-color: rgba(232, 0, 45, 0.2); color: #E8002D; }}

        /* ── Compound badges ──────────────────────────────── */
        .compound-soft {{ color: #E8002D; font-weight: 700; }}
        .compound-medium {{ color: #FFC107; font-weight: 700; }}
        .compound-hard {{ color: #CCCCCC; font-weight: 700; }}
        .compound-inter {{ color: #4CAF50; font-weight: 700; }}
        .compound-wet {{ color: #2196F3; font-weight: 700; }}

        /* ── Landing hero ─────────────────────────────────── */
        .hero-title {{
            font-size: 3rem;
            font-weight: 900;
            color: {WHITE};
            letter-spacing: -0.03em;
            line-height: 1.1;
        }}
        .hero-accent {{
            color: {RED_ACCENT};
        }}
        .hero-subtitle {{
            font-size: 1.1rem;
            color: {GREY_TEXT};
            margin-top: 0.5rem;
        }}

        /* ── Timeline ─────────────────────────────────────── */
        .timeline-container {{
            display: flex;
            align-items: center;
            gap: 0;
            padding: 1.5rem 0;
        }}
        .timeline-stint {{
            display: flex;
            align-items: center;
            justify-content: center;
            height: 40px;
            border-radius: 4px;
            font-weight: 600;
            font-size: 0.8rem;
            color: {BG_PRIMARY};
            position: relative;
        }}
        .timeline-pit {{
            width: 2px;
            height: 55px;
            background: {WHITE};
            position: relative;
        }}
        .timeline-pit-label {{
            position: absolute;
            top: -18px;
            left: 50%;
            transform: translateX(-50%);
            font-size: 0.65rem;
            color: {GREY_TEXT};
            white-space: nowrap;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
