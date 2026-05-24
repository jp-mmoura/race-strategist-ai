"""
F1 Race Strategist AI — Home Page.

Main landing page of the application, rendering the F1 telemetry styling,
features highlight, architecture, and quick access.
"""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path for imports
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st
from dotenv import load_dotenv
from app.styles.theme import inject_css, RED_ACCENT

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Race Control Hub — F1 Strategist AI",
    page_icon="🏎️",
    layout="wide",
)
inject_css()
load_dotenv()

# ── Hero Banner ───────────────────────────────────────────────────
st.markdown(
    f"""
    <div style="padding: 2rem 0; border-bottom: 2px solid {RED_ACCENT}; margin-bottom: 2rem;">
        <span class="hero-title">F1 RACE <span class="hero-accent">STRATEGIST AI</span></span>
        <div class="hero-subtitle">Telemetry-driven, agentic race planning control panel.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Layout: Two Columns ───────────────────────────────────────────
left_col, right_col = st.columns([3, 2])

with left_col:
    st.markdown("### 🏎️ System Overview")
    st.markdown(
        """
        The **F1 Race Strategist AI** is a state-of-the-art decision-support system designed to assist pit wall engineers. 
        It integrates real-time and historical Formula 1 data with agentic AI workflows to recommend optimal race strategies.
        
        #### Key Features:
        1. **Telemetry Analysis**: Real-time tire degradation calculation and optimal pit stop windows via FastF1 data.
        2. **Multi-Agent Orchestration**: A LangGraph workflow routing through specialized sub-agents:
           - **Tire Engineer Agent**: Classifies wear rates and builds decay regression models.
           - **Weather Analyst Agent**: Fetches historical and live rainfall risks.
           - **Strategist Agent**: Generates comprehensive strategy recommendations.
           - **Evaluator Agent**: Scores recommendations against the F1 rulebook and physics constraints.
        3. **RAG Integration**: Accesses past race records, rules, and track layouts semantic indexes.
        4. **Reflection Loop**: Loops strategy proposals back to the planner if the coherence score is low.
        """
    )
    
    st.markdown("---")
    st.markdown("### 🛠️ Open Control Panel Pages")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 🏁 01. Strategy")
        st.markdown("Run interactive scenario simulation chat with the LangGraph orchestrator.")
        if st.button("Open Strategy Hub", key="go_strat"):
            st.switch_page("pages/01_🏁_Strategy.py")
            
    with c2:
        st.markdown("#### 🔧 02. Tires")
        st.markdown("Inspect tire wear charts, stint telemetry data, and pit window models.")
        if st.button("Open Tire Analysis", key="go_tires"):
            st.switch_page("pages/02_🔧_Tire_Analysis.py")
            
    with c3:
        st.markdown("#### 📊 03. History")
        st.markdown("Explore historical race data, winning strategies, and driver lap charts.")
        if st.button("Open History Hub", key="go_hist"):
            st.switch_page("pages/03_📊_History.py")

with right_col:
    st.markdown("### 🧬 Architecture Topology")
    st.markdown(
        """
        ```
        ┌─────────────────────────────────────────────────────────────┐
        │                       Streamlit UI                          │
        └────────────────────────────┬────────────────────────────────┘
                                     │
        ┌────────────────────────────▼────────────────────────────────┐
        │                    LangGraph Workflow                       │
        │                                                             │
        │  ┌────────────┐  ┌─────────────┐  ┌─────────────┐           │
        │  │Tire Agent  │  │Weather Agent│  │  RAG Agent  │           │
        │  └─────┬──────┘  └──────┬──────┘  └──────┬──────┘           │
        │        │                │                │                  │
        │        └────────────────┼────────────────┘                  │
        │                         ▼                                   │
        │               ┌───────────────────┐                         │
        │               │  Strategy Agent   │                         │
        │               └─────────┬─────────┘                         │
        │                         ▼                                   │
        │               ┌───────────────────┐                         │
        │               │  Evaluator Agent  │ ── (Score < 45?) ──┐    │
        │               └─────────┬─────────┘                    │    │
        │                         │ (Score >= 45)                ▼    │
        │                         │                         [Revision]│
        │                         ▼                              │    │
        │                      [Finish] ◀────────────────────────┘    │
        └─────────────────────────────────────────────────────────────┘
        ```
        """
    )
    
    st.markdown("---")
    st.markdown("### 📡 System Status")

    _root = Path(_PROJECT_ROOT)

    # — LLM provider ——————————————————————————————————————————————
    _provider = os.getenv("STRATEGY_LLM_PROVIDER", "google").strip().lower()
    if _provider not in ("google", "openai"):
        st.error(
            f"🧠 LLM: `STRATEGY_LLM_PROVIDER={_provider!r}` is invalid — "
            "valid values: `'google'` or `'openai'`."
        )
    elif _provider == "google":
        if os.getenv("GOOGLE_GENAI_API_KEY", "").strip():
            st.success("🧠 LLM: Google Gemini — `GOOGLE_GENAI_API_KEY` configured")
        else:
            st.warning("🧠 LLM: Google Gemini — `GOOGLE_GENAI_API_KEY` not set")
    else:
        if os.getenv("OPENAI_API_KEY", "").strip():
            st.success("🧠 LLM: OpenAI — `OPENAI_API_KEY` configured")
        else:
            st.warning("🧠 LLM: OpenAI — `OPENAI_API_KEY` not set")

    # — FastF1 cache ——————————————————————————————————————————————
    _f1_raw = os.getenv("FASTF1_CACHE_DIR", "data/fastf1_cache")
    _f1_cache = Path(_f1_raw) if Path(_f1_raw).is_absolute() else _root / _f1_raw
    if not _f1_cache.exists():
        st.warning(f"📡 FastF1 cache: not found — will be created on first use (`{_f1_cache}`)")
    elif not os.access(_f1_cache, os.W_OK):
        st.error(f"📡 FastF1 cache: no write permission (`{_f1_cache}`)")
    else:
        st.success(f"📡 FastF1 cache: ready (`{_f1_cache}`)")

    # — ChromaDB ——————————————————————————————————————————————————
    _chroma_raw = os.getenv("CHROMA_PERSIST_DIRECTORY", "data/chroma_db")
    _chroma = Path(_chroma_raw) if Path(_chroma_raw).is_absolute() else _root / _chroma_raw
    _chroma_db = _chroma / "chroma.sqlite3"
    if not _chroma.exists():
        st.warning(f"📚 ChromaDB: directory not found — run `python -m rag.ingestor` (`{_chroma}`)")
    elif not _chroma_db.exists():
        st.warning(f"📚 ChromaDB: not populated — run `python -m rag.ingestor` (`{_chroma}`)")
    elif not os.access(_chroma_db, os.R_OK):
        st.error(f"📚 ChromaDB: no read permission (`{_chroma_db}`)")
    else:
        st.success(f"📚 ChromaDB: vector store ready (`{_chroma}`)")
