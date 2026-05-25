"""
F1 Race Strategist AI — Home Page.

Main landing page: hero banner, features overview, architecture diagram,
navigation cards, and live system status panel.
"""

import os
import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st
from dotenv import load_dotenv
from app.styles.theme import inject_css, RED_ACCENT

st.set_page_config(
    page_title="Race Control Hub — F1 Strategist AI",
    page_icon="🏎️",
    layout="wide",
)
inject_css()
load_dotenv()


# ── Helpers ───────────────────────────────────────────────────────────
_PLACEHOLDER_PREFIXES = ("your_", "YOUR_", "<", "sk-placeholder")


def _is_configured(env_key: str) -> bool:
    """Return True only if the env var is set and is not a placeholder value."""
    val = os.getenv(env_key, "").strip()
    return bool(val) and not any(val.startswith(p) for p in _PLACEHOLDER_PREFIXES)


# ── Hero Banner ───────────────────────────────────────────────────────
st.markdown(
    f"""
    <div style="padding: 2rem 0; border-bottom: 2px solid {RED_ACCENT}; margin-bottom: 2rem;">
        <span class="hero-title">F1 RACE <span class="hero-accent">STRATEGIST AI</span></span>
        <div class="hero-subtitle">Telemetry-driven, agentic race planning control panel.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Layout ────────────────────────────────────────────────────────────
left_col, right_col = st.columns([3, 2])

with left_col:
    st.markdown("### System Overview")
    st.markdown(
        """
        The **F1 Race Strategist AI** is a decision-support system for pit-wall engineers.
        It combines real-time and historical Formula 1 data with an agentic LangGraph workflow
        to generate optimal race strategies.

        **Key capabilities:**
        - **Tire analysis** — degradation modelling, wear classification, and pit-window prediction via FastF1.
        - **Weather analysis** — historical and live rainfall risk, temperature, and wind impact.
        - **RAG knowledge base** — semantic search over past race data, rules, and track layouts in ChromaDB.
        - **Reflection loop** — the Evaluator scores each strategy; proposals scoring below 45 are revised automatically.
        """
    )

    # ── Navigation cards ──────────────────────────────────────────
    st.markdown("### Control Panel Pages")

    _pages = [
        (
            "🏁", "01. Strategy",
            "Interactive scenario simulation with the full LangGraph orchestrator.",
            "go_strat", "pages/01_🏁_Strategy.py",
        ),
        (
            "🔧", "02. Tire Analysis",
            "Tire wear charts, stint telemetry, and pit-window forecasts.",
            "go_tires", "pages/02_🔧_Tire_Analysis.py",
        ),
        (
            "📊", "03. History",
            "Historical race data, winning strategies, and driver lap charts.",
            "go_hist", "pages/03_📊_History.py",
        ),
    ]

    nav_cols = st.columns(3)
    for col, (icon, title, desc, key, path) in zip(nav_cols, _pages):
        with col:
            st.markdown(
                f"""
                <div class="f1-card" style="min-height:110px; margin-bottom:0.5rem;">
                    <div class="f1-card-title">{icon} {title}</div>
                    <div class="f1-card-body" style="font-size:0.82rem;">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            _label = title.split(". ", 1)[1]
            if st.button(f"Open {_label}", key=key, use_container_width=True):
                st.switch_page(path)

    # ── Quick Start ───────────────────────────────────────────────
    st.markdown("---")
    with st.expander("Quick Start — first-time setup"):
        st.markdown(
            """
            1. **Copy `.env.example` → `.env`** and fill in your API key(s).
            2. **Ingest the knowledge base** (only needed once):
               ```bash
               python -m rag.ingestor
               ```
            3. **Launch the app**:
               ```bash
               streamlit run app/Home.py
               ```
            4. Open **01 Strategy** and enter a circuit name (e.g. `Silverstone`) to generate your first strategy.

            > The app works without a ChromaDB index or LLM key — it falls back to a rule-based offline strategy automatically.
            """
        )

with right_col:
    st.markdown("### Architecture Topology")
    st.markdown(
        """
        ```
        ┌─────────────────────────────────────────────┐
        │               Streamlit UI                  │
        └────────────────────┬────────────────────────┘
                             │
        ┌────────────────────▼────────────────────────┐
        │             LangGraph Workflow               │
        │                                              │
        │  ┌───────────┐ ┌────────────┐ ┌──────────┐   │
        │  │ Tire Agent│ │Weather Agt │ │RAG Agent │   │
        │  └─────┬─────┘ └─────┬──────┘ └────┬─────┘   │
        │        └─────────────┼──────────────┘        │
        │                      ▼                       │
        │            ┌─────────────────┐               │
        │            │ Strategy Agent  │               │
        │            └────────┬────────┘               │
        │                     ▼                        │
        │            ┌─────────────────┐               │
        │            │ Evaluator Agent │─(score<45?)─┐ │
        │            └────────┬────────┘             │ │
        │                     │ (score ≥ 45)         ▼ │
        │                     │              [Revision]│
        │                     ▼                    │   │
        │                  [Finish]◀──────────────┘   |
        └─────────────────────────────────────────────┘
        ```
        """
    )

    st.markdown("---")
    st.markdown("### System Status")

    _root = Path(_PROJECT_ROOT)
    _provider = os.getenv("STRATEGY_LLM_PROVIDER", "google").strip().lower()
    _model = os.getenv("STRATEGY_LLM_MODEL", "gemini-2.0-flash").strip()

    # LLM provider & API key
    if _provider not in ("google", "openai", "deepseek"):
        st.error(
            f"🧠 LLM: `STRATEGY_LLM_PROVIDER={_provider!r}` is invalid — "
            "expected `'google'`, `'openai'`, or `'deepseek'`."
        )
    elif _provider == "google":
        if _is_configured("GOOGLE_GENAI_API_KEY"):
            st.success(f"🧠 LLM: Google Gemini `{_model}` — API key configured")
        else:
            st.warning(f"🧠 LLM: Google Gemini `{_model}` — `GOOGLE_GENAI_API_KEY` not set")
    elif _provider == "openai":
        if _is_configured("OPENAI_API_KEY"):
            st.success(f"🧠 LLM: OpenAI `{_model}` — API key configured")
        else:
            st.warning(f"🧠 LLM: OpenAI `{_model}` — `OPENAI_API_KEY` not set")
    else:  # deepseek
        if _is_configured("DEEPSEEK_API_KEY"):
            st.success(f"🧠 LLM: DeepSeek `{_model}` — API key configured")
        else:
            st.warning(f"🧠 LLM: DeepSeek `{_model}` — `DEEPSEEK_API_KEY` not set")

    # FastF1 cache
    _f1_raw = os.getenv("FASTF1_CACHE_DIR", "data/fastf1_cache")
    _f1_cache = Path(_f1_raw) if Path(_f1_raw).is_absolute() else _root / _f1_raw
    if not _f1_cache.exists():
        st.warning(f"📡 FastF1 cache: will be created on first use (`{_f1_cache.name}`)")
    elif not os.access(_f1_cache, os.W_OK):
        st.error(f"📡 FastF1 cache: no write permission (`{_f1_cache}`)")
    else:
        st.success(f"📡 FastF1 cache: ready (`{_f1_cache.name}`)")

    # ChromaDB vector store
    _chroma_raw = os.getenv("CHROMA_PERSIST_DIRECTORY", "data/chroma_db")
    _chroma = Path(_chroma_raw) if Path(_chroma_raw).is_absolute() else _root / _chroma_raw
    _chroma_db = _chroma / "chroma.sqlite3"
    if not _chroma.exists():
        st.warning(f"📚 ChromaDB: run `python -m rag.ingestor` to populate (`{_chroma.name}`)")
    elif not _chroma_db.exists():
        st.warning(f"📚 ChromaDB: database missing — run `python -m rag.ingestor` (`{_chroma.name}`)")
    elif not os.access(_chroma_db, os.R_OK):
        st.error(f"📚 ChromaDB: no read permission (`{_chroma_db.name}`)")
    else:
        st.success(f"📚 ChromaDB: vector store ready (`{_chroma.name}`)")

    # LangSmith tracing
    _tracing_on = os.getenv("LANGCHAIN_TRACING_V2", "false").strip().lower() == "true"
    _ls_key_ok = _is_configured("LANGCHAIN_API_KEY")
    _ls_project = os.getenv("LANGCHAIN_PROJECT", "default").strip()
    if _tracing_on and _ls_key_ok:
        st.success(f"🔭 LangSmith: tracing active (project `{_ls_project}`)")
    elif _tracing_on and not _ls_key_ok:
        st.warning("🔭 LangSmith: tracing enabled but `LANGCHAIN_API_KEY` not set")
    else:
        st.info("🔭 LangSmith: tracing disabled")
