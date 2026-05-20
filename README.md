# 🏎️ F1 Race Strategist AI

An AI-powered Formula 1 race strategy assistant built with **LangGraph**, **FastF1**, and **ChromaDB**. It combines real telemetry data, tire compound analysis, weather conditions, and vector-based retrieval to generate and evaluate race strategies.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Streamlit UI                      │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│               LangGraph Workflow                    │
│  ┌───────────┐  ┌────────────┐  ┌───────────────┐  │
│  │ Data Node │→ │ Strategy   │→ │  Evaluation   │  │
│  │ (FastF1)  │  │   Agent    │  │    Agent      │  │
│  └───────────┘  └────────────┘  └───────────────┘  │
└──────────────────────┬──────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
   ┌─────────┐   ┌──────────┐   ┌─────────┐
   │ FastF1  │   │ ChromaDB │   │ Weather │
   │  Tool   │   │   RAG    │   │  Tool   │
   └─────────┘   └──────────┘   └─────────┘
```

## Project Structure

```
f1-raceStrategistAI/
├── graph/                  # LangGraph state & workflow
│   └── state.py            # RaceStrategyState (TypedDict)
├── tools/                  # Data acquisition tools
│   ├── fastf1_tool.py      # FastF1 wrapper (sessions, laps, telemetry, tires)
│   └── weather_tool.py     # Open-Meteo weather API
├── rag/                    # Retrieval-Augmented Generation
│   ├── ingestor.py         # Populates ChromaDB from raw data
│   ├── embedder.py         # ChromaDB client & collection access
│   └── retriever.py        # Semantic search + FastF1 enrichment
├── agents/                 # LangGraph agent nodes
│   ├── strategist_agent.py # Strategy generation
│   ├── evaluator_agent.py  # Strategy evaluation
│   ├── tire_agent.py       # Tire analysis
│   └── weather_agent.py    # Weather impact assessment
├── data/
│   └── raw/                # Source data (circuits.csv, FIA regulations)
├── notebooks/
│   └── rag_testing.ipynb   # RAG pipeline validation
├── tests/                  # Test suite
├── requirements.txt        # Python dependencies
├── .env                    # API keys (not tracked)
└── .gitignore
```

## Tech Stack

| Component | Technology |
|---|---|
| Agent Framework | [LangGraph](https://github.com/langchain-ai/langgraph) + [LangChain](https://github.com/langchain-ai/langchain) |
| F1 Data | [FastF1](https://github.com/theOehrly/Fast-F1) |
| Vector Store | [ChromaDB](https://www.trychroma.com/) |
| Weather | [Open-Meteo](https://open-meteo.com/) |
| UI | [Streamlit](https://streamlit.io/) |
| Data Processing | [Pandas](https://pandas.pydata.org/) |

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/jp-mmoura/race-strategist-ai.git
cd race-strategist-ai
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env` and add your API keys:

```bash
cp .env.example .env
```

```env
OPENAI_API_KEY=your_openai_api_key_here
LANGCHAIN_API_KEY=your_langchain_api_key_here
```

### 3. Ingest Data

Populate ChromaDB with circuit data:

```bash
python -m rag.ingestor
```

### 4. Validate Setup

Test that FastF1 and the RAG pipeline work:

```bash
# Test FastF1 data fetching
python tools/fastf1_tool.py

# Test RAG retrieval
python -m rag.retriever
```

## Key Features

### 🔧 FastF1 Tool (`tools/fastf1_tool.py`)

Full wrapper around the FastF1 library:

- **Session loading** — any GP, any year, any session type (FP1–Race)
- **Lap data** — all laps or filtered by driver
- **Telemetry** — speed, throttle, brake, RPM, gear, GPS (X/Y/Z)
- **Tire analysis** — compound, tyre life, stints, fresh-tyre flags
- **Race results** — official classification
- **Weather** — air/track temp, humidity, rainfall

### 🔍 RAG Pipeline (`rag/`)

Two retrieval modes:

1. **Circuit search** — semantic vector search over 77 F1 circuits
2. **Race context** — combines ChromaDB + FastF1 to build rich context (results, stint strategies, weather) for LLM consumption

```python
from rag.retriever import retrieve_race_context

ctx = retrieve_race_context(
    query="what strategy won at Silverstone 2022?",
    year=2022,
    circuit="Silverstone",
)
print(ctx["context_text"])
# → Winner: SAI (Ferrari), Strategy: MEDIUM → HARD → SOFT, Rainfall: Yes
```

### 📊 Shared State (`graph/state.py`)

`RaceStrategyState` TypedDict flows through LangGraph:

| Field | Type | Purpose |
|---|---|---|
| `circuit` | `str` | GP name |
| `year` | `int` | Season year |
| `lap_data` | `DataFrame` | Session laps |
| `tire_data` | `DataFrame` | Compound & stint info |
| `weather_data` | `DataFrame` | Weather readings |
| `strategy_recommendation` | `dict` | Strategy agent output |
| `evaluation_result` | `dict` | Evaluation agent output |

## Development

```bash
# Run tests
pytest tests/

# Run Streamlit UI (coming soon)
streamlit run app.py
```

## License

MIT

---

Built with ❤️ and data by [@jp-mmoura](https://github.com/jp-mmoura)
