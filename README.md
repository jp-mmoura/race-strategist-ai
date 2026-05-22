# 🏎️ F1 Race Strategist AI

An AI-powered Formula 1 race strategy assistant built with **LangGraph**, **FastF1**, and **ChromaDB**. It combines real telemetry data, tire compound analysis, weather conditions, and vector-based retrieval to generate and evaluate race strategies.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       Streamlit UI                          │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│                    LangGraph Workflow                        │
│                                                             │
│  ┌────────────┐  ┌─────────────┐                            │
│  │Tire Agent  │  │Weather Agent│                            │
│  │(degradation│  │(Open-Meteo  │                            │
│  │ pit window)│  │ rain/temp)  │                            │
│  └─────┬──────┘  └──────┬──────┘                            │
│        │                │                                   │
│        └───────┬────────┘                                   │
│                ▼                                            │
│  ┌─────────────────────┐     ┌──────────────────────────┐   │
│  │  Strategy Agent     │ ──▶ │   Evaluator Agent        │   │
│  │  (LLM + rule-based) │     │   (coherence scoring)    │   │
│  └─────────────────────┘     └──────────────────────────┘   │
└────────────────────────────┬────────────────────────────────┘
                             │
          ┌──────────────────┼──────────────────┐
          ▼                  ▼                  ▼
     ┌─────────┐       ┌──────────┐       ┌─────────┐
     │ FastF1  │       │ ChromaDB │       │ Weather │
     │  Tool   │       │   RAG    │       │  Tool   │
     └─────────┘       └──────────┘       └─────────┘
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
│   ├── tire_agent.py       # Tire wear, degradation & pit-window analysis
│   ├── strategist_agent.py # Strategy generation
│   ├── evaluator_agent.py  # Strategy evaluation
│   └── weather_agent.py    # Weather impact assessment
├── data/
│   └── raw/                # Source data
│       ├── circuits.csv        # 77 F1 circuits
│       ├── track_features.csv  # Track characteristics (tire stress, grip, etc.)
│       ├── track_history.csv   # Historical track data with temperatures
│       └── *.pdf               # FIA 2026 regulations
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

Test that FastF1, RAG, and agents work:

```bash
# Test FastF1 data fetching
python tools/fastf1_tool.py

# Test RAG retrieval
python -m rag.retriever

# Test Tire Agent
python -m agents.tire_agent

# Test Weather Agent
python -m agents.weather_agent

# Test Strategy Agent (offline / rule-based — no API key needed)
python -m agents.strategist_agent

# Test Evaluator Agent (full pipeline)
python -m agents.evaluator_agent
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

### 🏁 Tire Agent (`agents/tire_agent.py`)

Combines static track data (CSVs) with live FastF1 session data to produce strategy recommendations:

| Function | Description |
|---|---|
| `classify_track_tire_wear(circuit)` | Classifies wear level (High/Medium/Low) using weighted scoring on tire stress, lateral load, asphalt abrasion & grip |
| `calculate_tire_degradation(session, driver)` | Per-stint degradation rate (sec/lap) via linear regression on lap times |
| `estimate_pit_window(session, driver)` | Optimal pit-stop lap range using a crossover model (deg loss vs pit time loss) |
| `analyze_tire_strategy(circuit, year)` | Full analysis combining all above + compound recommendation + weather impact |

**Track wear classification examples:**

| Circuit | Classification | Score |
|---|---|---|
| Monaco | 🟢 Low Tire Wear | 1.00 |
| Singapore | 🟡 Medium Tire Wear | 2.05 |
| Monza | 🟡 Medium Tire Wear | 3.40 |
| Bahrain | 🔴 High Tire Wear | 3.55 |
| Spa | 🔴 High Tire Wear | 4.20 |
| Silverstone | 🔴 High Tire Wear | 4.45 |

```python
from agents.tire_agent import analyze_tire_strategy

analysis = analyze_tire_strategy("Silverstone", 2022)
print(analysis["track_wear"]["classification"])  # → "High Tire Wear"
print(analysis["compound_rec"]["recommended_order"])  # → ["MEDIUM", "MEDIUM", "HARD", "SOFT"]
print(analysis["pit_window"]["strategy_type"])  # → "2-stop"
```

### 🌦️ Weather Agent (`agents/weather_agent.py`)

Integrates the Open-Meteo API (via `tools/weather_tool.py`) to provide real-time and forecast weather data for any F1 circuit:

| Function | Description |
|---|---|
| `get_race_forecast(circuit, race_date)` | Hourly forecast filtered to the race-day window (12:00–18:00 UTC) |
| `assess_rain_risk(circuit, race_date)` | Classifies precipitation risk (High/Medium/Low/None) with wet windows |
| `analyze_weather_impact(circuit, race_date, year)` | Full analysis: current conditions, forecast, rain risk, temperature/wind, strategic notes |
| `compare_historical_weather(circuit, year)` | Compares forecast vs historical session weather from FastF1 |

```python
from agents.weather_agent import analyze_weather_impact

impact = analyze_weather_impact("Monaco")
print(impact["rain_risk"]["risk_level"])       # → "None"
print(impact["temperature"]["air_temp_avg_c"]) # → 22.2
print(impact["strategy_notes"][0])             # → "☀️ DRY CONDITIONS — ..."
```

### 🧠 Strategy Agent (`agents/strategist_agent.py`)

Consumes outputs from Tire Agent + Weather Agent + RAG to generate a unified strategy recommendation:

| Function | Description |
|---|---|
| `build_strategy_context(circuit, year, ...)` | Gathers all upstream data into a single context dict |
| `generate_strategy(circuit, year, ...)` | LLM-powered recommendation via LangChain/OpenAI (with offline fallback) |
| `generate_strategy_offline(circuit, year, ...)` | Rule-based fallback — no LLM needed |
| `run_strategy_node(state)` | LangGraph node entry-point |

Output includes: strategy type, compound order, target pit laps, full recommendation text with justification, and confidence level.

```python
from agents.strategist_agent import generate_strategy_offline

strategy = generate_strategy_offline("Silverstone", 2023)
print(strategy["strategy_type"])  # → "1-stop"
print(strategy["compounds"])     # → ["MEDIUM", "SOFT"]
print(strategy["pit_laps"])      # → [52]
print(strategy["confidence"])    # → "medium"
```

### ✅ Evaluator Agent (`agents/evaluator_agent.py`)

Verifies the coherence of a strategy recommendation through 8 rule-based checks:

| Rule | What it checks | Severity |
|---|---|---|
| `SOFT_HIGH_WEAR` | SOFT compound on high tire-wear circuit | critical / minor |
| `DRY_UNDER_RAIN` | All-dry compounds under high/medium rain risk | critical / major |
| `PIT_WINDOW_MISS` | Pit laps outside the computed optimal window | major |
| `DEG_VS_STOPS` | Strategy type vs degradation rate mismatch | major / minor |
| `SOFT_START` | Starting on SOFT at high/medium-wear circuit | major |
| `TEMP_COMPOUND` | Compound choice vs extreme temperatures | major |
| `STINT_COVERAGE` | Planned stints covering race distance | minor |
| `WET_NO_RAIN` | Wet/intermediate tyres without rain forecast | major |

Produces a coherence score (0–100) with verdicts: ✅ Approved (≥75), ⚠️ Review (≥45), ❌ Rejected (<45).

```python
from agents.evaluator_agent import evaluate_full_pipeline

result = evaluate_full_pipeline("Silverstone", 2023)
print(result["evaluation"]["score"])    # → 90
print(result["evaluation"]["verdict"])  # → "✅ Approved"
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
| `tire_analysis` | `dict` | Tire agent output |
| `weather_analysis` | `dict` | Weather agent output |
| `strategy_recommendation` | `dict` | Strategy agent output |
| `evaluation_result` | `dict` | Evaluation agent output |

## Development & Running the App

### 📊 Streamlit UI Control Panel

To start the telemetry-inspired multi-page dashboard:

```bash
streamlit run app/Home.py
```

Features available in the UI:
- **01. Strategy Hub**: Chat with the LangGraph orchestrator, model live race incidents (Safety Car, weather, laps remaining), and view a Pirelli stint-colored strategy timeline.
- **02. Tire Analysis**: Inspect live tire wear charts, stint telemetry data, regression decay rates, and track wear classification.
- **03. Race History**: Fetch past GP official classifications, look up winner strategy reference cards, and compare strategy options via interactive Plotly charts.

### 🧬 LangGraph Workflow Execution

You can run the full multi-agent workflow from Python or CLI:

```bash
# Run CLI test
python3 -m agents.supervisor
```

Or invoke the compiled graph directly in Python:

```python
from agents.supervisor import run_graph

result = run_graph("Silverstone 2023 race strategy")
print("Verdict:", result["evaluation_result"]["verdict"])
print("Score:", result["evaluation_result"]["score"])
```

## License

MIT


