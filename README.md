# 🧠 Universal AI-Powered Trading Intelligence Platform

A fully autonomous trading intelligence system that treats markets as **adaptive, living organisms** — not static price charts. It ingests real-time data from 5 sources, analyzes through 6 AI-driven engines, and executes trades autonomously via Alpaca paper trading.

> Built with Python, C++, FinBERT AI, Redis, and Monte Carlo simulations.

---

##  Architecture

```
                          ┌──────────────────────┐
                          │     FastAPI Server    │
                          │    (23 endpoints)     │
                          └──────────┬───────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              ▼                      ▼                      ▼
    ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
    │ Prediction Engine│   │Scenario Simulator│  │  Trade Executor  │
    │  (6 dimensions)  │   │ (Monte Carlo GBM)│  │ (Alpaca Paper)  │
    └───────┬─────────┘   └─────────────────┘   └────────┬────────┘
            │                                             │
            ▼                                             ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                     Intelligence Engines                     │
    │  Market Intelligence │ Biological Modeling │ Derivatives (C++)│
    └───────────────────────────────┬─────────────────────────────┘
                                    │
    ┌───────────────────────────────┼─────────────────────────────┐
    │                         Redis Pub/Sub                        │
    └───┬──────────┬──────────┬──────────┬──────────┬─────────────┘
        ▼          ▼          ▼          ▼          ▼
    Binance WS  Alpaca API  Yahoo Fin  Finnhub   Macro Poller
    (BTC Book)  (US Quotes) (Options)  (News)    (VIX/10Y/DXY)
```

---

##  Quick Start

### Prerequisites

- Python 3.11+
- Redis server running on `localhost:6379`
- C++ compiler (for gamma engine)

### 1. Install Dependencies

```bash
pip install -e .
pip install transformers torch --index-url https://download.pytorch.org/whl/cpu
```

### 2. Configure API Keys

Copy `.env.example` to `.env` and add your keys:

```bash
cp .env.example .env
```

```env
# Required
REDIS_URL=redis://localhost:6379/0

# Optional — enhances data quality
ALPACA_API_KEY=your_key          # Paper trading (https://alpaca.markets)
ALPACA_API_SECRET=your_secret
FINNHUB_API_KEY=your_key         # News headlines (https://finnhub.io)

# Tracked symbols
EQUITY_SYMBOLS=AAPL,NVDA,TSLA,MSFT,GOOGL,AMZN,META
CRYPTO_SYMBOLS=BTC
```

> **Note:** The platform works without Alpaca/Finnhub keys — it falls back to Yahoo Finance for data and yfinance for news.

### 3. Build the C++ Gamma Engine

```bash
python setup_cpp.py build_ext --inplace
```

### 4. Start the Platform

```bash
python main.py
```

The platform launches **9 concurrent services**:
- 5 data ingestion pipelines
- 3 intelligence engines
- 1 FastAPI server on `http://localhost:8000`

API docs available at: **http://localhost:8000/docs**

### 5. Run Tests

```bash
python -m pytest tests/ -v
```

---

##  The 6 Intelligence Dimensions

Every stock is scored **0–100** on 6 dimensions, combined into a weighted composite:

| Dimension | Weight | Source | What It Measures |
|-----------|--------|--------|-----------------|
| **Fundamentals** | 25% | Yahoo Finance | P/E, cash flow, margins, debt, revenue growth |
| **Options Flow** | 20% | Yahoo + C++ Engine | GEX, IV skew, convexity risk, dealer positioning |
| **Market Health** | 15% | Alpaca + Binance | Volatility regime, liquidity, spreads |
| **Macro Environment** | 15% | VIX, 10Y, DXY | Fear index, rates, dollar strength |
| **Momentum** | 15% | Yahoo Finance | Daily change, 52-week range positioning |
| **Sentiment** | 10% | Finnhub + FinBERT | AI-scored news headlines |

### Decision Thresholds

| Composite Score | Signal | Action |
|----------------|--------|--------|
| ≥ 68 | 🟢 **BUY** | Strong conviction — enter position |
| 45 – 67 | 🟡 **HOLD** | Insufficient edge — wait |
| < 45 | 🔴 **SELL** | Bearish — exit or avoid |

---

##  API Endpoints (23 total)

### Core Data
```bash
GET  /api/v1/health                          # System health
GET  /api/v1/lookup/{symbol}                 # Full analysis for any ticker
GET  /api/v1/fundamentals/{symbol}           # Company financials
GET  /api/v1/market/{symbol}                 # Real-time market state
GET  /api/v1/macro/indicators                # VIX, 10Y yield, DXY
GET  /api/v1/health-index/{symbol}           # Biological health score
GET  /api/v1/ecosystem/map                   # Competitive sector rankings
```

### Sentiment & Correlations
```bash
GET  /api/v1/sentiment?headline=...          # FinBERT AI sentiment scoring
GET  /api/v1/correlate?symbols=AAPL,TSLA     # Cross-asset correlations
```

### Derivatives Intelligence
```bash
GET  /api/v1/derivatives/gamma/{symbol}             # Gamma exposure map
GET  /api/v1/derivatives/vol-surface/{symbol}        # Volatility surface
GET  /api/v1/derivatives/convexity-risk/{symbol}     # Convexity risk
GET  /api/v1/derivatives/vol-regime/{symbol}         # Volatility regime
```

### Prediction & Simulation
```bash
GET  /api/v1/predict/{symbol}                        # BUY/HOLD/SELL signal
GET  /api/v1/simulate/{symbol}?horizon=1m            # Monte Carlo simulation
GET  /api/v1/simulate/scenarios/list                  # Stress scenario list
```

### Trading (Alpaca Paper)
```bash
GET    /api/v1/trade/account                          # Account balance
GET    /api/v1/trade/positions                        # Open positions + P&L
GET    /api/v1/trade/orders                           # Order history
POST   /api/v1/trade/order?symbol=AAPL&side=buy&qty=5 # Place order
POST   /api/v1/trade/smart/{symbol}                   # AI-driven trade
DELETE /api/v1/trade/position/{symbol}                # Close position
```

### Auto-Trader Bot
```bash
POST /api/v1/trade/auto/start?duration_minutes=30    # Start autonomous trading
POST /api/v1/trade/auto/stop                          # Emergency stop
GET  /api/v1/trade/auto/status                        # Live status + trade log
```

---

##  Auto-Trader Bot

The platform includes an autonomous trading bot that:

1. **Runs for a configurable duration** (5 min to 24 hr)
2. **Checks predictions** every 30–120 seconds for all tracked symbols
3. **Executes trades** automatically when signals are strong
4. **Tracks P&L** and logs every decision with reasoning

```bash
# Start: 30 min, check every 60s, max $5K/trade, max 5 positions
curl -X POST "http://localhost:8000/api/v1/trade/auto/start?\
duration_minutes=30&check_interval=60&max_investment=5000&max_positions=5"

# Monitor
curl http://localhost:8000/api/v1/trade/auto/status | python -m json.tool

# Stop
curl -X POST http://localhost:8000/api/v1/trade/auto/stop
```

---

##  Scenario Simulation

Monte Carlo engine using **Geometric Brownian Motion**:

- **10,000 simulations** under normal conditions
- **6 stress scenarios**: Market Crash, VIX Spike, Rate Hike, Bull Rally, Liquidity Crisis, Stagflation
- **Risk metrics**: VaR (95%/99%), Conditional VaR, max drawdown
- **Probabilities**: chance of loss, chance of 10%+ gain/loss
- **Horizons**: 1 day, 1 week, 1 month, 3 months

```bash
curl "http://localhost:8000/api/v1/simulate/AAPL?horizon=1m" | python -m json.tool
```

---

## 🛠️ Tech Stack

| Technology | Purpose |
|-----------|---------|
| **Python 3.11+ / FastAPI** | API server + async engines |
| **C++ / pybind11** | High-performance gamma exposure computation |
| **FinBERT (HuggingFace)** | Financial sentiment analysis AI |
| **Redis** | Real-time pub/sub data bus |
| **Alpaca API** | Paper trading + US equity data |
| **Binance WebSocket** | Live BTC order book |
| **Yahoo Finance** | Fundamentals, options chains, macro |
| **Finnhub** | Real-time financial news |
| **NumPy** | Monte Carlo simulations |

---

##  Project Structure

```
Trading/
├── main.py                          # Platform entry point (orchestrator)
├── config/
│   └── settings.py                  # Environment configuration
├── ingestion/                       # Data pipelines
│   ├── alpaca_client.py             # US stock quotes (every 2s)
│   ├── binance_ws.py                # BTC order book (WebSocket)
│   ├── yahoo_finance.py             # Fundamentals + options chains
│   ├── finnhub_poller.py            # News headlines
│   └── macro_poller.py              # VIX, 10Y Treasury, DXY
├── engines/                         # Intelligence engines
│   ├── market_intelligence.py       # Volatility regime + liquidity
│   ├── biological_modeling.py       # Asset health (biological model)
│   ├── derivatives_intelligence.py  # Options analysis
│   ├── cpp/gamma_engine.cpp         # C++ gamma exposure engine
│   ├── prediction.py                # 6-dimension scoring → BUY/HOLD/SELL
│   ├── scenario_simulation.py       # Monte Carlo + stress tests
│   ├── trade_execution.py           # Alpaca paper trading
│   ├── auto_trader.py               # Autonomous trading bot
│   └── lookup.py                    # On-demand symbol lookup
├── api/
│   ├── app.py                       # FastAPI app factory
│   └── routes.py                    # 23 API endpoints
├── state/
│   └── redis_client.py              # Redis pub/sub client
├── models/
│   └── schemas.py                   # Pydantic data models
├── tests/                           # 55 unit tests
│   ├── test_api.py
│   ├── test_engines.py
│   ├── test_gamma_engine.py
│   ├── test_ingestion.py
│   ├── test_new_features.py
│   └── test_redis_roundtrip.py
├── .env                             # API keys (not committed)
├── .env.example                     # Template for .env
├── pyproject.toml                   # Dependencies
└── setup_cpp.py                     # C++ build script
```

---

##  Disclaimer

This platform is for **educational and paper trading purposes only**. It is not financial advice. Past performance and simulations do not guarantee future results. Always do your own research before making trading decisions.
