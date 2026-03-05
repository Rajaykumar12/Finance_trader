#  Universal AI-Powered Trading Intelligence Platform

A fully autonomous trading intelligence system that treats markets as **adaptive, living organisms** — not static price charts. It ingests real-time data from 5 sources, analyzes through 6 AI-driven engines, and executes trades autonomously via Alpaca paper trading.

> Built with Python, C++, FinBERT AI, Redis, and Monte Carlo simulations.

---

##  Architecture

```
                          ┌──────────────────────┐
                          │     FastAPI Server    │
                          │    (32 endpoints)     │
                          └──────────┬───────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              ▼                      ▼                      ▼
    ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────────┐
    │ Prediction Engine│   │Scenario Simulator│  │   Trading Bots       │
    │  (6 dimensions)  │   │ (Monte Carlo GBM)│  │ Rigorous + Day Bot  │
    └───────┬─────────┘   └─────────────────┘   └────────┬────────────┘
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

> **Note:** The platform works without Alpaca/Finnhub keys — it falls back to Yahoo Finance for data.

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

**56 tests** — all pass.

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

##  Trading Bots

The platform includes **two autonomous trading bots** with different strategies.

### 🤖 Rigorous Auto-Trader

An AI-driven bot that uses the full 6-dimension prediction engine. Designed for **high conviction, low frequency** — it skips most signals and only acts when all gates agree.

**Entry Gates** (ALL must pass to enter):

| Gate | Threshold |
|------|-----------|
| Composite AI score | ≥ 78 (was 60) |
| Prediction confidence | ≥ 65% |
| Macro regime | Score ≥ 55 (not a risk-off day) |
| Liquidity | Not `"vanishing"` |
| Bullish signals | ≥ 2 independent reasons |
| PDT day-trades remaining | ≥ 2 |
| Symbol cooldown | 10 min post-close |
| Signal type | Must be `BUY` |
| Macro circuit breaker | Score < 35 → block all buys |

**Position Sizing** — Half-Kelly:
```
edge     = (composite_score − 50) / 50
notional = edge × confidence × buying_power   (capped at max_investment)
```

**Exit** — Bracket order locked in at entry (default: TP +1.5%, SL -0.8%, R:R ≈ 1.9:1)

```bash
# Start (default: 30 min, strict gates)
curl -X POST "http://localhost:8000/api/v1/trade/auto/start?\
duration_minutes=30&min_score=78&min_confidence=0.65&take_profit_pct=0.015&stop_loss_pct=0.008"

# Monitor
curl http://localhost:8000/api/v1/trade/auto/status | python -m json.tool

# Stop
curl -X POST http://localhost:8000/api/v1/trade/auto/stop
```

---

### 📈 Day Trading Bot

An intraday bot that reacts to **technical price signals** on 5-minute bars. Market-hours aware — manages its own session lifecycle automatically.

**Intraday Signals** (score 0–100, BUY if ≥ 70):

| Signal | Description | Score Impact |
|--------|-------------|-------------|
| **VWAP** | Price above/below Volume-Weighted Avg Price | ±12 |
| **EMA Crossover** | 9-period EMA vs 21-period EMA | ±15 |
| **ORB Breakout** | Break above/below 30-min opening range | ±15 |
| **Momentum** | 3 consecutive rising/falling closes | ±8 |

**Session Lifecycle** (automatic, IST times shown):

| ET Time | IST Time | Action |
|---------|----------|--------|
| 9:10–9:25 AM | 7:40–7:55 PM | Pre-market watchlist scan (ranks symbols by AI score) |
| 9:30 AM | 8:00 PM | Market open — intraday loop starts |
| 3:45 PM | 2:15 AM | EOD flatten — all positions closed |
| 4:00 PM | 2:30 AM | Market close — bot waits for next session |

**Safety Controls:**

| Control | Default | Description |
|---------|---------|-------------|
| Daily drawdown limit | −2% | Stop all new buys if portfolio drops this much from day open |
| PDT rolling tracker | 3 per 5 days | Enforces Pattern Day Trader rule, stored in Redis |
| EOD flatten | 15:45 ET | All positions closed before market close |
| Max positions | 5 | Maximum simultaneous open positions |

```bash
# Start day trading bot
curl -X POST "http://localhost:8000/api/v1/trade/day/start?\
take_profit_pct=0.015&stop_loss_pct=0.008&daily_drawdown_limit=0.02"

# Check live intraday signals for a symbol
curl http://localhost:8000/api/v1/trade/day/signals/AAPL

# Current market session (ET timezone)
curl http://localhost:8000/api/v1/trade/session

# Status (P&L, positions, PDT remaining, log)
curl http://localhost:8000/api/v1/trade/day/status

# Stop
curl -X POST http://localhost:8000/api/v1/trade/day/stop
```

---

### Bracket Orders

Both bots use **bracket orders** — the take-profit and stop-loss are submitted atomically at entry. Alpaca manages them automatically; no polling loop needed.

```
Entry (market buy)
  ├── Take-profit limit sell at entry × (1 + tp%)   → filled when target reached
  └── Stop-loss stop sell at entry × (1 − sl%)      → filled when loss limit hit
```

---

##  API Endpoints (32 total)

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

### Rigorous Auto-Trader
```bash
POST /api/v1/trade/auto/start   # Start (params: min_score, min_confidence, take_profit_pct, stop_loss_pct)
POST /api/v1/trade/auto/stop    # Emergency stop
GET  /api/v1/trade/auto/status  # Live status + full decision log
```

### Day Trading Bot
```bash
GET  /api/v1/trade/session                     # Current market session (ET)
POST /api/v1/trade/day/start                   # Start day bot (params: daily_drawdown_limit, min_intraday_score)
POST /api/v1/trade/day/stop                    # Stop
GET  /api/v1/trade/day/status                  # Status + PDT remaining + P&L
GET  /api/v1/trade/day/signals/{symbol}        # Live VWAP + EMA + ORB signal
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
| **Redis** | Real-time pub/sub data bus + PDT state |
| **Alpaca API** | Paper trading + US equity data + OHLCV bars |
| **Binance WebSocket** | Live BTC order book (100ms) |
| **Yahoo Finance** | Fundamentals, options chains, macro |
| **Finnhub** | Real-time financial news |
| **NumPy / Polars** | Monte Carlo simulations + tick analytics |

---

##  Project Structure

```
Trading/
├── main.py                          # Platform entry point (orchestrator)
├── config/
│   └── settings.py                  # Environment configuration
├── ingestion/                       # Data pipelines
│   ├── alpaca_client.py             # US stock quotes (every 2s)
│   ├── alpaca_bars.py               # 5-min OHLCV bar fetcher       ← NEW
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
│   ├── trade_execution.py           # Alpaca paper trading + bracket orders ← UPDATED
│   ├── auto_trader.py               # Rigorous AI bot (9-gate filter)       ← UPDATED
│   ├── market_hours.py              # US ET session utilities                ← NEW
│   ├── intraday_signals.py          # VWAP + EMA + ORB signals               ← NEW
│   ├── day_trader.py                # Day trading bot (IST/ET aware)         ← NEW
│   └── lookup.py                    # On-demand symbol lookup
├── api/
│   ├── app.py                       # FastAPI app factory
│   └── routes.py                    # 32 API endpoints                       ← UPDATED
├── state/
│   └── redis_client.py              # Redis pub/sub client
├── models/
│   └── schemas.py                   # Pydantic data models
├── tests/                           # 56 unit tests
│   ├── test_engines.py
│   ├── test_gamma_engine.py
│   ├── test_rigorous_auto_trader.py # ← NEW
│   ├── test_day_trader.py           # ← NEW
│   └── ...
├── .env                             # API keys (not committed)
├── .env.example                     # Template for .env
├── pyproject.toml                   # Dependencies
└── setup_cpp.py                     # C++ build script
```

---

##  Disclaimer

This platform is for **educational and paper trading purposes only**. It is not financial advice. Past performance and simulations do not guarantee future results. Always do your own research before making trading decisions.
