"""FastAPI routes — REST endpoints + WebSocket streaming."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from state.redis_client import redis_client
from engines.lookup import lookup_symbol
from engines.prediction import predict
from engines.scenario_simulation import simulate_scenarios, STRESS_SCENARIOS
from engines.trade_execution import (
    get_account, get_positions, get_orders, place_order,
    cancel_order, cancel_all_orders, close_position, close_all_positions,
    smart_trade,
)
from engines.auto_trader import auto_trader
from engines.day_trader import day_trader

logger = logging.getLogger(__name__)
router = APIRouter()


# ────────────────────────────────────────────────────────────────────
# Trade Execution (Alpaca Paper Trading)
# ────────────────────────────────────────────────────────────────────

@router.get("/trade/account", tags=["Trading"])
async def trading_account():
    """Get paper trading account info — cash, portfolio value, buying power."""
    try:
        return await get_account()
    except Exception as e:
        return {"error": str(e)}


@router.get("/trade/positions", tags=["Trading"])
async def trading_positions():
    """Get all open positions with unrealized P&L."""
    try:
        return await get_positions()
    except Exception as e:
        return {"error": str(e)}


@router.get("/trade/orders", tags=["Trading"])
async def trading_orders(status: str = "all", limit: int = 20):
    """Get recent orders. Status: all, open, closed."""
    try:
        return await get_orders(status, limit)
    except Exception as e:
        return {"error": str(e)}


@router.post("/trade/order", tags=["Trading"])
async def trading_place_order(
    symbol: str,
    side: str,
    qty: int = 1,
    order_type: str = "market",
    limit_price: float = None,
    stop_price: float = None,
    time_in_force: str = "day",
):
    """Place a paper trade order.

    Args:
        symbol: Stock ticker (AAPL, TSLA, NVDA)
        side: buy or sell
        qty: Number of shares
        order_type: market, limit, stop, stop_limit
        limit_price: Required for limit orders
        stop_price: Required for stop orders
        time_in_force: day, gtc (good til cancelled)

    Example: POST /api/v1/trade/order?symbol=AAPL&side=buy&qty=5&order_type=market
    """
    try:
        return await place_order(symbol, side, qty, order_type, limit_price, stop_price, time_in_force)
    except Exception as e:
        return {"error": str(e)}


@router.delete("/trade/order/{order_id}", tags=["Trading"])
async def trading_cancel_order(order_id: str):
    """Cancel a specific pending order."""
    try:
        return await cancel_order(order_id)
    except Exception as e:
        return {"error": str(e)}


@router.delete("/trade/orders", tags=["Trading"])
async def trading_cancel_all():
    """Cancel all open orders."""
    try:
        return await cancel_all_orders()
    except Exception as e:
        return {"error": str(e)}


@router.delete("/trade/position/{symbol}", tags=["Trading"])
async def trading_close_position(symbol: str):
    """Close (sell) entire position in a stock."""
    try:
        return await close_position(symbol)
    except Exception as e:
        return {"error": str(e)}


@router.delete("/trade/positions", tags=["Trading"])
async def trading_close_all():
    """Liquidate ALL positions (emergency exit)."""
    try:
        return await close_all_positions()
    except Exception as e:
        return {"error": str(e)}


@router.post("/trade/smart/{symbol}", tags=["Trading"])
async def trading_smart(symbol: str, max_investment: float = 5000):
    """AI-powered smart trade — uses prediction engine to auto-decide.

    The prediction engine analyzes fundamentals, sentiment, options flow,
    macro, and momentum. If the score is strong enough, it executes a trade.

    - Score >= 68: BUY (position size based on confidence)
    - Score < 45 + holding position: SELL
    - Otherwise: HOLD (no trade)

    Example: POST /api/v1/trade/smart/AAPL?max_investment=5000
    """
    try:
        return await smart_trade(symbol, max_investment)
    except Exception as e:
        return {"error": str(e)}


# ────────────────────────────────────────────────────────────────────
# Auto-Trader (Autonomous Live Trading)
# ────────────────────────────────────────────────────────────────────

@router.post("/trade/auto/start", tags=["Auto Trading"])
async def auto_trade_start(
    duration_minutes: int = 30,
    check_interval: int = 60,
    max_investment: float = 5000,
    max_positions: int = 5,
    min_score: float = 78.0,
    min_confidence: float = 0.65,
    take_profit_pct: float = 0.015,
    stop_loss_pct: float = 0.008,
):
    """Start the rigorous autonomous trading bot.

    Only enters trades when ALL conviction gates pass simultaneously.
    Every trade has a bracket exit (TP + SL) wired in at submission time.

    Args:
        duration_minutes:  How long to run (default 30 min)
        check_interval:    Seconds between checks (default 60s)
        max_investment:    Hard cap per trade in $ (default $5,000)
        max_positions:     Max simultaneous open positions (default 5)
        min_score:         Minimum composite score to enter (default 78)
        min_confidence:    Minimum prediction confidence, 0–1 (default 0.65)
        take_profit_pct:   Take-profit above entry, e.g. 0.015 = 1.5% (default)
        stop_loss_pct:     Stop-loss below entry, e.g. 0.008 = 0.8% (default)

    Example:
        POST /api/v1/trade/auto/start?duration_minutes=30&min_score=78&take_profit_pct=0.015
    """
    try:
        return await auto_trader.start(
            duration_minutes=duration_minutes,
            check_interval=check_interval,
            max_investment=max_investment,
            max_positions=max_positions,
            min_score=min_score,
            min_confidence=min_confidence,
            take_profit_pct=take_profit_pct,
            stop_loss_pct=stop_loss_pct,
        )
    except Exception as e:
        return {"error": str(e)}


@router.post("/trade/auto/stop", tags=["Auto Trading"])
async def auto_trade_stop():
    """Stop the autonomous trading bot."""
    try:
        return await auto_trader.stop()
    except Exception as e:
        return {"error": str(e)}


@router.get("/trade/auto/status", tags=["Auto Trading"])
async def auto_trade_status():
    """Get auto-trader status: running state, trades, P&L, log."""
    try:
        return await auto_trader.status()
    except Exception as e:
        return {"error": str(e)}


# ────────────────────────────────────────────────────────────────────
# Day Trading Bot
# ────────────────────────────────────────────────────────────────────

@router.get("/trade/session", tags=["Day Trading"])
async def market_session():
    """Get the current US market session state (ET timezone).

    Returns session label (PRE-MARKET, REGULAR, EOD-FLATTEN, AFTER-HOURS, CLOSED, WEEKEND),
    whether the market is open, minutes to open/close, and today's date ET.
    """
    from engines.market_hours import session_summary
    return session_summary()


@router.post("/trade/day/start", tags=["Day Trading"])
async def day_trade_start(
    check_interval:       int   = 60,
    max_investment:       float = 5000.0,
    max_positions:        int   = 5,
    take_profit_pct:      float = 0.015,
    stop_loss_pct:        float = 0.008,
    daily_drawdown_limit: float = 0.02,
    min_intraday_score:   float = 70.0,
):
    """Start the autonomous day-trading bot.

    The bot manages its own market hours — it waits during off-hours,
    wakes at 9:30 ET, scans at 9:15 ET, and closes all positions at 15:45 ET.

    Args:
        check_interval:       Seconds between cycles (default 60s, min 30s)
        max_investment:       Max $ per trade (default $5,000)
        max_positions:        Max simultaneous open positions (default 5)
        take_profit_pct:      TP above entry, e.g. 0.015 = +1.5%
        stop_loss_pct:        SL below entry, e.g. 0.008 = -0.8%
        daily_drawdown_limit: Stop buying if portfolio drops this % (default 2%)
        min_intraday_score:   Minimum VWAP+EMA+ORB score to enter (default 70)

    Example:
        POST /api/v1/trade/day/start?take_profit_pct=0.02&stop_loss_pct=0.01
    """
    try:
        return await day_trader.start(
            check_interval=check_interval,
            max_investment=max_investment,
            max_positions=max_positions,
            take_profit_pct=take_profit_pct,
            stop_loss_pct=stop_loss_pct,
            daily_drawdown_limit=daily_drawdown_limit,
            min_intraday_score=min_intraday_score,
        )
    except Exception as e:
        return {"error": str(e)}


@router.post("/trade/day/stop", tags=["Day Trading"])
async def day_trade_stop():
    """Stop the day-trading bot."""
    try:
        return await day_trader.stop()
    except Exception as e:
        return {"error": str(e)}


@router.get("/trade/day/status", tags=["Day Trading"])
async def day_trade_status():
    """Get day-trader status: session, P&L, positions, PDT remaining, log."""
    try:
        return await day_trader.status()
    except Exception as e:
        return {"error": str(e)}


@router.get("/trade/day/signals/{symbol}", tags=["Day Trading"])
async def intraday_signals(symbol: str):
    """Get live intraday signals for a symbol (VWAP, EMA, ORB).

    Fetches today's 5-min bars and computes the intraday signal immediately.
    Useful for inspecting the signal engine without running the full bot.

    Example: GET /api/v1/trade/day/signals/AAPL
    """
    try:
        from ingestion.alpaca_bars import fetch_today_bars
        from engines.intraday_signals import generate_intraday_signal
        bars = await fetch_today_bars(symbol)
        if not bars:
            return {"error": f"No intraday bars for {symbol.upper()} — is market open?", "symbol": symbol.upper()}
        return generate_intraday_signal(symbol.upper(), bars)
    except Exception as e:
        return {"error": str(e), "symbol": symbol.upper()}


# ────────────────────────────────────────────────────────────────────
# Scenario Simulation (Monte Carlo + Stress Testing)
# ────────────────────────────────────────────────────────────────────

@router.get("/simulate/{symbol}", tags=["Scenario Simulation"])
async def scenario_simulation(symbol: str, horizon: str = "1m", scenario: str = ""):
    """Monte Carlo scenario simulation for ANY ticker.

    Runs 10,000 simulations to forecast price distributions,
    plus stress tests (crash, VIX spike, rate hike, etc.).

    Args:
        symbol: Ticker (AAPL, TSLA, BTC-USD, RELIANCE.NS)
        horizon: 1d, 1w, 1m, 3m
        scenario: Optional specific stress scenario to run (leave empty for all)

    Example: GET /api/v1/simulate/AAPL?horizon=1m
    """
    try:
        return await simulate_scenarios(symbol, horizon)
    except Exception as e:
        return {"error": str(e), "symbol": symbol.upper()}


@router.get("/simulate/scenarios/list", tags=["Scenario Simulation"])
async def list_scenarios():
    """List all available stress scenarios."""
    return {
        "scenarios": {
            k: {"name": v["name"], "description": v["description"], "vix_level": v["vix_level"]}
            for k, v in STRESS_SCENARIOS.items()
        }
    }


# ────────────────────────────────────────────────────────────────────
# Trading Prediction (combines ALL signals)
# ────────────────────────────────────────────────────────────────────

@router.get("/predict/{symbol}", tags=["Prediction"])
async def trading_prediction(symbol: str):
    """AI-powered trading prediction for ANY ticker.

    Combines fundamentals, sentiment, market microstructure, options flow,
    macro environment, and momentum into a single BUY / HOLD / SELL signal.

    Example: GET /api/v1/predict/TSLA
    """
    try:
        return await predict(symbol)
    except Exception as e:
        return {"error": str(e), "symbol": symbol.upper()}

# ────────────────────────────────────────────────────────────────────
# On-Demand Lookup (any ticker, instant)
# ────────────────────────────────────────────────────────────────────

@router.get("/lookup/{symbol}", tags=["Lookup"])
async def on_demand_lookup(symbol: str):
    """Instant full analysis for ANY ticker — fundamentals, options,
    gamma exposure, vol surface, convexity risk, health scoring.

    Example: GET /api/v1/lookup/TSLA
    """
    try:
        return await lookup_symbol(symbol)
    except Exception as e:
        return {"error": str(e), "symbol": symbol.upper()}


# ────────────────────────────────────────────────────────────────────
# On-Demand Sentiment (Service 3)
# ────────────────────────────────────────────────────────────────────

@router.get("/sentiment", tags=["AI Sentiment"])
async def score_headline_sentiment(headline: str):
    """Score any headline's sentiment using FinBERT AI.

    Returns a score from -1.0 (very negative) to +1.0 (very positive).

    Example: GET /api/v1/sentiment?headline=Apple reports record revenue
    """
    try:
        from engines.biological_modeling import score_sentiment
        loop = asyncio.get_running_loop()
        score = await loop.run_in_executor(None, score_sentiment, headline)
        if score > 0.3:
            label, emoji = "POSITIVE", "🟢"
        elif score < -0.3:
            label, emoji = "NEGATIVE", "🔴"
        else:
            label, emoji = "NEUTRAL", "⚪"
        return {
            "headline": headline,
            "sentiment_score": score,
            "label": label,
            "emoji": emoji,
        }
    except Exception as e:
        return {"error": str(e), "hint": "FinBERT model may need to download on first use (~500MB)"}


# ────────────────────────────────────────────────────────────────────
# On-Demand Correlation (Service 4 — works without Alpaca)
# ────────────────────────────────────────────────────────────────────

@router.get("/correlate", tags=["Market Intelligence"])
async def on_demand_correlations(symbols: str, period: str = "3mo"):
    """Compute pairwise correlations between ANY stocks using historical data.

    No API keys needed — uses yfinance.

    Args:
        symbols: comma-separated tickers (e.g. AAPL,TSLA,BTC-USD,RELIANCE.NS)
        period: lookback period (1mo, 3mo, 6mo, 1y, 2y)

    Example: GET /api/v1/correlate?symbols=AAPL,TSLA,NVDA,BTC-USD
    """
    import numpy as np

    sym_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    if len(sym_list) < 2:
        return {"error": "Need at least 2 symbols (comma-separated)"}

    loop = asyncio.get_running_loop()

    def _fetch_and_correlate():
        import yfinance as yf
        import pandas as pd

        # Download all tickers at once
        data = yf.download(sym_list, period=period, auto_adjust=True, progress=False)

        if data.empty:
            return {"error": "No data returned for these symbols"}

        # Handle MultiIndex columns (yfinance returns this for multiple tickers)
        if isinstance(data.columns, pd.MultiIndex):
            close = data["Close"]
        else:
            close = data[["Close"]]
            close.columns = sym_list

        # Drop rows where ALL values are NaN, but keep partial data
        close = close.dropna(how="all")
        # Forward-fill small gaps
        close = close.ffill().bfill()

        if close.empty or len(close) < 5:
            return {"error": "Not enough price data for these symbols"}

        # Compute daily returns
        returns = close.pct_change().dropna()

        if len(returns) < 5:
            return {"error": f"Not enough data points ({len(returns)}) for correlation"}

        # Use actual column names from the DataFrame
        actual_cols = returns.columns.tolist()

        # Pairwise correlations
        corr_matrix = returns.corr()
        pairs = {}
        for i in range(len(actual_cols)):
            for j in range(i + 1, len(actual_cols)):
                s1, s2 = actual_cols[i], actual_cols[j]
                try:
                    val = float(corr_matrix.loc[s1, s2])
                    if np.isnan(val):
                        val = 0.0
                    pairs[f"{s1}_vs_{s2}"] = {
                        "correlation": round(val, 4),
                        "strength": "strong" if abs(val) > 0.7 else "moderate" if abs(val) > 0.4 else "weak",
                        "direction": "move together" if val > 0.3 else "move opposite" if val < -0.3 else "independent",
                    }
                except (KeyError, ValueError):
                    pairs[f"{s1}_vs_{s2}"] = {"correlation": None, "error": "data unavailable"}

        return {
            "symbols": actual_cols,
            "period": period,
            "data_points": len(returns),
            "pairs": pairs,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    try:
        result = await loop.run_in_executor(None, _fetch_and_correlate)
        return result
    except Exception as e:
        return {"error": str(e)}

# ────────────────────────────────────────────────────────────────────
# Health check
# ────────────────────────────────────────────────────────────────────

@router.get("/health", tags=["System"])
async def health_check():
    """Platform health check."""
    try:
        await redis_client.pool.ping()
        redis_ok = True
    except Exception:
        redis_ok = False

    return {
        "status": "ok" if redis_ok else "degraded",
        "redis": redis_ok,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ────────────────────────────────────────────────────────────────────
# REST: Market State Vector
# ────────────────────────────────────────────────────────────────────

@router.get("/market/{symbol}", tags=["Market Intelligence"])
async def get_market_state(symbol: str):
    """Get the latest Market State Vector (incl. volatility regime + liquidity)."""
    data = await redis_client.hgetall(f"market_state:{symbol.upper()}")
    if not data:
        return {"error": f"No market state data for {symbol.upper()}", "hint": "Is the ingestion + engine running?"}
    return data


# ────────────────────────────────────────────────────────────────────
# REST: Cross-Market Correlations
# ────────────────────────────────────────────────────────────────────

@router.get("/market/correlations/all", tags=["Market Intelligence"])
async def get_cross_correlations():
    """Get pairwise cross-market correlations for all tracked symbols."""
    data = await redis_client.hgetall("cross_correlations")
    if not data:
        return {"error": "No correlation data yet", "hint": "Need ≥2 symbols with ticks"}
    # Parse nested JSON fields
    for key in ("pairs",):
        if key in data and isinstance(data[key], str):
            try:
                data[key] = json.loads(data[key])
            except json.JSONDecodeError:
                pass
    return data


# ────────────────────────────────────────────────────────────────────
# REST: Asset Health Index
# ────────────────────────────────────────────────────────────────────

@router.get("/health-index/{symbol}", tags=["Biological Modeling"])
async def get_health_index(symbol: str):
    """Get the latest Asset Health Index for a symbol."""
    data = await redis_client.hgetall(f"health_index:{symbol.upper()}")
    if not data:
        return {"error": f"No health index for {symbol.upper()}", "hint": "Is Module B running?"}
    return data


# ────────────────────────────────────────────────────────────────────
# REST: Ecosystem Competitive Map
# ────────────────────────────────────────────────────────────────────

@router.get("/ecosystem/map", tags=["Biological Modeling"])
async def get_ecosystem_map():
    """Get the ecosystem competitive positioning map across all tracked assets."""
    data = await redis_client.hgetall("ecosystem_map")
    if not data:
        return {"error": "No ecosystem map yet"}
    # Parse the JSON blob
    raw = data.get("data", "{}")
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return data
    return raw


# ────────────────────────────────────────────────────────────────────
# REST: Macroeconomic Indicators
# ────────────────────────────────────────────────────────────────────

@router.get("/macro/indicators", tags=["Data"])
async def get_macro_indicators():
    """Get the latest macroeconomic indicators (VIX, yields, DXY, etc.)."""
    data = await redis_client.hgetall("macro_indicators")
    if not data:
        return {"error": "No macro data yet", "hint": "Has the macro poller run?"}
    return data


# ────────────────────────────────────────────────────────────────────
# REST: Gamma Exposure
# ────────────────────────────────────────────────────────────────────

@router.get("/derivatives/gamma/{symbol}", tags=["Derivatives Intelligence"])
async def get_gamma_exposure(symbol: str):
    """Get the latest Gamma Exposure result for a symbol."""
    data = await redis_client.hgetall(f"gamma_exposure:{symbol.upper()}")
    if not data:
        return {"error": f"No gamma exposure data for {symbol.upper()}", "hint": "Has the Yahoo Finance poller run yet?"}
    zones = data.get("hedging_pressure_zones")
    if isinstance(zones, str):
        try:
            data["hedging_pressure_zones"] = json.loads(zones)
        except json.JSONDecodeError:
            pass
    return data


# ────────────────────────────────────────────────────────────────────
# REST: Volatility Surface
# ────────────────────────────────────────────────────────────────────

@router.get("/derivatives/vol-surface/{symbol}", tags=["Derivatives Intelligence"])
async def get_vol_surface(symbol: str):
    """Get the implied volatility surface for a symbol."""
    data = await redis_client.hgetall(f"vol_surface:{symbol.upper()}")
    if not data:
        return {"error": f"No vol surface for {symbol.upper()}"}
    # Parse surface points
    pts = data.get("surface_points")
    if isinstance(pts, str):
        try:
            data["surface_points"] = json.loads(pts)
        except json.JSONDecodeError:
            pass
    return data


# ────────────────────────────────────────────────────────────────────
# REST: Convexity Risk
# ────────────────────────────────────────────────────────────────────

@router.get("/derivatives/convexity-risk/{symbol}", tags=["Derivatives Intelligence"])
async def get_convexity_risk(symbol: str):
    """Get the convexity risk assessment for a symbol."""
    data = await redis_client.hgetall(f"convexity_risk:{symbol.upper()}")
    if not data:
        return {"error": f"No convexity risk data for {symbol.upper()}"}
    return data


# ────────────────────────────────────────────────────────────────────
# REST: Volatility Regime Indicator
# ────────────────────────────────────────────────────────────────────

@router.get("/derivatives/vol-regime/{symbol}", tags=["Derivatives Intelligence"])
async def get_vol_regime(symbol: str):
    """Get the derivatives-implied volatility regime indicator."""
    data = await redis_client.hgetall(f"vol_regime_indicator:{symbol.upper()}")
    if not data:
        return {"error": f"No vol regime data for {symbol.upper()}"}
    return data


# ────────────────────────────────────────────────────────────────────
# REST: Fundamentals
# ────────────────────────────────────────────────────────────────────

@router.get("/fundamentals/{symbol}", tags=["Data"])
async def get_fundamentals(symbol: str):
    """Get the latest fundamentals snapshot for a symbol."""
    data = await redis_client.hgetall(f"fundamentals:{symbol.upper()}")
    if not data:
        return {"error": f"No fundamentals for {symbol.upper()}"}
    return data


# ────────────────────────────────────────────────────────────────────
# WebSocket: Live streaming
# ────────────────────────────────────────────────────────────────────

@router.websocket("/ws/stream/{symbol}")
async def stream_symbol(ws: WebSocket, symbol: str):
    """Stream live Market State + Health Index + Vol Regime for a symbol."""
    await ws.accept()
    sym = symbol.upper()
    logger.info("WebSocket client connected for %s", sym)

    try:
        while True:
            market = await redis_client.hgetall(f"market_state:{sym}")
            health = await redis_client.hgetall(f"health_index:{sym}")
            vol_regime = await redis_client.hgetall(f"vol_regime_indicator:{sym}")

            payload = {
                "symbol": sym,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "market_state": market or None,
                "health_index": health or None,
                "vol_regime": vol_regime or None,
            }
            await ws.send_json(payload)
            await asyncio.sleep(1.0)

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected (%s)", sym)
    except Exception:
        logger.exception("WebSocket error for %s", sym)
        await ws.close()
