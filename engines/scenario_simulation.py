"""Scenario Simulation Engine — Monte Carlo & stress testing.

Forecasts market movement through scenario simulation:
  - Monte Carlo price path simulation using historical volatility
  - Stress scenario testing (VIX spike, rate hike, market crash, liquidity crisis)
  - Probability distributions of outcomes
  - Value at Risk (VaR) and Conditional VaR (CVaR)
  - Multi-horizon forecasting (1d, 1w, 1m, 3m)

Fulfills the paper requirement:
  "Forecasts market movement through scenario simulation"
"""

from __future__ import annotations

import asyncio
import logging
import math
from datetime import datetime, timezone
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Stress scenario definitions ──────────────────────────────────────

STRESS_SCENARIOS = {
    "market_crash": {
        "name": "Market Crash (2020-style)",
        "description": "Sudden market-wide sell-off — VIX spikes, correlations go to 1",
        "volatility_multiplier": 3.0,
        "drift_override": -0.15,  # -15% expected over the period
        "vix_level": 65,
    },
    "vix_spike": {
        "name": "VIX Spike (Fear Event)",
        "description": "Volatility doubles — geopolitical event, unexpected news",
        "volatility_multiplier": 2.0,
        "drift_override": -0.05,
        "vix_level": 40,
    },
    "rate_hike": {
        "name": "Aggressive Rate Hike",
        "description": "Fed raises rates 75bps — growth stocks punished",
        "volatility_multiplier": 1.5,
        "drift_override": -0.08,
        "vix_level": 30,
    },
    "bull_run": {
        "name": "Bull Market Rally",
        "description": "Strong earnings, rate cuts, FOMO buying",
        "volatility_multiplier": 0.7,
        "drift_override": 0.12,
        "vix_level": 12,
    },
    "liquidity_crisis": {
        "name": "Liquidity Crisis",
        "description": "Flash crash — order books thin, spreads explode",
        "volatility_multiplier": 4.0,
        "drift_override": -0.20,
        "vix_level": 80,
    },
    "stagflation": {
        "name": "Stagflation",
        "description": "High inflation + slow growth — persistent grind lower",
        "volatility_multiplier": 1.3,
        "drift_override": -0.04,
        "vix_level": 28,
    },
}

HORIZONS = {
    "1d": 1,
    "1w": 5,
    "1m": 21,
    "3m": 63,
}


def _fetch_historical_stats(symbol: str, period: str = "6mo") -> dict:
    """Fetch historical price data and compute volatility/drift stats."""
    import yfinance as yf

    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=period, auto_adjust=True)

    if hist.empty or len(hist) < 20:
        raise ValueError(f"Not enough historical data for {symbol}")

    close = hist["Close"].dropna()
    returns = close.pct_change().dropna()

    current_price = float(close.iloc[-1])
    daily_vol = float(returns.std())
    daily_drift = float(returns.mean())
    annualized_vol = daily_vol * math.sqrt(252)
    annualized_return = daily_drift * 252

    # Recent stats (last 20 days)
    recent_returns = returns.tail(20)
    recent_vol = float(recent_returns.std())

    info = ticker.info or {}

    return {
        "current_price": current_price,
        "daily_volatility": daily_vol,
        "daily_drift": daily_drift,
        "annualized_volatility": annualized_vol,
        "annualized_return": annualized_return,
        "recent_volatility": recent_vol,
        "historical_returns": returns.values.tolist(),
        "name": info.get("shortName", symbol),
        "data_points": len(returns),
    }


def _monte_carlo_simulation(
    current_price: float,
    daily_vol: float,
    daily_drift: float,
    days: int,
    n_simulations: int = 10000,
    vol_multiplier: float = 1.0,
    drift_override: Optional[float] = None,
) -> dict:
    """Run Monte Carlo simulation of price paths.

    Uses Geometric Brownian Motion (GBM):
      S(t+1) = S(t) * exp((drift - 0.5*vol^2)*dt + vol*sqrt(dt)*Z)
    """
    np.random.seed(42)  # Reproducible

    vol = daily_vol * vol_multiplier
    drift = (drift_override / days) if drift_override is not None else daily_drift
    dt = 1.0  # daily steps

    # Generate random walks
    Z = np.random.standard_normal((n_simulations, days))
    daily_returns = np.exp((drift - 0.5 * vol ** 2) * dt + vol * np.sqrt(dt) * Z)

    # Build price paths
    price_paths = np.zeros((n_simulations, days + 1))
    price_paths[:, 0] = current_price
    for t in range(days):
        price_paths[:, t + 1] = price_paths[:, t] * daily_returns[:, t]

    # Final prices
    final_prices = price_paths[:, -1]
    final_returns = (final_prices - current_price) / current_price

    # Statistics
    mean_price = float(np.mean(final_prices))
    median_price = float(np.median(final_prices))
    std_price = float(np.std(final_prices))

    # Percentiles
    p5 = float(np.percentile(final_prices, 5))
    p10 = float(np.percentile(final_prices, 10))
    p25 = float(np.percentile(final_prices, 25))
    p50 = float(np.percentile(final_prices, 50))
    p75 = float(np.percentile(final_prices, 75))
    p90 = float(np.percentile(final_prices, 90))
    p95 = float(np.percentile(final_prices, 95))

    # Value at Risk
    var_95 = current_price - p5  # 95% VaR
    var_99 = current_price - float(np.percentile(final_prices, 1))

    # Conditional VaR (Expected Shortfall) — average loss below VaR
    tail_losses = final_prices[final_prices <= p5]
    cvar_95 = current_price - float(np.mean(tail_losses)) if len(tail_losses) > 0 else var_95

    # Probability of loss
    prob_loss = float(np.mean(final_returns < 0))
    prob_gain_10 = float(np.mean(final_returns > 0.10))
    prob_loss_10 = float(np.mean(final_returns < -0.10))
    prob_loss_20 = float(np.mean(final_returns < -0.20))

    # Sample paths for visualization (5 representative paths)
    indices = np.linspace(0, n_simulations - 1, 5, dtype=int)
    sample_paths = []
    for idx in indices:
        path = price_paths[idx].tolist()
        # Downsample for JSON size
        step = max(1, len(path) // 20)
        sample_paths.append([round(p, 2) for p in path[::step]])

    return {
        "simulations": n_simulations,
        "days": days,
        "current_price": round(current_price, 2),
        "mean_price": round(mean_price, 2),
        "median_price": round(median_price, 2),
        "std_dev": round(std_price, 2),
        "price_range": {
            "worst_case_5pct": round(p5, 2),
            "bearish_10pct": round(p10, 2),
            "lower_25pct": round(p25, 2),
            "median": round(p50, 2),
            "upper_75pct": round(p75, 2),
            "bullish_90pct": round(p90, 2),
            "best_case_95pct": round(p95, 2),
        },
        "expected_return": round(float(np.mean(final_returns)) * 100, 2),
        "risk_metrics": {
            "value_at_risk_95": round(var_95, 2),
            "value_at_risk_99": round(var_99, 2),
            "conditional_var_95": round(cvar_95, 2),
            "max_drawdown": round(float(current_price - np.min(final_prices)), 2),
        },
        "probabilities": {
            "prob_any_loss": round(prob_loss * 100, 1),
            "prob_gain_over_10pct": round(prob_gain_10 * 100, 1),
            "prob_loss_over_10pct": round(prob_loss_10 * 100, 1),
            "prob_loss_over_20pct": round(prob_loss_20 * 100, 1),
        },
        "sample_paths": sample_paths,
    }


async def simulate_scenarios(symbol: str, horizon: str = "1m") -> dict:
    """Run Monte Carlo simulation under normal and stress scenarios.

    Args:
        symbol: Stock ticker (AAPL, TSLA, RELIANCE.NS, BTC-USD, etc.)
        horizon: Time horizon — 1d, 1w, 1m, 3m

    Returns a comprehensive scenario analysis with normal + stress outcomes.
    """
    if horizon not in HORIZONS:
        return {"error": f"Invalid horizon. Use: {list(HORIZONS.keys())}"}

    days = HORIZONS[horizon]
    loop = asyncio.get_running_loop()

    def _run():
        # Fetch historical data
        stats = _fetch_historical_stats(symbol)
        price = stats["current_price"]
        vol = stats["daily_volatility"]
        drift = stats["daily_drift"]

        # 1) Normal market simulation
        normal = _monte_carlo_simulation(price, vol, drift, days)

        # 2) Stress scenario simulations
        stress_results = {}
        for scenario_id, params in STRESS_SCENARIOS.items():
            sim = _monte_carlo_simulation(
                price, vol, drift, days,
                n_simulations=5000,
                vol_multiplier=params["volatility_multiplier"],
                drift_override=params["drift_override"],
            )
            stress_results[scenario_id] = {
                "name": params["name"],
                "description": params["description"],
                "vix_level": params["vix_level"],
                "expected_price": sim["median_price"],
                "expected_return": sim["expected_return"],
                "worst_case_5pct": sim["price_range"]["worst_case_5pct"],
                "best_case_95pct": sim["price_range"]["best_case_95pct"],
                "prob_loss": sim["probabilities"]["prob_any_loss"],
                "value_at_risk_95": sim["risk_metrics"]["value_at_risk_95"],
            }

        # 3) Plain-English summary
        prob_loss = normal["probabilities"]["prob_any_loss"]
        expected_ret = normal["expected_return"]
        var95 = normal["risk_metrics"]["value_at_risk_95"]

        if prob_loss < 40 and expected_ret > 2:
            outlook = "FAVORABLE"
            outlook_emoji = "🟢"
            summary = f"Simulations suggest a {expected_ret:+.1f}% expected move with {prob_loss:.0f}% chance of loss. Odds are in your favor."
        elif prob_loss > 60 or expected_ret < -3:
            outlook = "RISKY"
            outlook_emoji = "🔴"
            summary = f"Simulations suggest a {expected_ret:+.1f}% expected move with {prob_loss:.0f}% chance of loss. High risk — consider waiting."
        else:
            outlook = "NEUTRAL"
            outlook_emoji = "🟡"
            summary = f"Simulations suggest a {expected_ret:+.1f}% expected move with {prob_loss:.0f}% chance of loss. Could go either way."

        return {
            "symbol": symbol.upper(),
            "name": stats["name"],
            "current_price": round(price, 2),
            "horizon": horizon,
            "horizon_days": days,
            "annualized_volatility": round(stats["annualized_volatility"] * 100, 1),
            "outlook": outlook,
            "outlook_emoji": outlook_emoji,
            "summary": summary,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "normal_simulation": normal,
            "stress_scenarios": stress_results,
        }

    try:
        result = await loop.run_in_executor(None, _run)
        return result
    except Exception as e:
        return {"error": str(e), "symbol": symbol.upper()}
