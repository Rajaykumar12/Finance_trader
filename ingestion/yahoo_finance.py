"""Yahoo Finance background poller — options chains + fundamentals.

Runs every ~10 minutes and stores data in Redis hashes:
  - ``fundamentals:<SYMBOL>``
  - ``options_chain:<SYMBOL>``
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from datetime import datetime, timezone
from functools import partial

import yfinance as yf


def _safe_int(val, default: int = 0) -> int:
    """Convert a value to int, treating NaN / None as *default*."""
    if val is None:
        return default
    try:
        if math.isnan(val):
            return default
    except TypeError:
        pass
    return int(val)


def _safe_float(val, default: float = 0.0) -> float:
    """Convert a value to float, treating NaN / None as *default*."""
    if val is None:
        return default
    try:
        f = float(val)
        return default if math.isnan(f) else f
    except (TypeError, ValueError):
        return default

from config.settings import settings
from state.redis_client import redis_client

logger = logging.getLogger(__name__)


def _fetch_fundamentals_sync(symbol: str) -> dict:
    """Blocking call — run in executor."""
    ticker = yf.Ticker(symbol)
    info = ticker.info or {}
    return {
        "symbol": symbol,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "market_cap": info.get("marketCap"),
        "pe_ratio": info.get("trailingPE"),
        "total_debt": info.get("totalDebt"),
        "total_cash": info.get("totalCash"),
        "free_cash_flow": info.get("freeCashflow"),
        "revenue": info.get("totalRevenue"),
        "profit_margin": info.get("profitMargins"),
    }


def _fetch_options_chain_sync(symbol: str) -> dict:
    """Blocking call — run in executor."""
    ticker = yf.Ticker(symbol)
    expirations = ticker.options
    if not expirations:
        return {"symbol": symbol, "timestamp": datetime.now(timezone.utc).isoformat(), "contracts": [], "underlying_price": 0}

    # Use the nearest expiration
    nearest = expirations[0]
    chain = ticker.option_chain(nearest)
    info = ticker.info or {}
    underlying = info.get("currentPrice") or info.get("regularMarketPrice") or 0

    contracts = []
    for _, row in chain.calls.iterrows():
        contracts.append({
            "strike": _safe_float(row.get("strike")),
            "expiry": nearest,
            "contract_type": "call",
            "last_price": _safe_float(row.get("lastPrice")),
            "bid": _safe_float(row.get("bid")),
            "ask": _safe_float(row.get("ask")),
            "volume": _safe_int(row.get("volume")),
            "open_interest": _safe_int(row.get("openInterest")),
            "implied_volatility": _safe_float(row.get("impliedVolatility")),
        })
    for _, row in chain.puts.iterrows():
        contracts.append({
            "strike": _safe_float(row.get("strike")),
            "expiry": nearest,
            "contract_type": "put",
            "last_price": _safe_float(row.get("lastPrice")),
            "bid": _safe_float(row.get("bid")),
            "ask": _safe_float(row.get("ask")),
            "volume": _safe_int(row.get("volume")),
            "open_interest": _safe_int(row.get("openInterest")),
            "implied_volatility": _safe_float(row.get("impliedVolatility")),
        })

    return {
        "symbol": symbol,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "contracts": contracts,
        "underlying_price": float(underlying),
    }


async def _poll_once() -> None:
    """Fetch fundamentals + options for every tracked equity symbol."""
    loop = asyncio.get_running_loop()
    symbols = settings.equity_symbol_list

    for sym in symbols:
        try:
            fundamentals = await loop.run_in_executor(
                None, partial(_fetch_fundamentals_sync, sym)
            )
            await redis_client.hset(f"fundamentals:{sym}", fundamentals)
            logger.info("Updated fundamentals:%s", sym)
        except Exception:
            logger.exception("yfinance fundamentals error for %s", sym)

        try:
            chain = await loop.run_in_executor(
                None, partial(_fetch_options_chain_sync, sym)
            )
            # Store chain as a single JSON blob (it's large)
            await redis_client.hset(
                f"options_chain:{sym}",
                {
                    "data": json.dumps(chain, default=str),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            logger.info(
                "Updated options_chain:%s  (%d contracts)",
                sym,
                len(chain.get("contracts", [])),
            )
        except Exception:
            logger.exception("yfinance options error for %s", sym)


async def run_yahoo_finance_poller() -> None:
    """Main loop — runs every yahoo_poll_interval seconds."""
    await redis_client.connect()
    logger.info(
        "Yahoo Finance poller starting (interval=%ds, symbols=%s) …",
        settings.yahoo_poll_interval,
        settings.equity_symbol_list,
    )

    while True:
        await _poll_once()
        await asyncio.sleep(settings.yahoo_poll_interval)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_yahoo_finance_poller())
