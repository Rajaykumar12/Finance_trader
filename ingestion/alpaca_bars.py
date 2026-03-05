"""Alpaca OHLCV Bar Ingestion — 5-minute candle fetcher.

Fetches intraday OHLCV bars from Alpaca's Market Data v2 API.
Used by the intraday signal engine for VWAP, EMA, and ORB computation.

Alpaca bar structure returned:
    {
        "t": "2024-01-15T14:30:00Z",  # bar start timestamp (UTC)
        "o": 182.50,                   # open
        "h": 183.10,                   # high
        "l": 182.20,                   # low
        "c": 182.95,                   # close
        "v": 1234567.0,                # volume (shares)
        "vw": 182.73,                  # VWAP for this bar
        "n": 4521,                     # number of trades
    }
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, date
from typing import Optional

import httpx

from config.settings import settings

logger = logging.getLogger(__name__)

DATA_API = "https://data.alpaca.markets"


def _headers() -> dict:
    return {
        "APCA-API-KEY-ID":     settings.alpaca_api_key,
        "APCA-API-SECRET-KEY": settings.alpaca_api_secret,
        "Accept":              "application/json",
    }


async def fetch_bars(
    symbol: str,
    timeframe: str = "5Min",
    limit: int = 80,
    start: Optional[str] = None,
) -> list[dict]:
    """Fetch OHLCV bars for a symbol from Alpaca.

    Args:
        symbol:    Ticker, e.g. "AAPL"
        timeframe: Bar width — "1Min", "5Min", "15Min", "1Hour", "1Day"
        limit:     Max bars to return (80 × 5-min = ~6.7 hours of history)
        start:     ISO datetime string for start of range (optional).
                   If omitted, returns the most recent `limit` bars.

    Returns:
        List of bar dicts sorted oldest-first, or [] on error.
    """
    if not settings.alpaca_api_key or settings.alpaca_api_key.startswith("your_"):
        logger.warning("Alpaca API key not configured — cannot fetch bars for %s", symbol)
        return []

    params: dict = {
        "timeframe": timeframe,
        "limit":     limit,
        "sort":      "asc",
        "feed":      "iex",   # IEX is free; use "sip" if you have a paid data subscription
    }
    if start:
        params["start"] = start

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"{DATA_API}/v2/stocks/{symbol.upper()}/bars",
                headers=_headers(),
                params=params,
            )
            resp.raise_for_status()
            data = resp.json()
            bars = data.get("bars") or []
            logger.debug("Fetched %d bars for %s (%s)", len(bars), symbol, timeframe)
            return bars

    except httpx.HTTPStatusError as e:
        logger.warning("Alpaca bars HTTP error for %s: %s", symbol, e.response.status_code)
        return []
    except Exception:
        logger.exception("Failed to fetch bars for %s", symbol)
        return []


async def fetch_today_bars(symbol: str, timeframe: str = "5Min") -> list[dict]:
    """Fetch all bars since today's market open (9:30 ET).

    Returns only today's intraday bars — useful for VWAP and ORB which
    must reset at the start of each trading day.
    """
    from engines.market_hours import market_date
    from zoneinfo import ZoneInfo

    ET = ZoneInfo("America/New_York")
    today = market_date()
    market_open_et = datetime(today.year, today.month, today.day, 9, 30, 0, tzinfo=ET)
    start_iso = market_open_et.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    return await fetch_bars(symbol, timeframe=timeframe, limit=100, start=start_iso)
