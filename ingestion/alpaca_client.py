"""Alpaca API ingestion — equity quotes via paper-trading sandbox.

Streams real-time equity quotes for configured symbols (e.g. AAPL, NVDA)
and publishes normalised ticks to Redis ``market_ticks:<SYMBOL>``.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from config.settings import settings
from state.redis_client import redis_client

logger = logging.getLogger(__name__)


async def _poll_alpaca_quotes() -> None:
    """Poll Alpaca REST API for latest quotes and publish to Redis.

    We use REST polling instead of Alpaca's WebSocket to keep the
    dependency surface small and because the sandbox WS can be flaky.
    """
    import httpx

    base = settings.alpaca_base_url
    headers = {
        "APCA-API-KEY-ID": settings.alpaca_api_key,
        "APCA-API-SECRET-KEY": settings.alpaca_api_secret,
        "Accept": "application/json",
    }

    symbols = settings.equity_symbol_list
    if not symbols:
        logger.warning("No equity symbols configured — Alpaca ingestion idle.")
        return

    async with httpx.AsyncClient(timeout=10) as client:
        while True:
            for sym in symbols:
                try:
                    # Alpaca Market Data v2 — latest quote
                    url = f"https://data.alpaca.markets/v2/stocks/{sym}/quotes/latest"
                    resp = await client.get(url, headers=headers)
                    resp.raise_for_status()
                    data = resp.json().get("quote", {})

                    tick = {
                        "symbol": sym,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "best_bid": float(data.get("bp", 0)),
                        "best_ask": float(data.get("ap", 0)),
                        "bid_qty": float(data.get("bs", 0)),
                        "ask_qty": float(data.get("as", 0)),
                        "spread": round(
                            float(data.get("ap", 0)) - float(data.get("bp", 0)), 4
                        ),
                        "source": "alpaca",
                    }
                    if tick["best_bid"] > 0:
                        await redis_client.publish(f"market_ticks:{sym}", tick)
                        logger.debug("Published Alpaca tick for %s", sym)

                except httpx.HTTPStatusError as exc:
                    logger.warning("Alpaca HTTP error for %s: %s", sym, exc.response.status_code)
                except Exception:
                    logger.exception("Alpaca polling error for %s", sym)

            await asyncio.sleep(2)  # Poll every 2 seconds


async def run_alpaca_ingestion() -> None:
    """Entry point — requires valid API keys in .env."""
    if not settings.alpaca_api_key or settings.alpaca_api_key.startswith("your_"):
        logger.warning("Alpaca API key not configured — skipping equity ingestion.")
        return

    await redis_client.connect()
    logger.info("Alpaca ingestion starting for %s …", settings.equity_symbol_list)
    await _poll_alpaca_quotes()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_alpaca_ingestion())
