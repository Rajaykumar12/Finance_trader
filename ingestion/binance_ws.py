"""Binance WebSocket ingestion — BTC order book depth stream.

Connects to the unauthenticated Binance public WebSocket and publishes
normalised order-book updates to Redis ``market_ticks:BTC``.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone

import websockets

from state.redis_client import redis_client

logger = logging.getLogger(__name__)

BINANCE_WS_URL = "wss://stream.binance.com:9443/ws/btcusdt@depth@100ms"
CHANNEL = "market_ticks:BTC"


async def _parse_depth(raw: dict) -> dict:
    """Extract best bid/ask from a Binance depth update.

    The Binance depth stream sends incremental updates with arrays of
    [price, qty] for bids and asks.  We take the top-of-book entry.
    """
    bids = raw.get("b", [])
    asks = raw.get("a", [])

    best_bid = float(bids[0][0]) if bids else 0.0
    best_bid_qty = float(bids[0][1]) if bids else 0.0
    best_ask = float(asks[0][0]) if asks else 0.0
    best_ask_qty = float(asks[0][1]) if asks else 0.0

    return {
        "symbol": "BTC",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "best_bid": best_bid,
        "best_ask": best_ask,
        "bid_qty": best_bid_qty,
        "ask_qty": best_ask_qty,
        "spread": round(best_ask - best_bid, 2),
        "source": "binance",
    }


async def run_binance_ingestion() -> None:
    """Main loop — connect, parse, publish to Redis.  Reconnects on failure."""
    await redis_client.connect()
    logger.info("Binance ingestion starting …")

    while True:
        try:
            async with websockets.connect(BINANCE_WS_URL) as ws:
                logger.info("Connected to Binance depth stream")
                async for message in ws:
                    data = json.loads(message)
                    tick = await _parse_depth(data)

                    # Only publish if we got meaningful data
                    if tick["best_bid"] > 0 and tick["best_ask"] > 0:
                        await redis_client.publish(CHANNEL, tick)

        except (websockets.ConnectionClosed, ConnectionError) as exc:
            logger.warning("Binance WS disconnected (%s), reconnecting in 3s …", exc)
            await asyncio.sleep(3)
        except Exception:
            logger.exception("Unexpected error in Binance ingestion")
            await asyncio.sleep(5)


# Allow standalone testing: python -m ingestion.binance_ws
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_binance_ingestion())
