"""Stage 2 integration test — verifies Binance ingestion pushes data to Redis.

Starts the Binance WebSocket ingestion for a few seconds, then checks that
at least one tick arrived on the ``market_ticks:BTC`` Redis channel.
"""

import asyncio
import logging
import pytest

from state.redis_client import RedisClient

logging.basicConfig(level=logging.INFO)


@pytest.mark.asyncio
async def test_binance_to_redis_roundtrip():
    """Start Binance ingestion, subscribe to its channel, verify ≥1 tick."""
    from ingestion.binance_ws import run_binance_ingestion

    # Subscriber — will collect ticks
    sub = RedisClient()
    await sub.connect()
    received: list[dict] = []

    async def collect():
        async for msg in sub.subscribe("market_ticks:BTC"):
            received.append(msg)
            if len(received) >= 3:
                break

    # Start ingestion + collector concurrently, with a timeout
    ingest_task = asyncio.create_task(run_binance_ingestion())
    collect_task = asyncio.create_task(collect())

    try:
        await asyncio.wait_for(collect_task, timeout=15.0)
    except asyncio.TimeoutError:
        pass  # We'll check what we got
    finally:
        ingest_task.cancel()
        try:
            await ingest_task
        except asyncio.CancelledError:
            pass
        await sub.close()

    # Assertions
    assert len(received) >= 1, f"Expected ≥1 tick, got {len(received)}"
    tick = received[0]
    assert tick["symbol"] == "BTC"
    assert tick["source"] == "binance"
    assert tick["best_bid"] > 0
    assert tick["best_ask"] > 0
    assert tick["spread"] >= 0
    assert "timestamp" in tick
    print(f"\n✅ Received {len(received)} ticks. Sample: {tick}")


@pytest.mark.asyncio
async def test_yahoo_finance_fundamentals():
    """Fetch fundamentals for AAPL and verify they land in Redis."""
    from ingestion.yahoo_finance import _poll_once
    from state.redis_client import redis_client

    await redis_client.connect()
    await _poll_once()

    data = await redis_client.hgetall("fundamentals:AAPL")
    assert data, "No fundamentals found in Redis for AAPL"
    assert data.get("symbol") == "AAPL"
    assert data.get("market_cap") is not None
    print(f"\n✅ Fundamentals for AAPL: market_cap={data.get('market_cap')}")

    # Check options chain was stored too
    chain_data = await redis_client.hgetall("options_chain:AAPL")
    assert chain_data, "No options chain found in Redis for AAPL"
    assert "data" in chain_data
    print(f"✅ Options chain for AAPL stored ({len(chain_data.get('data', ''))} chars)")

    # Cleanup
    await redis_client.pool.delete("fundamentals:AAPL", "fundamentals:NVDA",
                                    "options_chain:AAPL", "options_chain:NVDA")
    await redis_client.close()
