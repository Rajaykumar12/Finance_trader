"""Stage 1 verification — Redis pub/sub + hash round-trip tests."""

import asyncio
import pytest
import pytest_asyncio

from state.redis_client import RedisClient


@pytest_asyncio.fixture
async def rc():
    """Provide a fresh RedisClient connected to the local Redis server."""
    client = RedisClient()
    await client.connect()
    yield client
    # Cleanup test keys
    await client.pool.delete(
        "test:hash:roundtrip",
        "test:hash:types",
    )
    await client.close()


# ── Hash round-trip ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_hset_hgetall_roundtrip(rc: RedisClient):
    """Write a hash, read it back, assert equality."""
    data = {
        "volatility": 0.035,
        "spread_mean": 12.5,
        "symbol": "BTC",
        "tick_count": 42,
    }
    await rc.hset("test:hash:roundtrip", data)
    result = await rc.hgetall("test:hash:roundtrip")

    assert result["symbol"] == "BTC"
    assert float(result["volatility"]) == pytest.approx(0.035)
    assert float(result["spread_mean"]) == pytest.approx(12.5)
    assert int(result["tick_count"]) == 42


@pytest.mark.asyncio
async def test_hset_complex_values(rc: RedisClient):
    """Non-primitive values (lists, dicts) should survive JSON round-trip."""
    data = {
        "zones": [{"strike": 150, "gex": 1200.5}, {"strike": 155, "gex": -800.3}],
        "metadata": {"source": "test", "version": 1},
    }
    await rc.hset("test:hash:types", data)
    result = await rc.hgetall("test:hash:types")

    assert isinstance(result["zones"], list)
    assert len(result["zones"]) == 2
    assert result["zones"][0]["strike"] == 150
    assert isinstance(result["metadata"], dict)
    assert result["metadata"]["source"] == "test"


@pytest.mark.asyncio
async def test_hget_single_field(rc: RedisClient):
    """hget should read a single field from a hash."""
    await rc.hset("test:hash:roundtrip", {"alpha": 0.99, "beta": 1.5})
    val = await rc.hget("test:hash:roundtrip", "alpha")
    assert float(val) == pytest.approx(0.99)


@pytest.mark.asyncio
async def test_hget_missing_field(rc: RedisClient):
    """hget returns None for a non-existent field."""
    val = await rc.hget("test:hash:roundtrip", "does_not_exist")
    assert val is None


# ── Pub/Sub round-trip ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_pubsub_roundtrip(rc: RedisClient):
    """Publish a message and verify it is received by a subscriber."""
    channel = "test:pubsub:roundtrip"
    received: list[dict] = []

    async def listener():
        async for msg in rc.subscribe(channel):
            received.append(msg)
            break  # We only need one message

    # Start listener and give it a moment to subscribe
    task = asyncio.create_task(listener())
    await asyncio.sleep(0.3)

    # Publish with a *second* client (pub and sub can't share the same
    # connection once it's in subscribe mode on some Redis versions)
    pub_client = RedisClient()
    await pub_client.connect()
    sent = {"symbol": "BTC", "price": 62000.5, "qty": 1.2}
    await pub_client.publish(channel, sent)
    await pub_client.close()

    # Wait for listener to finish
    await asyncio.wait_for(task, timeout=3.0)

    assert len(received) == 1
    assert received[0]["symbol"] == "BTC"
    assert received[0]["price"] == 62000.5
