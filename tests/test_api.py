"""Stage 5 verification — FastAPI endpoint tests."""

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from api.app import app
from state.redis_client import redis_client


@pytest_asyncio.fixture(autouse=True)
async def setup_redis():
    """Ensure Redis is connected for all tests."""
    await redis_client.connect()
    yield
    await redis_client.close()


@pytest.mark.asyncio
async def test_health_endpoint():
    """GET /api/v1/health should return 200 with redis=true."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["redis"] is True
    print(f"\n✅ Health: {body}")


@pytest.mark.asyncio
async def test_market_endpoint_no_data():
    """GET /api/v1/market/UNKNOWN should return a helpful hint."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/v1/market/UNKNOWN")
    assert resp.status_code == 200
    body = resp.json()
    assert "error" in body


@pytest.mark.asyncio
async def test_market_endpoint_with_data():
    """Seed Redis with market data, then verify the endpoint returns it."""
    await redis_client.hset("market_state:TESTAPI", {
        "symbol": "TESTAPI",
        "rolling_volatility": 0.003,
        "bid_ask_spread_mean": 0.05,
        "order_book_imbalance": 0.15,
        "tick_count": 42,
    })

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/v1/market/TESTAPI")
    assert resp.status_code == 200
    body = resp.json()
    assert body["symbol"] == "TESTAPI"
    assert float(body["rolling_volatility"]) == pytest.approx(0.003)

    # Cleanup
    await redis_client.pool.delete("market_state:TESTAPI")
    print(f"\n✅ Market endpoint: {body}")


@pytest.mark.asyncio
async def test_gamma_endpoint_with_data():
    """Seed Redis with GEX data, then verify the endpoint returns it."""
    import json
    zones = [{"strike": 150, "gex": 5000, "type": "call"}]
    await redis_client.hset("gamma_exposure:GEXTEST", {
        "symbol": "GEXTEST",
        "total_gex": 12345.67,
        "hedging_pressure_zones": json.dumps(zones),
        "flip_point": 152.5,
        "contracts_analyzed": 100,
    })

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/v1/derivatives/gamma/GEXTEST")
    assert resp.status_code == 200
    body = resp.json()
    assert body["symbol"] == "GEXTEST"
    assert isinstance(body["hedging_pressure_zones"], list)
    assert body["hedging_pressure_zones"][0]["strike"] == 150

    await redis_client.pool.delete("gamma_exposure:GEXTEST")
    print(f"\n✅ Gamma endpoint: total_gex={body['total_gex']}")


@pytest.mark.asyncio
async def test_health_index_endpoint():
    """Seed Redis with AHI data, verify endpoint."""
    await redis_client.hset("health_index:AHITEST", {
        "symbol": "AHITEST",
        "health_index": 72.5,
        "sentiment_score": 0.3,
        "fundamental_score": 75.0,
        "market_score": 80.0,
        "metabolic_stress_alert": False,
    })

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/v1/health-index/AHITEST")
    assert resp.status_code == 200
    body = resp.json()
    assert body["symbol"] == "AHITEST"
    assert float(body["health_index"]) == pytest.approx(72.5)

    await redis_client.pool.delete("health_index:AHITEST")
    print(f"\n✅ Health Index: AHI={body['health_index']}")
