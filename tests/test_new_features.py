"""Tests for all new features — vol regime, liquidity shifts, correlations,
ecosystem map, vol surface, convexity risk, macro scoring, and API endpoints.
"""

import asyncio
import json
import pytest
import pytest_asyncio
import numpy as np
from datetime import datetime, timezone

from httpx import AsyncClient, ASGITransport
from api.app import app
from state.redis_client import RedisClient


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def rc():
    """Fresh Redis connection with cleanup."""
    client = RedisClient()
    await client.connect()
    yield client
    # Clean up test keys
    for pattern in [
        "market_state:TEST*", "health_index:TEST*", "cross_correlations",
        "ecosystem_map", "macro_indicators", "vol_surface:TEST*",
        "convexity_risk:TEST*", "vol_regime_indicator:TEST*",
        "fundamentals:TEST*",
    ]:
        cursor = 0
        while True:
            cursor, keys = await client.pool.scan(cursor, match=pattern.replace("*", "*"))
            if keys:
                await client.pool.delete(*keys)
            if cursor == 0:
                break
    await client.close()


def _make_tick(symbol, bid, ask, bid_qty=1.0, ask_qty=1.0):
    return {
        "symbol": symbol,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "best_bid": bid,
        "best_ask": ask,
        "bid_qty": bid_qty,
        "ask_qty": ask_qty,
        "spread": round(ask - bid, 4),
        "source": "test",
    }


# ══════════════════════════════════════════════════════════════════════
# Feature 1: Volatility Regime + Liquidity Shift (Module A)
# ══════════════════════════════════════════════════════════════════════

class TestVolatilityRegime:
    def test_low_regime(self):
        from engines.market_intelligence import MarketIntelligenceEngine
        engine = MarketIntelligenceEngine()
        # Constant price → near-zero volatility → "low" regime
        for i in range(20):
            engine.ingest_tick(_make_tick("REG", 100.0, 100.01))
        state = engine.compute_state_vector("REG")
        assert state["volatility_regime"] == "low"
        print(f"\n✅ Regime: {state['volatility_regime']} (vol={state['rolling_volatility']})")

    def test_high_regime(self):
        from engines.market_intelligence import MarketIntelligenceEngine
        engine = MarketIntelligenceEngine()
        # Large price swings → high volatility
        for i in range(20):
            bid = 100 + (10 * ((-1) ** i))  # Oscillates between 90 and 110
            engine.ingest_tick(_make_tick("VOLATILE", bid, bid + 0.05))
        state = engine.compute_state_vector("VOLATILE")
        assert state["volatility_regime"] in ("high", "crisis")
        print(f"\n✅ Regime: {state['volatility_regime']} (vol={state['rolling_volatility']})")

    def test_liquidity_shift_detection(self):
        from engines.market_intelligence import MarketIntelligenceEngine
        engine = MarketIntelligenceEngine()

        # First window: tight spreads
        for i in range(10):
            engine.ingest_tick(_make_tick("LIQ", 100, 100.01))
        state1 = engine.compute_state_vector("LIQ")
        assert state1["liquidity_status"] == "stable"

        # Second window: much wider, varying spreads → spread var jumps
        for i in range(10):
            spread = 0.01 + i * 0.1  # Spreads from 0.01 to 0.91
            engine.ingest_tick(_make_tick("LIQ", 100, 100 + spread))
        state2 = engine.compute_state_vector("LIQ")
        assert state2["shift_magnitude"] > 1.0
        print(f"\n✅ Liquidity: status={state2['liquidity_status']}, magnitude={state2['shift_magnitude']}")

    def test_state_vector_has_new_fields(self):
        from engines.market_intelligence import MarketIntelligenceEngine
        engine = MarketIntelligenceEngine()
        for i in range(10):
            engine.ingest_tick(_make_tick("FIELDS", 100 + i, 100.1 + i))
        state = engine.compute_state_vector("FIELDS")
        assert "volatility_regime" in state
        assert "liquidity_shift" in state
        assert "shift_magnitude" in state
        assert "liquidity_status" in state


# ══════════════════════════════════════════════════════════════════════
# Feature 2: Cross-Market Correlations (Module A)
# ══════════════════════════════════════════════════════════════════════

class TestCrossCorrelations:
    def test_correlated_pairs(self):
        from engines.market_intelligence import MarketIntelligenceEngine
        engine = MarketIntelligenceEngine()

        # Two symbols moving together → positive correlation
        for i in range(50):
            price = 100 + i * 0.5
            engine.ingest_tick(_make_tick("SYM_A", price, price + 0.01))
            engine.ingest_tick(_make_tick("SYM_B", price * 2, price * 2 + 0.01))

        corr = engine.compute_cross_correlations()
        assert corr is not None
        assert "SYM_A_SYM_B" in corr["pairs"]
        assert corr["pairs"]["SYM_A_SYM_B"] > 0.8, f"Expected high positive correlation, got {corr['pairs']['SYM_A_SYM_B']}"
        print(f"\n✅ Cross-correlation: {corr['pairs']}")

    def test_insufficient_data(self):
        from engines.market_intelligence import MarketIntelligenceEngine
        engine = MarketIntelligenceEngine()
        # Only 5 ticks → not enough for correlation
        for i in range(5):
            engine.ingest_tick(_make_tick("ONLY_ONE", 100, 100.01))
        assert engine.compute_cross_correlations() is None


# ══════════════════════════════════════════════════════════════════════
# Feature 3: Ecosystem Competitive Map (Module B)
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_ecosystem_map(rc):
    from engines.biological_modeling import compute_ecosystem_map

    # Seed health data for two symbols
    await rc.hset("health_index:TEST_A", {
        "health_index": 80, "sentiment_score": 0.5,
        "fundamental_score": 90, "market_score": 75,
    })
    await rc.hset("health_index:TEST_B", {
        "health_index": 40, "sentiment_score": -0.2,
        "fundamental_score": 30, "market_score": 45,
    })
    await rc.hset("market_state:TEST_A", {"volatility_regime": "normal", "liquidity_status": "stable"})
    await rc.hset("market_state:TEST_B", {"volatility_regime": "crisis", "liquidity_status": "vanishing"})

    eco = await compute_ecosystem_map(["TEST_A", "TEST_B"])
    assert eco["total_assets"] == 2
    assert eco["rankings"][0]["symbol"] == "TEST_A"  # Higher health → rank 1
    assert eco["rankings"][0]["rank"] == 1
    assert eco["rankings"][1]["symbol"] == "TEST_B"
    assert "stable_volatility" in eco["rankings"][0]["strengths"]
    assert "crisis_volatility" in eco["rankings"][1]["weaknesses"]
    print(f"\n✅ Ecosystem map: avg_health={eco['sector_average_health']}")
    for e in eco["rankings"]:
        print(f"   #{e['rank']} {e['symbol']}: AHI={e['health_index']}, strengths={e['strengths']}, weaknesses={e['weaknesses']}")

    # Cleanup
    for key in ["health_index:TEST_A", "health_index:TEST_B", "market_state:TEST_A", "market_state:TEST_B"]:
        await rc.pool.delete(key)


# ══════════════════════════════════════════════════════════════════════
# Feature 4: Volatility Surface (Module C)
# ══════════════════════════════════════════════════════════════════════

class TestVolSurface:
    def test_vol_surface_skew(self):
        from engines.derivatives_intelligence import _compute_vol_surface

        # Chain with higher OTM put IV → positive skew
        chain = {
            "symbol": "SKEW_TEST",
            "underlying_price": 100,
            "contracts": [
                {"strike": 80, "contract_type": "put", "implied_volatility": 0.40, "open_interest": 100},
                {"strike": 85, "contract_type": "put", "implied_volatility": 0.35, "open_interest": 200},
                {"strike": 90, "contract_type": "put", "implied_volatility": 0.28, "open_interest": 300},
                {"strike": 95, "contract_type": "call", "implied_volatility": 0.22, "open_interest": 500},
                {"strike": 100, "contract_type": "call", "implied_volatility": 0.20, "open_interest": 1000},
                {"strike": 105, "contract_type": "call", "implied_volatility": 0.22, "open_interest": 500},
                {"strike": 110, "contract_type": "call", "implied_volatility": 0.25, "open_interest": 200},
                {"strike": 115, "contract_type": "call", "implied_volatility": 0.28, "open_interest": 100},
            ],
        }
        surface = _compute_vol_surface(chain)
        assert surface is not None
        assert surface["skew"] > 0, "OTM puts have higher IV → positive skew"
        assert surface["atm_iv"] > 0
        print(f"\n✅ Vol Surface: ATM_IV={surface['atm_iv']}, skew={surface['skew']}, status={surface['skew_status']}")

    def test_vol_surface_empty(self):
        from engines.derivatives_intelligence import _compute_vol_surface
        assert _compute_vol_surface({"symbol": "X", "underlying_price": 0, "contracts": []}) is None


# ══════════════════════════════════════════════════════════════════════
# Feature 5: Convexity Risk (Module C)
# ══════════════════════════════════════════════════════════════════════

class TestConvexityRisk:
    def test_convexity_risk_scoring(self):
        from engines.derivatives_intelligence import _compute_convexity_risk

        chain = {
            "symbol": "CONV_TEST",
            "underlying_price": 100,
            "contracts": [
                {"strike": 95, "contract_type": "call", "implied_volatility": 0.25, "open_interest": 5000},
                {"strike": 100, "contract_type": "call", "implied_volatility": 0.20, "open_interest": 10000},
                {"strike": 105, "contract_type": "put", "implied_volatility": 0.25, "open_interest": 8000},
                {"strike": 110, "contract_type": "put", "implied_volatility": 0.30, "open_interest": 3000},
            ],
        }
        risk = _compute_convexity_risk(chain, None)
        assert risk is not None
        assert "convexity_risk_score" in risk
        assert risk["risk_level"] in ("low", "moderate", "elevated", "critical")
        assert risk["total_dollar_gamma"] != 0
        print(f"\n✅ Convexity Risk: score={risk['convexity_risk_score']}, level={risk['risk_level']}")

    def test_vol_regime_indicator(self):
        from engines.derivatives_intelligence import _compute_vol_regime_indicator

        # Elevated IV + moderate convexity
        indicator = _compute_vol_regime_indicator(
            {"atm_iv": 0.35, "skew": 0.08},
            {"convexity_risk_score": 45},
        )
        assert indicator["iv_regime"] == "elevated"
        assert indicator["overall_regime"] in ("normal", "stressed", "crisis")
        print(f"\n✅ Vol Regime: iv={indicator['iv_regime']}, overall={indicator['overall_regime']}, score={indicator['regime_score']}")


# ══════════════════════════════════════════════════════════════════════
# Feature 6: Macro Scoring (Module B)
# ══════════════════════════════════════════════════════════════════════

class TestMacroScoring:
    def test_favorable_macro(self):
        from engines.biological_modeling import _compute_macro_score
        score = _compute_macro_score({
            "fed_funds_rate": 1.5,
            "unemployment_rate": 3.5,
            "cpi_yoy": 2.0,
            "vix": 12,
        })
        assert score > 80, f"Favorable macro should score >80, got {score}"
        print(f"\n✅ Favorable macro score: {score}")

    def test_unfavorable_macro(self):
        from engines.biological_modeling import _compute_macro_score
        score = _compute_macro_score({
            "fed_funds_rate": 6.0,
            "unemployment_rate": 8.0,
            "cpi_yoy": 7.0,
            "vix": 35,
        })
        assert score < 30, f"Unfavorable macro should score <30, got {score}"
        print(f"\n✅ Unfavorable macro score: {score}")

    def test_neutral_when_empty(self):
        from engines.biological_modeling import _compute_macro_score
        assert _compute_macro_score({}) == 50.0


# ══════════════════════════════════════════════════════════════════════
# Feature 7: New API Endpoints
# ══════════════════════════════════════════════════════════════════════

@pytest_asyncio.fixture(autouse=True)
async def setup_redis_api():
    from state.redis_client import redis_client
    await redis_client.connect()
    yield
    await redis_client.close()


@pytest.mark.asyncio
async def test_api_correlations_endpoint():
    """GET /api/v1/market/correlations/all"""
    from state.redis_client import redis_client
    await redis_client.hset("cross_correlations", {
        "pairs": json.dumps({"BTC_AAPL": 0.42}),
        "symbols": json.dumps(["BTC", "AAPL"]),
        "window_size": 100,
    })
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/v1/market/correlations/all")
    assert resp.status_code == 200
    body = resp.json()
    assert "pairs" in body
    await redis_client.pool.delete("cross_correlations")
    print(f"\n✅ Correlations endpoint: {body}")


@pytest.mark.asyncio
async def test_api_ecosystem_endpoint():
    """GET /api/v1/ecosystem/map"""
    from state.redis_client import redis_client
    eco = {"sector_average_health": 72.5, "total_assets": 2, "rankings": []}
    await redis_client.hset("ecosystem_map", {
        "data": json.dumps(eco),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    })
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/v1/ecosystem/map")
    assert resp.status_code == 200
    body = resp.json()
    assert body["sector_average_health"] == 72.5
    await redis_client.pool.delete("ecosystem_map")


@pytest.mark.asyncio
async def test_api_macro_endpoint():
    """GET /api/v1/macro/indicators"""
    from state.redis_client import redis_client
    await redis_client.hset("macro_indicators", {"vix": 18.5, "treasury_10y": 4.3})
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/v1/macro/indicators")
    assert resp.status_code == 200
    body = resp.json()
    assert float(body["vix"]) == 18.5
    await redis_client.pool.delete("macro_indicators")
    print(f"\n✅ Macro endpoint: {body}")


@pytest.mark.asyncio
async def test_api_vol_surface_endpoint():
    """GET /api/v1/derivatives/vol-surface/TEST"""
    from state.redis_client import redis_client
    await redis_client.hset("vol_surface:TEST", {
        "symbol": "TEST",
        "atm_iv": 0.22,
        "skew": 0.05,
        "smile": 0.02,
        "skew_status": "moderate_skew",
        "smile_status": "mild_smile",
        "surface_points": json.dumps([{"moneyness": 1.0, "iv": 0.22}]),
    })
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/v1/derivatives/vol-surface/TEST")
    assert resp.status_code == 200
    body = resp.json()
    assert body["skew_status"] == "moderate_skew"
    assert isinstance(body["surface_points"], list)
    await redis_client.pool.delete("vol_surface:TEST")


@pytest.mark.asyncio
async def test_api_vol_regime_endpoint():
    """GET /api/v1/derivatives/vol-regime/TEST"""
    from state.redis_client import redis_client
    await redis_client.hset("vol_regime_indicator:TEST", {
        "symbol": "TEST",
        "iv_regime": "normal",
        "overall_regime": "calm",
        "regime_score": 25.0,
    })
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/v1/derivatives/vol-regime/TEST")
    assert resp.status_code == 200
    body = resp.json()
    assert body["overall_regime"] == "calm"
    await redis_client.pool.delete("vol_regime_indicator:TEST")
