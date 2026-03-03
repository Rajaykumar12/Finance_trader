"""Stage 3 verification — Intelligence Engine unit tests.

Tests Module A (Market Intelligence) computations directly and
Module B (health index scoring) with pre-seeded Redis data.
"""

import asyncio
import pytest
import pytest_asyncio
from datetime import datetime, timezone

from engines.market_intelligence import MarketIntelligenceEngine
from state.redis_client import RedisClient


# ── Module A: Market Intelligence ──────────────────────────────────────

def _make_tick(symbol: str, bid: float, ask: float, bid_qty: float = 1.0, ask_qty: float = 1.0) -> dict:
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


def test_market_engine_computes_state_vector():
    """Feed synthetic ticks and verify the state vector is computed."""
    engine = MarketIntelligenceEngine()

    # Feed 20 ticks with slightly varying prices
    for i in range(20):
        bid = 100.0 + i * 0.1
        ask = bid + 0.05
        engine.ingest_tick(_make_tick("TEST", bid, ask, bid_qty=5.0, ask_qty=3.0))

    state = engine.compute_state_vector("TEST")

    assert state is not None
    assert state["symbol"] == "TEST"
    assert state["tick_count"] == 20
    assert state["rolling_volatility"] > 0, "Volatility should be > 0 for varying prices"
    assert state["bid_ask_spread_mean"] > 0
    assert state["order_book_imbalance"] > 0, "More bids than asks → positive OBI"
    print(f"\n✅ State vector: vol={state['rolling_volatility']}, "
          f"spread_mean={state['bid_ask_spread_mean']}, OBI={state['order_book_imbalance']}")


def test_market_engine_needs_minimum_ticks():
    """Engine should return None with < 5 ticks."""
    engine = MarketIntelligenceEngine()
    for i in range(3):
        engine.ingest_tick(_make_tick("THIN", 100 + i, 100.1 + i))

    assert engine.compute_state_vector("THIN") is None


def test_market_engine_buffer_capped():
    """Buffer should not grow beyond BUFFER_SIZE."""
    engine = MarketIntelligenceEngine()
    for i in range(1500):
        engine.ingest_tick(_make_tick("BIG", 100, 100.01))

    buf = engine._buffers["BIG"]
    assert len(buf) <= 1000


# ── Module A: Integration (feeds Redis) ───────────────────────────────

@pytest_asyncio.fixture
async def rc():
    client = RedisClient()
    await client.connect()
    yield client
    await client.pool.delete("market_state:TEST_INT")
    await client.close()


@pytest.mark.asyncio
async def test_market_engine_writes_to_redis(rc: RedisClient):
    """Compute a state vector and write it to Redis, then read it back."""
    engine = MarketIntelligenceEngine()
    for i in range(10):
        engine.ingest_tick(_make_tick("TEST_INT", 200.0 + i * 0.5, 200.1 + i * 0.5))

    state = engine.compute_state_vector("TEST_INT")
    assert state is not None

    await rc.hset("market_state:TEST_INT", state)
    stored = await rc.hgetall("market_state:TEST_INT")

    assert stored["symbol"] == "TEST_INT"
    assert float(stored["rolling_volatility"]) > 0
    print(f"\n✅ Redis round-trip: market_state:TEST_INT stored with {stored['tick_count']} ticks")


# ── Module B: Health scoring (unit test — no FinBERT download) ─────────

def test_fundamental_score():
    """Test the fundamental scoring heuristic."""
    from engines.biological_modeling import _compute_fundamental_score

    # Perfect company (5 buckets × 20 = 100)
    perfect = {
        "free_cash_flow": 1_000_000,
        "total_cash": 5_000_000,
        "total_debt": 1_000_000,
        "pe_ratio": 20,
        "profit_margin": 0.15,
        "market_cap": 1_000_000_000,
    }
    assert _compute_fundamental_score(perfect) == 100.0

    # Terrible company
    terrible = {
        "free_cash_flow": -500_000,
        "total_cash": 100,
        "total_debt": 10_000_000,
        "pe_ratio": 500,
        "profit_margin": -0.3,
    }
    assert _compute_fundamental_score(terrible) == 0.0


def test_market_score():
    """Test the market scoring heuristic."""
    from engines.biological_modeling import _compute_market_score

    # Healthy market: tight spread, balanced book, low vol
    healthy = {"bid_ask_spread_var": 0.00001, "order_book_imbalance": 0.05, "volatility_regime": "low"}
    score = _compute_market_score(healthy)
    assert score == 100.0, f"Expected 100, got {score}"

    # Stressed market: wide spread variance, unbalanced book
    stressed = {"bid_ask_spread_var": 0.1, "order_book_imbalance": 0.8}
    score = _compute_market_score(stressed)
    assert score < 50, f"Expected < 50, got {score}"
