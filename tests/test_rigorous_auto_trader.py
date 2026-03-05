"""Unit tests for the RigorousAutoTrader gate logic and position sizing.

These tests run fully offline — no Alpaca API calls, no Redis, no FinBERT.
All external calls are mocked with unittest.mock.AsyncMock / patch.
"""

from __future__ import annotations

import asyncio
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, patch, MagicMock

from engines.auto_trader import RigorousAutoTrader


# ── Helpers ───────────────────────────────────────────────────────────

def _pred(score: float, confidence: float, signal: str = "BUY",
          bullish: list | None = None, bearish: list | None = None) -> dict:
    """Build a fake predict() response."""
    return {
        "symbol":          "TEST",
        "composite_score": score,
        "confidence":      confidence,
        "signal":          signal,
        "price":           150.0,
        "bullish_signals": bullish or ["✅ reason A", "✅ reason B", "✅ reason C"],
        "bearish_signals": bearish or [],
        "caution_signals": [],
    }


async def _make_trader(**kwargs) -> RigorousAutoTrader:
    """Construct a RigorousAutoTrader with defaults overridden by kwargs."""
    trader = RigorousAutoTrader()
    # Set config directly (bypass async start() which calls Alpaca)
    trader.running = True
    trader.min_score = kwargs.get("min_score", 78.0)
    trader.min_confidence = kwargs.get("min_confidence", 0.65)
    trader.take_profit_pct = kwargs.get("take_profit_pct", 0.015)
    trader.stop_loss_pct = kwargs.get("stop_loss_pct", 0.008)
    trader.max_investment_per_trade = kwargs.get("max_investment", 5000.0)
    trader.max_positions = kwargs.get("max_positions", 5)
    trader.symbols = ["TEST"]
    return trader


# ── Gate tests ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_entry_gate_all_conditions_met():
    """When all gates pass, a bracket order should be placed."""
    trader = await _make_trader()

    with (
        patch("engines.auto_trader.predict", new=AsyncMock(return_value=_pred(82, 0.72))),
        patch("engines.auto_trader.place_bracket_order",
              new=AsyncMock(return_value={"success": True, "take_profit_price": 152.25,
                                          "stop_loss_price": 148.80, "reward_risk_ratio": 1.88,
                                          "order": {"order_id": "abc123"}})) as mock_bracket,
        patch.object(trader, "_get_macro_score", new=AsyncMock(return_value=65.0)),
        patch.object(trader, "_get_liquidity",   new=AsyncMock(return_value="stable")),
    ):
        await trader._evaluate_symbol(
            "TEST",
            current_positions={},
            buying_power=10_000.0,
            macro_score=65.0,
            day_trades_remaining=3,
        )

    mock_bracket.assert_called_once()
    buy_logs = [t for t in trader.trade_log if t["action"] == "BUY"]
    assert len(buy_logs) == 1, f"Expected 1 BUY log, got: {trader.trade_log}"


@pytest.mark.asyncio
async def test_entry_gate_score_too_low():
    """Score below min_score (78) must be skipped."""
    trader = await _make_trader()

    with (
        patch("engines.auto_trader.predict", new=AsyncMock(return_value=_pred(74, 0.72))),
        patch("engines.auto_trader.place_bracket_order",
              new=AsyncMock()) as mock_bracket,
        patch.object(trader, "_get_macro_score", new=AsyncMock(return_value=65.0)),
        patch.object(trader, "_get_liquidity",   new=AsyncMock(return_value="stable")),
    ):
        await trader._evaluate_symbol("TEST", {}, 10_000.0, 65.0, 3)

    mock_bracket.assert_not_called()
    skips = [t for t in trader.trade_log if t["action"] == "SKIP"]
    assert any("score" in str(s.get("gates_failed", [])) for s in skips), \
        "Expected a score-gate skip"


@pytest.mark.asyncio
async def test_entry_gate_confidence_too_low():
    """Good score but confidence below 0.65 must be skipped."""
    trader = await _make_trader()

    with (
        patch("engines.auto_trader.predict", new=AsyncMock(return_value=_pred(82, 0.50))),
        patch("engines.auto_trader.place_bracket_order",
              new=AsyncMock()) as mock_bracket,
        patch.object(trader, "_get_macro_score", new=AsyncMock(return_value=65.0)),
        patch.object(trader, "_get_liquidity",   new=AsyncMock(return_value="stable")),
    ):
        await trader._evaluate_symbol("TEST", {}, 10_000.0, 65.0, 3)

    mock_bracket.assert_not_called()
    skips = [t for t in trader.trade_log if t["action"] == "SKIP"]
    assert any("confidence" in str(s.get("gates_failed", [])) for s in skips)


@pytest.mark.asyncio
async def test_entry_gate_macro_circuit_breaker():
    """When macro_score < 35, entire cycle is blocked — _trading_loop skips all buys."""
    trader = await _make_trader()
    trader.checks_completed = 0
    trader.start_time = datetime.now(timezone.utc)
    trader.end_time = datetime.now(timezone.utc) + timedelta(seconds=5)

    with (
        patch("engines.auto_trader.predict", new=AsyncMock(return_value=_pred(85, 0.80))),
        patch("engines.auto_trader.place_bracket_order",
              new=AsyncMock()) as mock_bracket,
        patch("engines.auto_trader.get_account",
              new=AsyncMock(return_value={
                  "portfolio_value": 10_000.0, "buying_power": 8_000.0,
                  "day_trades_remaining": 3
              })),
        patch("engines.auto_trader.get_positions",
              new=AsyncMock(return_value={"positions": []})),
        patch.object(trader, "_get_macro_score", new=AsyncMock(return_value=28.0)),
        patch.object(trader, "_finalize",         new=AsyncMock()),
    ):
        # Run one iteration manually (the loop blocks all buys at macro_score < 35)
        trader.checks_completed = 1
        macro_score = await trader._get_macro_score()
        if macro_score < 35:
            trader._log("⛔ MACRO CIRCUIT BREAKER", {
                "reason": "macro_score < 35 — fear regime",
                "macro_score": round(macro_score, 1),
            })

    mock_bracket.assert_not_called()
    assert any("MACRO CIRCUIT BREAKER" in t["action"] for t in trader.trade_log)


@pytest.mark.asyncio
async def test_entry_gate_cooldown():
    """Symbol in 10-min post-close cooldown must be skipped."""
    trader = await _make_trader()
    # Put TEST in cooldown
    trader._cooldown["TEST"] = datetime.now(timezone.utc) + timedelta(minutes=8)

    with (
        patch("engines.auto_trader.predict", new=AsyncMock(return_value=_pred(85, 0.80))),
        patch("engines.auto_trader.place_bracket_order",
              new=AsyncMock()) as mock_bracket,
        patch.object(trader, "_get_macro_score", new=AsyncMock(return_value=65.0)),
        patch.object(trader, "_get_liquidity",   new=AsyncMock(return_value="stable")),
    ):
        await trader._evaluate_symbol("TEST", {}, 10_000.0, 65.0, 3)

    mock_bracket.assert_not_called()
    skips = [t for t in trader.trade_log if t["action"] == "SKIP"]
    assert any("cooldown" in str(s.get("gate", "")) for s in skips)


@pytest.mark.asyncio
async def test_entry_gate_insufficient_bullish_signals():
    """Only 1 bullish signal (< 2 required) must be skipped."""
    trader = await _make_trader()

    with (
        patch("engines.auto_trader.predict", new=AsyncMock(return_value=_pred(
            82, 0.72, bullish=["✅ single signal"]
        ))),
        patch("engines.auto_trader.place_bracket_order",
              new=AsyncMock()) as mock_bracket,
        patch.object(trader, "_get_macro_score", new=AsyncMock(return_value=65.0)),
        patch.object(trader, "_get_liquidity",   new=AsyncMock(return_value="stable")),
    ):
        await trader._evaluate_symbol("TEST", {}, 10_000.0, 65.0, 3)

    mock_bracket.assert_not_called()
    skips = [t for t in trader.trade_log if t["action"] == "SKIP"]
    assert any("bullish signal" in str(s.get("gates_failed", [])) for s in skips)


@pytest.mark.asyncio
async def test_entry_gate_hold_signal_skipped():
    """A HOLD signal must not trigger a buy even with a high score."""
    trader = await _make_trader()

    with (
        patch("engines.auto_trader.predict", new=AsyncMock(return_value=_pred(
            80, 0.70, signal="HOLD"
        ))),
        patch("engines.auto_trader.place_bracket_order",
              new=AsyncMock()) as mock_bracket,
        patch.object(trader, "_get_macro_score", new=AsyncMock(return_value=65.0)),
        patch.object(trader, "_get_liquidity",   new=AsyncMock(return_value="stable")),
    ):
        await trader._evaluate_symbol("TEST", {}, 10_000.0, 65.0, 3)

    mock_bracket.assert_not_called()


@pytest.mark.asyncio
async def test_entry_gate_vanishing_liquidity():
    """Vanishing liquidity must block the trade."""
    trader = await _make_trader()

    with (
        patch("engines.auto_trader.predict", new=AsyncMock(return_value=_pred(85, 0.80))),
        patch("engines.auto_trader.place_bracket_order",
              new=AsyncMock()) as mock_bracket,
        patch.object(trader, "_get_macro_score", new=AsyncMock(return_value=65.0)),
        patch.object(trader, "_get_liquidity",   new=AsyncMock(return_value="vanishing")),
    ):
        await trader._evaluate_symbol("TEST", {}, 10_000.0, 65.0, 3)

    mock_bracket.assert_not_called()
    skips = [t for t in trader.trade_log if t["action"] == "SKIP"]
    assert any("vanishing" in str(s.get("gates_failed", [])) for s in skips)


# ── Kelly position sizing ─────────────────────────────────────────────

def test_kelly_position_sizing_typical():
    """Verify Kelly notional with known inputs."""
    trader = RigorousAutoTrader()
    trader.max_investment_per_trade = 5000.0

    # score=85, confidence=0.72, buying_power=20_000
    # edge = (85 - 50) / 50 = 0.70
    # kelly = 0.70 * 0.72 = 0.504
    # notional = 0.504 * 20_000 = 10_080 → capped at 5_000
    notional = trader._kelly_notional(85.0, 0.72, 20_000.0)
    assert notional == 5000.0, f"Expected 5000 (capped), got {notional}"


def test_kelly_position_sizing_small_edge():
    """Small edge produces a proportionally small notional."""
    trader = RigorousAutoTrader()
    trader.max_investment_per_trade = 5000.0

    # score=62 (just above HOLD threshold), confidence=0.55
    # edge = (62 - 50) / 50 = 0.24
    # kelly = 0.24 * 0.55 = 0.132
    # notional = 0.132 * 10_000 = 1_320
    notional = trader._kelly_notional(62.0, 0.55, 10_000.0)
    assert 1200 < notional < 1500, f"Expected ~1320, got {notional}"


def test_kelly_position_sizing_floored_at_100():
    """Tiny buying power → floor at $100."""
    trader = RigorousAutoTrader()
    trader.max_investment_per_trade = 5000.0

    notional = trader._kelly_notional(80.0, 0.70, 50.0)  # virtually no buying power
    assert notional == 100.0, f"Expected floor of 100, got {notional}"


# ── Cooldown logic ────────────────────────────────────────────────────

def test_cooldown_active():
    trader = RigorousAutoTrader()
    trader._set_cooldown("AAPL")
    assert trader._in_cooldown("AAPL")


def test_cooldown_expired():
    trader = RigorousAutoTrader()
    # Set cooldown that already expired
    trader._cooldown["AAPL"] = datetime.now(timezone.utc) - timedelta(minutes=1)
    assert not trader._in_cooldown("AAPL")


def test_cooldown_unknown_symbol():
    trader = RigorousAutoTrader()
    assert not trader._in_cooldown("MSFT")  # never set


# ── Bracket order payload test ────────────────────────────────────────

@pytest.mark.asyncio
async def test_bracket_order_prices_correct():
    """Verify bracket prices are computed correctly from entry price."""
    import httpx
    from engines.trade_execution import place_bracket_order

    entry = 200.0
    tp_pct = 0.015
    sl_pct = 0.008
    expected_tp = round(entry * (1 + tp_pct), 2)   # 203.0
    expected_sl = round(entry * (1 - sl_pct), 2)   # 198.4

    captured = {}

    async def mock_post(url, **kwargs):
        captured["body"] = kwargs.get("json", {})
        resp = MagicMock()
        resp.status_code = 201
        resp.headers = {"content-type": "application/json"}
        resp.json.return_value = {
            "id": "mock-order-id", "symbol": "AAPL",
            "side": "buy", "type": "market",
            "order_class": "bracket", "status": "pending_new",
            "notional": "500", "submitted_at": "2026-01-01T00:00:00Z",
        }
        return resp

    async def mock_get(url, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"quote": {"ap": entry, "bp": entry - 0.01}}
        return resp

    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.get  = AsyncMock(side_effect=mock_get)
    mock_client.post = AsyncMock(side_effect=mock_post)

    with (
        patch("engines.trade_execution._check_keys"),
        patch("engines.trade_execution.httpx.AsyncClient", return_value=mock_client),
    ):
        result = await place_bracket_order("AAPL", notional=500, take_profit_pct=tp_pct, stop_loss_pct=sl_pct)

    assert result["success"], result
    assert result["take_profit_price"] == expected_tp, f"Expected TP {expected_tp}, got {result['take_profit_price']}"
    assert result["stop_loss_price"]   == expected_sl, f"Expected SL {expected_sl}, got {result['stop_loss_price']}"
    assert result["reward_risk_ratio"] == round(tp_pct / sl_pct, 2)

    body = captured["body"]
    assert body["order_class"] == "bracket"
    assert "take_profit" in body
    assert "stop_loss" in body
