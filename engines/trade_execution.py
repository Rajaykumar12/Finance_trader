"""Trade Execution Engine — Alpaca Paper Trading.

Executes strategies seamlessly on the Alpaca broker API sandbox.
Supports:
  - Market, Limit, and Stop orders
  - Buy / Sell / Short
  - Position tracking and portfolio overview
  - Order history
  - Prediction-driven trade suggestions

Fulfills the paper requirement:
  "Executes strategies seamlessly on any broker or exchange API"
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import httpx

from config.settings import settings

logger = logging.getLogger(__name__)

# ── Alpaca API configuration ────────────────────────────────────────

PAPER_API = "https://paper-api.alpaca.markets"
DATA_API = "https://data.alpaca.markets"


def _headers() -> dict:
    """Build Alpaca authentication headers."""
    return {
        "APCA-API-KEY-ID": settings.alpaca_api_key,
        "APCA-API-SECRET-KEY": settings.alpaca_api_secret,
        "Accept": "application/json",
    }


def _check_keys():
    """Raise if Alpaca keys not configured."""
    if not settings.alpaca_api_key or settings.alpaca_api_key.startswith("your_"):
        raise ValueError("Alpaca API keys not configured in .env")


# ── Account & Portfolio ─────────────────────────────────────────────

async def get_account() -> dict:
    """Get paper trading account info (cash, portfolio value, buying power)."""
    _check_keys()
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(f"{PAPER_API}/v2/account", headers=_headers())
        r.raise_for_status()
        acct = r.json()

        return {
            "account_id": acct["account_number"],
            "status": acct["status"],
            "cash": round(float(acct["cash"]), 2),
            "portfolio_value": round(float(acct["portfolio_value"]), 2),
            "buying_power": round(float(acct["buying_power"]), 2),
            "equity": round(float(acct["equity"]), 2),
            "long_market_value": round(float(acct["long_market_value"]), 2),
            "short_market_value": round(float(acct["short_market_value"]), 2),
            "initial_margin": round(float(acct.get("initial_margin", 0)), 2),
            "day_trades_remaining": 3 - int(acct.get("daytrade_count", 0)),
            "pattern_day_trader": acct.get("pattern_day_trader", False),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


async def get_positions() -> dict:
    """Get all open positions with P&L."""
    _check_keys()
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(f"{PAPER_API}/v2/positions", headers=_headers())
        r.raise_for_status()
        positions = r.json()

        if not positions:
            return {"positions": [], "total_positions": 0, "total_pl": 0.0}

        result = []
        total_pl = 0.0
        for p in positions:
            pl = float(p.get("unrealized_pl", 0))
            total_pl += pl
            result.append({
                "symbol": p["symbol"],
                "qty": int(float(p["qty"])),
                "side": p["side"],
                "avg_entry": round(float(p["avg_entry_price"]), 2),
                "current_price": round(float(p["current_price"]), 2),
                "market_value": round(float(p["market_value"]), 2),
                "unrealized_pl": round(pl, 2),
                "unrealized_pl_pct": round(float(p.get("unrealized_plpc", 0)) * 100, 2),
                "change_today": round(float(p.get("change_today", 0)) * 100, 2),
            })

        # Sort by market value
        result.sort(key=lambda x: abs(x["market_value"]), reverse=True)

        return {
            "positions": result,
            "total_positions": len(result),
            "total_unrealized_pl": round(total_pl, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


async def get_orders(status: str = "all", limit: int = 20) -> dict:
    """Get recent orders."""
    _check_keys()
    async with httpx.AsyncClient(timeout=10) as client:
        params = {"status": status, "limit": limit, "direction": "desc"}
        r = await client.get(f"{PAPER_API}/v2/orders", headers=_headers(), params=params)
        r.raise_for_status()
        orders = r.json()

        result = []
        for o in orders:
            result.append({
                "order_id": o["id"],
                "symbol": o["symbol"],
                "side": o["side"],
                "type": o["type"],
                "qty": o.get("qty"),
                "filled_qty": o.get("filled_qty", "0"),
                "limit_price": o.get("limit_price"),
                "stop_price": o.get("stop_price"),
                "filled_avg_price": o.get("filled_avg_price"),
                "status": o["status"],
                "submitted_at": o.get("submitted_at"),
                "filled_at": o.get("filled_at"),
            })

        return {
            "orders": result,
            "total": len(result),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


# ── Order Execution ─────────────────────────────────────────────────

async def place_order(
    symbol: str,
    side: str,           # "buy" or "sell"
    qty: Optional[float] = None,
    order_type: str = "market",  # "market", "limit", "stop", "stop_limit"
    limit_price: Optional[float] = None,
    stop_price: Optional[float] = None,
    time_in_force: str = "day",  # "day", "gtc", "ioc", "fok"
    notional: Optional[float] = None,
) -> dict:
    """Place a paper trade order on Alpaca.

    Args:
        symbol: Stock ticker (AAPL, TSLA, etc.)
        side: "buy" or "sell"
        qty: Number of shares (mutually exclusive with notional)
        order_type: "market", "limit", "stop", "stop_limit"
        limit_price: Required for limit/stop_limit orders
        stop_price: Required for stop/stop_limit orders
        time_in_force: "day" (expires end of day), "gtc" (good til cancelled)
        notional: Dollar amount to invest (mutually exclusive with qty)

    Returns:
        Order confirmation with ID and status
    """
    _check_keys()

    # Validate inputs
    side = side.lower()
    if side not in ("buy", "sell"):
        return {"error": "Side must be 'buy' or 'sell'"}

    order_type = order_type.lower()
    if order_type not in ("market", "limit", "stop", "stop_limit"):
        return {"error": "Order type must be: market, limit, stop, stop_limit"}

    if order_type in ("limit", "stop_limit") and limit_price is None:
        return {"error": "limit_price required for limit/stop_limit orders"}

    if order_type in ("stop", "stop_limit") and stop_price is None:
        return {"error": "stop_price required for stop/stop_limit orders"}

    if qty is None and notional is None:
        qty = 1 # default to 1 share if neither provided

    if qty is not None and notional is not None:
        return {"error": "Cannot provide both qty and notional"}

    if qty is not None and float(qty) <= 0:
        return {"error": "Quantity must be positive"}

    if notional is not None and float(notional) <= 0:
        return {"error": "Notional must be positive"}

    # Build order payload
    order_data = {
        "symbol": symbol.upper(),
        "side": side,
        "type": order_type,
        "time_in_force": time_in_force,
    }
    if qty is not None:
        order_data["qty"] = str(qty)
    elif notional is not None:
        order_data["notional"] = str(notional)

    if limit_price is not None:
        order_data["limit_price"] = str(limit_price)
    if stop_price is not None:
        order_data["stop_price"] = str(stop_price)

    async with httpx.AsyncClient(timeout=10) as client:
        # Get current price for context
        try:
            quote_r = await client.get(
                f"{DATA_API}/v2/stocks/{symbol.upper()}/quotes/latest",
                headers=_headers(),
            )
            quote = quote_r.json().get("quote", {})
            current_price = float(quote.get("ap", 0))  # ask for buy, bid for sell
            if side == "sell":
                current_price = float(quote.get("bp", 0))
        except Exception:
            current_price = None

        # Submit order
        r = await client.post(
            f"{PAPER_API}/v2/orders",
            headers=_headers(),
            json=order_data,
        )

        if r.status_code in (200, 201):
            o = r.json()
            emoji = "🟢" if side == "buy" else "🔴"
            return {
                "success": True,
                "emoji": emoji,
                "message": f"{emoji} {side.upper()} {qty} x {symbol.upper()} — {order_type} order placed",
                "order": {
                    "order_id": o["id"],
                    "symbol": o["symbol"],
                    "side": o["side"],
                    "qty": o.get("qty"),
                    "notional": o.get("notional"),
                    "type": o["type"],
                    "status": o["status"],
                    "limit_price": o.get("limit_price"),
                    "stop_price": o.get("stop_price"),
                    "submitted_at": o.get("submitted_at"),
                    "time_in_force": o.get("time_in_force"),
                },
                "current_price": current_price,
                "estimated_cost": round(current_price * (qty or 1), 2) if current_price and qty is not None else float(o.get("notional", 0)),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        else:
            error = r.json()
            return {
                "success": False,
                "error": error.get("message", r.text),
                "symbol": symbol.upper(),
                "side": side,
                "qty": qty,
            }


async def cancel_order(order_id: str) -> dict:
    """Cancel a pending order."""
    _check_keys()
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.delete(f"{PAPER_API}/v2/orders/{order_id}", headers=_headers())
        if r.status_code in (200, 204):
            return {"success": True, "message": f"Order {order_id} cancelled"}
        else:
            return {"success": False, "error": r.text}


async def cancel_all_orders() -> dict:
    """Cancel all open orders."""
    _check_keys()
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.delete(f"{PAPER_API}/v2/orders", headers=_headers())
        if r.status_code in (200, 207):
            return {"success": True, "message": "All open orders cancelled"}
        else:
            return {"success": False, "error": r.text}


async def close_position(symbol: str) -> dict:
    """Close all shares of a position."""
    _check_keys()
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.delete(
            f"{PAPER_API}/v2/positions/{symbol.upper()}",
            headers=_headers(),
        )
        if r.status_code == 200:
            o = r.json()
            return {
                "success": True,
                "message": f"Closed position in {symbol.upper()}",
                "order_id": o.get("id"),
                "symbol": o.get("symbol"),
            }
        else:
            error = r.json() if r.headers.get("content-type", "").startswith("application/json") else {"message": r.text}
            return {"success": False, "error": error.get("message", r.text)}


async def close_all_positions() -> dict:
    """Liquidate all positions."""
    _check_keys()
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.delete(f"{PAPER_API}/v2/positions", headers=_headers())
        if r.status_code in (200, 207):
            return {"success": True, "message": "All positions closed (liquidated)"}
        else:
            return {"success": False, "error": r.text}


# ── Bracket Order (Entry + TP + SL in one submission) ───────────────

async def place_bracket_order(
    symbol: str,
    notional: float,
    take_profit_pct: float = 0.015,
    stop_loss_pct: float = 0.008,
) -> dict:
    """Place a bracket buy order: market entry + take-profit limit + stop-loss stop.

    Alpaca executes this as a single OTO (One-Triggers-Other) bracket:
      - Parent: market buy of `notional` dollars
      - When filled, two OCO legs activate automatically:
        - Limit sell at entry_price × (1 + take_profit_pct)  → take profit
        - Stop  sell at entry_price × (1 - stop_loss_pct)   → stop loss

    The exits are wired in at submission time — no polling required.

    Args:
        symbol:           Ticker (AAPL, TSLA, …)
        notional:         Dollar amount to invest
        take_profit_pct:  e.g. 0.015 = 1.5% above entry
        stop_loss_pct:    e.g. 0.008 = 0.8% below entry
    """
    _check_keys()

    if notional <= 0:
        return {"error": "Notional must be positive", "success": False}

    # Fetch current ask price to calculate bracket legs
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            quote_r = await client.get(
                f"{DATA_API}/v2/stocks/{symbol.upper()}/quotes/latest",
                headers=_headers(),
            )
            quote = quote_r.json().get("quote", {})
            ask_price = float(quote.get("ap", 0))
        except Exception:
            ask_price = 0.0

        if ask_price <= 0:
            return {"error": f"Could not fetch current price for {symbol}", "success": False}

        take_profit_price = round(ask_price * (1 + take_profit_pct), 2)
        stop_loss_price   = round(ask_price * (1 - stop_loss_pct), 2)

        order_data = {
            "symbol":        symbol.upper(),
            "side":          "buy",
            "type":          "market",
            "time_in_force": "day",
            "notional":      str(round(notional, 2)),
            "order_class":   "bracket",
            "take_profit":   {"limit_price": str(take_profit_price)},
            "stop_loss":     {"stop_price":  str(stop_loss_price)},
        }

        r = await client.post(
            f"{PAPER_API}/v2/orders",
            headers=_headers(),
            json=order_data,
        )

        if r.status_code in (200, 201):
            o = r.json()
            return {
                "success":          True,
                "emoji":            "🟢",
                "message":          f"🟢 BRACKET BUY {symbol.upper()} — ${notional:.0f} notional",
                "order": {
                    "order_id":         o["id"],
                    "symbol":           o["symbol"],
                    "side":             o["side"],
                    "notional":         o.get("notional"),
                    "type":             o["type"],
                    "order_class":      o.get("order_class"),
                    "status":           o["status"],
                    "submitted_at":     o.get("submitted_at"),
                },
                "entry_price":      ask_price,
                "take_profit_price": take_profit_price,
                "stop_loss_price":   stop_loss_price,
                "take_profit_pct":  take_profit_pct,
                "stop_loss_pct":    stop_loss_pct,
                "reward_risk_ratio": round(take_profit_pct / stop_loss_pct, 2),
                "timestamp":        datetime.now(timezone.utc).isoformat(),
            }
        else:
            error = r.json() if r.headers.get("content-type", "").startswith("application/json") else {"message": r.text}
            return {
                "success": False,
                "error":   error.get("message", r.text),
                "symbol":  symbol.upper(),
            }


# ── Smart Trade (Prediction-Driven) ─────────────────────────────────

async def smart_trade(symbol: str, max_investment: float = 5000.0) -> dict:
    """Execute a trade based on the prediction engine's recommendation.

    Uses the prediction composite score to decide:
      - BUY if score >= 68 (strong conviction)
      - SELL if score < 45 and we hold a position
      - HOLD otherwise (no action)

    Position sizing is proportional to confidence.

    Args:
        symbol: Stock ticker
        max_investment: Maximum $ to invest per trade (default $5,000)
    """
    _check_keys()
    from engines.prediction import predict

    # Get prediction
    pred = await predict(symbol)
    if "error" in pred:
        return {"error": pred["error"], "symbol": symbol.upper()}

    signal = pred["signal"]
    score = pred["composite_score"]
    confidence = pred["confidence"]
    price = pred.get("price", 0)

    if not price or price <= 0:
        return {"error": "Cannot determine price", "symbol": symbol.upper()}

    # Check current positions
    positions = await get_positions()
    current_pos = None
    for p in positions.get("positions", []):
        if p["symbol"] == symbol.upper():
            current_pos = p
            break

    # Decision logic
    if signal == "BUY" and not current_pos:
        # Position size based on confidence (20%–100% of max_investment)
        allocation = max_investment * max(0.2, confidence)
        qty = max(1, int(allocation / price))

        result = await place_order(symbol, "buy", qty, "market")
        result["reasoning"] = {
            "signal": signal,
            "score": score,
            "confidence": confidence,
            "allocation": round(allocation, 2),
            "bullish_signals": pred.get("bullish_signals", [])[:3],
        }
        return result

    elif signal == "SELL" and current_pos:
        qty = abs(current_pos["qty"])
        result = await place_order(symbol, "sell", qty, "market")
        result["reasoning"] = {
            "signal": signal,
            "score": score,
            "confidence": confidence,
            "position_pl": current_pos["unrealized_pl"],
            "bearish_signals": pred.get("bearish_signals", [])[:3],
        }
        return result

    else:
        action = "Already holding" if current_pos else "No position"
        return {
            "success": True,
            "action": "HOLD — no trade executed",
            "emoji": "🟡",
            "message": f"🟡 HOLD {symbol.upper()} — score {score}/100, confidence {confidence:.0%}. {action}.",
            "reasoning": {
                "signal": signal,
                "score": score,
                "confidence": confidence,
                "current_position": current_pos,
                "caution_signals": pred.get("caution_signals", [])[:3],
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
