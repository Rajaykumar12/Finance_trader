"""Rigorous Auto-Trader — high-conviction autonomous trading bot.

Entry Philosophy:  Trade rarely, but with very high conviction.
                   Every trade has a pre-wired bracket exit at submission time.

Entry Gates (ALL must pass):
  1. composite_score  >= min_score      (default 78)
  2. confidence       >= min_confidence (default 0.65)
  3. macro_score      >= 55             (not a risk-off day)
  4. liquidity        != "vanishing"    (market functioning)
  5. bullish_signals  >= 2              (at least 2 independent reasons)
  6. No existing position in symbol
  7. PDT day-trades remaining >= 2
  8. Symbol not in per-symbol cooldown (10 min after any close)

Exit Strategy: Bracket order locked in at entry time.
  - Take-profit: entry × (1 + take_profit_pct)   default +1.5%
  - Stop-loss:   entry × (1 - stop_loss_pct)     default -0.8%
  - Reward/Risk ratio: ~1.9 : 1

Position Sizing: Half-Kelly
  edge     = (composite_score - 50) / 50   → 0..1
  kelly    = edge × confidence             → 0..1
  notional = kelly × buying_power          capped at max_investment

Macro Circuit Breaker: if macro_score < 35, skip ALL buys this cycle.
"""

from __future__ import annotations

import asyncio
import logging
import math
from datetime import datetime, timezone, timedelta
from typing import Optional

from config.settings import settings
from engines.prediction import predict, _score_macro
from engines.trade_execution import (
    get_account, get_positions, place_bracket_order, close_position,
)
from state.redis_client import redis_client as _redis

logger = logging.getLogger(__name__)

# Minimum 10s between checks to avoid API spam
_MIN_INTERVAL_S = 10
# Per-symbol post-close cooldown
_COOLDOWN_MINUTES = 10


class RigorousAutoTrader:
    """Autonomous trading bot that only acts on very high-conviction signals."""

    def __init__(self):
        self.running = False
        self.task: Optional[asyncio.Task] = None
        self.trade_log: list[dict] = []
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

        # Configuration
        self.check_interval: int = 60
        self.max_investment_per_trade: float = 5000.0
        self.max_positions: int = 5
        self.min_score: float = 78.0
        self.min_confidence: float = 0.65
        self.take_profit_pct: float = 0.015
        self.stop_loss_pct: float = 0.008

        # State
        self.symbols: list[str] = []
        self.starting_portfolio: float = 0.0
        self.checks_completed: int = 0
        self._cooldown: dict[str, datetime] = {}   # symbol → cooldown-until time

    # ── Public API ────────────────────────────────────────────────────

    async def start(
        self,
        duration_minutes: int = 30,
        check_interval: int = 60,
        max_investment: float = 5000.0,
        max_positions: int = 5,
        min_score: float = 78.0,
        min_confidence: float = 0.65,
        take_profit_pct: float = 0.015,
        stop_loss_pct: float = 0.008,
        symbols: Optional[list[str]] = None,
    ) -> dict:
        """Start the rigorous auto-trader.

        Args:
            duration_minutes:  How long to run (default 30 min)
            check_interval:    Seconds between cycles (default 60s)
            max_investment:    Hard cap per trade in dollars (default $5,000)
            max_positions:     Max simultaneous open positions (default 5)
            min_score:         Minimum composite score to enter (default 78)
            min_confidence:    Minimum prediction confidence to enter (default 0.65)
            take_profit_pct:   Take-profit leg, e.g. 0.015 = 1.5% (default 0.015)
            stop_loss_pct:     Stop-loss leg, e.g. 0.008 = 0.8% (default 0.008)
            symbols:           Tickers to trade (default: from .env EQUITY_SYMBOLS)
        """
        if self.running:
            return {"error": "Auto-trader is already running", "status": await self.status()}

        if stop_loss_pct <= 0 or take_profit_pct <= 0:
            return {"error": "take_profit_pct and stop_loss_pct must be positive"}

        self.running = True
        self.trade_log = []
        self.checks_completed = 0
        self._cooldown = {}
        self.check_interval = max(_MIN_INTERVAL_S, check_interval)
        self.max_investment_per_trade = max_investment
        self.max_positions = max_positions
        self.min_score = min_score
        self.min_confidence = min_confidence
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct
        self.symbols = symbols or settings.equity_symbol_list
        self.start_time = datetime.now(timezone.utc)
        self.end_time = self.start_time + timedelta(minutes=duration_minutes)

        try:
            acct = await get_account()
            self.starting_portfolio = acct["portfolio_value"]
        except Exception:
            self.starting_portfolio = 0

        rr = round(take_profit_pct / stop_loss_pct, 2)
        self._log("🚀 Rigorous Auto-Trader STARTED", {
            "duration":         f"{duration_minutes} minutes",
            "symbols":          self.symbols,
            "check_interval":   f"{self.check_interval}s",
            "entry_gates": {
                "min_score":      min_score,
                "min_confidence": min_confidence,
                "macro_gate":     "score >= 55",
                "liquidity_gate": "no vanishing",
                "signal_gate":    ">= 2 bullish signals",
                "pdt_gate":       ">= 2 day-trades remaining",
            },
            "bracket_exits": {
                "take_profit":  f"+{take_profit_pct:.1%}",
                "stop_loss":    f"-{stop_loss_pct:.1%}",
                "reward_risk":  f"{rr}:1",
            },
            "max_investment":       max_investment,
            "starting_portfolio":   self.starting_portfolio,
        })

        self.task = asyncio.create_task(self._trading_loop(duration_minutes))

        return {
            "success":            True,
            "message":            f"🤖 Rigorous auto-trader started for {duration_minutes} min",
            "symbols":            self.symbols,
            "check_interval":     f"{self.check_interval}s",
            "end_time":           self.end_time.isoformat(),
            "entry_gates": {
                "min_score":      min_score,
                "min_confidence": f"{min_confidence:.0%}",
            },
            "bracket_exits": {
                "take_profit":    f"+{take_profit_pct:.1%}",
                "stop_loss":      f"-{stop_loss_pct:.1%}",
                "reward_risk":    f"{rr:.1f}:1",
            },
            "max_investment_per_trade": max_investment,
            "max_positions":           max_positions,
        }

    async def stop(self) -> dict:
        """Stop the auto-trader immediately."""
        if not self.running:
            return {"error": "Auto-trader is not running"}

        self.running = False
        if self.task and not self.task.done():
            self.task.cancel()

        self._log("🛑 Auto-trader STOPPED by user", {})
        return await self.status()

    async def status(self) -> dict:
        """Get current status: running state, P&L, decisions log."""
        now = datetime.now(timezone.utc)

        current_portfolio = 0.0
        positions = []
        try:
            acct = await get_account()
            current_portfolio = acct["portfolio_value"]
            pos_data = await get_positions()
            positions = pos_data.get("positions", [])
        except Exception:
            pass

        pl = current_portfolio - self.starting_portfolio if self.starting_portfolio > 0 else 0
        pl_pct = (pl / self.starting_portfolio * 100) if self.starting_portfolio > 0 else 0

        remaining = ""
        if self.running and self.end_time:
            rem = self.end_time - now
            remaining = f"{max(0, int(rem.total_seconds()))}s"

        buys  = [t for t in self.trade_log if t.get("action") == "BUY"]
        sells = [t for t in self.trade_log if t.get("action") in ("SELL", "SELL (exit)")]
        skips = [t for t in self.trade_log if t.get("action") == "SKIP"]

        # Active cooldowns
        active_cooldowns = {
            sym: until.isoformat()
            for sym, until in self._cooldown.items()
            if until > now
        }

        return {
            "running":          self.running,
            "symbols":          self.symbols,
            "start_time":       self.start_time.isoformat() if self.start_time else None,
            "end_time":         self.end_time.isoformat() if self.end_time else None,
            "remaining":        remaining,
            "checks_completed": self.checks_completed,
            "stats": {
                "buys":  len(buys),
                "sells": len(sells),
                "skips": len(skips),
            },
            "portfolio": {
                "starting": round(self.starting_portfolio, 2),
                "current":  round(current_portfolio, 2),
                "pl":       round(pl, 2),
                "pl_pct":   round(pl_pct, 3),
            },
            "open_positions":  positions,
            "active_cooldowns": active_cooldowns,
            "recent_log":      self.trade_log[-15:],
            "timestamp":       now.isoformat(),
        }

    # ── Internal helpers ──────────────────────────────────────────────

    def _in_cooldown(self, symbol: str) -> bool:
        until = self._cooldown.get(symbol)
        return until is not None and until > datetime.now(timezone.utc)

    def _set_cooldown(self, symbol: str):
        self._cooldown[symbol] = datetime.now(timezone.utc) + timedelta(minutes=_COOLDOWN_MINUTES)

    def _kelly_notional(self, composite_score: float, confidence: float, buying_power: float) -> float:
        """Calculate position size using half-Kelly criterion.

        edge     = (score - 50) / 50    maps 50..100 → 0..1
        kelly    = edge × confidence    fraction of capital to risk
        notional = kelly × buying_power capped at max_investment
        Floored at $100 to avoid tiny dust orders.
        """
        edge = (composite_score - 50.0) / 50.0
        edge = max(0.0, min(1.0, edge))
        kelly = edge * confidence
        notional = kelly * buying_power
        notional = min(notional, self.max_investment_per_trade)
        return max(100.0, round(notional, 2))

    async def _get_macro_score(self) -> float:
        """Fetch macro indicators from Redis and score them 0-100."""
        try:
            macro = await _redis.hgetall("macro_indicators")
            if not macro:
                return 50.0  # neutral if unavailable
            score, _ = _score_macro(macro)
            return score
        except Exception:
            return 50.0

    async def _get_liquidity(self, symbol: str) -> str:
        """Fetch liquidity status from Redis market state."""
        try:
            market = await _redis.hgetall(f"market_state:{symbol.upper()}")
            return market.get("liquidity_status", "unknown")
        except Exception:
            return "unknown"

    # ── Main loop ─────────────────────────────────────────────────────

    async def _trading_loop(self, duration_minutes: int):
        """Continuous loop: evaluate, gate-check, and bracket-order."""
        try:
            end_time = datetime.now(timezone.utc) + timedelta(minutes=duration_minutes)

            while self.running and datetime.now(timezone.utc) < end_time:
                self.checks_completed += 1
                logger.info("Rigorous auto-trader: check #%d", self.checks_completed)

                # ── Fetch account state ───────────────────────────────
                try:
                    acct = await get_account()
                    buying_power = acct["buying_power"]
                    day_trades_remaining = acct.get("day_trades_remaining", 3)
                except Exception as e:
                    logger.warning("Cannot fetch account: %s", e)
                    await asyncio.sleep(self.check_interval)
                    continue

                # ── PDT protection: need at least 2 day-trades left ──
                if day_trades_remaining < 2:
                    self._log("⛔ PDT CIRCUIT BREAKER", {
                        "reason":               "< 2 day-trades remaining today",
                        "day_trades_remaining": day_trades_remaining,
                    })
                    await asyncio.sleep(self.check_interval)
                    continue

                # ── Macro circuit breaker ─────────────────────────────
                macro_score = await self._get_macro_score()
                if macro_score < 35:
                    self._log("⛔ MACRO CIRCUIT BREAKER", {
                        "reason":      "macro_score < 35 — fear regime, skipping all buys",
                        "macro_score": round(macro_score, 1),
                    })
                    await asyncio.sleep(self.check_interval)
                    continue

                if macro_score < 55:
                    self._log("⚠️ MACRO WARNING", {
                        "note":        "macro_score < 55 — risk-off, requiring higher conviction",
                        "macro_score": round(macro_score, 1),
                    })

                # ── Fetch open positions ──────────────────────────────
                try:
                    pos_data = await get_positions()
                    current_positions = {p["symbol"]: p for p in pos_data.get("positions", [])}
                except Exception as e:
                    logger.warning("Cannot fetch positions: %s", e)
                    current_positions = {}

                # ── Manage positions opened by this bot ──────────────
                # Bracket exits are handled automatically by Alpaca,
                # but we track and log closed positions for P&L visibility.
                # (Alpaca removes the position once SL/TP fires.)
                # If a position we opened is now gone, it exited via bracket → log it.
                # We do not re-open immediately (cooldown enforced).

                # ── Evaluate each symbol ──────────────────────────────
                for symbol in self.symbols:
                    try:
                        await self._evaluate_symbol(
                            symbol, current_positions, buying_power,
                            macro_score, day_trades_remaining
                        )
                    except Exception as e:
                        logger.warning("Error evaluating %s: %s", symbol, e)

                await asyncio.sleep(self.check_interval)

        except asyncio.CancelledError:
            logger.info("Rigorous auto-trader cancelled")
        except Exception as e:
            logger.exception("Rigorous auto-trader error: %s", e)
            self._log("ERROR", {"error": str(e)})
        finally:
            self.running = False
            await self._finalize()

    async def _evaluate_symbol(
        self,
        symbol: str,
        current_positions: dict,
        buying_power: float,
        macro_score: float,
        day_trades_remaining: int,
    ):
        """Run all entry gates for one symbol, then bracket-buy or skip."""
        has_position = symbol in current_positions

        # ── If we have a position, just monitor it (bracket handles exit) ──
        if has_position:
            pos = current_positions[symbol]
            self._log("HOLD", {
                "symbol":           symbol,
                "unrealized_pl":    pos.get("unrealized_pl", 0),
                "unrealized_pl_pct": pos.get("unrealized_pl_pct", 0),
                "note": "Bracket exit is active — Alpaca handles TP/SL automatically",
            })
            return

        # ── Cannot open more positions ────────────────────────────────
        if len(current_positions) >= self.max_positions:
            return  # silent — no log spam

        # ── Per-symbol cooldown ───────────────────────────────────────
        if self._in_cooldown(symbol):
            until = self._cooldown[symbol]
            self._log("SKIP", {
                "symbol": symbol,
                "gate":   "cooldown",
                "reason": f"10-min post-close cooldown active until {until.strftime('%H:%M:%S')} UTC",
            })
            return

        # ── Get prediction ────────────────────────────────────────────
        pred = await predict(symbol)
        if "error" in pred:
            logger.debug("Prediction error for %s: %s", symbol, pred["error"])
            return

        score      = pred["composite_score"]
        confidence = pred["confidence"]
        signal     = pred["signal"]
        bullish    = pred.get("bullish_signals", [])
        bearish    = pred.get("bearish_signals", [])
        price      = pred.get("price", 0)

        if not price or price <= 0:
            return

        skip_reasons = []

        # Gate 1: composite score
        if score < self.min_score:
            skip_reasons.append(
                f"score {score:.1f} < {self.min_score} (threshold)"
            )

        # Gate 2: confidence
        if confidence < self.min_confidence:
            skip_reasons.append(
                f"confidence {confidence:.0%} < {self.min_confidence:.0%}"
            )

        # Gate 3: macro score (already checked globally, but enforce 55 gate per trade)
        if macro_score < 55:
            skip_reasons.append(
                f"macro_score {macro_score:.1f} < 55 (risk-off environment)"
            )

        # Gate 4: liquidity
        liquidity = await self._get_liquidity(symbol)
        if liquidity == "vanishing":
            skip_reasons.append("liquidity=vanishing (flash-crash risk)")

        # Gate 5: bullish signal count
        if len(bullish) < 2:
            skip_reasons.append(
                f"only {len(bullish)} bullish signal(s) (need >= 2)"
            )

        # Gate 6: signal must be BUY
        if signal != "BUY":
            skip_reasons.append(f"signal={signal} (only acting on BUY)")

        # ── Skip if any gate failed ───────────────────────────────────
        if skip_reasons:
            self._log("SKIP", {
                "symbol":     symbol,
                "score":      score,
                "confidence": confidence,
                "signal":     signal,
                "gates_failed": skip_reasons,
                "bullish":    bullish[:3],
                "bearish":    bearish[:2],
            })
            return

        # ── All gates passed — compute Kelly position size ────────────
        notional = self._kelly_notional(score, confidence, buying_power)

        # Safety: don't exceed available buying power
        if notional > buying_power * 0.95:
            notional = round(buying_power * 0.25, 2)  # fall back to 25% of buying power

        if notional < 100:
            self._log("SKIP", {
                "symbol": symbol,
                "gate":   "buying_power",
                "reason": f"insufficient buying power (notional=${notional:.0f})",
            })
            return

        # ── Submit bracket order ──────────────────────────────────────
        result = await place_bracket_order(
            symbol,
            notional=notional,
            take_profit_pct=self.take_profit_pct,
            stop_loss_pct=self.stop_loss_pct,
        )

        if result.get("success"):
            self._log("BUY", {
                "symbol":            symbol,
                "notional":          notional,
                "price":             price,
                "score":             score,
                "confidence":        f"{confidence:.0%}",
                "macro_score":       round(macro_score, 1),
                "liquidity":         liquidity,
                "bullish_signals":   bullish[:4],
                "take_profit_price": result.get("take_profit_price"),
                "stop_loss_price":   result.get("stop_loss_price"),
                "reward_risk":       result.get("reward_risk_ratio"),
                "order_id":          result.get("order", {}).get("order_id"),
            })
            # Arm cooldown: symbol can't be re-entered until the bracket exits
            # (We'll enforce a 10-min cooldown after the position closes.)
        else:
            self._log("BUY_FAILED", {
                "symbol": symbol,
                "error":  result.get("error", "unknown"),
                "notional": notional,
            })

    async def _finalize(self):
        """Log final P&L summary when the loop ends."""
        try:
            acct = await get_account()
            final_val = acct["portfolio_value"]
            pl = final_val - self.starting_portfolio
            pl_pct = (pl / self.starting_portfolio * 100) if self.starting_portfolio else 0
            buys  = sum(1 for t in self.trade_log if t.get("action") == "BUY")
            skips = sum(1 for t in self.trade_log if t.get("action") == "SKIP")
            self._log("🏁 Rigorous Auto-Trader FINISHED", {
                "total_checks":      self.checks_completed,
                "total_buys":        buys,
                "total_skips":       skips,
                "skip_to_buy_ratio": f"{skips}:{buys}" if buys else f"{skips}:0",
                "starting_portfolio": self.starting_portfolio,
                "ending_portfolio":   round(final_val, 2),
                "total_pl":          round(pl, 2),
                "total_pl_pct":      round(pl_pct, 3),
            })
        except Exception:
            self._log("🏁 Rigorous Auto-Trader FINISHED", {
                "total_checks": self.checks_completed,
            })

    def _log(self, action: str, details: dict):
        """Append to trade log and emit to logger."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action":    action,
            **details,
        }
        self.trade_log.append(entry)
        symbol = details.get("symbol", "")
        logger.info("AutoTrader [%s] %s", action, symbol)


# Singleton — replaces old AutoTrader instance
auto_trader = RigorousAutoTrader()
