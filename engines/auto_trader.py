"""Auto-Trader — autonomous trading loop that runs for a given duration.

Continuously monitors predictions for tracked symbols and executes
trades automatically based on the prediction engine's BUY/HOLD/SELL signals.

Features:
  - Runs for a configurable duration (e.g. 30m, 1h, 4h)
  - Checks predictions every N seconds
  - Executes via Alpaca paper trading
  - Tracks all actions, P&L, and reasoning in a trade log
  - Can be started, stopped, and monitored via API
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

from config.settings import settings
from engines.prediction import predict
from engines.trade_execution import (
    get_account, get_positions, place_order, close_position,
)

logger = logging.getLogger(__name__)


class AutoTrader:
    """Autonomous trading bot that runs for a set duration."""

    def __init__(self):
        self.running = False
        self.task: Optional[asyncio.Task] = None
        self.trade_log: list[dict] = []
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.check_interval: int = 60  # seconds between prediction checks
        self.max_investment_per_trade: float = 5000.0
        self.max_positions: int = 5
        self.symbols: list[str] = []
        self.starting_portfolio: float = 0.0
        self.checks_completed: int = 0

    async def start(
        self,
        duration_minutes: int = 30,
        check_interval: int = 60,
        max_investment: float = 5000.0,
        max_positions: int = 5,
        symbols: Optional[list[str]] = None,
    ) -> dict:
        """Start the auto-trader for a given duration.

        Args:
            duration_minutes: How long to run (default 30 min)
            check_interval: Seconds between prediction checks (default 60s)
            max_investment: Max $ per trade (default $5,000)
            max_positions: Max simultaneous positions (default 5)
            symbols: Stocks to trade (default: from .env EQUITY_SYMBOLS)
        """
        if self.running:
            return {"error": "Auto-trader is already running", "status": await self.status()}

        self.running = True
        self.trade_log = []
        self.checks_completed = 0
        self.check_interval = max(10, check_interval)  # minimum 30s
        self.max_investment_per_trade = max_investment
        self.max_positions = max_positions
        self.symbols = symbols or settings.equity_symbol_list
        self.start_time = datetime.now(timezone.utc)
        self.end_time = self.start_time + timedelta(minutes=duration_minutes)

        # Record starting portfolio value
        try:
            acct = await get_account()
            self.starting_portfolio = acct["portfolio_value"]
        except Exception:
            self.starting_portfolio = 0

        self._log("🚀 Auto-trader STARTED", {
            "duration": f"{duration_minutes} minutes",
            "symbols": self.symbols,
            "check_interval": f"{self.check_interval}s",
            "max_investment": max_investment,
            "starting_portfolio": self.starting_portfolio,
        })

        # Launch the trading loop as a background task
        self.task = asyncio.create_task(self._trading_loop(duration_minutes))

        return {
            "success": True,
            "message": f"🤖 Auto-trader started for {duration_minutes} minutes",
            "symbols": self.symbols,
            "check_interval": f"{self.check_interval}s",
            "end_time": self.end_time.isoformat(),
            "max_investment_per_trade": max_investment,
            "max_positions": max_positions,
        }

    async def stop(self) -> dict:
        """Stop the auto-trader."""
        if not self.running:
            return {"error": "Auto-trader is not running"}

        self.running = False
        if self.task and not self.task.done():
            self.task.cancel()

        self._log("🛑 Auto-trader STOPPED by user", {})

        return await self.status()

    async def status(self) -> dict:
        """Get current status of the auto-trader."""
        now = datetime.now(timezone.utc)

        # Get current portfolio
        current_portfolio = 0
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

        return {
            "running": self.running,
            "symbols": self.symbols,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "remaining": remaining,
            "checks_completed": self.checks_completed,
            "trades_executed": sum(1 for t in self.trade_log if t.get("action") in ("BUY", "SELL")),
            "portfolio": {
                "starting": round(self.starting_portfolio, 2),
                "current": round(current_portfolio, 2),
                "pl": round(pl, 2),
                "pl_pct": round(pl_pct, 3),
            },
            "open_positions": positions,
            "recent_log": self.trade_log[-10:],  # last 10 entries
            "timestamp": now.isoformat(),
        }

    async def _trading_loop(self, duration_minutes: int):
        """Main trading loop — runs until duration expires or stopped."""
        try:
            end_time = datetime.now(timezone.utc) + timedelta(minutes=duration_minutes)

            while self.running and datetime.now(timezone.utc) < end_time:
                self.checks_completed += 1
                logger.info("Auto-trader: check #%d", self.checks_completed)

                # Get current positions
                try:
                    pos_data = await get_positions()
                    current_positions = {p["symbol"]: p for p in pos_data.get("positions", [])}
                except Exception as e:
                    logger.warning("Auto-trader: couldn't fetch positions: %s", e)
                    current_positions = {}

                # Check each symbol
                for symbol in self.symbols:
                    try:
                        pred = await predict(symbol)
                        if "error" in pred:
                            continue

                        signal = pred["signal"]
                        score = pred["composite_score"]
                        confidence = pred["confidence"]
                        price = pred.get("price", 0)

                        if not price or price <= 0:
                            continue

                        has_position = symbol in current_positions

                        # BUY logic: strong signal + not already holding + room for more
                        if signal == "BUY" and not has_position and len(current_positions) < self.max_positions:
                            allocation = self.max_investment_per_trade * max(0.2, confidence)
                            qty = max(1, int(allocation / price))

                            result = await place_order(symbol, "buy", qty, "market")
                            if result.get("success"):
                                self._log("BUY", {
                                    "symbol": symbol,
                                    "qty": qty,
                                    "price": price,
                                    "score": score,
                                    "confidence": confidence,
                                    "cost": round(qty * price, 2),
                                    "reason": pred.get("bullish_signals", [])[:2],
                                })
                                current_positions[symbol] = {"symbol": symbol, "qty": qty}
                            else:
                                self._log("BUY_FAILED", {
                                    "symbol": symbol,
                                    "error": result.get("error", "unknown"),
                                })

                        # SELL logic: bearish signal + currently holding
                        elif signal == "SELL" and has_position:
                            result = await close_position(symbol)
                            if result.get("success"):
                                pos = current_positions[symbol]
                                self._log("SELL", {
                                    "symbol": symbol,
                                    "qty": pos.get("qty", 0),
                                    "price": price,
                                    "score": score,
                                    "pl": pos.get("unrealized_pl", 0),
                                    "reason": pred.get("bearish_signals", [])[:2],
                                })
                                del current_positions[symbol]
                            else:
                                self._log("SELL_FAILED", {
                                    "symbol": symbol,
                                    "error": result.get("error", "unknown"),
                                })

                        else:
                            # HOLD — just log the check
                            self._log("HOLD", {
                                "symbol": symbol,
                                "score": score,
                                "signal": signal,
                                "has_position": has_position,
                            })

                    except Exception as e:
                        logger.warning("Auto-trader error for %s: %s", symbol, e)

                # Wait for next check
                await asyncio.sleep(self.check_interval)

        except asyncio.CancelledError:
            logger.info("Auto-trader cancelled")
        except Exception as e:
            logger.exception("Auto-trader error: %s", e)
            self._log("ERROR", {"error": str(e)})
        finally:
            self.running = False

            # Final P&L
            try:
                acct = await get_account()
                final_val = acct["portfolio_value"]
                pl = final_val - self.starting_portfolio
                self._log("🏁 Auto-trader FINISHED", {
                    "total_checks": self.checks_completed,
                    "total_trades": sum(1 for t in self.trade_log if t.get("action") in ("BUY", "SELL")),
                    "starting_portfolio": self.starting_portfolio,
                    "ending_portfolio": final_val,
                    "total_pl": round(pl, 2),
                    "total_pl_pct": round(pl / self.starting_portfolio * 100, 3) if self.starting_portfolio else 0,
                })
            except Exception:
                self._log("🏁 Auto-trader FINISHED", {"total_checks": self.checks_completed})

    def _log(self, action: str, details: dict):
        """Add entry to trade log."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            **details,
        }
        self.trade_log.append(entry)
        logger.info("Auto-trader: %s %s", action, details.get("symbol", ""))


# Singleton instance
auto_trader = AutoTrader()
