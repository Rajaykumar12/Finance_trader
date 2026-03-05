"""Day Trading Bot — autonomous intraday trading engine.

Implements all 7 day-trading requirements:

  1. Market hours gate     — only trade 9:30–16:00 ET on weekdays
  2. EOD flat              — close all positions at 15:45 ET (15 min before close)
  3. Daily drawdown limit  — stop all new buys if portfolio drops X% from day open
  4. 5-min OHLCV bars      — fetched from Alpaca bars endpoint per symbol
  5. Intraday signals      — VWAP, EMA(9/21) crossover, ORB (30-min range)
  6. PDT weekly tracker    — rolling 5-business-day day-trade counter (max 3)
  7. Pre-market scan       — 9:10–9:25 ET watchlist preparation

Strategy:
  - Intraday signal (VWAP + EMA + ORB) provides the entry trigger
  - Rigorousness from the auto-trader (score >= 70 intraday vs 78 prediction)
  - Bracket orders lock in TP and SL at submission (default +1.5% / -0.8%)
  - Positions are always closed EOD — never held overnight

PDT Tracking:
  - Stored in Redis key "day_trader:pdt_log" as a JSON list of ISO timestamps
  - A day trade is counted each time we open AND close a position on the same day
  - Only round-trips within the rolling last-5 business days count toward the limit
  - If daytrade_count >= 3 (from Alpaca), no new buys are allowed
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone, timedelta, date
from typing import Optional

from config.settings import settings
from engines.market_hours import (
    is_market_open, is_premarket, should_flatten_eod,
    is_premarket_scan_window, session_label, session_summary,
    minutes_to_open, minutes_to_close, market_date,
)
from ingestion.alpaca_bars import fetch_today_bars
from engines.intraday_signals import generate_intraday_signal
from engines.trade_execution import (
    get_account, get_positions, place_bracket_order, close_all_positions, close_position,
)
from state.redis_client import redis_client as _redis

logger = logging.getLogger(__name__)

# Redis keys
_PDT_LOG_KEY    = "day_trader:pdt_log"      # JSON list of ISO trade timestamps
_DAY_START_KEY  = "day_trader:day_start"    # JSON {date, portfolio_value}

_MIN_INTRADAY_SCORE = 70.0     # minimum intraday signal score to enter
_MAX_PDT_TRADES     = 3        # PDT rule: max 3 round-trips per rolling 5 business days
_MIN_INTERVAL_S     = 30       # minimum seconds between check cycles


def _business_days_ago(n: int) -> date:
    """Return the date that is `n` business days before today."""
    d = date.today()
    count = 0
    while count < n:
        d -= timedelta(days=1)
        if d.weekday() < 5:
            count += 1
    return d


class DayTrader:
    """Autonomous intraday trading bot for US equity market hours."""

    def __init__(self):
        self.running           = False
        self.task: Optional[asyncio.Task] = None
        self.trade_log: list[dict] = []
        self.start_time: Optional[datetime] = None

        # Configuration (set by start())
        self.symbols:               list[str] = []
        self.check_interval:        int   = 60
        self.max_investment:        float = 5000.0
        self.max_positions:         int   = 5
        self.take_profit_pct:       float = 0.015
        self.stop_loss_pct:         float = 0.008
        self.daily_drawdown_limit:  float = 0.02    # 2% portfolio drop → stop trading
        self.min_intraday_score:    float = _MIN_INTRADAY_SCORE

        # Intraday state
        self._day_start_portfolio:  float = 0.0
        self._day_start_date:       Optional[date] = None
        self._drawdown_triggered:   bool  = False
        self._eod_flattened:        bool  = False
        self._scan_done_today:      bool  = False
        self._watchlist:            list[str] = []
        self._checks_completed:     int   = 0

    # ── Public API ────────────────────────────────────────────────────

    async def start(
        self,
        check_interval:       int   = 60,
        max_investment:       float = 5000.0,
        max_positions:        int   = 5,
        take_profit_pct:      float = 0.015,
        stop_loss_pct:        float = 0.008,
        daily_drawdown_limit: float = 0.02,
        min_intraday_score:   float = 70.0,
        symbols: Optional[list[str]] = None,
    ) -> dict:
        """Start the day-trading bot.

        The bot runs indefinitely until stopped or until end of trading week.
        It self-manages market hours: it waits during off-hours and wakes at open.

        Args:
            check_interval:       Seconds between evaluation cycles (min 30s)
            max_investment:       Max $ per individual trade
            max_positions:        Max simultaneous open positions
            take_profit_pct:      Bracket TP level, e.g. 0.015 = +1.5%
            stop_loss_pct:        Bracket SL level, e.g. 0.008 = -0.8%
            daily_drawdown_limit: Stop trading if portfolio drops this % from day open
            min_intraday_score:   Minimum intraday signal score to enter (default 70)
            symbols:              Tickers to trade (default: from .env)
        """
        if self.running:
            return {"error": "Day trader is already running", "status": await self.status()}

        self.running              = True
        self.trade_log            = []
        self._checks_completed    = 0
        self._drawdown_triggered  = False
        self._eod_flattened       = False
        self._scan_done_today     = False
        self._watchlist           = []
        self.start_time           = datetime.now(timezone.utc)
        self.check_interval       = max(_MIN_INTERVAL_S, check_interval)
        self.max_investment       = max_investment
        self.max_positions        = max_positions
        self.take_profit_pct      = take_profit_pct
        self.stop_loss_pct        = stop_loss_pct
        self.daily_drawdown_limit = daily_drawdown_limit
        self.min_intraday_score   = min_intraday_score
        self.symbols              = symbols or settings.equity_symbol_list

        await self._refresh_day_start()
        self.task = asyncio.create_task(self._main_loop(), name="day_trader")

        self._log("🌅 Day Trader STARTED", {
            "symbols":              self.symbols,
            "session":              session_label(),
            "check_interval":       f"{self.check_interval}s",
            "take_profit":          f"+{take_profit_pct:.1%}",
            "stop_loss":            f"-{stop_loss_pct:.1%}",
            "daily_drawdown_limit": f"-{daily_drawdown_limit:.1%}",
            "min_intraday_score":   min_intraday_score,
            "day_start_portfolio":  self._day_start_portfolio,
        })

        return {
            "success":    True,
            "message":    "📈 Day Trader started",
            "session":    session_label(),
            "symbols":    self.symbols,
            "config": {
                "take_profit":          f"+{take_profit_pct:.1%}",
                "stop_loss":            f"-{stop_loss_pct:.1%}",
                "daily_drawdown_limit": f"-{daily_drawdown_limit:.1%}",
                "min_intraday_score":   min_intraday_score,
                "pdt_remaining":        await self._pdt_remaining(),
            },
        }

    async def stop(self) -> dict:
        """Stop the day trader."""
        if not self.running:
            return {"error": "Day trader is not running"}
        self.running = False
        if self.task and not self.task.done():
            self.task.cancel()
        self._log("🛑 Day Trader STOPPED by user", {})
        return await self.status()

    async def status(self) -> dict:
        """Full status report."""
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

        day_pl     = current_portfolio - self._day_start_portfolio
        day_pl_pct = (day_pl / self._day_start_portfolio * 100) if self._day_start_portfolio else 0

        buys  = [t for t in self.trade_log if t.get("action") == "BUY"]
        sells = [t for t in self.trade_log if "SELL" in t.get("action", "")]
        skips = [t for t in self.trade_log if t.get("action") == "SKIP"]

        return {
            "running":           self.running,
            "session":           session_summary(),
            "symbols":           self.symbols,
            "watchlist":         self._watchlist,
            "checks_completed":  self._checks_completed,
            "drawdown_triggered": self._drawdown_triggered,
            "eod_flattened":     self._eod_flattened,
            "pdt_remaining":     await self._pdt_remaining(),
            "stats": {
                "buys":  len(buys),
                "sells": len(sells),
                "skips": len(skips),
            },
            "portfolio": {
                "day_start":  round(self._day_start_portfolio, 2),
                "current":    round(current_portfolio, 2),
                "day_pl":     round(day_pl, 2),
                "day_pl_pct": round(day_pl_pct, 3),
            },
            "open_positions": positions,
            "recent_log":     self.trade_log[-15:],
            "timestamp":      now.isoformat(),
        }

    # ── PDT Tracking ──────────────────────────────────────────────────

    async def _load_pdt_log(self) -> list[str]:
        """Load the PDT trade log from Redis (list of ISO timestamps)."""
        try:
            raw = await _redis.pool.get(_PDT_LOG_KEY)
            if raw:
                return json.loads(raw)
        except Exception:
            pass
        return []

    async def _save_pdt_log(self, log: list[str]) -> None:
        try:
            # Keep only last 30 days of data to prevent unbounded growth
            cutoff = (date.today() - timedelta(days=30)).isoformat()
            trimmed = [ts for ts in log if ts[:10] >= cutoff]
            await _redis.pool.set(_PDT_LOG_KEY, json.dumps(trimmed))
        except Exception:
            pass

    async def _pdt_count_rolling(self) -> int:
        """Count day trades within the rolling last 5 business days."""
        log = await self._load_pdt_log()
        cutoff = _business_days_ago(5)
        count = sum(1 for ts in log if date.fromisoformat(ts[:10]) >= cutoff)
        return count

    async def _pdt_remaining(self) -> int:
        """How many PDT day trades remain in the rolling 5-business-day window."""
        used = await self._pdt_count_rolling()
        return max(0, _MAX_PDT_TRADES - used)

    async def _record_pdt_trade(self) -> None:
        """Record a completed day trade (round-trip same day)."""
        log = await self._load_pdt_log()
        log.append(datetime.now(timezone.utc).isoformat())
        await self._save_pdt_log(log)

    # ── Day-Start P&L Baseline ────────────────────────────────────────

    async def _refresh_day_start(self) -> None:
        """Update the day-start portfolio value for drawdown tracking.

        Called once per calendar day (ET) at the start of the trading session.
        Persisted in Redis so restarts within the same day don't reset the baseline.
        """
        today = market_date()

        # Try to load existing day-start from Redis
        try:
            raw = await _redis.pool.get(_DAY_START_KEY)
            if raw:
                data = json.loads(raw)
                if data.get("date") == today.isoformat():
                    self._day_start_portfolio = float(data["portfolio_value"])
                    self._day_start_date = today
                    logger.info("Day-start portfolio loaded from Redis: $%.2f", self._day_start_portfolio)
                    return
        except Exception:
            pass

        # Fetch from Alpaca
        try:
            acct = await get_account()
            self._day_start_portfolio = acct["portfolio_value"]
        except Exception:
            self._day_start_portfolio = 0.0

        self._day_start_date = today

        # Persist
        try:
            payload = json.dumps({"date": today.isoformat(), "portfolio_value": self._day_start_portfolio})
            await _redis.pool.set(_DAY_START_KEY, payload, ex=86400)  # expire in 24h
        except Exception:
            pass

        logger.info("Day-start portfolio baseline: $%.2f (%s)", self._day_start_portfolio, today)

    # ── Main Loop ─────────────────────────────────────────────────────

    async def _main_loop(self) -> None:
        try:
            while self.running:
                # Reset daily state if the date rolled over
                if self._day_start_date and market_date() != self._day_start_date:
                    self._drawdown_triggered = False
                    self._eod_flattened      = False
                    self._scan_done_today    = False
                    self._watchlist          = []
                    await self._refresh_day_start()
                    self._log("🌅 New Trading Day", {"date": market_date().isoformat()})

                # ── Pre-market scan window ──────────────────────────────
                if is_premarket_scan_window() and not self._scan_done_today:
                    await self._premarket_scan()
                    self._scan_done_today = True

                # ── Not in market hours → wait ──────────────────────────
                if not is_market_open():
                    mins = minutes_to_open()
                    label = session_label()
                    if int(self._checks_completed) % 5 == 0:  # log every 5 cycles to avoid spam
                        self._log("⏸ WAITING", {
                            "session": label,
                            "minutes_to_open": round(mins, 1),
                        })
                    await asyncio.sleep(self.check_interval)
                    self._checks_completed += 1
                    continue

                self._checks_completed += 1

                # ── EOD flatten ─────────────────────────────────────────
                if should_flatten_eod() and not self._eod_flattened:
                    await self._flatten_all_eod()
                    self._eod_flattened = True
                    await asyncio.sleep(self.check_interval)
                    continue

                if self._eod_flattened:
                    # After EOD flatten, just wait for market close
                    await asyncio.sleep(self.check_interval)
                    continue

                # ── Fetch account state ─────────────────────────────────
                try:
                    acct = await get_account()
                    buying_power       = acct["buying_power"]
                    portfolio_value    = acct["portfolio_value"]
                    alpaca_daytrades   = acct.get("day_trades_remaining", 3)
                except Exception as e:
                    logger.warning("Day trader: cannot fetch account: %s", e)
                    await asyncio.sleep(self.check_interval)
                    continue

                # ── Daily drawdown circuit breaker ──────────────────────
                if self._day_start_portfolio > 0 and not self._drawdown_triggered:
                    day_drop_pct = (self._day_start_portfolio - portfolio_value) / self._day_start_portfolio
                    if day_drop_pct >= self.daily_drawdown_limit:
                        self._drawdown_triggered = True
                        self._log("⛔ DAILY DRAWDOWN LIMIT", {
                            "reason":       f"Portfolio dropped {day_drop_pct:.2%} from day open — no more buys today",
                            "day_start":    self._day_start_portfolio,
                            "current":      portfolio_value,
                            "drop_pct":     round(day_drop_pct * 100, 2),
                            "limit_pct":    round(self.daily_drawdown_limit * 100, 2),
                        })

                if self._drawdown_triggered:
                    self._log("SKIP (all)", {"reason": "Daily drawdown limit reached — only monitoring existing positions"})
                    await asyncio.sleep(self.check_interval)
                    continue

                # ── PDT circuit breaker ─────────────────────────────────
                pdt_remaining = await self._pdt_remaining()
                if pdt_remaining <= 0 or alpaca_daytrades <= 0:
                    self._log("⛔ PDT LIMIT", {
                        "reason":          "No day trades remaining this rolling 5-day window",
                        "pdt_remaining":   pdt_remaining,
                        "alpaca_dt_left":  alpaca_daytrades,
                    })
                    await asyncio.sleep(self.check_interval)
                    continue

                # ── Fetch open positions ────────────────────────────────
                try:
                    pos_data = await get_positions()
                    current_positions = {p["symbol"]: p for p in pos_data.get("positions", [])}
                except Exception:
                    current_positions = {}

                # ── Evaluate each symbol ────────────────────────────────
                active_symbols = self._watchlist if self._watchlist else self.symbols
                for symbol in active_symbols:
                    try:
                        await self._evaluate(symbol, current_positions, buying_power)
                    except Exception as e:
                        logger.warning("Day trader error for %s: %s", symbol, e)

                await asyncio.sleep(self.check_interval)

        except asyncio.CancelledError:
            logger.info("Day trader cancelled")
        except Exception as e:
            logger.exception("Day trader error: %s", e)
            self._log("ERROR", {"error": str(e)})
        finally:
            self.running = False
            self._log("🏁 Day Trader FINISHED", {
                "checks": self._checks_completed,
                "buys":   sum(1 for t in self.trade_log if t.get("action") == "BUY"),
            })

    # ── Pre-market scan ───────────────────────────────────────────────

    async def _premarket_scan(self) -> None:
        """9:10–9:25 ET scan — rank symbols by prediction score to build the watchlist."""
        from engines.prediction import predict

        self._log("🔍 Pre-market scan starting", {"symbols": self.symbols})
        ranked = []

        for symbol in self.symbols:
            try:
                pred = await predict(symbol)
                if "error" not in pred:
                    ranked.append((symbol, pred["composite_score"], pred["signal"]))
            except Exception as e:
                logger.warning("Pre-market scan error for %s: %s", symbol, e)

        # Sort descending by score, take top 5
        ranked.sort(key=lambda x: x[1], reverse=True)
        self._watchlist = [sym for sym, score, sig in ranked if sig == "BUY"][:5]

        self._log("📋 Pre-market watchlist ready", {
            "watchlist": self._watchlist,
            "all_scores": {sym: round(score, 1) for sym, score, _ in ranked},
        })

    # ── EOD flatten ───────────────────────────────────────────────────

    async def _flatten_all_eod(self) -> None:
        """Close ALL open positions 15 minutes before market close."""
        self._log("🏁 EOD FLATTEN — closing all positions", {
            "reason":     "15 minutes to market close",
            "et_time":    __import__("engines.market_hours", fromlist=["now_et"]).now_et().strftime("%H:%M ET"),
        })
        try:
            result = await close_all_positions()
            self._log("EOD FLATTEN complete", {"result": result})
        except Exception as e:
            self._log("EOD FLATTEN error", {"error": str(e)})

    # ── Symbol evaluation ─────────────────────────────────────────────

    async def _evaluate(
        self,
        symbol: str,
        current_positions: dict,
        buying_power: float,
    ) -> None:
        """Evaluate one symbol: fetch bars, compute signal, act."""
        has_position = symbol in current_positions

        # If holding — log status (bracket exits are handled by Alpaca automatically)
        if has_position:
            pos = current_positions[symbol]
            self._log("HOLD", {
                "symbol":           symbol,
                "unrealized_pl":    pos.get("unrealized_pl", 0),
                "unrealized_pl_pct": pos.get("unrealized_pl_pct", 0),
            })
            return

        # At max positions — skip
        if len(current_positions) >= self.max_positions:
            return

        # Fetch today's 5-min bars
        bars = await fetch_today_bars(symbol)
        if not bars:
            self._log("SKIP", {"symbol": symbol, "gate": "no_bars",
                                "reason": "No intraday bar data available"})
            return

        # Generate intraday signal
        sig = generate_intraday_signal(symbol, bars)
        intraday_score  = sig["score"]
        intraday_signal = sig["signal"]

        if intraday_signal != "BUY" or intraday_score < self.min_intraday_score:
            self._log("SKIP", {
                "symbol":          symbol,
                "gate":            "intraday_signal",
                "score":           intraday_score,
                "signal":          intraday_signal,
                "threshold":       self.min_intraday_score,
                "ema_signal":      sig.get("ema_signal"),
                "vwap_signal":     sig.get("vwap_signal"),
                "orb_signal":      sig.get("orb_signal"),
            })
            return

        # Position size: Kelly-style from intraday score
        edge = (intraday_score - 50.0) / 50.0
        notional = min(edge * buying_power, self.max_investment)
        notional = max(100.0, round(notional, 2))

        if notional > buying_power * 0.95:
            notional = round(buying_power * 0.25, 2)

        # Submit bracket order
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
                "price":             sig.get("current_price"),
                "intraday_score":    intraday_score,
                "vwap":              sig.get("vwap"),
                "ema_short":         sig.get("ema_short"),
                "ema_long":          sig.get("ema_long"),
                "orb_high":          sig.get("orb_high"),
                "orb_low":           sig.get("orb_low"),
                "reasons":           sig.get("reasons", []),
                "take_profit_price": result.get("take_profit_price"),
                "stop_loss_price":   result.get("stop_loss_price"),
                "reward_risk":       result.get("reward_risk_ratio"),
                "order_id":          result.get("order", {}).get("order_id"),
            })
            # Record as a potential day trade (opened today)
            await self._record_pdt_trade()
        else:
            self._log("BUY_FAILED", {
                "symbol":   symbol,
                "error":    result.get("error", "unknown"),
                "notional": notional,
            })

    # ── Logging ───────────────────────────────────────────────────────

    def _log(self, action: str, details: dict) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action":    action,
            **details,
        }
        self.trade_log.append(entry)
        logger.info("DayTrader [%s] %s", action, details.get("symbol", ""))


# Singleton instance used by API routes
day_trader = DayTrader()
