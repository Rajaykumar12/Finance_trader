"""Module A — Real-Time Market Intelligence Engine.

Subscribes to ``market_ticks:*`` Redis channels, maintains a rolling buffer
of recent ticks per symbol, and computes:
  1. **Market State Vector** — volatility, spread stats, OBI
  2. **Volatility Regime** — classifies market into low/normal/high/crisis
  3. **Liquidity Shift Detection** — detects when liquidity is vanishing
  4. **Cross-Market Correlations** — pairwise rolling correlations

Results are written to Redis hashes:
  - ``market_state:<SYMBOL>``
  - ``cross_correlations``
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from datetime import datetime, timezone

import numpy as np

try:
    import polars as pl
except ImportError:
    pl = None

from state.redis_client import redis_client

logger = logging.getLogger(__name__)

BUFFER_SIZE = 1_000
COMPUTE_INTERVAL = 1.0  # seconds
CORRELATION_INTERVAL = 5.0  # seconds (cross-correlations are heavier)

# ── Volatility Regime thresholds ──────────────────────────────────────
# These are calibrated for crypto (BTC) tick-level log-return std.
# For equities the thresholds would be lower.
REGIME_THRESHOLDS = {
    "low":    0.00005,   # Very calm market
    "normal": 0.0003,    # Typical conditions
    "high":   0.001,     # Elevated volatility
    # Anything above "high" → "crisis"
}

# Liquidity shift: if spread variance increases by this factor
# vs the previous window, flag a liquidity shift.
LIQUIDITY_SHIFT_FACTOR = 3.0


class MarketIntelligenceEngine:
    """Continuously processes market ticks into intelligence outputs."""

    def __init__(self) -> None:
        self._buffers: dict[str, deque[dict]] = {}
        # Store recent mid-price series for correlation computation
        self._mid_price_series: dict[str, deque[float]] = {}
        # Track previous spread variance for liquidity shift detection
        self._prev_spread_var: dict[str, float] = {}

    def _ensure_buffer(self, symbol: str) -> deque:
        if symbol not in self._buffers:
            self._buffers[symbol] = deque(maxlen=BUFFER_SIZE)
            self._mid_price_series[symbol] = deque(maxlen=BUFFER_SIZE)
        return self._buffers[symbol]

    def ingest_tick(self, tick: dict) -> None:
        """Add a normalised tick to the rolling buffer."""
        symbol = tick.get("symbol", "UNKNOWN")
        buf = self._ensure_buffer(symbol)
        buf.append(tick)

        # Also track mid prices for correlation
        bid = tick.get("best_bid", 0)
        ask = tick.get("best_ask", 0)
        if bid > 0 and ask > 0:
            self._mid_price_series[symbol].append((bid + ask) / 2)

    def _classify_volatility_regime(self, rolling_vol: float) -> str:
        """Classify volatility into a regime label."""
        if rolling_vol < REGIME_THRESHOLDS["low"]:
            return "low"
        elif rolling_vol < REGIME_THRESHOLDS["normal"]:
            return "normal"
        elif rolling_vol < REGIME_THRESHOLDS["high"]:
            return "high"
        else:
            return "crisis"

    def _detect_liquidity_shift(self, symbol: str, spread_var: float) -> dict:
        """Detect if a liquidity shift is occurring.

        Returns a dict with shift info:
          - liquidity_shift: bool
          - shift_magnitude: ratio of current vs previous spread variance
          - liquidity_status: "stable" | "thinning" | "vanishing"
        """
        prev = self._prev_spread_var.get(symbol, spread_var)
        self._prev_spread_var[symbol] = spread_var

        if prev > 0:
            magnitude = spread_var / prev
        else:
            magnitude = 1.0

        if magnitude > LIQUIDITY_SHIFT_FACTOR * 2:
            status = "vanishing"
            shift = True
        elif magnitude > LIQUIDITY_SHIFT_FACTOR:
            status = "thinning"
            shift = True
        else:
            status = "stable"
            shift = False

        return {
            "liquidity_shift": shift,
            "shift_magnitude": round(magnitude, 4),
            "liquidity_status": status,
        }

    def compute_state_vector(self, symbol: str) -> dict | None:
        """Compute the full Market State Vector for *symbol*."""
        buf = self._buffers.get(symbol)
        if not buf or len(buf) < 5:
            return None

        ticks = list(buf)

        if pl is not None:
            df = pl.DataFrame({
                "best_bid": [t["best_bid"] for t in ticks],
                "best_ask": [t["best_ask"] for t in ticks],
                "bid_qty": [t["bid_qty"] for t in ticks],
                "ask_qty": [t["ask_qty"] for t in ticks],
                "spread": [t["spread"] for t in ticks],
            })
            mid_prices = ((df["best_bid"] + df["best_ask"]) / 2).to_numpy()
            spreads = df["spread"].to_numpy()
            bid_qty_arr = df["bid_qty"].to_numpy()
            ask_qty_arr = df["ask_qty"].to_numpy()
        else:
            mid_prices = np.array([(t["best_bid"] + t["best_ask"]) / 2 for t in ticks])
            spreads = np.array([t["spread"] for t in ticks])
            bid_qty_arr = np.array([t["bid_qty"] for t in ticks])
            ask_qty_arr = np.array([t["ask_qty"] for t in ticks])

        # ── Rolling volatility ────────────────────────────────────────
        if len(mid_prices) > 1 and np.all(mid_prices > 0):
            log_returns = np.diff(np.log(mid_prices))
            rolling_vol = float(np.std(log_returns))
        else:
            rolling_vol = 0.0

        # ── Spread statistics ─────────────────────────────────────────
        spread_mean = float(np.mean(spreads))
        spread_var = float(np.var(spreads))

        # ── Order book imbalance ──────────────────────────────────────
        total_bid = float(np.sum(bid_qty_arr))
        total_ask = float(np.sum(ask_qty_arr))
        obi = (total_bid - total_ask) / (total_bid + total_ask) if (total_bid + total_ask) > 0 else 0.0

        # ── NEW: Volatility regime classification ─────────────────────
        vol_regime = self._classify_volatility_regime(rolling_vol)

        # ── NEW: Liquidity shift detection ────────────────────────────
        liq = self._detect_liquidity_shift(symbol, spread_var)

        return {
            "symbol": symbol,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "rolling_volatility": round(rolling_vol, 8),
            "bid_ask_spread_mean": round(spread_mean, 6),
            "bid_ask_spread_var": round(spread_var, 10),
            "order_book_imbalance": round(obi, 6),
            "tick_count": len(ticks),
            # New fields
            "volatility_regime": vol_regime,
            "liquidity_shift": liq["liquidity_shift"],
            "shift_magnitude": liq["shift_magnitude"],
            "liquidity_status": liq["liquidity_status"],
        }

    def compute_cross_correlations(self) -> dict | None:
        """Compute pairwise rolling correlations across all tracked symbols.

        Uses the last 200 mid-price log returns for each symbol pair.
        Returns a dict like:
          {"BTC_AAPL": 0.35, "BTC_NVDA": 0.12, "AAPL_NVDA": 0.85}
        """
        symbols = [s for s, series in self._mid_price_series.items() if len(series) >= 30]
        if len(symbols) < 2:
            return None

        # Build log-return arrays (aligned by length)
        min_len = min(len(self._mid_price_series[s]) for s in symbols)
        min_len = min(min_len, 200)  # Cap at 200

        returns: dict[str, np.ndarray] = {}
        for sym in symbols:
            prices = list(self._mid_price_series[sym])[-min_len:]
            arr = np.array(prices)
            if np.all(arr > 0) and len(arr) > 1:
                returns[sym] = np.diff(np.log(arr))

        if len(returns) < 2:
            return None

        # Pairwise correlations
        corr_result: dict[str, float] = {}
        sym_list = list(returns.keys())
        for i in range(len(sym_list)):
            for j in range(i + 1, len(sym_list)):
                s1, s2 = sym_list[i], sym_list[j]
                r1, r2 = returns[s1], returns[s2]
                # Align lengths
                n = min(len(r1), len(r2))
                if n > 5:
                    corr = float(np.corrcoef(r1[-n:], r2[-n:])[0, 1])
                    if np.isnan(corr):
                        corr = 0.0
                    corr_result[f"{s1}_{s2}"] = round(corr, 4)

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pairs": corr_result,
            "symbols": sym_list,
            "window_size": min_len,
        }


# ── Background workers ────────────────────────────────────────────────

async def _tick_listener(engine: MarketIntelligenceEngine, symbols: list[str]) -> None:
    """Subscribe to market tick channels and feed the engine."""
    channels = [f"market_ticks:{s}" for s in symbols]
    async for tick in redis_client.subscribe(*channels):
        engine.ingest_tick(tick)


async def _state_publisher(engine: MarketIntelligenceEngine, symbols: list[str]) -> None:
    """Periodically compute and publish the Market State Vector."""
    while True:
        for sym in symbols:
            state = engine.compute_state_vector(sym)
            if state:
                await redis_client.hset(f"market_state:{sym}", state)
                logger.debug(
                    "market_state:%s  regime=%s  liq=%s  (ticks=%d)",
                    sym, state["volatility_regime"], state["liquidity_status"],
                    state["tick_count"],
                )
        await asyncio.sleep(COMPUTE_INTERVAL)


async def _correlation_publisher(engine: MarketIntelligenceEngine) -> None:
    """Periodically compute and publish cross-market correlations."""
    while True:
        corr = engine.compute_cross_correlations()
        if corr:
            await redis_client.hset("cross_correlations", corr)
            logger.debug("Cross-correlations updated: %s", list(corr["pairs"].keys()))
        await asyncio.sleep(CORRELATION_INTERVAL)


async def run_market_intelligence(symbols: list[str] | None = None) -> None:
    """Entry point — starts listener + publisher + correlation as concurrent tasks."""
    from config.settings import settings

    if symbols is None:
        symbols = settings.crypto_symbol_list + settings.equity_symbol_list

    await redis_client.connect()
    logger.info("Market Intelligence Engine starting for %s …", symbols)

    engine = MarketIntelligenceEngine()
    await asyncio.gather(
        _tick_listener(engine, symbols),
        _state_publisher(engine, symbols),
        _correlation_publisher(engine),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_market_intelligence())
