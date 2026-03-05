"""Intraday Technical Signal Engine.

Computes three classic day-trading signals from 5-minute OHLCV bars:

  1. VWAP (Volume-Weighted Average Price)
     - Buy signal when price is above VWAP (momentum + market support)
     - Sell signal when price drops below VWAP (breakdown)

  2. EMA Crossover (9-period vs 21-period)
     - Bullish when short (9) EMA crosses above long (21) EMA
     - Bearish when short EMA crosses below long EMA

  3. ORB — Opening Range Breakout (first 30 minutes)
     - Buy signal when price breaks above the 30-min opening range high
     - Sell signal when price breaks below the 30-min opening range low

All three work on the bar list returned by ingestion.alpaca_bars.fetch_today_bars().
Minimum 6 bars (30 minutes of 5-min data) required before generating signals.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Minimum bars before generating any signal
MIN_BARS = 6               # 30 minutes at 5-min resolution
EMA_SHORT_PERIOD = 9       # fast EMA
EMA_LONG_PERIOD  = 21      # slow EMA
ORB_BARS         = 6       # first 6 bars = 30-minute opening range


# ── Primitive computations ─────────────────────────────────────────────

def compute_ema(prices: list[float], period: int) -> Optional[float]:
    """Exponential moving average of the last `period` closes.

    Uses the standard EMA formula:
        multiplier = 2 / (period + 1)
        EMA_t = close_t * multiplier + EMA_{t-1} * (1 - multiplier)

    Returns None if there are fewer than `period` prices.
    """
    if len(prices) < period:
        return None

    k = 2 / (period + 1)
    ema = prices[0]
    for p in prices[1:]:
        ema = p * k + ema * (1 - k)
    return round(ema, 4)


def compute_vwap(bars: list[dict]) -> Optional[float]:
    """Cumulative VWAP from the first bar (resets each trading day).

    VWAP = Σ(typical_price × volume) / Σ(volume)
    typical_price = (high + low + close) / 3

    Returns None if no bars have volume.
    """
    if not bars:
        return None

    cum_pv = 0.0
    cum_vol = 0.0
    for bar in bars:
        h  = float(bar.get("h", 0))
        l  = float(bar.get("l", 0))
        c  = float(bar.get("c", 0))
        v  = float(bar.get("v", 0))
        if v <= 0:
            continue
        typical = (h + l + c) / 3
        cum_pv  += typical * v
        cum_vol += v

    if cum_vol <= 0:
        return None
    return round(cum_pv / cum_vol, 4)


def compute_orb(bars: list[dict], orb_bars: int = ORB_BARS) -> Optional[dict]:
    """Compute the Opening Range (first `orb_bars` bars = 30 min by default).

    Returns:
        {"orb_high": float, "orb_low": float, "orb_range": float}
        or None if there aren't enough bars yet.
    """
    if len(bars) < orb_bars:
        return None

    opening_bars = bars[:orb_bars]
    highs = [float(b.get("h", 0)) for b in opening_bars]
    lows  = [float(b.get("l", float("inf"))) for b in opening_bars]

    orb_high = max(highs)
    orb_low  = min(lows)
    return {
        "orb_high":  round(orb_high, 4),
        "orb_low":   round(orb_low, 4),
        "orb_range": round(orb_high - orb_low, 4),
    }


# ── Combined signal generator ──────────────────────────────────────────

def generate_intraday_signal(symbol: str, bars: list[dict]) -> dict:
    """Combine VWAP, EMA crossover, and ORB into a single intraday signal.

    Args:
        symbol: Ticker (for logging only)
        bars:   Sorted oldest-first list of 5-min OHLCV bar dicts

    Returns a dict with:
        signal:       "BUY", "SELL", or "HOLD"
        score:        0–100 conviction score (50 = neutral)
        reasons:      list of human-readable signal descriptions
        vwap:         current VWAP
        ema_short:    9-period EMA
        ema_long:     21-period EMA
        ema_signal:   "bullish" | "bearish" | "neutral"
        orb_high:     opening range high
        orb_low:      opening range low
        current_price: last close price
        bar_count:    number of bars used
    """
    if len(bars) < MIN_BARS:
        return {
            "symbol":   symbol,
            "signal":   "HOLD",
            "score":    50,
            "reasons":  [f"⏳ Only {len(bars)} bars — need {MIN_BARS}+ (30 min of data)"],
            "bar_count": len(bars),
        }

    closes  = [float(b.get("c", 0)) for b in bars]
    current = closes[-1]

    if current <= 0:
        return {"symbol": symbol, "signal": "HOLD", "score": 50, "reasons": ["⚪ No price data"]}

    # ── Compute indicators ────────────────────────────────────────────
    vwap      = compute_vwap(bars)
    ema_short = compute_ema(closes, EMA_SHORT_PERIOD)
    ema_long  = compute_ema(closes, EMA_LONG_PERIOD)
    orb       = compute_orb(bars)

    score   = 50.0
    reasons = []

    # ── VWAP signal ───────────────────────────────────────────────────
    vwap_signal = "neutral"
    if vwap and current > 0:
        vwap_dist_pct = (current - vwap) / vwap * 100
        if vwap_dist_pct > 0.2:
            score += 12
            vwap_signal = "bullish"
            reasons.append(f"✅ Price {current:.2f} above VWAP {vwap:.2f} (+{vwap_dist_pct:.2f}%)")
        elif vwap_dist_pct > 0:
            score += 5
            vwap_signal = "bullish"
            reasons.append(f"✅ Slightly above VWAP ({vwap_dist_pct:+.2f}%)")
        elif vwap_dist_pct < -0.2:
            score -= 12
            vwap_signal = "bearish"
            reasons.append(f"🔴 Price {current:.2f} below VWAP {vwap:.2f} ({vwap_dist_pct:.2f}%)")
        else:
            reasons.append(f"⚪ Near VWAP ({vwap:.2f}, dist {vwap_dist_pct:+.2f}%)")

    # ── EMA crossover signal ──────────────────────────────────────────
    ema_signal = "neutral"
    if ema_short and ema_long:
        ema_gap_pct = (ema_short - ema_long) / ema_long * 100
        if ema_gap_pct > 0.15:
            score += 15
            ema_signal = "bullish"
            reasons.append(f"✅ EMA9 ({ema_short:.2f}) above EMA21 ({ema_long:.2f}) — uptrend")
        elif ema_gap_pct > 0:
            score += 7
            ema_signal = "bullish"
            reasons.append(f"✅ EMA9 crossing above EMA21 (+{ema_gap_pct:.2f}%)")
        elif ema_gap_pct < -0.15:
            score -= 15
            ema_signal = "bearish"
            reasons.append(f"🔴 EMA9 ({ema_short:.2f}) below EMA21 ({ema_long:.2f}) — downtrend")
        else:
            reasons.append(f"⚪ EMAs converging ({ema_gap_pct:+.2f}%)")

    # ── ORB signal ────────────────────────────────────────────────────
    orb_signal = "neutral"
    if orb:
        orb_high = orb["orb_high"]
        orb_low  = orb["orb_low"]
        if current > orb_high:
            breakout_pct = (current - orb_high) / orb_high * 100
            score += 15
            orb_signal = "bullish"
            reasons.append(f"✅ ORB breakout above ${orb_high:.2f} (+{breakout_pct:.2f}%)")
        elif current < orb_low:
            breakdown_pct = (orb_low - current) / orb_low * 100
            score -= 15
            orb_signal = "bearish"
            reasons.append(f"🔴 ORB breakdown below ${orb_low:.2f} (-{breakdown_pct:.2f}%)")
        else:
            pct_in_range = (current - orb_low) / max(orb["orb_range"], 0.01) * 100
            reasons.append(f"⚪ Inside opening range (${orb_low:.2f}–${orb_high:.2f}), {pct_in_range:.0f}% up in range")

    # ── Momentum: consecutive closes ──────────────────────────────────
    if len(closes) >= 3:
        last3 = closes[-3:]
        if all(last3[i] < last3[i+1] for i in range(len(last3)-1)):
            score += 8
            reasons.append("✅ 3 consecutive rising closes")
        elif all(last3[i] > last3[i+1] for i in range(len(last3)-1)):
            score -= 8
            reasons.append("🔴 3 consecutive falling closes")

    score = max(0.0, min(100.0, score))

    # ── Translate score to signal ─────────────────────────────────────
    # Use tighter thresholds than the prediction engine — intraday is faster-moving
    if score >= 70:
        signal = "BUY"
    elif score <= 35:
        signal = "SELL"
    else:
        signal = "HOLD"

    return {
        "symbol":        symbol,
        "signal":        signal,
        "score":         round(score, 1),
        "reasons":       reasons,
        "current_price": current,
        "vwap":          vwap,
        "ema_short":     ema_short,
        "ema_long":      ema_long,
        "ema_signal":    ema_signal,
        "vwap_signal":   vwap_signal,
        "orb_signal":    orb_signal,
        "orb_high":      orb["orb_high"] if orb else None,
        "orb_low":       orb["orb_low"] if orb else None,
        "bar_count":     len(bars),
    }
