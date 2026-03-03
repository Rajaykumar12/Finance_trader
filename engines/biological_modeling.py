"""Module B — Asset Biological Modeling Engine.

Combines multiple signals into a unified 0-100 Asset Health Index (AHI):
  1. **Sentiment** — FinBERT-scored news headlines  (weight: 25%)
  2. **Fundamentals** — yfinance balance sheet ratios  (weight: 25%)
  3. **Market** — real-time Market State Vector from Module A  (weight: 35%)
  4. **Macro** — macroeconomic context  (weight: 15%)

Also computes the **Ecosystem Competitive Map** — a relative positioning
of all tracked assets within their sector.

Results are written to Redis hashes:
  - ``health_index:<SYMBOL>``
  - ``ecosystem_map``
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from state.redis_client import redis_client

logger = logging.getLogger(__name__)

# ── FinBERT (loaded lazily) ───────────────────────────────────────────
_sentiment_pipeline = None


def _get_sentiment_pipeline():
    """Load ProsusAI/finbert on first call."""
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        logger.info("Loading FinBERT model (first time only) …")
        from transformers import pipeline as hf_pipeline

        _sentiment_pipeline = hf_pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            top_k=None,
            truncation=True,
        )
        logger.info("FinBERT loaded ✓")
    return _sentiment_pipeline


def score_sentiment(headline: str) -> float:
    """Return a sentiment score in [-1, +1] for a headline."""
    pipe = _get_sentiment_pipeline()
    results = pipe(headline)[0]
    scores = {r["label"]: r["score"] for r in results}
    return round(scores.get("positive", 0) - scores.get("negative", 0), 4)


# ── Fundamental scoring ───────────────────────────────────────────────

def _compute_fundamental_score(data: dict) -> float:
    """Convert raw fundamentals into a 0-100 score.

    Biological mapping:
      - Cash flow = Energy flow  → +20
      - Cash > Debt = Low structural stress  → +20
      - P/E in healthy range  → +20
      - Positive profit margin  → +20
      - Market cap stability (exists and > 0)  → +20
    """
    score = 0.0

    fcf = data.get("free_cash_flow")
    if fcf is not None and float(fcf) > 0:
        score += 20

    cash = data.get("total_cash")
    debt = data.get("total_debt")
    if cash is not None and debt is not None:
        if float(cash) > float(debt):
            score += 20

    pe = data.get("pe_ratio")
    if pe is not None:
        pe_val = float(pe)
        if 5 <= pe_val <= 30:
            score += 20

    margin = data.get("profit_margin")
    if margin is not None and float(margin) > 0:
        score += 20

    mcap = data.get("market_cap")
    if mcap is not None and float(mcap) > 0:
        score += 20

    return score


# ── Market scoring ────────────────────────────────────────────────────

def _compute_market_score(state: dict) -> float:
    """Convert Market State Vector into a 0-100 score.

    Biological mapping:
      - Volatility = Nervous system activity
      - Order flow = Blood circulation
      - Liquidity = Vascular health
    """
    score = 50.0

    # Spread variance (vascular health)
    spread_var = float(state.get("bid_ask_spread_var", 0))
    if spread_var < 0.0001:
        score += 20
    elif spread_var < 0.01:
        score += 10
    else:
        score -= 15

    # OBI (blood circulation balance)
    obi = abs(float(state.get("order_book_imbalance", 0)))
    if obi < 0.1:
        score += 20
    elif obi < 0.3:
        score += 10
    else:
        score -= 10

    # Volatility regime (nervous system)
    regime = state.get("volatility_regime", "normal")
    if regime == "low":
        score += 10
    elif regime == "normal":
        score += 5
    elif regime == "high":
        score -= 10
    elif regime == "crisis":
        score -= 25

    return max(0.0, min(100.0, score))


# ── Macro scoring ─────────────────────────────────────────────────────

def _compute_macro_score(macro: dict) -> float:
    """Convert macroeconomic indicators into a 0-100 score.

    Higher score = more favorable macro environment.
    """
    if not macro:
        return 50.0  # Neutral if no data

    score = 50.0

    # Federal Funds Rate — lower is generally more stimulative
    ffr = macro.get("fed_funds_rate")
    if ffr is not None:
        ffr_val = float(ffr)
        if ffr_val < 2.0:
            score += 15
        elif ffr_val < 4.0:
            score += 5
        elif ffr_val > 5.0:
            score -= 15

    # Unemployment — lower is better
    unemp = macro.get("unemployment_rate")
    if unemp is not None:
        unemp_val = float(unemp)
        if unemp_val < 4.0:
            score += 15
        elif unemp_val < 6.0:
            score += 5
        else:
            score -= 10

    # CPI (inflation) — moderate is best
    cpi = macro.get("cpi_yoy")
    if cpi is not None:
        cpi_val = float(cpi)
        if 1.0 <= cpi_val <= 3.0:
            score += 15
        elif cpi_val > 6.0:
            score -= 20
        elif cpi_val < 0:
            score -= 10  # Deflation is also bad

    # VIX — lower is calmer
    vix = macro.get("vix")
    if vix is not None:
        vix_val = float(vix)
        if vix_val < 15:
            score += 10
        elif vix_val < 25:
            score += 0
        elif vix_val > 30:
            score -= 15

    return max(0.0, min(100.0, score))


# ── Health index computation ──────────────────────────────────────────

SENTIMENT_WEIGHT = 0.25
FUNDAMENTAL_WEIGHT = 0.25
MARKET_WEIGHT = 0.35
MACRO_WEIGHT = 0.15
STRESS_THRESHOLD = 50.0

_sentiment_accum: dict[str, list[float]] = {}


async def compute_health_index(symbol: str) -> dict | None:
    """Pull all signals from Redis and compute the AHI for *symbol*."""

    # 1. Sentiment
    sentiments = _sentiment_accum.get(symbol, [])
    if sentiments:
        avg_sentiment = sum(sentiments) / len(sentiments)
        sentiment_score_0_100 = (avg_sentiment + 1) / 2 * 100
    else:
        avg_sentiment = 0.0
        sentiment_score_0_100 = 50.0

    # 2. Fundamentals
    fund_data = await redis_client.hgetall(f"fundamentals:{symbol}")
    fundamental_score = _compute_fundamental_score(fund_data) if fund_data else 50.0

    # 3. Market State
    market_data = await redis_client.hgetall(f"market_state:{symbol}")
    market_score = _compute_market_score(market_data) if market_data else 50.0

    # 4. Macro (shared across all symbols)
    macro_data = await redis_client.hgetall("macro_indicators")
    macro_score = _compute_macro_score(macro_data)

    # Weighted combination
    health_index = (
        SENTIMENT_WEIGHT * sentiment_score_0_100
        + FUNDAMENTAL_WEIGHT * fundamental_score
        + MARKET_WEIGHT * market_score
        + MACRO_WEIGHT * macro_score
    )
    health_index = round(max(0, min(100, health_index)), 2)

    stress_alert = health_index < STRESS_THRESHOLD

    return {
        "symbol": symbol,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "health_index": health_index,
        "sentiment_score": round(avg_sentiment, 4),
        "fundamental_score": round(fundamental_score, 2),
        "market_score": round(market_score, 2),
        "macro_score": round(macro_score, 2),
        "metabolic_stress_alert": stress_alert,
    }


# ── Ecosystem Competitive Map ─────────────────────────────────────────

async def compute_ecosystem_map(symbols: list[str]) -> dict:
    """Compare all tracked assets and rank them by health + positioning.

    Outputs a competitive map:
      - Relative health ranking
      - Per-symbol breakdown of strengths/weaknesses
      - Sector health average
    """
    entries = []

    for sym in symbols:
        health = await redis_client.hgetall(f"health_index:{sym}")
        fund = await redis_client.hgetall(f"fundamentals:{sym}")
        market = await redis_client.hgetall(f"market_state:{sym}")

        hi = float(health.get("health_index", 50)) if health else 50.0
        mcap = float(fund.get("market_cap", 0)) if fund else 0
        vol_regime = market.get("volatility_regime", "unknown") if market else "unknown"
        liq_status = market.get("liquidity_status", "unknown") if market else "unknown"

        # Strength / weakness analysis
        strengths = []
        weaknesses = []

        sentiment = float(health.get("sentiment_score", 0)) if health else 0
        if sentiment > 0.2:
            strengths.append("positive_narrative")
        elif sentiment < -0.2:
            weaknesses.append("negative_narrative")

        fund_score = float(health.get("fundamental_score", 50)) if health else 50
        if fund_score >= 80:
            strengths.append("strong_fundamentals")
        elif fund_score <= 20:
            weaknesses.append("weak_fundamentals")

        if vol_regime in ("low", "normal"):
            strengths.append("stable_volatility")
        elif vol_regime == "crisis":
            weaknesses.append("crisis_volatility")

        if liq_status == "stable":
            strengths.append("healthy_liquidity")
        elif liq_status == "vanishing":
            weaknesses.append("liquidity_crisis")

        entries.append({
            "symbol": sym,
            "health_index": hi,
            "market_cap": mcap,
            "volatility_regime": vol_regime,
            "liquidity_status": liq_status,
            "strengths": strengths,
            "weaknesses": weaknesses,
        })

    # Rank by health index
    entries.sort(key=lambda e: e["health_index"], reverse=True)
    for rank, entry in enumerate(entries, 1):
        entry["rank"] = rank

    avg_health = sum(e["health_index"] for e in entries) / len(entries) if entries else 50

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sector_average_health": round(avg_health, 2),
        "total_assets": len(entries),
        "rankings": entries,
    }


# ── Background workers ────────────────────────────────────────────────

async def _news_listener() -> None:
    """Subscribe to ``news_feed`` and compute sentiment."""
    async for hl in redis_client.subscribe("news_feed"):
        headline = hl.get("headline", "")
        symbol = hl.get("symbol")
        if not headline:
            continue

        try:
            score = score_sentiment(headline)
            logger.debug("Sentiment: %.4f  |  %s", score, headline[:80])

            from config.settings import settings
            targets = [symbol] if symbol else settings.equity_symbol_list
            for sym in targets:
                buf = _sentiment_accum.setdefault(sym, [])
                buf.append(score)
                if len(buf) > 50:
                    buf.pop(0)
        except Exception:
            logger.exception("Sentiment scoring error for: %s", headline[:80])


async def _health_publisher(symbols: list[str]) -> None:
    """Periodically recompute and publish AHI + ecosystem map."""
    while True:
        for sym in symbols:
            try:
                result = await compute_health_index(sym)
                if result:
                    await redis_client.hset(f"health_index:{sym}", result)
                    status = "🔴 STRESS" if result["metabolic_stress_alert"] else "🟢 OK"
                    logger.debug("AHI %s = %.1f  %s", sym, result["health_index"], status)
            except Exception:
                logger.exception("Health index error for %s", sym)

        # Ecosystem map (computed after all AHIs are updated)
        try:
            eco_map = await compute_ecosystem_map(symbols)
            import json
            await redis_client.hset("ecosystem_map", {
                "data": json.dumps(eco_map, default=str),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            })
        except Exception:
            logger.exception("Ecosystem map error")

        await asyncio.sleep(2.0)


async def run_biological_modeling(symbols: list[str] | None = None) -> None:
    """Entry point."""
    from config.settings import settings

    if symbols is None:
        symbols = settings.equity_symbol_list

    await redis_client.connect()
    logger.info("Biological Modeling Engine starting for %s …", symbols)

    await asyncio.gather(
        _news_listener(),
        _health_publisher(symbols),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_biological_modeling())
