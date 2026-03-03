"""Prediction Engine — combines all platform signals into a trading decision.

Aggregates:
  - Fundamentals (health score, P/E, cash flow, debt)
  - Sentiment (FinBERT news scoring)
  - Market microstructure (volatility regime, liquidity)
  - Derivatives intelligence (GEX, vol surface skew, convexity risk)
  - Macroeconomic context (VIX, rates, DXY)
  - Cross-market correlations

Outputs: BUY / HOLD / SELL with confidence score and reasoning.
"""

from __future__ import annotations

import asyncio
import logging
import math
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


# ── Signal weights (sum to 1.0) ──────────────────────────────────────

WEIGHTS = {
    "fundamentals": 0.25,
    "sentiment": 0.10,
    "market_health": 0.15,
    "options_flow": 0.20,
    "macro": 0.15,
    "momentum": 0.15,
}


def _safe_float(val, default=0.0):
    """Safely convert to float, handling None/NaN."""
    if val is None:
        return default
    try:
        v = float(val)
        return default if (math.isnan(v) or math.isinf(v)) else v
    except (ValueError, TypeError):
        return default


def _score_fundamentals(data: dict) -> tuple[float, list[str]]:
    """Score fundamentals from 0 (bearish) to 100 (bullish)."""
    score = 50.0
    reasons = []

    fcf = _safe_float(data.get("free_cash_flow"))
    if fcf > 0:
        score += 10
        reasons.append("✅ Positive free cash flow")
    elif fcf < 0:
        score -= 15
        reasons.append("🔴 Negative cash flow — burning cash")

    debt = _safe_float(data.get("total_debt"))
    cash = _safe_float(data.get("total_cash"))
    if cash > debt and debt > 0:
        score += 10
        reasons.append("✅ More cash than debt")
    elif debt > cash * 3 and cash > 0:
        score -= 15
        reasons.append("🔴 Debt is 3x+ cash — heavy leverage")
    elif debt > cash and cash > 0:
        score -= 5
        reasons.append("⚠️ Debt exceeds cash reserves")

    pe = _safe_float(data.get("pe_ratio"))
    if pe > 0:
        if pe < 15:
            score += 10
            reasons.append(f"✅ Low P/E ({pe:.1f}) — undervalued")
        elif pe <= 30:
            score += 5
            reasons.append(f"✅ Fair P/E ({pe:.1f})")
        elif pe <= 60:
            score -= 5
            reasons.append(f"⚠️ High P/E ({pe:.1f}) — pricey")
        else:
            score -= 10
            reasons.append(f"🔴 Very high P/E ({pe:.1f}) — overvalued risk")

    margin = _safe_float(data.get("profit_margin"))
    if margin > 0.20:
        score += 10
        reasons.append(f"✅ Strong profit margin ({margin:.0%})")
    elif margin > 0.05:
        score += 5
        reasons.append(f"✅ Decent margin ({margin:.0%})")
    elif margin > 0:
        reasons.append(f"⚠️ Thin margin ({margin:.0%})")
    elif margin < 0:
        score -= 10
        reasons.append(f"🔴 Losing money (margin {margin:.0%})")

    rev_growth = _safe_float(data.get("revenue_growth"))
    if rev_growth > 0.15:
        score += 10
        reasons.append(f"✅ Strong revenue growth ({rev_growth:.0%})")
    elif rev_growth > 0:
        score += 5
        reasons.append(f"✅ Revenue growing ({rev_growth:.0%})")
    elif rev_growth < -0.05:
        score -= 10
        reasons.append(f"🔴 Revenue declining ({rev_growth:.0%})")

    return max(0, min(100, score)), reasons


def _score_sentiment(sentiment_score: float | None) -> tuple[float, list[str]]:
    """Score sentiment from 0 (bearish) to 100 (bullish)."""
    if sentiment_score is None:
        return 50.0, ["⚪ No sentiment data available"]

    s = _safe_float(sentiment_score)
    # Map -1..+1 to 0..100
    score = (s + 1) * 50
    reasons = []

    if s > 0.5:
        reasons.append(f"✅ Very positive news sentiment ({s:+.2f})")
    elif s > 0.2:
        reasons.append(f"✅ Positive news sentiment ({s:+.2f})")
    elif s > -0.2:
        reasons.append(f"⚪ Neutral news sentiment ({s:+.2f})")
    elif s > -0.5:
        reasons.append(f"🔴 Negative news sentiment ({s:+.2f})")
    else:
        reasons.append(f"🔴 Very negative news coverage ({s:+.2f})")

    return max(0, min(100, score)), reasons


def _score_market_health(market_data: dict) -> tuple[float, list[str]]:
    """Score market microstructure from 0 (dangerous) to 100 (healthy)."""
    score = 50.0
    reasons = []

    vol_regime = market_data.get("volatility_regime", "unknown")
    regime_scores = {"low": 90, "normal": 70, "high": 35, "crisis": 10}
    score = regime_scores.get(vol_regime, 50)

    regime_labels = {
        "low": "✅ Low volatility — calm market",
        "normal": "✅ Normal volatility",
        "high": "⚠️ High volatility — increased risk",
        "crisis": "🔴 CRISIS volatility — extreme risk",
    }
    reasons.append(regime_labels.get(vol_regime, f"⚪ Volatility regime: {vol_regime}"))

    liq = market_data.get("liquidity_status", "unknown")
    if liq == "stable":
        score += 10
        reasons.append("✅ Liquidity is stable")
    elif liq == "thinning":
        score -= 15
        reasons.append("⚠️ Liquidity thinning — spreads widening")
    elif liq == "vanishing":
        score -= 30
        reasons.append("🔴 Liquidity vanishing — danger of flash crash")

    return max(0, min(100, score)), reasons


def _score_options(options_data: dict) -> tuple[float, list[str]]:
    """Score derivatives signals from 0 (bearish) to 100 (bullish)."""
    score = 50.0
    reasons = []

    if not options_data or options_data.get("unavailable"):
        return 50.0, ["⚪ Options data unavailable for this ticker"]

    # Vol surface skew
    skew_status = options_data.get("skew_status", "normal")
    if skew_status == "normal":
        score += 15
        reasons.append("✅ Options skew is normal — no fear in the market")
    elif skew_status == "moderate_skew":
        score -= 5
        reasons.append("⚠️ Moderate options skew — some hedging activity")
    elif skew_status == "severe_skew":
        score -= 20
        reasons.append("🔴 Severe skew — heavy put buying (fear of crash)")

    # Convexity risk
    risk_level = options_data.get("convexity_risk", "low")
    risk_scores = {"low": 15, "moderate": 0, "elevated": -15, "critical": -30}
    score += risk_scores.get(risk_level, 0)
    risk_labels = {
        "low": "✅ Low convexity risk — stable options positioning",
        "moderate": "⚪ Moderate convexity risk",
        "elevated": "⚠️ Elevated convexity risk — big move possible",
        "critical": "🔴 CRITICAL convexity risk — explosive move imminent",
    }
    reasons.append(risk_labels.get(risk_level, f"⚪ Convexity risk: {risk_level}"))

    # GEX direction
    total_gex = _safe_float(options_data.get("total_gex"))
    if total_gex > 0:
        score += 10
        reasons.append("✅ Positive gamma — dealers stabilize price (support)")
    elif total_gex < 0:
        score -= 10
        reasons.append("🔴 Negative gamma — dealers amplify moves (volatile)")

    # ATM IV
    atm_iv = _safe_float(options_data.get("atm_iv"))
    if atm_iv > 0:
        if atm_iv < 0.20:
            score += 5
            reasons.append(f"✅ Low implied volatility ({atm_iv:.0%}) — cheap options")
        elif atm_iv > 0.50:
            score -= 10
            reasons.append(f"🔴 Very high IV ({atm_iv:.0%}) — market expects big move")

    return max(0, min(100, score)), reasons


def _score_macro(macro: dict) -> tuple[float, list[str]]:
    """Score macroeconomic environment from 0 (bearish) to 100 (bullish)."""
    score = 50.0
    reasons = []

    if not macro:
        return 50.0, ["⚪ Macro data not yet available (start the platform)"]

    vix = _safe_float(macro.get("vix"))
    if vix > 0:
        if vix < 15:
            score += 15
            reasons.append(f"✅ VIX is low ({vix:.1f}) — market calm, favorable")
        elif vix < 20:
            score += 5
            reasons.append(f"✅ VIX normal ({vix:.1f})")
        elif vix < 30:
            score -= 10
            reasons.append(f"⚠️ VIX elevated ({vix:.1f}) — market nervous")
        else:
            score -= 25
            reasons.append(f"🔴 VIX spike ({vix:.1f}) — FEAR in markets")

    sp500 = _safe_float(macro.get("sp500_price"))
    sp500_prev = _safe_float(macro.get("sp500_prev"))
    if sp500 > 0 and sp500_prev > 0:
        sp_change = (sp500 - sp500_prev) / sp500_prev
        if sp_change > 0.005:
            score += 5
            reasons.append(f"✅ S&P 500 rising ({sp_change:+.1%})")
        elif sp_change < -0.01:
            score -= 10
            reasons.append(f"🔴 S&P 500 falling ({sp_change:+.1%})")

    dxy = _safe_float(macro.get("dxy"))
    if dxy > 105:
        score -= 5
        reasons.append("⚠️ Strong dollar — headwind for stocks")
    elif dxy > 0 and dxy < 95:
        score += 5
        reasons.append("✅ Weak dollar — tailwind for stocks")

    return max(0, min(100, score)), reasons


def _score_momentum(price_data: dict) -> tuple[float, list[str]]:
    """Score price momentum from 0 (bearish) to 100 (bullish)."""
    score = 50.0
    reasons = []

    current = _safe_float(price_data.get("current"))
    prev_close = _safe_float(price_data.get("previous_close"))
    high_52w = _safe_float(price_data.get("fifty_two_week_high"))
    low_52w = _safe_float(price_data.get("fifty_two_week_low"))

    # Daily change
    if current > 0 and prev_close > 0:
        daily_chg = (current - prev_close) / prev_close
        if daily_chg > 0.02:
            score += 15
            reasons.append(f"✅ Strong daily move ({daily_chg:+.1%})")
        elif daily_chg > 0:
            score += 5
            reasons.append(f"✅ Price up today ({daily_chg:+.1%})")
        elif daily_chg < -0.03:
            score -= 15
            reasons.append(f"🔴 Sharp drop today ({daily_chg:+.1%})")
        elif daily_chg < 0:
            score -= 5
            reasons.append(f"⚠️ Price down today ({daily_chg:+.1%})")

    # 52-week position
    if current > 0 and high_52w > 0 and low_52w > 0 and high_52w > low_52w:
        position = (current - low_52w) / (high_52w - low_52w)
        if position > 0.9:
            score += 5
            reasons.append(f"✅ Near 52-week high ({position:.0%} of range)")
        elif position < 0.2:
            score -= 5
            reasons.append(f"⚠️ Near 52-week low ({position:.0%} of range) — could be opportunity or red flag")
        else:
            reasons.append(f"⚪ At {position:.0%} of 52-week range")

    return max(0, min(100, score)), reasons


async def predict(symbol: str) -> dict:
    """Generate a trading prediction by combining all platform signals."""
    from engines.lookup import lookup_symbol
    from state.redis_client import redis_client

    # ── Fetch all data ────────────────────────────────────────────
    lookup = await lookup_symbol(symbol)

    if "error" in lookup:
        return {"error": lookup["error"], "symbol": symbol.upper()}

    # Read cached data from Redis
    try:
        market_data = await redis_client.hgetall(f"market_state:{symbol.upper()}")
    except Exception:
        market_data = {}

    try:
        macro = await redis_client.hgetall("macro_indicators")
    except Exception:
        macro = {}

    # ── Sentiment scoring ─────────────────────────────────────────
    # Strategy: 1) Check Redis for cached sentiment (from biological engine)
    #           2) Fall back to yfinance headlines + FinBERT scoring
    sentiment_val = None
    sentiment_headlines = []
    loop = asyncio.get_running_loop()

    # 1) Try Redis cached sentiment (updated by biological modeling engine)
    try:
        health_data = await redis_client.hgetall(f"health_index:{symbol.upper()}")
        cached_sent = _safe_float(health_data.get("sentiment_score"))
        if cached_sent != 0.0 and health_data:
            sentiment_val = cached_sent
            sentiment_headlines = [{"source": "cached", "note": "Pre-computed by platform's biological modeling engine"}]
            logger.info("Using cached sentiment for %s: %.3f", symbol, cached_sent)
    except Exception:
        pass

    # 2) If no cached sentiment, fetch live headlines from yfinance + score with FinBERT
    if sentiment_val is None:
        def _fetch_and_score_sentiment():
            """Fetch headlines from yfinance and score with FinBERT."""
            try:
                import yfinance as yf
                ticker_obj = yf.Ticker(symbol)
                info = ticker_obj.info or {}
                news = ticker_obj.news or []
                if not news:
                    return None, []

                # Build a set of keywords to filter relevant headlines
                company_name = info.get("shortName", "") or info.get("longName", "")
                filter_terms = {symbol.upper().split(".")[0]}  # e.g. "AAPL", "RELIANCE"
                if company_name:
                    # Add company name words (e.g. "Apple", "Tesla", "Reliance")
                    for word in company_name.split():
                        if len(word) > 2 and word.lower() not in ("inc", "inc.", "ltd", "ltd.", "corp", "corp.", "the", "and", "llc"):
                            filter_terms.add(word.lower())

                # yfinance returns news as [{"id": ..., "content": {"title": ...}}]
                all_headlines = []
                for item in news[:20]:  # scan more headlines to find relevant ones
                    title = None
                    if isinstance(item, dict):
                        content = item.get("content", {})
                        if isinstance(content, dict):
                            title = content.get("title")
                        if not title:
                            title = item.get("title")
                    if title:
                        all_headlines.append(title)

                # Filter: keep only headlines mentioning the company
                relevant = [h for h in all_headlines if any(t in h.lower() for t in filter_terms)]

                # If no relevant headlines found, use first few as fallback
                headlines = relevant[:8] if relevant else all_headlines[:4]

                if not headlines:
                    return None, []

                try:
                    from transformers import pipeline
                    pipe = pipeline("sentiment-analysis", model="ProsusAI/finbert",
                                    device=-1, top_k=None)
                    scores = []
                    scored_headlines = []
                    for h in headlines:
                        result = pipe(h[:512])
                        if result and isinstance(result[0], list):
                            result = result[0]
                        score_map = {r["label"]: r["score"] for r in result}
                        s = score_map.get("positive", 0) - score_map.get("negative", 0)
                        scores.append(s)
                        label = "positive" if s > 0.3 else "negative" if s < -0.3 else "neutral"
                        scored_headlines.append({"headline": h, "score": round(s, 3), "label": label})
                    avg = sum(scores) / len(scores) if scores else 0
                    return avg, scored_headlines
                except Exception as e:
                    logger.warning("FinBERT unavailable: %s", e)
                    return None, []
            except Exception as e:
                logger.warning("Failed to fetch news for %s: %s", symbol, e)
                return None, []

        try:
            sentiment_val, sentiment_headlines = await loop.run_in_executor(None, _fetch_and_score_sentiment)
        except Exception:
            sentiment_val = None

    # ── Score each dimension ──────────────────────────────────────
    fund_score, fund_reasons = _score_fundamentals(lookup.get("fundamentals", {}))
    sent_score, sent_reasons = _score_sentiment(sentiment_val)
    market_score, market_reasons = _score_market_health(market_data)
    momentum_score, momentum_reasons = _score_momentum(lookup.get("price", {}))
    macro_score, macro_reasons = _score_macro(macro)

    # Options scoring — combine GEX, vol surface, convexity
    options_combined = {}
    if lookup.get("vol_surface"):
        options_combined["skew_status"] = lookup["vol_surface"].get("skew_status", "normal")
        options_combined["atm_iv"] = lookup["vol_surface"].get("atm_iv", 0)
    if lookup.get("convexity_risk"):
        options_combined["convexity_risk"] = lookup["convexity_risk"].get("risk_level", "low")
    if lookup.get("gamma_exposure"):
        options_combined["total_gex"] = lookup["gamma_exposure"].get("total_gex", 0)
    if not options_combined:
        options_combined["unavailable"] = True
    options_score, options_reasons = _score_options(options_combined)

    # ── Weighted composite score ──────────────────────────────────
    composite = (
        fund_score * WEIGHTS["fundamentals"]
        + sent_score * WEIGHTS["sentiment"]
        + market_score * WEIGHTS["market_health"]
        + options_score * WEIGHTS["options_flow"]
        + macro_score * WEIGHTS["macro"]
        + momentum_score * WEIGHTS["momentum"]
    )

    # ── Generate signal ───────────────────────────────────────────
    if composite >= 68:
        signal = "BUY"
        signal_emoji = "🟢"
    elif composite >= 45:
        signal = "HOLD"
        signal_emoji = "🟡"
    else:
        signal = "SELL"
        signal_emoji = "🔴"

    # Confidence is how far from neutral (50)
    confidence = min(1.0, abs(composite - 50) / 35)

    # Collect all reasons, filter to most impactful
    all_reasons = fund_reasons + sent_reasons + market_reasons + options_reasons + macro_reasons + momentum_reasons
    bullish = [r for r in all_reasons if r.startswith("✅")]
    bearish = [r for r in all_reasons if r.startswith("🔴")]
    caution = [r for r in all_reasons if r.startswith("⚠️")]

    return {
        "symbol": symbol.upper(),
        "name": lookup.get("name", symbol),
        "signal": signal,
        "signal_emoji": signal_emoji,
        "composite_score": round(composite, 1),
        "confidence": round(confidence, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "price": lookup.get("price", {}).get("current"),
        "scores": {
            "fundamentals": {"score": round(fund_score, 1), "weight": f"{WEIGHTS['fundamentals']:.0%}"},
            "sentiment": {"score": round(sent_score, 1), "weight": f"{WEIGHTS['sentiment']:.0%}"},
            "market_health": {"score": round(market_score, 1), "weight": f"{WEIGHTS['market_health']:.0%}"},
            "options_flow": {"score": round(options_score, 1), "weight": f"{WEIGHTS['options_flow']:.0%}"},
            "macro_environment": {"score": round(macro_score, 1), "weight": f"{WEIGHTS['macro']:.0%}"},
            "momentum": {"score": round(momentum_score, 1), "weight": f"{WEIGHTS['momentum']:.0%}"},
        },
        "bullish_signals": bullish,
        "bearish_signals": bearish,
        "caution_signals": caution,
        "news_analysis": sentiment_headlines if sentiment_headlines else "No headlines available",
    }
