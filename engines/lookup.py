"""On-demand lookup — instant full analysis for any ticker.

Fetches fundamentals, options chain, computes GEX, vol surface,
convexity risk, and health scoring in a single request.
No need to add the symbol to the tracked list.
"""

from __future__ import annotations

import asyncio
import logging
import math
from datetime import datetime, timezone

import numpy as np

from state.redis_client import redis_client

logger = logging.getLogger(__name__)


def _fetch_ticker_data(symbol: str) -> dict:
    """Blocking call — fetch everything from yfinance for a single symbol."""
    import yfinance as yf

    ticker = yf.Ticker(symbol)
    info = ticker.info or {}
    result = {
        "symbol": symbol.upper(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "name": info.get("shortName") or info.get("longName", symbol),
        "sector": info.get("sector", "Unknown"),
        "industry": info.get("industry", "Unknown"),
        "exchange": info.get("exchange", "Unknown"),
        "currency": info.get("currency", "USD"),
    }

    # ── Price Data ────────────────────────────────────────────────
    result["price"] = {
        "current": info.get("regularMarketPrice") or info.get("currentPrice"),
        "previous_close": info.get("previousClose"),
        "open": info.get("regularMarketOpen") or info.get("open"),
        "day_high": info.get("regularMarketDayHigh") or info.get("dayHigh"),
        "day_low": info.get("regularMarketDayLow") or info.get("dayLow"),
        "volume": info.get("regularMarketVolume") or info.get("volume"),
        "market_cap": info.get("marketCap"),
        "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
        "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
    }

    # ── Fundamentals ──────────────────────────────────────────────
    def _safe(val):
        if val is None:
            return None
        try:
            v = float(val)
            return None if (math.isnan(v) or math.isinf(v)) else v
        except (ValueError, TypeError):
            return None

    result["fundamentals"] = {
        "pe_ratio": _safe(info.get("trailingPE")),
        "forward_pe": _safe(info.get("forwardPE")),
        "peg_ratio": _safe(info.get("pegRatio")),
        "price_to_book": _safe(info.get("priceToBook")),
        "total_debt": _safe(info.get("totalDebt")),
        "total_cash": _safe(info.get("totalCash")),
        "free_cash_flow": _safe(info.get("freeCashflow")),
        "revenue": _safe(info.get("totalRevenue")),
        "profit_margin": _safe(info.get("profitMargins")),
        "return_on_equity": _safe(info.get("returnOnEquity")),
        "dividend_yield": _safe(info.get("dividendYield")),
        "beta": _safe(info.get("beta")),
        "earnings_growth": _safe(info.get("earningsGrowth")),
        "revenue_growth": _safe(info.get("revenueGrowth")),
    }

    # ── Fundamental Health Score ──────────────────────────────────
    fund = result["fundamentals"]
    fund_score = 0
    if fund["free_cash_flow"] and fund["free_cash_flow"] > 0:
        fund_score += 20
    if fund["total_cash"] and fund["total_debt"] and fund["total_cash"] > fund["total_debt"]:
        fund_score += 20
    if fund["pe_ratio"] and 5 <= fund["pe_ratio"] <= 30:
        fund_score += 20
    if fund["profit_margin"] and fund["profit_margin"] > 0:
        fund_score += 20
    if result["price"].get("market_cap") and result["price"]["market_cap"] > 0:
        fund_score += 20
    result["fundamental_score"] = fund_score

    # ── Options Chain + Derivatives (if available) ────────────────
    result["options"] = None
    result["gamma_exposure"] = None
    result["vol_surface"] = None
    result["convexity_risk"] = None

    try:
        expiries = ticker.options
        if expiries:
            chain = ticker.option_chain(expiries[0])
            contracts = []

            for _, row in chain.calls.iterrows():
                iv = row.get("impliedVolatility", 0) or 0
                oi = row.get("openInterest", 0) or 0
                vol = row.get("volume", 0) or 0
                lp = row.get("lastPrice", 0) or 0
                try:
                    oi = int(oi) if not (math.isnan(oi) or math.isinf(oi)) else 0
                    vol = int(vol) if not (math.isnan(vol) or math.isinf(vol)) else 0
                except (ValueError, TypeError):
                    oi, vol = 0, 0
                contracts.append({
                    "strike": float(row["strike"]),
                    "contract_type": "call",
                    "implied_volatility": float(iv) if iv and not math.isnan(float(iv)) else 0.0,
                    "open_interest": oi,
                    "last_price": float(lp) if lp and not math.isnan(float(lp)) else 0.0,
                    "volume": vol,
                })

            for _, row in chain.puts.iterrows():
                iv = row.get("impliedVolatility", 0) or 0
                oi = row.get("openInterest", 0) or 0
                vol = row.get("volume", 0) or 0
                lp = row.get("lastPrice", 0) or 0
                try:
                    oi = int(oi) if not (math.isnan(oi) or math.isinf(oi)) else 0
                    vol = int(vol) if not (math.isnan(vol) or math.isinf(vol)) else 0
                except (ValueError, TypeError):
                    oi, vol = 0, 0
                contracts.append({
                    "strike": float(row["strike"]),
                    "contract_type": "put",
                    "implied_volatility": float(iv) if iv and not math.isnan(float(iv)) else 0.0,
                    "open_interest": oi,
                    "last_price": float(lp) if lp and not math.isnan(float(lp)) else 0.0,
                    "volume": vol,
                })

            spot = result["price"]["current"] or info.get("previousClose", 0)
            result["options"] = {
                "total_contracts": len(contracts),
                "nearest_expiry": expiries[0],
                "available_expiries": len(expiries),
            }

            if contracts and spot and spot > 0:
                chain_data = {
                    "symbol": symbol.upper(),
                    "underlying_price": spot,
                    "contracts": contracts,
                }
                # Compute derivatives intelligence
                result["gamma_exposure"] = _compute_gex(chain_data)
                result["vol_surface"] = _compute_vol_surface(chain_data, spot)
                result["convexity_risk"] = _compute_convexity(chain_data, spot)

    except Exception as e:
        logger.warning("Options data unavailable for %s: %s", symbol, e)

    # ── Macro context ─────────────────────────────────────────────
    # Read cached macro from Redis (if available)
    result["_needs_macro"] = True  # Flag for async step

    return result


def _compute_gex(chain_data: dict) -> dict | None:
    """Compute GEX using the C++ engine."""
    try:
        import gamma_engine
    except ImportError:
        return None

    contracts = chain_data["contracts"]
    spot = chain_data["underlying_price"]
    strikes = np.array([c["strike"] for c in contracts], dtype=np.float64)
    ivs = np.array([c["implied_volatility"] for c in contracts], dtype=np.float64)
    ois = np.array([c["open_interest"] for c in contracts], dtype=np.float64)
    is_calls = np.array([1 if c["contract_type"] == "call" else 0 for c in contracts], dtype=np.int32)
    ivs[ivs <= 0] = 0.01

    result = gamma_engine.compute_gamma_exposure(
        strikes, ivs, ois, is_calls,
        spot, 0.045, 30 / 365,
    )

    # Top zones
    zone_idx = sorted(range(len(result.gex_per_strike)),
                      key=lambda i: abs(result.gex_per_strike[i]), reverse=True)[:5]
    zones = [
        {"strike": result.strikes[i], "gex": round(result.gex_per_strike[i], 2), "type": result.types[i]}
        for i in zone_idx
    ]

    return {
        "total_gex": round(result.total_gex, 2),
        "flip_point": round(result.flip_point, 2) if result.flip_point > 0 else None,
        "top_hedging_zones": zones,
    }


def _compute_vol_surface(chain_data: dict, spot: float) -> dict | None:
    """Compute vol surface from chain."""
    contracts = [c for c in chain_data["contracts"] if c["implied_volatility"] > 0]
    if not contracts:
        return None

    moneyness = np.array([c["strike"] / spot for c in contracts])
    ivs = np.array([c["implied_volatility"] for c in contracts])

    atm_mask = (moneyness > 0.95) & (moneyness < 1.05)
    atm_iv = float(np.mean(ivs[atm_mask])) if atm_mask.any() else float(np.median(ivs))
    otm_put_iv = float(np.mean(ivs[moneyness < 0.95])) if (moneyness < 0.95).any() else atm_iv
    otm_call_iv = float(np.mean(ivs[moneyness > 1.05])) if (moneyness > 1.05).any() else atm_iv

    skew = otm_put_iv - otm_call_iv

    return {
        "atm_iv": round(atm_iv, 4),
        "otm_put_iv": round(otm_put_iv, 4),
        "otm_call_iv": round(otm_call_iv, 4),
        "skew": round(skew, 4),
        "skew_status": "severe_skew" if abs(skew) > 0.10 else "moderate_skew" if abs(skew) > 0.05 else "normal",
    }


def _compute_convexity(chain_data: dict, spot: float) -> dict | None:
    """Compute convexity risk."""
    try:
        import gamma_engine
    except ImportError:
        return None

    contracts = chain_data["contracts"]
    total_dollar_gamma = 0.0
    negative_gamma = 0.0

    for c in contracts:
        iv = c["implied_volatility"] if c["implied_volatility"] > 0 else 0.01
        bs = gamma_engine.black_scholes(spot, c["strike"], 30/365, 0.045, iv, c["contract_type"] == "call")
        dg = bs.gamma * c["open_interest"] * 100 * spot * spot
        if c["contract_type"] != "call":
            dg = -dg
        total_dollar_gamma += dg
        if dg < 0:
            negative_gamma += abs(dg)

    total_abs = abs(total_dollar_gamma) + negative_gamma
    neg_ratio = negative_gamma / total_abs if total_abs > 0 else 0

    risk_score = 0
    if neg_ratio > 0.6:
        risk_score += 50
    elif neg_ratio > 0.4:
        risk_score += 25

    risk_score = min(100, risk_score)

    return {
        "convexity_risk_score": risk_score,
        "risk_level": "critical" if risk_score > 70 else "elevated" if risk_score > 50 else "moderate" if risk_score > 25 else "low",
        "negative_gamma_ratio": round(neg_ratio, 4),
    }


async def lookup_symbol(symbol: str) -> dict:
    """Full on-demand analysis for any ticker. Called from API route."""
    loop = asyncio.get_running_loop()
    data = await loop.run_in_executor(None, _fetch_ticker_data, symbol)

    # Fetch cached macro from Redis
    macro = await redis_client.hgetall("macro_indicators")
    data.pop("_needs_macro", None)

    # Compute health index
    fund_score = data.get("fundamental_score", 50)
    macro_score = 50.0
    if macro:
        macro_score = 50.0
        ffr = macro.get("fed_funds_rate")
        if ffr:
            ffr = float(ffr)
            if ffr < 2: macro_score += 15
            elif ffr > 5: macro_score -= 15
        vix = macro.get("vix")
        if vix:
            vix = float(vix)
            if vix < 15: macro_score += 10
            elif vix > 30: macro_score -= 15
        macro_score = max(0, min(100, macro_score))

    health_index = round(0.40 * fund_score + 0.35 * 50 + 0.25 * macro_score, 2)
    data["health_index"] = health_index
    data["macro_indicators"] = macro or "Run `python main.py` to populate"

    return data
