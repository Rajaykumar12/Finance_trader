"""Module C — Derivatives Intelligence Engine.

Computes three derivative intelligence outputs:
  1. **Gamma Exposure (GEX)** — via C++ Black-Scholes engine
  2. **Volatility Surface** — IV mapped across strike × expiry with skew/smile detection
  3. **Convexity Risk** — detects dangerous gamma/vega relationships
  4. **Volatility Regime Indicator** — overall derivatives-implied vol regime

Results are written to Redis hashes:
  - ``gamma_exposure:<SYMBOL>``
  - ``vol_surface:<SYMBOL>``
  - ``vol_regime_indicator:<SYMBOL>``
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from datetime import datetime, timezone

import numpy as np

from config.settings import settings
from state.redis_client import redis_client

logger = logging.getLogger(__name__)

RISK_FREE_RATE = 0.045
DEFAULT_TTE = 30 / 365


# ── GEX computation (existing) ────────────────────────────────────────

def _compute_gex_for_chain(chain_data: dict) -> dict | None:
    """Run the C++ gamma engine on an options chain dict."""
    try:
        import gamma_engine
    except ImportError:
        logger.error(
            "gamma_engine C++ module not found. "
            "Build it with: python setup_cpp.py build_ext --inplace"
        )
        return None

    contracts = chain_data.get("contracts", [])
    spot = float(chain_data.get("underlying_price", 0))
    symbol = chain_data.get("symbol", "UNKNOWN")

    if not contracts or spot <= 0:
        return None

    strikes = np.array([c["strike"] for c in contracts], dtype=np.float64)
    ivs = np.array([c["implied_volatility"] for c in contracts], dtype=np.float64)
    ois = np.array([c["open_interest"] for c in contracts], dtype=np.float64)
    is_calls = np.array(
        [1 if c["contract_type"] == "call" else 0 for c in contracts],
        dtype=np.int32,
    )
    ivs[ivs <= 0] = 0.01

    result = gamma_engine.compute_gamma_exposure(
        strikes, ivs, ois, is_calls,
        spot, RISK_FREE_RATE, DEFAULT_TTE,
    )

    zone_indices = sorted(
        range(len(result.gex_per_strike)),
        key=lambda i: abs(result.gex_per_strike[i]),
        reverse=True,
    )[:10]

    zones = [
        {
            "strike": result.strikes[i],
            "gex": round(result.gex_per_strike[i], 2),
            "type": result.types[i],
        }
        for i in zone_indices
    ]

    return {
        "symbol": symbol,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_gex": round(result.total_gex, 2),
        "hedging_pressure_zones": json.dumps(zones),
        "flip_point": round(result.flip_point, 2) if result.flip_point > 0 else None,
        "spot_price": spot,
        "contracts_analyzed": len(contracts),
    }


# ── Volatility Surface Mapping ────────────────────────────────────────

def _compute_vol_surface(chain_data: dict) -> dict | None:
    """Build a volatility surface from the options chain.

    Maps IV across strikes, detects skew and smile distortions.
    """
    contracts = chain_data.get("contracts", [])
    spot = float(chain_data.get("underlying_price", 0))
    symbol = chain_data.get("symbol", "UNKNOWN")

    if not contracts or spot <= 0:
        return None

    # Separate calls and puts
    calls = [c for c in contracts if c["contract_type"] == "call" and c["implied_volatility"] > 0]
    puts = [c for c in contracts if c["contract_type"] == "put" and c["implied_volatility"] > 0]

    if not calls and not puts:
        return None

    # Build IV curve indexed by moneyness (strike / spot)
    all_contracts = calls + puts
    moneyness = np.array([c["strike"] / spot for c in all_contracts])
    ivs = np.array([c["implied_volatility"] for c in all_contracts])

    # Sort by moneyness
    sort_idx = np.argsort(moneyness)
    moneyness = moneyness[sort_idx]
    ivs = ivs[sort_idx]

    # ATM IV (moneyness ≈ 1.0)
    atm_mask = (moneyness > 0.95) & (moneyness < 1.05)
    atm_iv = float(np.mean(ivs[atm_mask])) if atm_mask.any() else float(np.median(ivs))

    # OTM Put IV (moneyness < 0.95 — downside protection demand)
    otm_put_mask = moneyness < 0.95
    otm_put_iv = float(np.mean(ivs[otm_put_mask])) if otm_put_mask.any() else atm_iv

    # OTM Call IV (moneyness > 1.05 — upside speculation)
    otm_call_mask = moneyness > 1.05
    otm_call_iv = float(np.mean(ivs[otm_call_mask])) if otm_call_mask.any() else atm_iv

    # Skew = OTM Put IV - OTM Call IV  (positive = more demand for downside protection)
    skew = otm_put_iv - otm_call_iv

    # Smile = average OTM IV - ATM IV  (positive = smile shape exists)
    avg_otm_iv = (otm_put_iv + otm_call_iv) / 2
    smile = avg_otm_iv - atm_iv

    # Skew distortion detection
    if abs(skew) > 0.10:
        skew_status = "severe_skew"
    elif abs(skew) > 0.05:
        skew_status = "moderate_skew"
    else:
        skew_status = "normal"

    # Smile distortion detection
    if smile > 0.05:
        smile_status = "pronounced_smile"
    elif smile > 0.02:
        smile_status = "mild_smile"
    else:
        smile_status = "flat"

    # Build surface data points for the API
    surface_points = [
        {"moneyness": round(float(m), 4), "iv": round(float(iv), 4)}
        for m, iv in zip(moneyness, ivs)
    ]

    return {
        "symbol": symbol,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "atm_iv": round(atm_iv, 4),
        "otm_put_iv": round(otm_put_iv, 4),
        "otm_call_iv": round(otm_call_iv, 4),
        "skew": round(skew, 4),
        "smile": round(smile, 4),
        "skew_status": skew_status,
        "smile_status": smile_status,
        "spot_price": spot,
        "surface_points": json.dumps(surface_points),
        "total_contracts": len(all_contracts),
    }


# ── Convexity Risk Detection ─────────────────────────────────────────

def _compute_convexity_risk(chain_data: dict, gex_data: dict | None) -> dict | None:
    """Detect convexity risk from gamma/vega relationships.

    Convexity risk arises when:
      1. Gamma is very high + OI is concentrated → explosive moves
      2. Vega exposure is large → vol-of-vol sensitivity
      3. Negative gamma (dealer short gamma) → amplified moves
    """
    contracts = chain_data.get("contracts", [])
    spot = float(chain_data.get("underlying_price", 0))
    symbol = chain_data.get("symbol", "UNKNOWN")

    if not contracts or spot <= 0:
        return None

    try:
        import gamma_engine
    except ImportError:
        return None

    # Compute per-contract gamma and vega
    total_dollar_gamma = 0.0
    total_dollar_vega = 0.0
    negative_gamma_exposure = 0.0
    max_single_strike_gex = 0.0

    for c in contracts:
        iv = c["implied_volatility"]
        if iv <= 0:
            iv = 0.01
        oi = c["open_interest"]
        strike = c["strike"]
        is_call = c["contract_type"] == "call"

        bs = gamma_engine.black_scholes(spot, strike, DEFAULT_TTE, RISK_FREE_RATE, iv, is_call)

        # Dollar gamma per contract: gamma × OI × 100 × spot²
        dollar_gamma = bs.gamma * oi * 100 * spot * spot
        if not is_call:
            dollar_gamma = -dollar_gamma

        total_dollar_gamma += dollar_gamma
        if dollar_gamma < 0:
            negative_gamma_exposure += abs(dollar_gamma)

        max_single_strike_gex = max(max_single_strike_gex, abs(dollar_gamma))

        # Vega approximation: gamma × S × σ × √T × OI × 100
        vega = bs.gamma * spot * iv * math.sqrt(DEFAULT_TTE) * oi * 100
        total_dollar_vega += vega

    # Risk scoring
    risk_score = 0.0

    # High gamma concentration
    if max_single_strike_gex > 0 and total_dollar_gamma != 0:
        concentration = max_single_strike_gex / abs(total_dollar_gamma) if total_dollar_gamma != 0 else 0
        if concentration > 0.5:
            risk_score += 30  # One strike dominates
        elif concentration > 0.3:
            risk_score += 15

    # Negative gamma ratio
    total_abs = abs(total_dollar_gamma) + negative_gamma_exposure
    if total_abs > 0:
        neg_ratio = negative_gamma_exposure / total_abs
        if neg_ratio > 0.6:
            risk_score += 35  # Dealers heavily short gamma
        elif neg_ratio > 0.4:
            risk_score += 20

    # High vega relative to gamma (vol-of-vol sensitivity)
    if abs(total_dollar_gamma) > 0:
        vega_gamma_ratio = abs(total_dollar_vega) / abs(total_dollar_gamma)
        if vega_gamma_ratio > 2.0:
            risk_score += 25
        elif vega_gamma_ratio > 1.0:
            risk_score += 10

    risk_score = min(100.0, risk_score)

    if risk_score > 70:
        risk_level = "critical"
    elif risk_score > 50:
        risk_level = "elevated"
    elif risk_score > 25:
        risk_level = "moderate"
    else:
        risk_level = "low"

    return {
        "symbol": symbol,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "convexity_risk_score": round(risk_score, 2),
        "risk_level": risk_level,
        "total_dollar_gamma": round(total_dollar_gamma, 2),
        "negative_gamma_exposure": round(negative_gamma_exposure, 2),
        "total_dollar_vega": round(total_dollar_vega, 2),
        "gamma_concentration": round(
            max_single_strike_gex / abs(total_dollar_gamma), 4
        ) if total_dollar_gamma != 0 else 0,
    }


# ── Volatility Regime Indicator ───────────────────────────────────────

def _compute_vol_regime_indicator(vol_surface: dict | None, convexity: dict | None) -> dict:
    """Combine vol surface and convexity risk into overall regime indicator."""
    atm_iv = float(vol_surface.get("atm_iv", 0.2)) if vol_surface else 0.2
    skew = abs(float(vol_surface.get("skew", 0))) if vol_surface else 0
    risk_score = float(convexity.get("convexity_risk_score", 0)) if convexity else 0

    # Vol regime from ATM IV level
    if atm_iv < 0.15:
        iv_regime = "compressed"
    elif atm_iv < 0.25:
        iv_regime = "normal"
    elif atm_iv < 0.40:
        iv_regime = "elevated"
    else:
        iv_regime = "extreme"

    # Combined regime
    regime_score = 0
    if iv_regime == "compressed":
        regime_score += 10
    elif iv_regime == "normal":
        regime_score += 30
    elif iv_regime == "elevated":
        regime_score += 60
    else:
        regime_score += 90

    regime_score += min(30, risk_score * 0.3)

    if skew > 0.10:
        regime_score += 20
    elif skew > 0.05:
        regime_score += 10

    regime_score = min(100, regime_score)

    if regime_score > 75:
        overall = "crisis"
    elif regime_score > 50:
        overall = "stressed"
    elif regime_score > 25:
        overall = "normal"
    else:
        overall = "calm"

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "iv_regime": iv_regime,
        "overall_regime": overall,
        "regime_score": round(regime_score, 2),
        "atm_iv": round(atm_iv, 4),
        "skew_magnitude": round(skew, 4),
        "convexity_risk": round(risk_score, 2),
    }


# ── Main loop ─────────────────────────────────────────────────────────

async def _poll_and_compute(symbols: list[str]) -> None:
    """Poll Redis for options data and compute all derivatives intelligence."""
    last_timestamps: dict[str, str] = {}

    while True:
        for sym in symbols:
            try:
                chain_raw = await redis_client.hgetall(f"options_chain:{sym}")
                if not chain_raw:
                    continue

                updated_at = chain_raw.get("updated_at", "")
                if updated_at == last_timestamps.get(sym):
                    continue

                data_str = chain_raw.get("data", "{}")
                chain_data = json.loads(data_str) if isinstance(data_str, str) else data_str

                # 1. GEX
                gex = _compute_gex_for_chain(chain_data)
                if gex:
                    await redis_client.hset(f"gamma_exposure:{sym}", gex)

                # 2. Vol Surface
                vol_surface = _compute_vol_surface(chain_data)
                if vol_surface:
                    await redis_client.hset(f"vol_surface:{sym}", vol_surface)

                # 3. Convexity Risk
                convexity = _compute_convexity_risk(chain_data, gex)
                if convexity:
                    await redis_client.hset(f"convexity_risk:{sym}", convexity)

                # 4. Vol Regime Indicator
                vol_regime = _compute_vol_regime_indicator(vol_surface, convexity)
                vol_regime["symbol"] = sym
                await redis_client.hset(f"vol_regime_indicator:{sym}", vol_regime)

                last_timestamps[sym] = updated_at
                logger.info(
                    "Derivatives %s: GEX=%.0f, ATM_IV=%.2f%%, skew=%s, risk=%s, regime=%s",
                    sym,
                    gex["total_gex"] if gex else 0,
                    (vol_surface["atm_iv"] * 100) if vol_surface else 0,
                    vol_surface["skew_status"] if vol_surface else "N/A",
                    convexity["risk_level"] if convexity else "N/A",
                    vol_regime["overall_regime"],
                )
            except Exception:
                logger.exception("Derivatives computation error for %s", sym)

        await asyncio.sleep(5.0)


async def run_derivatives_intelligence(symbols: list[str] | None = None) -> None:
    """Entry point."""
    if symbols is None:
        symbols = settings.equity_symbol_list

    await redis_client.connect()
    logger.info("Derivatives Intelligence Engine starting for %s …", symbols)
    await _poll_and_compute(symbols)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_derivatives_intelligence())
