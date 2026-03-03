"""Macroeconomic indicators ingestion — FRED API + VIX via yfinance.

Polls key macroeconomic data periodically and stores in Redis hash
``macro_indicators``. Uses free public data:
  - VIX (CBOE Volatility Index) via yfinance
  - Treasury yields via yfinance (^TNX for 10-year)
  - Other macro data estimated from market proxies

Runs every 10 minutes (macro data changes slowly).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from functools import partial

from config.settings import settings
from state.redis_client import redis_client

logger = logging.getLogger(__name__)


def _fetch_macro_sync() -> dict:
    """Blocking call — fetches macro indicators. Run in executor."""
    import yfinance as yf

    macro = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # VIX — CBOE Volatility Index
    try:
        vix = yf.Ticker("^VIX")
        vix_info = vix.info or {}
        vix_price = vix_info.get("regularMarketPrice") or vix_info.get("previousClose")
        if vix_price:
            macro["vix"] = round(float(vix_price), 2)
    except Exception as e:
        logger.warning("Failed to fetch VIX: %s", e)

    # 10-Year Treasury Yield (proxy for risk-free rate)
    try:
        tnx = yf.Ticker("^TNX")
        tnx_info = tnx.info or {}
        tnx_price = tnx_info.get("regularMarketPrice") or tnx_info.get("previousClose")
        if tnx_price:
            macro["treasury_10y"] = round(float(tnx_price), 3)
    except Exception as e:
        logger.warning("Failed to fetch 10Y yield: %s", e)

    # US Dollar Index (DXY) — strength of the dollar
    try:
        dxy = yf.Ticker("DX-Y.NYB")
        dxy_info = dxy.info or {}
        dxy_price = dxy_info.get("regularMarketPrice") or dxy_info.get("previousClose")
        if dxy_price:
            macro["dollar_index"] = round(float(dxy_price), 2)
    except Exception as e:
        logger.warning("Failed to fetch DXY: %s", e)

    # S&P 500 as market health indicator
    try:
        sp500 = yf.Ticker("^GSPC")
        sp500_info = sp500.info or {}
        sp500_price = sp500_info.get("regularMarketPrice") or sp500_info.get("previousClose")
        if sp500_price:
            macro["sp500"] = round(float(sp500_price), 2)
    except Exception as e:
        logger.warning("Failed to fetch S&P 500: %s", e)

    # Gold (safe haven indicator)
    try:
        gold = yf.Ticker("GC=F")
        gold_info = gold.info or {}
        gold_price = gold_info.get("regularMarketPrice") or gold_info.get("previousClose")
        if gold_price:
            macro["gold"] = round(float(gold_price), 2)
    except Exception as e:
        logger.warning("Failed to fetch Gold: %s", e)

    # Derive estimated macro scores from market proxies
    # Fed funds rate approximation from 2Y treasury
    try:
        gs2 = yf.Ticker("^IRX")  # 13-week treasury bill rate
        gs2_info = gs2.info or {}
        rate = gs2_info.get("regularMarketPrice") or gs2_info.get("previousClose")
        if rate:
            macro["fed_funds_rate"] = round(float(rate), 3)
    except Exception as e:
        logger.warning("Failed to fetch short rate: %s", e)

    return macro


async def run_macro_poller() -> None:
    """Main loop — polls macro data at the configured interval."""
    await redis_client.connect()
    logger.info("Macro indicators poller starting (interval=%ds) …", settings.yahoo_poll_interval)

    loop = asyncio.get_running_loop()
    while True:
        try:
            macro = await loop.run_in_executor(None, _fetch_macro_sync)
            if macro:
                await redis_client.hset("macro_indicators", macro)
                logger.info(
                    "Macro updated: VIX=%s, 10Y=%s, DXY=%s",
                    macro.get("vix", "N/A"),
                    macro.get("treasury_10y", "N/A"),
                    macro.get("dollar_index", "N/A"),
                )
        except Exception:
            logger.exception("Macro polling error")

        await asyncio.sleep(settings.yahoo_poll_interval)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_macro_poller())
