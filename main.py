"""Orchestrator — single entry point that launches the entire platform.

Usage:
    python main.py

Starts:
    - Ingestion workers (Binance WS, Alpaca, Finnhub, Yahoo Finance)
    - Intelligence engines (Market, Biological, Derivatives)
    - FastAPI server (REST + WebSocket)
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys

import uvicorn

from config.settings import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-35s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("orchestrator")


async def main() -> None:
    """Launch all platform components concurrently."""
    logger.info("=" * 60)
    logger.info("  Trading Intelligence Platform — Starting")
    logger.info("=" * 60)

    tasks: list[asyncio.Task] = []

    # ── Ingestion Layer ────────────────────────────────────────────
    from ingestion.binance_ws import run_binance_ingestion
    tasks.append(asyncio.create_task(run_binance_ingestion(), name="binance_ws"))

    from ingestion.alpaca_client import run_alpaca_ingestion
    tasks.append(asyncio.create_task(run_alpaca_ingestion(), name="alpaca"))

    from ingestion.finnhub_poller import run_finnhub_poller
    tasks.append(asyncio.create_task(run_finnhub_poller(), name="finnhub"))

    from ingestion.yahoo_finance import run_yahoo_finance_poller
    tasks.append(asyncio.create_task(run_yahoo_finance_poller(), name="yahoo_finance"))

    from ingestion.macro_poller import run_macro_poller
    tasks.append(asyncio.create_task(run_macro_poller(), name="macro_poller"))

    # ── Intelligence Engines ───────────────────────────────────────
    from engines.market_intelligence import run_market_intelligence
    tasks.append(asyncio.create_task(run_market_intelligence(), name="market_intel"))

    from engines.biological_modeling import run_biological_modeling
    tasks.append(asyncio.create_task(run_biological_modeling(), name="bio_modeling"))

    from engines.derivatives_intelligence import run_derivatives_intelligence
    tasks.append(asyncio.create_task(run_derivatives_intelligence(), name="derivatives"))

    # ── FastAPI Server ─────────────────────────────────────────────
    from api.app import app

    config = uvicorn.Config(
        app,
        host=settings.api_host,
        port=settings.api_port,
        log_level="info",
    )
    server = uvicorn.Server(config)
    tasks.append(asyncio.create_task(server.serve(), name="api_server"))

    logger.info("All components launched — %d tasks running", len(tasks))
    logger.info(
        "  API: http://localhost:%d/docs",
        settings.api_port,
    )
    logger.info("-" * 60)

    # Wait for any task to exit (usually means a crash)
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)

    for task in done:
        if task.exception():
            logger.error("Task %s crashed: %s", task.get_name(), task.exception())

    # Cancel remaining tasks
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)

    logger.info("Platform shut down.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹  Shutting down …")
        sys.exit(0)
