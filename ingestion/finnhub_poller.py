"""Finnhub news poller — fetches financial headlines at regular intervals.

Publishes parsed headlines to Redis channel ``news_feed``.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

import httpx

from config.settings import settings
from state.redis_client import redis_client

logger = logging.getLogger(__name__)

FINNHUB_NEWS_URL = "https://finnhub.io/api/v1/news"
CHANNEL = "news_feed"


async def _fetch_headlines(client: httpx.AsyncClient) -> list[dict]:
    """Fetch general market news from Finnhub REST API."""
    params = {
        "category": "general",
        "token": settings.finnhub_api_key,
    }
    resp = await client.get(FINNHUB_NEWS_URL, params=params)
    resp.raise_for_status()
    raw_items = resp.json()

    headlines = []
    for item in raw_items[:15]:  # Cap to avoid flooding
        headlines.append({
            "headline": item.get("headline", ""),
            "source": item.get("source", ""),
            "url": item.get("url", ""),
            "timestamp": datetime.fromtimestamp(
                item.get("datetime", 0), tz=timezone.utc
            ).isoformat(),
            "symbol": None,  # General news — no specific symbol
            "sentiment": None,  # To be computed by Module B
        })
    return headlines


async def run_finnhub_poller() -> None:
    """Main loop — polls Finnhub at the configured interval."""
    if not settings.finnhub_api_key or settings.finnhub_api_key.startswith("your_"):
        logger.warning("Finnhub API key not configured — skipping news ingestion.")
        return

    await redis_client.connect()
    logger.info(
        "Finnhub poller starting (interval=%ds) …", settings.finnhub_poll_interval
    )

    seen_urls: set[str] = set()

    async with httpx.AsyncClient(timeout=10) as client:
        while True:
            try:
                headlines = await _fetch_headlines(client)
                new_count = 0
                for hl in headlines:
                    if hl["url"] not in seen_urls:
                        seen_urls.add(hl["url"])
                        await redis_client.publish(CHANNEL, hl)
                        new_count += 1

                if new_count:
                    logger.info("Published %d new headlines", new_count)

                # Keep seen set bounded
                if len(seen_urls) > 500:
                    seen_urls.clear()

            except httpx.HTTPStatusError as exc:
                logger.warning("Finnhub HTTP error: %s", exc.response.status_code)
            except Exception:
                logger.exception("Finnhub polling error")

            await asyncio.sleep(settings.finnhub_poll_interval)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_finnhub_poller())
