"""Async Redis client — thin wrapper for pub/sub + hash operations."""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

import redis.asyncio as aioredis

from config.settings import settings

logger = logging.getLogger(__name__)


class RedisClient:
    """Async Redis helper for the Trading Intelligence Platform.

    Usage::

        rc = RedisClient()
        await rc.connect()

        # Publish / Subscribe
        await rc.publish("market_ticks:BTC", {"price": 62000})
        async for msg in rc.subscribe("market_ticks:BTC"):
            print(msg)

        # Hash get / set
        await rc.hset("market_state:BTC", {"volatility": 0.03})
        data = await rc.hgetall("market_state:BTC")

        await rc.close()
    """

    def __init__(self, url: str | None = None) -> None:
        self._url = url or settings.redis_url
        self._pool: aioredis.Redis | None = None

    # ── lifecycle ──────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Create the connection pool."""
        if self._pool is None:
            self._pool = aioredis.from_url(
                self._url,
                decode_responses=True,
                max_connections=20,
            )
            # Quick health check
            await self._pool.ping()
            logger.info("Redis connected: %s", self._url)

    async def close(self) -> None:
        """Drain and close the pool."""
        if self._pool is not None:
            await self._pool.aclose()
            self._pool = None
            logger.info("Redis connection closed.")

    @property
    def pool(self) -> aioredis.Redis:
        if self._pool is None:
            raise RuntimeError("RedisClient is not connected. Call .connect() first.")
        return self._pool

    # ── Pub / Sub ──────────────────────────────────────────────────────

    async def publish(self, channel: str, data: dict[str, Any]) -> int:
        """Publish a JSON-serialised dict to *channel*."""
        payload = json.dumps(data, default=str)
        receivers = await self.pool.publish(channel, payload)
        return receivers

    async def subscribe(self, *channels: str) -> AsyncIterator[dict[str, Any]]:
        """Yield parsed JSON messages from one or more channels.

        This is an infinite async generator — cancel the task to stop.
        """
        pubsub = self.pool.pubsub()
        await pubsub.subscribe(*channels)
        logger.info("Subscribed to: %s", ", ".join(channels))
        try:
            async for raw in pubsub.listen():
                if raw["type"] == "message":
                    try:
                        yield json.loads(raw["data"])
                    except json.JSONDecodeError:
                        logger.warning("Non-JSON message on %s: %s", raw["channel"], raw["data"])
        finally:
            await pubsub.unsubscribe(*channels)
            await pubsub.aclose()

    # ── Hash operations ────────────────────────────────────────────────

    async def hset(self, key: str, mapping: dict[str, Any]) -> int:
        """Write *mapping* to a Redis hash.

        Non-string values are JSON-serialised automatically.
        """
        serialised = {
            k: (json.dumps(v, default=str) if not isinstance(v, (str, int, float)) or isinstance(v, bool) else v)
            for k, v in mapping.items()
        }
        result = await self.pool.hset(key, mapping=serialised)  # type: ignore[arg-type]
        return result

    async def hgetall(self, key: str) -> dict[str, Any]:
        """Read an entire Redis hash, attempting to JSON-parse each value."""
        raw = await self.pool.hgetall(key)
        parsed: dict[str, Any] = {}
        for k, v in raw.items():
            try:
                parsed[k] = json.loads(v)
            except (json.JSONDecodeError, TypeError):
                parsed[k] = v
        return parsed

    async def hget(self, key: str, field: str) -> Any:
        """Read a single field from a Redis hash."""
        raw = await self.pool.hget(key, field)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return raw


# ── Module-level singleton ─────────────────────────────────────────────
redis_client = RedisClient()
