from __future__ import annotations

import json
import logging
from typing import Any, Optional

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

# TTL constants (seconds)
ODDS = 120
PROPS = 300
INJURIES = 900
STATS = 3600
SCHEDULE = 86400
HISTORICAL = 86400


class RedisCache:
    def __init__(self, redis_url: str) -> None:
        self._url = redis_url
        self._client: Optional[aioredis.Redis] = None

    async def _get_client(self) -> aioredis.Redis:
        if self._client is None:
            self._client = aioredis.from_url(self._url, decode_responses=True)
        return self._client

    async def get(self, key: str) -> Optional[Any]:
        client = await self._get_client()
        try:
            raw = await client.get(key)
            if raw is None:
                logger.debug("Cache MISS: %s", key)
                print(f"[CACHE MISS] {key}")
                return None
            logger.debug("Cache HIT: %s", key)
            print(f"[CACHE HIT]  {key}")
            return json.loads(raw)
        except Exception as exc:
            logger.warning("Redis GET error for key %s: %s", key, exc)
            return None

    async def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        client = await self._get_client()
        try:
            serialized = json.dumps(value, default=str)
            await client.setex(key, ttl_seconds, serialized)
            logger.debug("Cache SET: %s (ttl=%ds)", key, ttl_seconds)
        except Exception as exc:
            logger.warning("Redis SET error for key %s: %s", key, exc)

    async def delete(self, key: str) -> None:
        client = await self._get_client()
        try:
            await client.delete(key)
        except Exception as exc:
            logger.warning("Redis DELETE error for key %s: %s", key, exc)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
