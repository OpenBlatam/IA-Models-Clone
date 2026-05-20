import asyncio
import time
from typing import Any, Awaitable, Callable, Dict, Optional

try:
    from redis import asyncio as redis  # type: ignore
except Exception:
    redis = None  # type: ignore


class AsyncCache:
    def __init__(self, redis_url: Optional[str] = None):
        self._memory: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        self._redis = None
        self._redis_url = redis_url

    async def init(self):
        if self._redis_url and redis is not None:
            try:
                self._redis = await redis.from_url(self._redis_url, encoding="utf-8", decode_responses=True)  # type: ignore
            except Exception:
                self._redis = None

    async def get(self, key: str) -> Optional[str]:
        if self._redis is not None:
            try:
                return await self._redis.get(key)
            except Exception:
                pass
        async with self._lock:
            entry = self._memory.get(key)
            if not entry:
                return None
            if entry["exp"] < time.time():
                self._memory.pop(key, None)
                return None
            return entry["val"]

    async def set(self, key: str, value: str, ttl: int):
        if self._redis is not None:
            try:
                await self._redis.set(key, value, ex=ttl)
                return
            except Exception:
                pass
        async with self._lock:
            self._memory[key] = {"val": value, "exp": time.time() + ttl}


async def cached_json(cache: AsyncCache, key: str, ttl: int, producer: Callable[[], Awaitable[Dict[str, Any]]]) -> Dict[str, Any]:
    cached_val = await cache.get(key)
    if cached_val:
        try:
            import orjson
            return orjson.loads(cached_val)
        except Exception:
            pass
    data = await producer()
    try:
        import orjson
        await cache.set(key, orjson.dumps(data).decode("utf-8"), ttl)
    except Exception:
        pass
    return data







