from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import asyncio
from typing import Optional, Any
import redis.asyncio as redis
from ..utils.config import get_settings
from typing import Any, List, Dict, Optional
import logging
"""
Cache Utilities - High-performance caching
"""



_cache_client: Optional[redis.Redis] = None


async def get_cache_client() -> redis.Redis:
    """Get Redis cache client instance."""
    global _cache_client
    
    if _cache_client is None:
        settings = get_settings()
        _cache_client = redis.from_url(
            settings.cache.redis_url,
            encoding="utf-8",
            decode_responses=True,
        )
    
    return _cache_client


async def cache_get(key: str) -> Optional[str]:
    """Get value from cache."""
    client = await get_cache_client()
    return await client.get(key)


async def cache_set(
    key: str,
    value: str,
    expire: Optional[int] = None,
) -> bool:
    """Set value in cache with optional expiration."""
    client = await get_cache_client()
    return await client.set(key, value, ex=expire)


async def cache_delete(key: str) -> bool:
    """Delete key from cache."""
    client = await get_cache_client()
    return bool(await client.delete(key))


async def cache_exists(key: str) -> bool:
    """Check if key exists in cache."""
    client = await get_cache_client()
    return bool(await client.exists(key)) 