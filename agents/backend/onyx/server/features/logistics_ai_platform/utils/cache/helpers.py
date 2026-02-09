"""
Cache helper utilities

This module provides high-level cache operations to reduce code duplication
and ensure consistent caching patterns across the application.
"""

import asyncio
from typing import Optional, TypeVar, Callable, Awaitable, Any, List

from ..cache import cache_service
from ..logger import logger
from ..constants import DEFAULT_CACHE_TTL
from ..async_utils import run_in_background

T = TypeVar('T')


async def get_cached_or_fetch(
    cache_key: str,
    fetch_func: Callable[[], Awaitable[T]],
    ttl: int = DEFAULT_CACHE_TTL,
    model_class: Optional[type] = None
) -> T:
    """
    Get from cache or fetch using provided function
    
    Args:
        cache_key: Cache key to use
        fetch_func: Async function to fetch data if not in cache
        ttl: Time to live in seconds (default: 3600)
        model_class: Optional Pydantic model class to deserialize cached data
        
    Returns:
        Cached or freshly fetched data
        
    Example:
        quote = await get_cached_or_fetch(
            f"quote:{quote_id}",
            lambda: get_quote_domain(quote_id, repository),
            ttl=3600,
            model_class=QuoteResponse
        )
    """
    try:
        cached = await cache_service.get(cache_key)
        if cached is not None:
            if model_class:
                return model_class(**cached)
            return cached
    except Exception:
        pass
    
    result = await fetch_func()
    
    serialized = result.model_dump() if hasattr(result, 'model_dump') else result
    await run_in_background(cache_service.set(cache_key, serialized, ttl))
    return result


async def cache_entity(
    cache_key: str,
    entity: Any,
    ttl: int = DEFAULT_CACHE_TTL
) -> None:
    """
    Cache an entity (Pydantic model or dict)
    
    Args:
        cache_key: Cache key to use
        entity: Entity to cache (Pydantic model or dict)
        ttl: Time to live in seconds (default: 3600)
    """
    try:
        serialized = entity.model_dump() if hasattr(entity, 'model_dump') else entity
        await cache_service.set(cache_key, serialized, ttl)
    except Exception as e:
        logger.warning(f"Failed to cache entity {cache_key}: {e}")


async def invalidate_cache(cache_key: str) -> None:
    """
    Invalidate a cache key
    
    Args:
        cache_key: Cache key to invalidate
    """
    try:
        await cache_service.delete(cache_key)
        logger.debug(f"Invalidated cache: {cache_key}")
    except Exception as e:
        logger.warning(f"Failed to invalidate cache {cache_key}: {e}")


async def invalidate_cache_pattern(pattern: str) -> None:
    """
    Invalidate all cache keys matching a pattern
    
    Args:
        pattern: Pattern to match (e.g., "quote:*")
    """
    try:
        await cache_service.clear_pattern(pattern)
        logger.debug(f"Invalidated cache pattern: {pattern}")
    except Exception as e:
        logger.warning(f"Failed to invalidate cache pattern {pattern}: {e}")


async def invalidate_related_caches(cache_keys: List[str], patterns: Optional[List[str]] = None) -> None:
    """
    Invalidate multiple cache keys and patterns in parallel
    
    Args:
        cache_keys: List of cache keys to invalidate
        patterns: Optional list of cache patterns to invalidate
    """
    if not cache_keys and not patterns:
        return
    
    tasks = [cache_service.delete(key) for key in cache_keys]
    if patterns:
        tasks.extend(cache_service.clear_pattern(pattern) for pattern in patterns)
    
    if tasks:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                key_or_pattern = cache_keys[i] if i < len(cache_keys) else patterns[i - len(cache_keys)] if patterns else None
                if key_or_pattern:
                    logger.warning(f"Failed to invalidate {key_or_pattern}: {result}")







