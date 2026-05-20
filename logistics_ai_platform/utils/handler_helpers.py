"""
Handler helper utilities

This module provides common utilities for handlers to reduce code duplication
and ensure consistent patterns across the application.
"""

from typing import Optional, TypeVar, Callable, Awaitable, List

from utils.exceptions import NotFoundError
from utils.cache.helpers import get_cached_or_fetch, cache_entity, invalidate_cache
from utils.logger import logger
from utils.constants import DEFAULT_CACHE_TTL

T = TypeVar('T')


async def get_entity_or_raise(
    entity_id: str,
    fetch_func: Callable[[], Awaitable[Optional[T]]],
    entity_name: str,
    cache_key: Optional[str] = None,
    cache_ttl: int = DEFAULT_CACHE_TTL,
    model_class: Optional[type] = None
) -> T:
    """
    Get entity by ID or raise NotFoundError
    
    Args:
        entity_id: Entity identifier
        fetch_func: Async function to fetch entity
        entity_name: Name of entity type (for error messages)
        cache_key: Optional cache key (if provided, uses cache)
        cache_ttl: Cache TTL in seconds (default: 3600)
        model_class: Optional Pydantic model class for cached data
        
    Returns:
        Entity if found
        
    Raises:
        NotFoundError: If entity not found
    """
    if cache_key:
        entity = await get_cached_or_fetch(
            cache_key,
            fetch_func,
            ttl=cache_ttl,
            model_class=model_class
        )
    else:
        entity = await fetch_func()
    
    if entity is None:
        raise NotFoundError(entity_name, entity_id)
    
    return entity


async def create_entity_with_cache(
    entity: T,
    cache_key: str,
    cache_ttl: int = DEFAULT_CACHE_TTL
) -> T:
    """
    Create entity and cache it (non-blocking cache)
    
    Args:
        entity: Entity to cache
        cache_key: Cache key to use
        cache_ttl: Cache TTL in seconds (default: 3600)
        
    Returns:
        Created entity
    """
    from utils.async_utils import run_in_background
    await run_in_background(cache_entity(cache_key, entity, cache_ttl))
    return entity


async def update_entity_with_cache_invalidation(
    entity: T,
    cache_key: str,
    related_cache_patterns: Optional[List[str]] = None
) -> T:
    """
    Update entity and invalidate related caches in parallel
    
    Args:
        entity: Updated entity
        cache_key: Cache key to invalidate
        related_cache_patterns: Optional list of cache patterns to invalidate
        
    Returns:
        Updated entity
    """
    from utils.cache.helpers import invalidate_related_caches
    
    cache_keys = [cache_key]
    await invalidate_related_caches(cache_keys, related_cache_patterns)
    return entity

