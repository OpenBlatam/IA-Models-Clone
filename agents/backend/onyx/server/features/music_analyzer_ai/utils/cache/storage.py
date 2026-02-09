"""
Cache Storage Module

Cache storage operations.
"""

from typing import Any, Dict
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


def store_item(
    memory_cache: Dict[str, Dict[str, Any]],
    key: str,
    data: Any,
    ttl: int,
    max_items: int
) -> None:
    """
    Store item in cache.
    
    Args:
        memory_cache: Cache dictionary
        key: Cache key
        data: Data to store
        ttl: Time to live in seconds
        max_items: Maximum number of items
    """
    # Limpiar cache si está lleno
    if len(memory_cache) >= max_items:
        from .cleanup import cleanup_expired
        cleanup_expired(memory_cache)
        
        # Si aún está lleno, eliminar el más antiguo
        if len(memory_cache) >= max_items:
            oldest_key = min(
                memory_cache.keys(),
                key=lambda k: memory_cache[k].get("created_at", datetime.min)
            )
            del memory_cache[oldest_key]
    
    expires_at = datetime.now() + timedelta(seconds=ttl)
    
    memory_cache[key] = {
        "data": data,
        "expires_at": expires_at,
        "created_at": datetime.now()
    }
    
    logger.debug(f"Cache item stored: {key} (expires in {ttl}s)")


def set_item(
    memory_cache: Dict[str, Dict[str, Any]],
    key: str,
    data: Any,
    ttl: int,
    max_items: int
) -> None:
    """
    Set item in cache (alias for store_item).
    
    Args:
        memory_cache: Cache dictionary
        key: Cache key
        data: Data to store
        ttl: Time to live in seconds
        max_items: Maximum number of items
    """
    store_item(memory_cache, key, data, ttl, max_items)



