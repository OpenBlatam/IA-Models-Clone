from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

from fastapi import APIRouter, Depends, HTTPException
from typing import Optional
from ..dependencies import (
from ..core import CacheManager
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Cache routes for Ultra-Optimized SEO Service v15.

This module contains cache management endpoints including:
- Cache statistics
- Cache clearing operations
- Cache configuration
- Cache health monitoring
"""


    get_cache_manager,
    get_logger
)

# Create router with prefix and tags
router = APIRouter(
    prefix="/cache",
    tags=["Cache Management"],
    responses={
        400: {"description": "Bad Request"},
        500: {"description": "Internal Server Error"}
    }
)

@router.get("/stats")
async def cache_stats_endpoint(
    cache_manager: CacheManager = Depends(get_cache_manager),
    logger = Depends(get_logger)
):
    """
    Get cache statistics.
    
    Returns comprehensive cache statistics including:
    - Hit/miss ratios
    - Memory usage
    - Cache size
    - Performance metrics
    """
    try:
        stats = cache_manager.get_stats()
        return stats
    except Exception as e:
        logger.error("Failed to get cache stats", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve cache statistics")

@router.post("/clear")
async def clear_cache_endpoint(
    pattern: Optional[str] = None,
    cache_manager: CacheManager = Depends(get_cache_manager),
    logger = Depends(get_logger)
):
    """
    Clear cache entries.
    
    Clears cache entries based on pattern matching.
    If no pattern is provided, clears all cache entries.
    """
    try:
        await cache_manager.clear(pattern)
        return {
            "success": True,
            "message": f"Cache cleared successfully",
            "pattern": pattern
        }
    except Exception as e:
        logger.error("Failed to clear cache", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to clear cache")

@router.delete("/key/{key}")
async def delete_cache_key_endpoint(
    key: str,
    cache_manager: CacheManager = Depends(get_cache_manager),
    logger = Depends(get_logger)
):
    """
    Delete specific cache key.
    
    Deletes a specific cache entry by key.
    Returns success status and confirmation.
    """
    try:
        await cache_manager.delete(key)
        return {
            "success": True,
            "message": f"Cache key '{key}' deleted successfully"
        }
    except Exception as e:
        logger.error("Failed to delete cache key", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to delete cache key")

@router.get("/key/{key}")
async def get_cache_key_endpoint(
    key: str,
    cache_manager: CacheManager = Depends(get_cache_manager),
    logger = Depends(get_logger)
):
    """
    Get cache entry by key.
    
    Retrieves a specific cache entry by key.
    Returns the cached data if found, 404 if not found.
    """
    try:
        data = await cache_manager.get(key)
        if data:
            return {"key": key, "data": data}
        else:
            raise HTTPException(status_code=404, detail="Cache key not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get cache key", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve cache key")

@router.post("/key/{key}")
async def set_cache_key_endpoint(
    key: str,
    data: dict,
    ttl: int = 3600,
    cache_manager: CacheManager = Depends(get_cache_manager),
    logger = Depends(get_logger)
):
    """
    Set cache entry by key.
    
    Sets a cache entry with the specified key, data, and TTL.
    Returns success status and confirmation.
    """
    try:
        await cache_manager.set(key, data, ttl)
        return {
            "success": True,
            "message": f"Cache key '{key}' set successfully",
            "ttl": ttl
        }
    except Exception as e:
        logger.error("Failed to set cache key", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to set cache key")

@router.get("/health")
async def cache_health_endpoint(
    cache_manager: CacheManager = Depends(get_cache_manager),
    logger = Depends(get_logger)
):
    """
    Get cache health status.
    
    Returns cache health information including:
    - Connection status
    - Performance metrics
    - Error rates
    - Health indicators
    """
    try:
        health = cache_manager.get_health_status()
        return health
    except Exception as e:
        logger.error("Failed to get cache health", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve cache health")

@router.post("/warm")
async def warm_cache_endpoint(
    cache_manager: CacheManager = Depends(get_cache_manager),
    logger = Depends(get_logger)
):
    """
    Warm up cache with frequently accessed data.
    
    Pre-loads cache with commonly accessed data to improve performance.
    Useful after cache clearing or service restart.
    """
    try:
        await cache_manager.warm_cache()
        return {
            "success": True,
            "message": "Cache warming completed successfully"
        }
    except Exception as e:
        logger.error("Failed to warm cache", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to warm cache")

@router.get("/config")
async def get_cache_config_endpoint(
    cache_manager: CacheManager = Depends(get_cache_manager),
    logger = Depends(get_logger)
):
    """
    Get cache configuration.
    
    Returns current cache configuration including:
    - TTL settings
    - Memory limits
    - Eviction policies
    - Connection settings
    """
    try:
        config = cache_manager.get_config()
        return config
    except Exception as e:
        logger.error("Failed to get cache config", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve cache configuration") 