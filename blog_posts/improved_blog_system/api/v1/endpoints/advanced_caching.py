"""
Advanced Caching API endpoints
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel, Field

from ....services.advanced_caching_service import AdvancedCachingService
from ....api.dependencies import CurrentUserDep, DatabaseSessionDep
from ....core.exceptions import DatabaseError, ValidationError

router = APIRouter()


class CacheSetRequest(BaseModel):
    """Request model for setting cache values."""
    key: str = Field(..., description="Cache key")
    value: Any = Field(..., description="Cache value")
    ttl: Optional[int] = Field(default=3600, description="Time to live in seconds")
    layers: Optional[List[str]] = Field(default=None, description="Cache layers to use")
    serialize: bool = Field(default=True, description="Whether to serialize the value")


class CacheWarmRequest(BaseModel):
    """Request model for warming cache."""
    keys_and_values: Dict[str, Any] = Field(..., description="Key-value pairs to cache")
    ttl: int = Field(default=3600, description="Time to live in seconds")


class CacheClearRequest(BaseModel):
    """Request model for clearing cache."""
    pattern: Optional[str] = Field(default=None, description="Pattern to match keys")
    layer: Optional[str] = Field(default=None, description="Specific layer to clear")


async def get_caching_service(session: DatabaseSessionDep) -> AdvancedCachingService:
    """Get caching service instance."""
    return AdvancedCachingService(session)


@router.get("/get/{key}", response_model=Dict[str, Any])
async def get_cache_value(
    key: str,
    layer: Optional[str] = Query(None, description="Specific cache layer"),
    deserialize: bool = Query(default=True, description="Whether to deserialize the value"),
    caching_service: AdvancedCachingService = Depends(get_caching_service),
    current_user: CurrentUserDep = Depends()
):
    """Get value from cache."""
    try:
        value = await caching_service.get(key, layer=layer, deserialize=deserialize)
        
        return {
            "success": True,
            "data": {
                "key": key,
                "value": value,
                "found": value is not None
            },
            "message": "Cache value retrieved successfully"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get cache value"
        )


@router.post("/set", response_model=Dict[str, Any])
async def set_cache_value(
    request: CacheSetRequest = Depends(),
    caching_service: AdvancedCachingService = Depends(get_caching_service),
    current_user: CurrentUserDep = Depends()
):
    """Set value in cache."""
    try:
        success = await caching_service.set(
            key=request.key,
            value=request.value,
            ttl=request.ttl,
            layers=request.layers,
            serialize=request.serialize
        )
        
        return {
            "success": success,
            "data": {
                "key": request.key,
                "ttl": request.ttl,
                "layers": request.layers or caching_service.cache_layers
            },
            "message": "Cache value set successfully" if success else "Failed to set cache value"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to set cache value"
        )


@router.delete("/delete/{key}", response_model=Dict[str, Any])
async def delete_cache_value(
    key: str,
    layers: Optional[List[str]] = Query(default=None, description="Specific layers to delete from"),
    caching_service: AdvancedCachingService = Depends(get_caching_service),
    current_user: CurrentUserDep = Depends()
):
    """Delete value from cache."""
    try:
        success = await caching_service.delete(key, layers=layers)
        
        return {
            "success": success,
            "data": {
                "key": key,
                "layers": layers or caching_service.cache_layers
            },
            "message": "Cache value deleted successfully" if success else "Failed to delete cache value"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete cache value"
        )


@router.get("/exists/{key}", response_model=Dict[str, Any])
async def check_cache_exists(
    key: str,
    layer: Optional[str] = Query(None, description="Specific cache layer"),
    caching_service: AdvancedCachingService = Depends(get_caching_service),
    current_user: CurrentUserDep = Depends()
):
    """Check if key exists in cache."""
    try:
        exists = await caching_service.exists(key, layer=layer)
        
        return {
            "success": True,
            "data": {
                "key": key,
                "exists": exists,
                "layer": layer or "all"
            },
            "message": "Cache existence check completed"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to check cache existence"
        )


@router.post("/clear", response_model=Dict[str, Any])
async def clear_cache(
    request: CacheClearRequest = Depends(),
    caching_service: AdvancedCachingService = Depends(get_caching_service),
    current_user: CurrentUserDep = Depends()
):
    """Clear cache entries."""
    try:
        cleared_count = await caching_service.clear(
            pattern=request.pattern,
            layer=request.layer
        )
        
        return {
            "success": True,
            "data": {
                "cleared_count": cleared_count,
                "pattern": request.pattern,
                "layer": request.layer or "all"
            },
            "message": f"Cache cleared successfully, {cleared_count} entries removed"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear cache"
        )


@router.get("/stats", response_model=Dict[str, Any])
async def get_cache_stats(
    caching_service: AdvancedCachingService = Depends(get_caching_service),
    current_user: CurrentUserDep = Depends()
):
    """Get cache statistics."""
    try:
        stats = await caching_service.get_stats()
        
        return {
            "success": True,
            "data": stats,
            "message": "Cache statistics retrieved successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get cache statistics"
        )


@router.post("/warm", response_model=Dict[str, Any])
async def warm_cache(
    request: CacheWarmRequest = Depends(),
    caching_service: AdvancedCachingService = Depends(get_caching_service),
    current_user: CurrentUserDep = Depends()
):
    """Warm cache with multiple key-value pairs."""
    try:
        success_count = await caching_service.warm_cache(
            keys_and_values=request.keys_and_values,
            ttl=request.ttl
        )
        
        return {
            "success": True,
            "data": {
                "total_keys": len(request.keys_and_values),
                "successful_keys": success_count,
                "ttl": request.ttl
            },
            "message": f"Cache warmed successfully, {success_count}/{len(request.keys_and_values)} keys cached"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to warm cache"
        )


@router.get("/memory-usage", response_model=Dict[str, Any])
async def get_memory_usage(
    caching_service: AdvancedCachingService = Depends(get_caching_service),
    current_user: CurrentUserDep = Depends()
):
    """Get memory usage statistics."""
    try:
        memory_stats = await caching_service.get_memory_usage()
        
        return {
            "success": True,
            "data": memory_stats,
            "message": "Memory usage statistics retrieved successfully"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get memory usage statistics"
        )


@router.post("/optimize", response_model=Dict[str, Any])
async def optimize_cache(
    caching_service: AdvancedCachingService = Depends(get_caching_service),
    current_user: CurrentUserDep = Depends()
):
    """Optimize cache by cleaning expired entries and optimizing memory usage."""
    try:
        optimization_results = await caching_service.optimize_cache()
        
        return {
            "success": True,
            "data": optimization_results,
            "message": "Cache optimization completed successfully"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to optimize cache"
        )


@router.get("/layers", response_model=Dict[str, Any])
async def get_cache_layers(
    caching_service: AdvancedCachingService = Depends(get_caching_service),
    current_user: CurrentUserDep = Depends()
):
    """Get available cache layers and their status."""
    try:
        layers_info = {
            "memory": {
                "name": "Memory Cache",
                "description": "In-memory cache for fastest access",
                "enabled": True,
                "size": len(caching_service.memory_cache)
            },
            "redis": {
                "name": "Redis Cache",
                "description": "Redis-based distributed cache",
                "enabled": caching_service.redis_client is not None,
                "connected": caching_service.redis_client is not None
            },
            "database": {
                "name": "Database Cache",
                "description": "Persistent cache stored in database",
                "enabled": True,
                "persistent": True
            }
        }
        
        return {
            "success": True,
            "data": {
                "layers": layers_info,
                "active_layers": caching_service.cache_layers,
                "default_ttl": caching_service.default_ttl
            },
            "message": "Cache layers information retrieved successfully"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get cache layers information"
        )


@router.get("/performance", response_model=Dict[str, Any])
async def get_cache_performance(
    caching_service: AdvancedCachingService = Depends(get_caching_service),
    current_user: CurrentUserDep = Depends()
):
    """Get cache performance metrics."""
    try:
        stats = await caching_service.get_stats()
        
        # Calculate performance metrics
        total_requests = stats["cache_stats"]["hits"] + stats["cache_stats"]["misses"]
        hit_rate = stats["hit_rate"]
        
        performance_metrics = {
            "hit_rate": hit_rate,
            "total_requests": total_requests,
            "cache_hits": stats["cache_stats"]["hits"],
            "cache_misses": stats["cache_stats"]["misses"],
            "cache_sets": stats["cache_stats"]["sets"],
            "cache_deletes": stats["cache_stats"]["deletes"],
            "cache_expires": stats["cache_stats"]["expires"],
            "performance_grade": "A" if hit_rate >= 90 else "B" if hit_rate >= 80 else "C" if hit_rate >= 70 else "D" if hit_rate >= 60 else "F"
        }
        
        return {
            "success": True,
            "data": performance_metrics,
            "message": "Cache performance metrics retrieved successfully"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get cache performance metrics"
        )


@router.get("/health", response_model=Dict[str, Any])
async def get_cache_health(
    caching_service: AdvancedCachingService = Depends(get_caching_service),
    current_user: CurrentUserDep = Depends()
):
    """Get cache system health status."""
    try:
        # Get cache stats
        stats = await caching_service.get_stats()
        
        # Calculate health metrics
        hit_rate = stats["hit_rate"]
        memory_usage = stats["memory_cache"]["size"]
        
        # Check Redis connection
        redis_healthy = stats["redis_cache"].get("connected", False)
        
        # Calculate health score
        health_score = 100
        if hit_rate < 70:
            health_score -= 30
        if memory_usage > 1000:
            health_score -= 20
        if not redis_healthy:
            health_score -= 10
        
        health_status = "excellent" if health_score >= 90 else "good" if health_score >= 70 else "fair" if health_score >= 50 else "poor"
        
        return {
            "success": True,
            "data": {
                "health_status": health_status,
                "health_score": health_score,
                "hit_rate": hit_rate,
                "memory_usage": memory_usage,
                "redis_connected": redis_healthy,
                "active_layers": len(stats["cache_layers"]),
                "timestamp": "2024-01-15T10:00:00Z"
            },
            "message": "Cache health status retrieved successfully"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get cache health status"
        )


@router.post("/invalidate-pattern", response_model=Dict[str, Any])
async def invalidate_cache_pattern(
    pattern: str = Query(..., description="Pattern to match keys for invalidation"),
    caching_service: AdvancedCachingService = Depends(get_caching_service),
    current_user: CurrentUserDep = Depends()
):
    """Invalidate cache entries matching a pattern."""
    try:
        invalidated_count = await caching_service.invalidate_pattern(pattern)
        
        return {
            "success": True,
            "data": {
                "pattern": pattern,
                "invalidated_count": invalidated_count
            },
            "message": f"Cache pattern '{pattern}' invalidated successfully, {invalidated_count} entries removed"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to invalidate cache pattern"
        )


@router.get("/keys", response_model=Dict[str, Any])
async def list_cache_keys(
    pattern: Optional[str] = Query(None, description="Pattern to filter keys"),
    limit: int = Query(default=100, ge=1, le=1000, description="Maximum number of keys to return"),
    caching_service: AdvancedCachingService = Depends(get_caching_service),
    current_user: CurrentUserDep = Depends()
):
    """List cache keys (memory cache only for security)."""
    try:
        # Only return memory cache keys for security
        memory_keys = list(caching_service.memory_cache.keys())
        
        if pattern:
            memory_keys = [key for key in memory_keys if pattern in key]
        
        # Limit results
        memory_keys = memory_keys[:limit]
        
        return {
            "success": True,
            "data": {
                "keys": memory_keys,
                "total_keys": len(memory_keys),
                "pattern": pattern,
                "limit": limit
            },
            "message": "Cache keys retrieved successfully"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list cache keys"
        )

























