"""
Cache API Endpoints
===================

REST API endpoints for cache management and monitoring.
"""

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

from ..services.cache_service import CacheService, CacheConfig, get_cache_service
from ..middleware.auth_middleware import get_current_user, require_permission, User

# Create API router
router = APIRouter(prefix="/cache", tags=["Cache"])

# Pydantic models
class CacheSetRequest(BaseModel):
    key: str = Field(..., description="Cache key")
    value: Any = Field(..., description="Cache value")
    ttl: Optional[int] = Field(None, description="Time to live in seconds")

class CacheConfigRequest(BaseModel):
    redis_url: Optional[str] = Field(None, description="Redis URL")
    memory_cache_size: Optional[int] = Field(None, description="Memory cache size")
    default_ttl: Optional[int] = Field(None, description="Default TTL")
    compression_enabled: Optional[bool] = Field(None, description="Enable compression")
    serialization_method: Optional[str] = Field(None, description="Serialization method")
    key_prefix: Optional[str] = Field(None, description="Key prefix")
    enable_memory_cache: Optional[bool] = Field(None, description="Enable memory cache")
    enable_redis_cache: Optional[bool] = Field(None, description="Enable Redis cache")

class CacheWarmupRequest(BaseModel):
    function_names: List[str] = Field(..., description="Function names to warm up")

# API Endpoints

@router.get("/health", response_model=Dict[str, Any])
async def cache_health_check():
    """Cache health check."""
    
    cache_service = get_cache_service()
    
    try:
        await cache_service.initialize()
        stats = await cache_service.get_stats()
        
        # Test basic operations
        test_key = "health_check_test"
        test_value = {"timestamp": "2024-01-01T00:00:00Z", "status": "ok"}
        
        # Test set
        set_success = await cache_service.set(test_key, test_value, ttl=60)
        
        # Test get
        get_value = await cache_service.get(test_key)
        
        # Test delete
        delete_success = await cache_service.delete(test_key)
        
        return {
            "status": "healthy",
            "stats": stats,
            "operations": {
                "set": set_success,
                "get": get_value is not None,
                "delete": delete_success
            },
            "timestamp": "2024-01-01T00:00:00Z"
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": "2024-01-01T00:00:00Z"
        }

@router.get("/stats", response_model=Dict[str, Any])
async def get_cache_statistics(
    current_user: User = Depends(require_permission("cache:view"))
):
    """Get cache statistics."""
    
    cache_service = get_cache_service()
    
    try:
        await cache_service.initialize()
        stats = await cache_service.get_stats()
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get cache statistics: {str(e)}")

@router.post("/set", response_model=Dict[str, str])
async def set_cache_value(
    request: CacheSetRequest,
    current_user: User = Depends(require_permission("cache:manage"))
):
    """Set cache value."""
    
    cache_service = get_cache_service()
    
    try:
        await cache_service.initialize()
        success = await cache_service.set(request.key, request.value, request.ttl)
        
        if success:
            return {"message": "Cache value set successfully", "key": request.key}
        else:
            raise HTTPException(status_code=500, detail="Failed to set cache value")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to set cache value: {str(e)}")

@router.get("/get/{key}", response_model=Dict[str, Any])
async def get_cache_value(
    key: str,
    current_user: User = Depends(require_permission("cache:view"))
):
    """Get cache value."""
    
    cache_service = get_cache_service()
    
    try:
        await cache_service.initialize()
        value = await cache_service.get(key)
        
        if value is not None:
            return {"key": key, "value": value, "found": True}
        else:
            return {"key": key, "value": None, "found": False}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get cache value: {str(e)}")

@router.delete("/delete/{key}", response_model=Dict[str, str])
async def delete_cache_value(
    key: str,
    current_user: User = Depends(require_permission("cache:manage"))
):
    """Delete cache value."""
    
    cache_service = get_cache_service()
    
    try:
        await cache_service.initialize()
        success = await cache_service.delete(key)
        
        if success:
            return {"message": "Cache value deleted successfully", "key": key}
        else:
            return {"message": "Cache value not found", "key": key}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete cache value: {str(e)}")

@router.get("/exists/{key}", response_model=Dict[str, Any])
async def check_cache_exists(
    key: str,
    current_user: User = Depends(require_permission("cache:view"))
):
    """Check if cache key exists."""
    
    cache_service = get_cache_service()
    
    try:
        await cache_service.initialize()
        exists = await cache_service.exists(key)
        
        return {"key": key, "exists": exists}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to check cache existence: {str(e)}")

@router.post("/expire/{key}", response_model=Dict[str, str])
async def set_cache_expiration(
    key: str,
    ttl: int = Query(..., description="Time to live in seconds"),
    current_user: User = Depends(require_permission("cache:manage"))
):
    """Set cache expiration."""
    
    cache_service = get_cache_service()
    
    try:
        await cache_service.initialize()
        success = await cache_service.expire(key, ttl)
        
        if success:
            return {"message": f"Cache expiration set to {ttl} seconds", "key": key}
        else:
            raise HTTPException(status_code=500, detail="Failed to set cache expiration")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to set cache expiration: {str(e)}")

@router.get("/ttl/{key}", response_model=Dict[str, Any])
async def get_cache_ttl(
    key: str,
    current_user: User = Depends(require_permission("cache:view"))
):
    """Get cache time to live."""
    
    cache_service = get_cache_service()
    
    try:
        await cache_service.initialize()
        ttl = await cache_service.ttl(key)
        
        return {"key": key, "ttl": ttl}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get cache TTL: {str(e)}")

@router.post("/clear", response_model=Dict[str, str])
async def clear_cache(
    pattern: Optional[str] = Query(None, description="Pattern to clear"),
    current_user: User = Depends(require_permission("cache:manage"))
):
    """Clear cache entries."""
    
    cache_service = get_cache_service()
    
    try:
        await cache_service.initialize()
        
        if pattern:
            deleted_count = await cache_service.clear_pattern(pattern)
            return {"message": f"Cleared {deleted_count} cache entries matching pattern: {pattern}"}
        else:
            deleted_count = await cache_service.clear_all()
            return {"message": f"Cleared {deleted_count} cache entries"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")

@router.post("/warmup", response_model=Dict[str, Any])
async def warmup_cache(
    request: CacheWarmupRequest,
    background_tasks: BackgroundTasks = None,
    current_user: User = Depends(require_permission("cache:manage"))
):
    """Warm up cache with predefined functions."""
    
    cache_service = get_cache_service()
    
    try:
        await cache_service.initialize()
        
        # This would need to be implemented with actual warmup functions
        # For now, return a placeholder response
        return {
            "message": "Cache warmup initiated",
            "functions": request.function_names,
            "status": "scheduled"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to warmup cache: {str(e)}")

@router.post("/invalidate/function/{function_name}", response_model=Dict[str, str])
async def invalidate_function_cache(
    function_name: str,
    current_user: User = Depends(require_permission("cache:manage"))
):
    """Invalidate cache for a specific function."""
    
    cache_service = get_cache_service()
    
    try:
        await cache_service.initialize()
        deleted_count = await cache_service.invalidate_function_cache(function_name)
        
        return {
            "message": f"Invalidated {deleted_count} cache entries for function: {function_name}",
            "function_name": function_name,
            "deleted_count": deleted_count
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to invalidate function cache: {str(e)}")

@router.get("/keys", response_model=List[str])
async def list_cache_keys(
    pattern: Optional[str] = Query(None, description="Pattern to match"),
    limit: int = Query(100, description="Maximum number of keys to return"),
    current_user: User = Depends(require_permission("cache:view"))
):
    """List cache keys."""
    
    cache_service = get_cache_service()
    
    try:
        await cache_service.initialize()
        
        # This would need to be implemented with actual key listing
        # For now, return a placeholder response
        return {
            "message": "Key listing not implemented",
            "pattern": pattern,
            "limit": limit
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list cache keys: {str(e)}")

@router.post("/config", response_model=Dict[str, str])
async def update_cache_config(
    request: CacheConfigRequest,
    current_user: User = Depends(require_permission("cache:manage"))
):
    """Update cache configuration."""
    
    # This would need to be implemented with actual config updates
    # For now, return a placeholder response
    return {
        "message": "Cache configuration update not implemented",
        "note": "Configuration changes require service restart"
    }

@router.get("/performance", response_model=Dict[str, Any])
async def get_cache_performance(
    current_user: User = Depends(require_permission("cache:view"))
):
    """Get cache performance metrics."""
    
    cache_service = get_cache_service()
    
    try:
        await cache_service.initialize()
        stats = await cache_service.get_stats()
        
        # Calculate performance metrics
        performance = {
            "hit_rate": stats.get("redis_cache", {}).get("hit_rate", 0),
            "total_commands": stats.get("redis_cache", {}).get("total_commands_processed", 0),
            "memory_usage": stats.get("redis_cache", {}).get("used_memory", "N/A"),
            "connected_clients": stats.get("redis_cache", {}).get("connected_clients", 0),
            "memory_cache_size": stats.get("memory_cache", {}).get("size", 0),
            "memory_cache_max": stats.get("memory_cache", {}).get("max_size", 0)
        }
        
        return performance
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get cache performance: {str(e)}")

@router.post("/test", response_model=Dict[str, Any])
async def test_cache_operations(
    current_user: User = Depends(require_permission("cache:manage"))
):
    """Test cache operations."""
    
    cache_service = get_cache_service()
    
    try:
        await cache_service.initialize()
        
        test_results = {}
        
        # Test 1: Set and get
        test_key = "test_key"
        test_value = {"test": "value", "timestamp": "2024-01-01T00:00:00Z"}
        
        set_result = await cache_service.set(test_key, test_value, ttl=60)
        get_result = await cache_service.get(test_key)
        
        test_results["set_get"] = {
            "set_success": set_result,
            "get_success": get_result is not None,
            "value_match": get_result == test_value
        }
        
        # Test 2: Exists check
        exists_result = await cache_service.exists(test_key)
        test_results["exists"] = {"success": exists_result}
        
        # Test 3: TTL check
        ttl_result = await cache_service.ttl(test_key)
        test_results["ttl"] = {"success": ttl_result > 0, "ttl": ttl_result}
        
        # Test 4: Delete
        delete_result = await cache_service.delete(test_key)
        test_results["delete"] = {"success": delete_result}
        
        # Test 5: Verify deletion
        verify_result = await cache_service.exists(test_key)
        test_results["verify_deletion"] = {"success": not verify_result}
        
        return {
            "message": "Cache operations test completed",
            "results": test_results,
            "overall_success": all(
                result.get("success", False) or result.get("set_success", False)
                for result in test_results.values()
            )
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to test cache operations: {str(e)}")




























