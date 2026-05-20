"""
Caching Routes - API endpoints for Advanced Caching Engine
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from ..core.advanced_caching_engine import (
    advanced_caching_engine,
    CacheConfig,
    CacheStrategy,
    SerializationFormat
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/caching", tags=["Advanced Caching"])


# Pydantic models
class CacheConfigRequest(BaseModel):
    """Request model for cache configuration"""
    max_size: int = Field(1000, description="Maximum cache size")
    ttl_seconds: int = Field(3600, description="Time to live in seconds")
    strategy: str = Field("lru", description="Cache strategy")
    serialization: str = Field("json", description="Serialization format")
    compression: bool = Field(False, description="Enable compression")
    compression_level: int = Field(6, description="Compression level")
    enable_redis: bool = Field(True, description="Enable Redis cache")
    enable_disk: bool = Field(False, description="Enable disk cache")
    disk_path: str = Field("./cache", description="Disk cache path")
    redis_url: str = Field("redis://localhost:6379", description="Redis URL")
    redis_db: int = Field(0, description="Redis database number")
    redis_password: Optional[str] = Field(None, description="Redis password")


class CacheOperationRequest(BaseModel):
    """Request model for cache operations"""
    key: str = Field(..., description="Cache key")
    value: Any = Field(..., description="Cache value")
    ttl: Optional[int] = Field(None, description="Time to live in seconds")


class CacheKeyRequest(BaseModel):
    """Request model for cache key operations"""
    key: str = Field(..., description="Cache key")


class CacheBatchRequest(BaseModel):
    """Request model for batch cache operations"""
    operations: List[Dict[str, Any]] = Field(..., description="List of cache operations")


class CacheFunctionRequest(BaseModel):
    """Request model for caching function results"""
    function_name: str = Field(..., description="Function name")
    key: str = Field(..., description="Cache key")
    ttl: Optional[int] = Field(None, description="Time to live in seconds")
    args: List[Any] = Field(default_factory=list, description="Function arguments")
    kwargs: Dict[str, Any] = Field(default_factory=dict, description="Function keyword arguments")


# Cache configuration endpoints
@router.get("/config", response_model=Dict[str, Any])
async def get_cache_config():
    """Get current cache configuration"""
    try:
        if not advanced_caching_engine:
            raise HTTPException(status_code=503, detail="Caching engine not initialized")
        
        config = advanced_caching_engine.config
        
        return {
            "success": True,
            "data": {
                "max_size": config.max_size,
                "ttl_seconds": config.ttl_seconds,
                "strategy": config.strategy.value,
                "serialization": config.serialization.value,
                "compression": config.compression,
                "compression_level": config.compression_level,
                "enable_redis": config.enable_redis,
                "enable_disk": config.enable_disk,
                "disk_path": config.disk_path,
                "redis_url": config.redis_url,
                "redis_db": config.redis_db,
                "redis_password": "***" if config.redis_password else None
            },
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error getting cache config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/config", response_model=Dict[str, Any])
async def update_cache_config(request: CacheConfigRequest):
    """Update cache configuration"""
    try:
        if not advanced_caching_engine:
            raise HTTPException(status_code=503, detail="Caching engine not initialized")
        
        # Create new config
        new_config = CacheConfig(
            max_size=request.max_size,
            ttl_seconds=request.ttl_seconds,
            strategy=CacheStrategy(request.strategy),
            serialization=SerializationFormat(request.serialization),
            compression=request.compression,
            compression_level=request.compression_level,
            enable_redis=request.enable_redis,
            enable_disk=request.enable_disk,
            disk_path=request.disk_path,
            redis_url=request.redis_url,
            redis_db=request.redis_db,
            redis_password=request.redis_password
        )
        
        # Update engine config
        advanced_caching_engine.config = new_config
        
        return {
            "success": True,
            "message": "Cache configuration updated successfully",
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error updating cache config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Cache operations endpoints
@router.get("/get/{key}", response_model=Dict[str, Any])
async def get_cache_value(key: str):
    """Get value from cache"""
    try:
        if not advanced_caching_engine:
            raise HTTPException(status_code=503, detail="Caching engine not initialized")
        
        value = await advanced_caching_engine.get(key)
        
        return {
            "success": True,
            "data": {
                "key": key,
                "value": value,
                "found": value is not None
            },
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error getting cache value for key '{key}': {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/set", response_model=Dict[str, Any])
async def set_cache_value(request: CacheOperationRequest):
    """Set value in cache"""
    try:
        if not advanced_caching_engine:
            raise HTTPException(status_code=503, detail="Caching engine not initialized")
        
        success = await advanced_caching_engine.set(request.key, request.value, request.ttl)
        
        return {
            "success": success,
            "message": f"Cache value {'set successfully' if success else 'failed to set'}",
            "data": {
                "key": request.key,
                "ttl": request.ttl
            },
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error setting cache value for key '{request.key}': {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/delete/{key}", response_model=Dict[str, Any])
async def delete_cache_value(key: str):
    """Delete value from cache"""
    try:
        if not advanced_caching_engine:
            raise HTTPException(status_code=503, detail="Caching engine not initialized")
        
        success = await advanced_caching_engine.delete(key)
        
        return {
            "success": success,
            "message": f"Cache value {'deleted successfully' if success else 'not found'}",
            "data": {
                "key": key
            },
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error deleting cache value for key '{key}': {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clear", response_model=Dict[str, Any])
async def clear_cache():
    """Clear all cache entries"""
    try:
        if not advanced_caching_engine:
            raise HTTPException(status_code=503, detail="Caching engine not initialized")
        
        await advanced_caching_engine.clear()
        
        return {
            "success": True,
            "message": "Cache cleared successfully",
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Batch operations endpoints
@router.post("/batch/get", response_model=Dict[str, Any])
async def batch_get_cache_values(request: List[str]):
    """Get multiple values from cache"""
    try:
        if not advanced_caching_engine:
            raise HTTPException(status_code=503, detail="Caching engine not initialized")
        
        results = {}
        for key in request:
            value = await advanced_caching_engine.get(key)
            results[key] = value
        
        return {
            "success": True,
            "data": results,
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error batch getting cache values: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch/set", response_model=Dict[str, Any])
async def batch_set_cache_values(request: CacheBatchRequest):
    """Set multiple values in cache"""
    try:
        if not advanced_caching_engine:
            raise HTTPException(status_code=503, detail="Caching engine not initialized")
        
        results = {}
        for operation in request.operations:
            key = operation.get("key")
            value = operation.get("value")
            ttl = operation.get("ttl")
            
            if key and value is not None:
                success = await advanced_caching_engine.set(key, value, ttl)
                results[key] = success
            else:
                results[key] = False
        
        return {
            "success": True,
            "data": results,
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error batch setting cache values: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch/delete", response_model=Dict[str, Any])
async def batch_delete_cache_values(request: List[str]):
    """Delete multiple values from cache"""
    try:
        if not advanced_caching_engine:
            raise HTTPException(status_code=503, detail="Caching engine not initialized")
        
        results = {}
        for key in request:
            success = await advanced_caching_engine.delete(key)
            results[key] = success
        
        return {
            "success": True,
            "data": results,
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error batch deleting cache values: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Cache statistics endpoints
@router.get("/stats", response_model=Dict[str, Any])
async def get_cache_stats():
    """Get cache statistics"""
    try:
        if not advanced_caching_engine:
            raise HTTPException(status_code=503, detail="Caching engine not initialized")
        
        stats = await advanced_caching_engine.get_stats()
        
        return {
            "success": True,
            "data": stats,
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/level/{level}", response_model=Dict[str, Any])
async def get_cache_level_stats(level: str):
    """Get statistics for specific cache level"""
    try:
        if not advanced_caching_engine:
            raise HTTPException(status_code=503, detail="Caching engine not initialized")
        
        stats = await advanced_caching_engine.get_stats()
        
        if level not in stats.get("cache_levels", {}):
            raise HTTPException(status_code=404, detail=f"Cache level '{level}' not found")
        
        level_stats = stats["cache_levels"][level]
        
        return {
            "success": True,
            "data": {
                "level": level,
                "stats": level_stats
            },
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error getting cache level stats for '{level}': {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Cache optimization endpoints
@router.post("/optimize", response_model=Dict[str, Any])
async def optimize_cache_strategy():
    """Optimize cache strategy"""
    try:
        if not advanced_caching_engine:
            raise HTTPException(status_code=503, detail="Caching engine not initialized")
        
        result = await advanced_caching_engine.optimize_cache_strategy()
        
        return {
            "success": True,
            "data": result,
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error optimizing cache strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Cache key generation endpoint
@router.post("/key/generate", response_model=Dict[str, Any])
async def generate_cache_key(
    prefix: str = Query(..., description="Key prefix"),
    args: List[str] = Query(default_factory=list, description="Key arguments"),
    kwargs: Dict[str, str] = Query(default_factory=dict, description="Key keyword arguments")
):
    """Generate consistent cache key"""
    try:
        if not advanced_caching_engine:
            raise HTTPException(status_code=503, detail="Caching engine not initialized")
        
        key = advanced_caching_engine.generate_cache_key(prefix, *args, **kwargs)
        
        return {
            "success": True,
            "data": {
                "prefix": prefix,
                "args": args,
                "kwargs": kwargs,
                "generated_key": key
            },
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error generating cache key: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Function caching endpoint
@router.post("/function/cache", response_model=Dict[str, Any])
async def cache_function_result(request: CacheFunctionRequest):
    """Cache function result"""
    try:
        if not advanced_caching_engine:
            raise HTTPException(status_code=503, detail="Caching engine not initialized")
        
        # This is a simplified example - in practice, you'd need to map function names to actual functions
        # For now, we'll just demonstrate the caching mechanism
        
        # Try to get from cache first
        cached_result = await advanced_caching_engine.get(request.key)
        if cached_result is not None:
            return {
                "success": True,
                "data": {
                    "function_name": request.function_name,
                    "key": request.key,
                    "result": cached_result,
                    "cached": True
                },
                "timestamp": datetime.now()
            }
        
        # Simulate function execution (in practice, this would call the actual function)
        result = f"Result of {request.function_name} with args {request.args} and kwargs {request.kwargs}"
        
        # Cache result
        await advanced_caching_engine.set(request.key, result, request.ttl)
        
        return {
            "success": True,
            "data": {
                "function_name": request.function_name,
                "key": request.key,
                "result": result,
                "cached": False
            },
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error caching function result: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Cache health check
@router.get("/health", response_model=Dict[str, Any])
async def cache_health_check():
    """Cache engine health check"""
    try:
        if not advanced_caching_engine:
            return {
                "status": "unhealthy",
                "service": "Advanced Caching Engine",
                "timestamp": datetime.now(),
                "error": "Caching engine not initialized"
            }
        
        # Test cache operations
        test_key = "health_check_test"
        test_value = "test_value"
        
        # Test set
        set_success = await advanced_caching_engine.set(test_key, test_value, 60)
        
        # Test get
        get_value = await advanced_caching_engine.get(test_key)
        
        # Test delete
        delete_success = await advanced_caching_engine.delete(test_key)
        
        # Get stats
        stats = await advanced_caching_engine.get_stats()
        
        return {
            "status": "healthy",
            "service": "Advanced Caching Engine",
            "timestamp": datetime.now(),
            "cache_levels": list(stats.get("cache_levels", {}).keys()),
            "test_operations": {
                "set": set_success,
                "get": get_value == test_value,
                "delete": delete_success
            },
            "overall_stats": stats.get("overall_stats", {})
        }
    except Exception as e:
        logger.error(f"Cache health check failed: {e}")
        return {
            "status": "unhealthy",
            "service": "Advanced Caching Engine",
            "timestamp": datetime.now(),
            "error": str(e)
        }


# Cache capabilities
@router.get("/capabilities", response_model=Dict[str, Any])
async def get_cache_capabilities():
    """Get cache capabilities"""
    return {
        "success": True,
        "data": {
            "cache_strategies": [strategy.value for strategy in CacheStrategy],
            "serialization_formats": [fmt.value for fmt in SerializationFormat],
            "cache_levels": {
                "memory": {
                    "description": "In-memory cache with multiple eviction strategies",
                    "features": ["LRU", "LFU", "TTL", "Random Replacement", "Adaptive"]
                },
                "redis": {
                    "description": "Distributed Redis cache",
                    "features": ["Distributed", "Persistent", "High Performance", "Clustering"]
                },
                "disk": {
                    "description": "Disk-based persistent cache",
                    "features": ["Persistent", "Large Capacity", "File-based"]
                }
            },
            "optimization_features": {
                "compression": "LZ4 and Brotli compression support",
                "serialization": "Multiple serialization formats",
                "hierarchy": "Multi-level cache hierarchy",
                "promotion": "Automatic cache promotion",
                "statistics": "Comprehensive cache statistics",
                "optimization": "Automatic cache strategy optimization"
            }
        },
        "timestamp": datetime.now()
    }