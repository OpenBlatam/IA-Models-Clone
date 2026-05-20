from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import asyncio
import time
import json
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Query, Depends, Request
from fastapi.responses import JSONResponse
import logging
from core.shared_resources import (
from types.optimized_schemas import (
        import psutil
from typing import Any, List, Dict, Optional
"""
Shared Resources Routes for Instagram Captions API v14.0

Specialized routes for:
- Connection pool management
- Cache operations
- AI model management
- Resource monitoring
- Health checks
- Performance metrics
"""


# Import shared resources components
    SharedResources, ResourceConfig, ResourceType, ResourceState,
    get_shared_resources, initialize_shared_resources, shutdown_shared_resources,
    database_session, http_client, redis_client, with_cache, with_ai_model
)

# Import schemas
    PerformanceMetrics, APIErrorResponse
)

logger = logging.getLogger(__name__)

# Create router
shared_resources_router = APIRouter(prefix="/resources", tags=["shared-resources"])


# =============================================================================
# RESOURCE MANAGEMENT ENDPOINTS
# =============================================================================

@shared_resources_router.post("/initialize")
async def initialize_resources(
    config: Optional[ResourceConfig] = None
) -> Dict[str, Any]:
    """
    Initialize shared resources
    
    Initializes all shared resources including:
    - Database connection pools
    - HTTP client pools
    - Redis connection pools
    - AI model pools
    - Shared caches
    """
    
    try:
        if config is None:
            config = ResourceConfig()
        
        resources = await initialize_shared_resources(config)
        
        return {
            "success": True,
            "message": "Shared resources initialized successfully",
            "timestamp": time.time(),
            "config": {
                "database_pool_size": config.database_pool_size,
                "http_max_connections": config.http_max_connections,
                "redis_pool_size": config.redis_pool_size,
                "model_cache_size": config.model_cache_size,
                "cache_size": config.cache_size
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to initialize shared resources: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize shared resources: {str(e)}"
        )


@shared_resources_router.post("/shutdown")
async def shutdown_resources() -> Dict[str, Any]:
    """
    Shutdown shared resources
    
    Gracefully shuts down all shared resources and cleans up connections.
    """
    
    try:
        await shutdown_shared_resources()
        
        return {
            "success": True,
            "message": "Shared resources shut down successfully",
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Failed to shutdown shared resources: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to shutdown shared resources: {str(e)}"
        )


# =============================================================================
# CONNECTION POOL MANAGEMENT
# =============================================================================

@shared_resources_router.get("/pools/database/status")
async def get_database_pool_status() -> Dict[str, Any]:
    """
    Get database pool status
    
    Returns information about database connection pool including:
    - Pool size and usage
    - Connection statistics
    - Health status
    """
    
    try:
        resources = await get_shared_resources()
        
        # Test database connection
        async with database_session() as session:
            # Simple query to test connection
            result = await session.execute("SELECT 1")
            await result.fetchone()
        
        return {
            "success": True,
            "pool_info": {
                "pool_size": resources.database_pool.pool_size,
                "max_overflow": resources.database_pool.max_overflow,
                "available_connections": len(resources.database_pool._pool),
                "in_use_connections": len(resources.database_pool._in_use),
                "total_connections": len(resources.database_pool._pool) + len(resources.database_pool._in_use)
            },
            "health": "healthy",
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Database pool status check failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "health": "unhealthy",
            "timestamp": time.time()
        }


@shared_resources_router.get("/pools/http/status")
async async def get_http_pool_status() -> Dict[str, Any]:
    """
    Get HTTP client pool status
    
    Returns information about HTTP client connection pool.
    """
    
    try:
        resources = await get_shared_resources()
        
        # Test HTTP connection
        async with http_client() as client:
            # Simple request to test connection
            async with client.get("https://httpbin.org/get") as response:
                await response.json()
        
        return {
            "success": True,
            "pool_info": {
                "pool_size": resources.http_pool.pool_size,
                "max_overflow": resources.http_pool.max_overflow,
                "timeout": resources.http_pool.timeout,
                "available_connections": len(resources.http_pool._pool),
                "in_use_connections": len(resources.http_pool._in_use)
            },
            "health": "healthy",
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"HTTP pool status check failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "health": "unhealthy",
            "timestamp": time.time()
        }


@shared_resources_router.get("/pools/redis/status")
async def get_redis_pool_status() -> Dict[str, Any]:
    """
    Get Redis pool status
    
    Returns information about Redis connection pool.
    """
    
    try:
        resources = await get_shared_resources()
        
        # Test Redis connection
        async with redis_client() as redis:
            await redis.ping()
        
        return {
            "success": True,
            "pool_info": {
                "pool_size": resources.redis_pool.pool_size,
                "max_overflow": resources.redis_pool.max_overflow,
                "available_connections": len(resources.redis_pool._pool),
                "in_use_connections": len(resources.redis_pool._in_use)
            },
            "health": "healthy",
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Redis pool status check failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "health": "unhealthy",
            "timestamp": time.time()
        }


# =============================================================================
# CACHE MANAGEMENT
# =============================================================================

@shared_resources_router.get("/cache/status")
async def get_cache_status() -> Dict[str, Any]:
    """
    Get cache status and statistics
    
    Returns detailed information about shared cache including:
    - Cache size and usage
    - Hit/miss statistics
    - Performance metrics
    """
    
    try:
        resources = await get_shared_resources()
        cache_stats = await resources.shared_cache.get_stats()
        
        return {
            "success": True,
            "cache_stats": cache_stats,
            "performance": {
                "cache_hits": resources.stats["cache_hits"],
                "total_requests": resources.stats["total_requests"],
                "hit_rate": (resources.stats["cache_hits"] / max(1, resources.stats["total_requests"])) * 100
            },
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Cache status check failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get cache status: {str(e)}"
        )


@shared_resources_router.post("/cache/clear")
async def clear_cache() -> Dict[str, Any]:
    """
    Clear all cache
    
    Clears all cached data and resets cache statistics.
    """
    
    try:
        resources = await get_shared_resources()
        await resources.shared_cache.clear()
        
        return {
            "success": True,
            "message": "Cache cleared successfully",
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear cache: {str(e)}"
        )


@shared_resources_router.get("/cache/{key}")
async def get_cache_value(key: str) -> Dict[str, Any]:
    """
    Get a specific cache value
    
    Retrieves a specific value from the shared cache.
    """
    
    try:
        resources = await get_shared_resources()
        value = await resources.get_cache(key)
        
        if value is None:
            return {
                "success": False,
                "message": "Key not found in cache",
                "key": key,
                "timestamp": time.time()
            }
        
        return {
            "success": True,
            "key": key,
            "value": value,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Failed to get cache value: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get cache value: {str(e)}"
        )


@shared_resources_router.post("/cache/{key}")
async def set_cache_value(
    key: str,
    value: Any,
    ttl: Optional[int] = Query(default=None, description="Time to live in seconds")
) -> Dict[str, Any]:
    """
    Set a cache value
    
    Sets a value in the shared cache with optional TTL.
    """
    
    try:
        resources = await get_shared_resources()
        await resources.set_cache(key, value, ttl)
        
        return {
            "success": True,
            "message": "Value set in cache successfully",
            "key": key,
            "ttl": ttl,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Failed to set cache value: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to set cache value: {str(e)}"
        )


# =============================================================================
# AI MODEL MANAGEMENT
# =============================================================================

@shared_resources_router.get("/models/status")
async def get_ai_models_status() -> Dict[str, Any]:
    """
    Get AI models status
    
    Returns information about loaded AI models including:
    - Model names and states
    - Memory usage
    - Access statistics
    """
    
    try:
        resources = await get_shared_resources()
        model_info = await resources.ai_model_pool.get_model_info()
        
        return {
            "success": True,
            "models": model_info,
            "pool_info": {
                "cache_size": resources.ai_model_pool.model_cache_size,
                "memory_limit_mb": resources.ai_model_pool.memory_limit_mb,
                "loaded_models": len(model_info)
            },
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Failed to get AI models status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get AI models status: {str(e)}"
        )


@shared_resources_router.post("/models/{model_name}/load")
async def load_ai_model(
    model_name: str,
    loader_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Load an AI model
    
    Loads a specific AI model into the model pool.
    """
    
    try:
        resources = await get_shared_resources()
        
        # Mock loader function - in real implementation, this would load actual models
        def mock_loader():
            
    """mock_loader function."""
return {"model_name": model_name, "loaded_at": time.time()}
        
        model = await resources.get_ai_model(model_name, mock_loader)
        
        return {
            "success": True,
            "message": f"Model {model_name} loaded successfully",
            "model_info": model,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Failed to load AI model {model_name}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load AI model: {str(e)}"
        )


@shared_resources_router.delete("/models/{model_name}")
async def unload_ai_model(model_name: str) -> Dict[str, Any]:
    """
    Unload an AI model
    
    Unloads a specific AI model from the model pool.
    """
    
    try:
        resources = await get_shared_resources()
        await resources.ai_model_pool.unload_model(model_name)
        
        return {
            "success": True,
            "message": f"Model {model_name} unloaded successfully",
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Failed to unload AI model {model_name}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to unload AI model: {str(e)}"
        )


# =============================================================================
# RESOURCE MONITORING
# =============================================================================

@shared_resources_router.get("/monitoring/stats")
async def get_resource_stats() -> Dict[str, Any]:
    """
    Get comprehensive resource statistics
    
    Returns detailed statistics about all shared resources including:
    - Connection pool usage
    - Cache performance
    - AI model statistics
    - System metrics
    """
    
    try:
        resources = await get_shared_resources()
        stats = await resources.get_stats()
        
        # Add system metrics
        memory_info = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent()
        
        stats["system_metrics"] = {
            "memory_usage_percent": memory_info.percent,
            "memory_available_mb": memory_info.available / (1024 * 1024),
            "cpu_usage_percent": cpu_percent,
            "disk_usage_percent": psutil.disk_usage('/').percent
        }
        
        return {
            "success": True,
            "stats": stats,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Failed to get resource stats: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get resource stats: {str(e)}"
        )


@shared_resources_router.get("/monitoring/health")
async def get_resource_health() -> Dict[str, Any]:
    """
    Get resource health status
    
    Performs health checks on all shared resources and returns overall status.
    """
    
    try:
        resources = await get_shared_resources()
        
        health_checks = {}
        
        # Database health check
        try:
            async with database_session() as session:
                result = await session.execute("SELECT 1")
                await result.fetchone()
            health_checks["database"] = "healthy"
        except Exception as e:
            health_checks["database"] = f"unhealthy: {str(e)}"
        
        # HTTP client health check
        try:
            async with http_client() as client:
                async with client.get("https://httpbin.org/get", timeout=5) as response:
                    await response.json()
            health_checks["http_client"] = "healthy"
        except Exception as e:
            health_checks["http_client"] = f"unhealthy: {str(e)}"
        
        # Redis health check
        try:
            async with redis_client() as redis:
                await redis.ping()
            health_checks["redis"] = "healthy"
        except Exception as e:
            health_checks["redis"] = f"unhealthy: {str(e)}"
        
        # Cache health check
        try:
            await resources.get_cache("health_check")
            health_checks["cache"] = "healthy"
        except Exception as e:
            health_checks["cache"] = f"unhealthy: {str(e)}"
        
        # Overall health
        all_healthy = all(status == "healthy" for status in health_checks.values())
        
        return {
            "success": True,
            "overall_health": "healthy" if all_healthy else "unhealthy",
            "health_checks": health_checks,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "success": False,
            "overall_health": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }


# =============================================================================
# PERFORMANCE TESTING
# =============================================================================

@shared_resources_router.post("/testing/performance")
async def test_resource_performance() -> Dict[str, Any]:
    """
    Test resource performance
    
    Performs performance tests on shared resources and returns metrics.
    """
    
    try:
        resources = await get_shared_resources()
        results = {}
        
        # Test cache performance
        start_time = time.time()
        for i in range(100):
            await resources.set_cache(f"test_key_{i}", f"test_value_{i}")
            await resources.get_cache(f"test_key_{i}")
        cache_time = time.time() - start_time
        results["cache"] = {
            "operations": 200,
            "time_seconds": cache_time,
            "ops_per_second": 200 / cache_time
        }
        
        # Test database performance
        start_time = time.time()
        async with database_session() as session:
            for i in range(10):
                result = await session.execute("SELECT 1")
                await result.fetchone()
        db_time = time.time() - start_time
        results["database"] = {
            "operations": 10,
            "time_seconds": db_time,
            "ops_per_second": 10 / db_time
        }
        
        # Test HTTP client performance
        start_time = time.time()
        async with http_client() as client:
            for i in range(5):
                async with client.get("https://httpbin.org/get") as response:
                    await response.json()
        http_time = time.time() - start_time
        results["http_client"] = {
            "operations": 5,
            "time_seconds": http_time,
            "ops_per_second": 5 / http_time
        }
        
        return {
            "success": True,
            "performance_results": results,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Performance test failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Performance test failed: {str(e)}"
        )


# =============================================================================
# UTILITY ENDPOINTS
# =============================================================================

@shared_resources_router.get("/info")
async def get_resources_info() -> Dict[str, Any]:
    """
    Get general resources information
    
    Returns general information about shared resources configuration and status.
    """
    
    try:
        resources = await get_shared_resources()
        
        return {
            "success": True,
            "resources_info": {
                "initialized": resources._initialized,
                "shutdown": resources._shutdown,
                "background_tasks": len(resources._background_tasks),
                "total_resources": len(resources.resources),
                "config": {
                    "database_pool_size": resources.config.database_pool_size,
                    "http_max_connections": resources.config.http_max_connections,
                    "redis_pool_size": resources.config.redis_pool_size,
                    "model_cache_size": resources.config.model_cache_size,
                    "cache_size": resources.config.cache_size
                }
            },
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Failed to get resources info: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get resources info: {str(e)}"
        )


# =============================================================================
# DEPENDENCY INJECTION
# =============================================================================

async def get_resources() -> SharedResources:
    """Dependency to get shared resources instance"""
    return await get_shared_resources()


async def get_resources_config() -> ResourceConfig:
    """Dependency to get shared resources configuration"""
    resources = await get_shared_resources()
    return resources.config 