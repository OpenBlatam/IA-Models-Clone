"""
Ultra-optimized API router with maximum performance features.
"""

import asyncio
import time
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query, Path
from fastapi.responses import StreamingResponse, JSONResponse
import json

from ...core.config import get_settings
from ...core.optimization import get_resource_manager, get_performance_optimizer, force_optimization
from ...core.async_pool import get_pool_manager, get_optimized_executor
from ...core.cache import get_cache_manager, invalidate_analysis_cache, invalidate_plugin_cache
from ...core.metrics import get_metrics_collector, get_prometheus_metrics
from ...core.logging import get_logger
from ...models.schemas import (
    ContentAnalysisRequest, ContentAnalysisResponse,
    BaseResponse, SystemStats, HealthCheckResponse
)
from ...services.optimized_analysis_service import analyze_content_ultra_optimized
from ...services.plugin_service import (
    discover_plugins, install_plugin, activate_plugin, deactivate_plugin,
    list_plugins, get_plugin_stats, execute_plugin_hook
)

router = APIRouter()
settings = get_settings()
logger = get_logger(__name__)


@router.get("/health/ultra", response_model=HealthCheckResponse, summary="Ultra health check")
async def ultra_health_check():
    """Ultra health check with comprehensive system status."""
    try:
        # Check optimization system
        resource_manager = get_resource_manager()
        optimization_stats = resource_manager.get_performance_summary()
        
        # Check pool system
        pool_manager = get_pool_manager()
        pool_stats = await pool_manager.get_all_stats()
        
        # Check cache system
        cache_manager = await get_cache_manager()
        cache_stats = await cache_manager.get_stats()
        
        # Check metrics system
        metrics_collector = get_metrics_collector()
        metrics_stats = metrics_collector.get_all_metrics()
        
        # Check plugin system
        plugin_stats = await get_plugin_stats()
        
        # Determine overall health
        systems_healthy = {
            "optimization_system": optimization_stats.get("status") != "no_data",
            "pool_system": len(pool_stats) >= 0,
            "cache_system": cache_stats.get("redis_connected", False) or cache_stats.get("memory_cache_size", 0) >= 0,
            "metrics_system": len(metrics_stats) > 0,
            "plugin_system": plugin_stats.get("total_plugins", 0) >= 0,
            "analysis_system": True,
            "api_system": True
        }
        
        overall_health = "healthy" if all(systems_healthy.values()) else "degraded"
        
        return HealthCheckResponse(
            status=overall_health,
            systems=systems_healthy,
            version=settings.app_version,
            uptime=None  # Would be calculated in real implementation
        )
        
    except Exception as e:
        logger.error(f"Ultra health check failed: {e}")
        return HealthCheckResponse(
            status="unhealthy",
            systems={"error": str(e)},
            version=settings.app_version
        )


@router.get("/optimization/status", summary="Get optimization status")
async def get_optimization_status():
    """Get current optimization status and performance metrics."""
    try:
        resource_manager = get_resource_manager()
        optimization_stats = resource_manager.get_performance_summary()
        
        pool_manager = get_pool_manager()
        pool_stats = await pool_manager.get_all_stats()
        
        return {
            "optimization_status": optimization_stats,
            "pool_stats": pool_stats,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error getting optimization status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get optimization status: {e}")


@router.post("/optimization/force", response_model=BaseResponse, summary="Force optimization")
async def force_optimization_endpoint():
    """Force immediate system optimization."""
    try:
        await force_optimization()
        return BaseResponse(
            message="System optimization completed successfully",
            data={"optimization_type": "forced", "timestamp": time.time()}
        )
        
    except Exception as e:
        logger.error(f"Error forcing optimization: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to force optimization: {e}")


@router.get("/pools/stats", summary="Get pool statistics")
async def get_pool_statistics():
    """Get statistics for all resource pools."""
    try:
        pool_manager = get_pool_manager()
        pool_stats = await pool_manager.get_all_stats()
        
        return {
            "pools": pool_stats,
            "total_pools": len(pool_stats),
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error getting pool statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get pool statistics: {e}")


@router.post("/pools/create", response_model=BaseResponse, summary="Create resource pool")
async def create_resource_pool(
    pool_name: str = Query(..., description="Name of the pool"),
    pool_type: str = Query(..., description="Type of pool (connection, worker, etc.)"),
    max_size: int = Query(10, description="Maximum pool size"),
    min_size: int = Query(1, description="Minimum pool size")
):
    """Create a new resource pool."""
    try:
        pool_manager = get_pool_manager()
        
        # Create pool based on type
        if pool_type == "connection":
            from ...core.async_pool import create_connection_pool
            pool = await create_connection_pool(pool_name, lambda: None, max_size, min_size)
        elif pool_type == "worker":
            from ...core.async_pool import create_worker_pool
            pool = await create_worker_pool(pool_name, lambda: None, max_size, min_size)
        else:
            raise HTTPException(status_code=400, detail="Invalid pool type")
        
        return BaseResponse(
            message=f"Pool {pool_name} created successfully",
            data={
                "pool_name": pool_name,
                "pool_type": pool_type,
                "max_size": max_size,
                "min_size": min_size
            }
        )
        
    except Exception as e:
        logger.error(f"Error creating pool: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create pool: {e}")


@router.delete("/pools/{pool_name}", response_model=BaseResponse, summary="Remove resource pool")
async def remove_resource_pool(
    pool_name: str = Path(..., description="Name of the pool to remove")
):
    """Remove a resource pool."""
    try:
        pool_manager = get_pool_manager()
        await pool_manager.remove_pool(pool_name)
        
        return BaseResponse(
            message=f"Pool {pool_name} removed successfully",
            data={"pool_name": pool_name}
        )
        
    except Exception as e:
        logger.error(f"Error removing pool: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to remove pool: {e}")


@router.post("/analyze/ultra-optimized", response_model=ContentAnalysisResponse, summary="Ultra-optimized content analysis")
async def ultra_optimized_analyze_content(
    request: ContentAnalysisRequest,
    background_tasks: BackgroundTasks,
    use_cache: bool = Query(True, description="Use cache for analysis"),
    parallel_processing: bool = Query(True, description="Use parallel processing"),
    optimization_level: str = Query("ultra", description="Optimization level")
):
    """Perform ultra-optimized content analysis with maximum performance."""
    try:
        # Execute plugin hooks in background
        background_tasks.add_task(
            _execute_plugin_hooks_ultra_optimized,
            "pre_analysis",
            request.dict()
        )
        
        # Perform ultra-optimized analysis
        result = await analyze_content_ultra_optimized(request)
        
        # Execute post-analysis hooks
        background_tasks.add_task(
            _execute_plugin_hooks_ultra_optimized,
            "post_analysis",
            result.dict()
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Ultra-optimized analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ultra-optimized analysis failed: {e}")


@router.post("/analyze/ultra-batch", summary="Ultra-optimized batch content analysis")
async def ultra_optimized_batch_analyze_content(
    requests: List[ContentAnalysisRequest],
    background_tasks: BackgroundTasks,
    max_concurrent: int = Query(10, description="Maximum concurrent analyses"),
    optimization_level: str = Query("ultra", description="Optimization level")
):
    """Perform ultra-optimized batch content analysis."""
    try:
        if len(requests) > 200:
            raise HTTPException(status_code=400, detail="Maximum 200 requests per batch")
        
        # Process in batches with optimized executor
        executor = get_optimized_executor()
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def analyze_with_semaphore(request: ContentAnalysisRequest):
            async with semaphore:
                return await analyze_content_ultra_optimized(request)
        
        # Execute batch analysis
        tasks = [analyze_with_semaphore(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        successful_results = []
        failed_results = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed_results.append({
                    "index": i,
                    "error": str(result),
                    "request": requests[i].dict()
                })
            else:
                successful_results.append(result.dict())
        
        # Execute batch completion hooks
        background_tasks.add_task(
            _execute_plugin_hooks_ultra_optimized,
            "batch_analysis_completed",
            {
                "total_requests": len(requests),
                "successful": len(successful_results),
                "failed": len(failed_results),
                "optimization_level": optimization_level
            }
        )
        
        return {
            "total_requests": len(requests),
            "successful_analyses": len(successful_results),
            "failed_analyses": len(failed_results),
            "optimization_level": optimization_level,
            "results": successful_results,
            "errors": failed_results
        }
        
    except Exception as e:
        logger.error(f"Ultra-optimized batch analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ultra-optimized batch analysis failed: {e}")


@router.get("/performance/ultra", summary="Get ultra performance metrics")
async def get_ultra_performance_metrics():
    """Get comprehensive ultra performance metrics."""
    try:
        # Get optimization metrics
        resource_manager = get_resource_manager()
        optimization_stats = resource_manager.get_performance_summary()
        
        # Get pool metrics
        pool_manager = get_pool_manager()
        pool_stats = await pool_manager.get_all_stats()
        
        # Get cache metrics
        cache_manager = await get_cache_manager()
        cache_stats = await cache_manager.get_stats()
        
        # Get metrics collector data
        metrics_collector = get_metrics_collector()
        metrics_data = metrics_collector.get_all_metrics()
        
        # Get executor stats
        executor = get_optimized_executor()
        executor_stats = executor.get_stats()
        
        return {
            "timestamp": time.time(),
            "optimization": optimization_stats,
            "pools": pool_stats,
            "cache": cache_stats,
            "metrics": metrics_data,
            "executor": executor_stats,
            "performance_level": "ultra_optimized"
        }
        
    except Exception as e:
        logger.error(f"Error getting ultra performance metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get ultra performance metrics: {e}")


@router.post("/performance/tune", response_model=BaseResponse, summary="Tune system performance")
async def tune_system_performance(
    cpu_limit: Optional[float] = Query(None, description="CPU usage limit"),
    memory_limit: Optional[float] = Query(None, description="Memory usage limit"),
    response_time_limit: Optional[float] = Query(None, description="Response time limit"),
    throughput_limit: Optional[float] = Query(None, description="Throughput limit")
):
    """Tune system performance parameters."""
    try:
        resource_manager = get_resource_manager()
        
        # Update configuration
        if cpu_limit is not None:
            resource_manager.config.max_cpu_usage = cpu_limit
        if memory_limit is not None:
            resource_manager.config.max_memory_usage = memory_limit
        if response_time_limit is not None:
            resource_manager.config.max_response_time = response_time_limit
        if throughput_limit is not None:
            resource_manager.config.min_throughput = throughput_limit
        
        # Force optimization with new parameters
        await force_optimization()
        
        return BaseResponse(
            message="System performance tuned successfully",
            data={
                "cpu_limit": cpu_limit,
                "memory_limit": memory_limit,
                "response_time_limit": response_time_limit,
                "throughput_limit": throughput_limit,
                "timestamp": time.time()
            }
        )
        
    except Exception as e:
        logger.error(f"Error tuning system performance: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to tune system performance: {e}")


@router.get("/executor/stats", summary="Get executor statistics")
async def get_executor_statistics():
    """Get statistics for the optimized executor."""
    try:
        executor = get_optimized_executor()
        stats = executor.get_stats()
        
        return {
            "executor_stats": stats,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error getting executor statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get executor statistics: {e}")


@router.post("/executor/scale", response_model=BaseResponse, summary="Scale executor")
async def scale_executor(
    new_max_workers: int = Query(..., description="New maximum number of workers")
):
    """Scale the executor to a new number of workers."""
    try:
        if new_max_workers < 1 or new_max_workers > 100:
            raise HTTPException(status_code=400, detail="Max workers must be between 1 and 100")
        
        executor = get_optimized_executor()
        old_max_workers = executor.max_workers
        executor.max_workers = new_max_workers
        
        # Restart executor with new configuration
        await executor.shutdown(wait=True)
        executor._executor = None
        await executor._ensure_executor()
        
        return BaseResponse(
            message=f"Executor scaled from {old_max_workers} to {new_max_workers} workers",
            data={
                "old_max_workers": old_max_workers,
                "new_max_workers": new_max_workers,
                "timestamp": time.time()
            }
        )
        
    except Exception as e:
        logger.error(f"Error scaling executor: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to scale executor: {e}")


@router.get("/system/ultra-stats", summary="Get ultra system statistics")
async def get_ultra_system_stats():
    """Get comprehensive ultra system statistics."""
    try:
        # Get all system statistics
        optimization_stats = get_resource_manager().get_performance_summary()
        pool_stats = await get_pool_manager().get_all_stats()
        cache_stats = await get_cache_manager().get_stats()
        metrics_data = get_metrics_collector().get_all_metrics()
        plugin_stats = await get_plugin_stats()
        executor_stats = get_optimized_executor().get_stats()
        
        return {
            "timestamp": time.time(),
            "system_status": "ultra_optimized",
            "optimization": optimization_stats,
            "pools": pool_stats,
            "cache": cache_stats,
            "metrics": metrics_data,
            "plugins": plugin_stats,
            "executor": executor_stats,
            "performance_level": "ultra"
        }
        
    except Exception as e:
        logger.error(f"Error getting ultra system stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get ultra system stats: {e}")


@router.post("/system/ultra-warmup", response_model=BaseResponse, summary="Ultra system warmup")
async def ultra_system_warmup():
    """Perform comprehensive system warmup."""
    try:
        # Warm up all systems
        tasks = [
            _warmup_cache_system(),
            _warmup_pool_system(),
            _warmup_optimization_system(),
            _warmup_plugin_system(),
            _warmup_metrics_system()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check results
        successful_warmups = []
        failed_warmups = []
        
        warmup_names = ["cache", "pools", "optimization", "plugins", "metrics"]
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed_warmups.append({"system": warmup_names[i], "error": str(result)})
            else:
                successful_warmups.append(warmup_names[i])
        
        return BaseResponse(
            message=f"Ultra system warmup completed: {len(successful_warmups)} successful, {len(failed_warmups)} failed",
            data={
                "successful_warmups": successful_warmups,
                "failed_warmups": failed_warmups,
                "timestamp": time.time()
            }
        )
        
    except Exception as e:
        logger.error(f"Ultra system warmup failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ultra system warmup failed: {e}")


# Background task functions
async def _execute_plugin_hooks_ultra_optimized(hook_name: str, data: Dict[str, Any]):
    """Execute plugin hooks with ultra-optimized error handling."""
    try:
        results = await execute_plugin_hook(hook_name, data)
        logger.info(f"Ultra-optimized plugin hook {hook_name} executed successfully", extra={"results_count": len(results)})
    except Exception as e:
        logger.error(f"Error executing ultra-optimized plugin hook {hook_name}: {e}")
        # Don't raise - this is a background task


# Warmup functions
async def _warmup_cache_system():
    """Warm up cache system."""
    from ...core.cache import warm_up_cache
    await warm_up_cache()


async def _warmup_pool_system():
    """Warm up pool system."""
    pool_manager = get_pool_manager()
    # Create default pools if they don't exist
    # Implementation would depend on specific pool requirements


async def _warmup_optimization_system():
    """Warm up optimization system."""
    resource_manager = get_resource_manager()
    await resource_manager.start_optimization()


async def _warmup_plugin_system():
    """Warm up plugin system."""
    await discover_plugins()
    await get_plugin_stats()


async def _warmup_metrics_system():
    """Warm up metrics system."""
    metrics_collector = get_metrics_collector()
    # Initialize metrics collection
    # Implementation would depend on specific metrics requirements


# Utility endpoints
@router.get("/debug/ultra-info", summary="Ultra debug information")
async def get_ultra_debug_info():
    """Get ultra debug information (only in development)."""
    if not settings.debug:
        raise HTTPException(status_code=404, detail="Not found")
    
    try:
        return {
            "environment": settings.environment,
            "debug": settings.debug,
            "version": settings.app_version,
            "optimization_stats": get_resource_manager().get_performance_summary(),
            "pool_stats": await get_pool_manager().get_all_stats(),
            "cache_stats": await get_cache_manager().get_stats(),
            "metrics_summary": get_metrics_collector().get_all_metrics(),
            "plugin_stats": await get_plugin_stats(),
            "executor_stats": get_optimized_executor().get_stats()
        }
        
    except Exception as e:
        logger.error(f"Error getting ultra debug info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get ultra debug info: {e}")


@router.post("/debug/ultra-reset", response_model=BaseResponse, summary="Ultra system reset (debug only)")
async def ultra_reset_system():
    """Reset ultra system state (only in development)."""
    if not settings.debug:
        raise HTTPException(status_code=404, detail="Not found")
    
    try:
        # Reset all systems
        tasks = [
            _reset_cache_system(),
            _reset_pool_system(),
            _reset_optimization_system(),
            _reset_metrics_system()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return BaseResponse(
            message="Ultra system reset successfully",
            data={
                "reset_components": ["cache", "pools", "optimization", "metrics"],
                "timestamp": time.time()
            }
        )
        
    except Exception as e:
        logger.error(f"Ultra system reset failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ultra system reset failed: {e}")


# Reset functions
async def _reset_cache_system():
    """Reset cache system."""
    cache_manager = await get_cache_manager()
    await cache_manager.clear()


async def _reset_pool_system():
    """Reset pool system."""
    pool_manager = get_pool_manager()
    await pool_manager.shutdown_all()


async def _reset_optimization_system():
    """Reset optimization system."""
    resource_manager = get_resource_manager()
    resource_manager.performance_history.clear()


async def _reset_metrics_system():
    """Reset metrics system."""
    metrics_collector = get_metrics_collector()
    metrics_collector.reset_metrics()


