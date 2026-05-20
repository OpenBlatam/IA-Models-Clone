"""
Ultra speed API router with extreme velocity optimizations.
"""

import asyncio
import time
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query, Path
from fastapi.responses import StreamingResponse, JSONResponse
import json

from ...core.config import get_settings
from ...core.ultra_speed_engine import get_ultra_speed_engine, get_ultra_speed_optimizer, force_ultra_speed_optimization
from ...core.optimization import get_resource_manager, get_performance_optimizer
from ...core.async_pool import get_pool_manager, get_optimized_executor
from ...core.cache import get_cache_manager, invalidate_analysis_cache, invalidate_plugin_cache
from ...core.metrics import get_metrics_collector, get_prometheus_metrics
from ...core.logging import get_logger
from ...models.schemas import (
    ContentAnalysisRequest, ContentAnalysisResponse,
    BaseResponse, SystemStats, HealthCheckResponse
)
from ...services.ultra_speed_analysis_service import analyze_content_ultra_speed
from ...services.plugin_service import (
    discover_plugins, install_plugin, activate_plugin, deactivate_plugin,
    list_plugins, get_plugin_stats, execute_plugin_hook
)

router = APIRouter()
settings = get_settings()
logger = get_logger(__name__)


@router.get("/health/ultra-speed", response_model=HealthCheckResponse, summary="Ultra speed health check")
async def ultra_speed_health_check():
    """Ultra speed health check with extreme velocity status."""
    try:
        # Check ultra speed optimization system
        ultra_speed_engine = get_ultra_speed_engine()
        ultra_speed_stats = ultra_speed_engine.get_ultra_speed_summary()
        
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
            "ultra_speed_optimization_system": ultra_speed_stats.get("status") != "no_data",
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
        logger.error(f"Ultra speed health check failed: {e}")
        return HealthCheckResponse(
            status="unhealthy",
            systems={"error": str(e)},
            version=settings.app_version
        )


@router.get("/ultra-speed/status", summary="Get ultra speed optimization status")
async def get_ultra_speed_optimization_status():
    """Get current ultra speed optimization status and performance metrics."""
    try:
        ultra_speed_engine = get_ultra_speed_engine()
        ultra_speed_stats = ultra_speed_engine.get_ultra_speed_summary()
        
        resource_manager = get_resource_manager()
        optimization_stats = resource_manager.get_performance_summary()
        
        pool_manager = get_pool_manager()
        pool_stats = await pool_manager.get_all_stats()
        
        return {
            "ultra_speed_optimization_status": ultra_speed_stats,
            "optimization_status": optimization_stats,
            "pool_stats": pool_stats,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error getting ultra speed optimization status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get ultra speed optimization status: {e}")


@router.post("/ultra-speed/force", response_model=BaseResponse, summary="Force ultra speed optimization")
async def force_ultra_speed_optimization_endpoint():
    """Force immediate ultra speed optimization."""
    try:
        await force_ultra_speed_optimization()
        return BaseResponse(
            message="Ultra speed optimization completed successfully",
            data={"optimization_type": "ultra_speed_forced", "timestamp": time.time()}
        )
        
    except Exception as e:
        logger.error(f"Error forcing ultra speed optimization: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to force ultra speed optimization: {e}")


@router.get("/ultra-speed/metrics", summary="Get ultra speed performance metrics")
async def get_ultra_speed_performance_metrics():
    """Get comprehensive ultra speed performance metrics."""
    try:
        ultra_speed_engine = get_ultra_speed_engine()
        ultra_speed_stats = ultra_speed_engine.get_ultra_speed_summary()
        
        resource_manager = get_resource_manager()
        optimization_stats = resource_manager.get_performance_summary()
        
        pool_manager = get_pool_manager()
        pool_stats = await pool_manager.get_all_stats()
        
        cache_manager = await get_cache_manager()
        cache_stats = await cache_manager.get_stats()
        
        metrics_collector = get_metrics_collector()
        metrics_data = metrics_collector.get_all_metrics()
        
        executor = get_optimized_executor()
        executor_stats = executor.get_stats()
        
        return {
            "timestamp": time.time(),
            "ultra_speed_optimization": ultra_speed_stats,
            "optimization": optimization_stats,
            "pools": pool_stats,
            "cache": cache_stats,
            "metrics": metrics_data,
            "executor": executor_stats,
            "performance_level": "ultra_speed"
        }
        
    except Exception as e:
        logger.error(f"Error getting ultra speed performance metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get ultra speed performance metrics: {e}")


@router.post("/ultra-speed/tune", response_model=BaseResponse, summary="Tune ultra speed performance")
async def tune_ultra_speed_performance(
    target_ops_per_second: Optional[float] = Query(None, description="Target operations per second"),
    max_latency_p50: Optional[float] = Query(None, description="Maximum P50 latency"),
    max_latency_p95: Optional[float] = Query(None, description="Maximum P95 latency"),
    max_latency_p99: Optional[float] = Query(None, description="Maximum P99 latency"),
    max_latency_p999: Optional[float] = Query(None, description="Maximum P999 latency"),
    min_throughput_gbps: Optional[float] = Query(None, description="Minimum throughput in GB/s"),
    target_cpu_efficiency: Optional[float] = Query(None, description="Target CPU efficiency"),
    target_memory_efficiency: Optional[float] = Query(None, description="Target memory efficiency"),
    target_cache_hit_rate: Optional[float] = Query(None, description="Target cache hit rate"),
    target_gpu_utilization: Optional[float] = Query(None, description="Target GPU utilization")
):
    """Tune ultra speed performance parameters."""
    try:
        ultra_speed_engine = get_ultra_speed_engine()
        
        # Update configuration
        if target_ops_per_second is not None:
            ultra_speed_engine.config.target_ops_per_second = target_ops_per_second
        if max_latency_p50 is not None:
            ultra_speed_engine.config.max_latency_p50 = max_latency_p50
        if max_latency_p95 is not None:
            ultra_speed_engine.config.max_latency_p95 = max_latency_p95
        if max_latency_p99 is not None:
            ultra_speed_engine.config.max_latency_p99 = max_latency_p99
        if max_latency_p999 is not None:
            ultra_speed_engine.config.max_latency_p999 = max_latency_p999
        if min_throughput_gbps is not None:
            ultra_speed_engine.config.min_throughput_gbps = min_throughput_gbps
        if target_cpu_efficiency is not None:
            ultra_speed_engine.config.target_cpu_efficiency = target_cpu_efficiency
        if target_memory_efficiency is not None:
            ultra_speed_engine.config.target_memory_efficiency = target_memory_efficiency
        if target_cache_hit_rate is not None:
            ultra_speed_engine.config.target_cache_hit_rate = target_cache_hit_rate
        if target_gpu_utilization is not None:
            ultra_speed_engine.config.target_gpu_utilization = target_gpu_utilization
        
        # Force ultra speed optimization with new parameters
        await force_ultra_speed_optimization()
        
        return BaseResponse(
            message="Ultra speed performance tuned successfully",
            data={
                "target_ops_per_second": target_ops_per_second,
                "max_latency_p50": max_latency_p50,
                "max_latency_p95": max_latency_p95,
                "max_latency_p99": max_latency_p99,
                "max_latency_p999": max_latency_p999,
                "min_throughput_gbps": min_throughput_gbps,
                "target_cpu_efficiency": target_cpu_efficiency,
                "target_memory_efficiency": target_memory_efficiency,
                "target_cache_hit_rate": target_cache_hit_rate,
                "target_gpu_utilization": target_gpu_utilization,
                "timestamp": time.time()
            }
        )
        
    except Exception as e:
        logger.error(f"Error tuning ultra speed performance: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to tune ultra speed performance: {e}")


@router.post("/analyze/ultra-speed", response_model=ContentAnalysisResponse, summary="Ultra-speed content analysis")
async def ultra_speed_analyze_content(
    request: ContentAnalysisRequest,
    background_tasks: BackgroundTasks,
    use_cache: bool = Query(True, description="Use cache for analysis"),
    parallel_processing: bool = Query(True, description="Use parallel processing"),
    speed_level: str = Query("ultra_speed", description="Speed optimization level")
):
    """Perform ultra-speed content analysis with extreme velocity optimizations."""
    try:
        # Execute plugin hooks in background
        background_tasks.add_task(
            _execute_plugin_hooks_ultra_speed,
            "pre_analysis",
            request.dict()
        )
        
        # Perform ultra-speed analysis
        result = await analyze_content_ultra_speed(request)
        
        # Execute post-analysis hooks
        background_tasks.add_task(
            _execute_plugin_hooks_ultra_speed,
            "post_analysis",
            result.dict()
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Ultra-speed analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ultra-speed analysis failed: {e}")


@router.post("/analyze/ultra-speed-batch", summary="Ultra-speed batch content analysis")
async def ultra_speed_batch_analyze_content(
    requests: List[ContentAnalysisRequest],
    background_tasks: BackgroundTasks,
    max_concurrent: int = Query(50, description="Maximum concurrent analyses"),
    speed_level: str = Query("ultra_speed", description="Speed optimization level")
):
    """Perform ultra-speed batch content analysis."""
    try:
        if len(requests) > 1000:
            raise HTTPException(status_code=400, detail="Maximum 1000 requests per batch")
        
        # Process in batches with ultra-speed executor
        executor = get_optimized_executor()
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def analyze_with_semaphore(request: ContentAnalysisRequest):
            async with semaphore:
                return await analyze_content_ultra_speed(request)
        
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
            _execute_plugin_hooks_ultra_speed,
            "batch_analysis_completed",
            {
                "total_requests": len(requests),
                "successful": len(successful_results),
                "failed": len(failed_results),
                "speed_level": speed_level
            }
        )
        
        return {
            "total_requests": len(requests),
            "successful_analyses": len(successful_results),
            "failed_analyses": len(failed_results),
            "speed_level": speed_level,
            "results": successful_results,
            "errors": failed_results
        }
        
    except Exception as e:
        logger.error(f"Ultra-speed batch analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ultra-speed batch analysis failed: {e}")


@router.get("/performance/ultra-speed", summary="Get ultra speed performance metrics")
async def get_ultra_speed_performance_metrics_detailed():
    """Get comprehensive ultra speed performance metrics."""
    try:
        # Get ultra speed optimization metrics
        ultra_speed_engine = get_ultra_speed_engine()
        ultra_speed_stats = ultra_speed_engine.get_ultra_speed_summary()
        
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
            "ultra_speed_optimization": ultra_speed_stats,
            "optimization": optimization_stats,
            "pools": pool_stats,
            "cache": cache_stats,
            "metrics": metrics_data,
            "executor": executor_stats,
            "performance_level": "ultra_speed"
        }
        
    except Exception as e:
        logger.error(f"Error getting ultra speed performance metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get ultra speed performance metrics: {e}")


@router.post("/performance/ultra-speed-tune", response_model=BaseResponse, summary="Tune ultra speed performance")
async def tune_ultra_speed_performance_advanced(
    ultra_aggressive_optimization: Optional[bool] = Query(None, description="Enable ultra aggressive optimization"),
    use_gpu_acceleration: Optional[bool] = Query(None, description="Enable GPU acceleration"),
    use_memory_mapping: Optional[bool] = Query(None, description="Enable memory mapping"),
    use_zero_copy: Optional[bool] = Query(None, description="Enable zero-copy operations"),
    use_vectorization: Optional[bool] = Query(None, description="Enable vectorization"),
    use_parallel_processing: Optional[bool] = Query(None, description="Enable parallel processing"),
    use_prefetching: Optional[bool] = Query(None, description="Enable prefetching"),
    use_batching: Optional[bool] = Query(None, description="Enable batching"),
    optimization_interval: Optional[float] = Query(None, description="Optimization interval in seconds")
):
    """Tune ultra speed performance parameters."""
    try:
        ultra_speed_engine = get_ultra_speed_engine()
        
        # Update configuration
        if ultra_aggressive_optimization is not None:
            ultra_speed_engine.config.ultra_aggressive_optimization = ultra_aggressive_optimization
        if use_gpu_acceleration is not None:
            ultra_speed_engine.config.use_gpu_acceleration = use_gpu_acceleration
        if use_memory_mapping is not None:
            ultra_speed_engine.config.use_memory_mapping = use_memory_mapping
        if use_zero_copy is not None:
            ultra_speed_engine.config.use_zero_copy = use_zero_copy
        if use_vectorization is not None:
            ultra_speed_engine.config.use_vectorization = use_vectorization
        if use_parallel_processing is not None:
            ultra_speed_engine.config.use_parallel_processing = use_parallel_processing
        if use_prefetching is not None:
            ultra_speed_engine.config.use_prefetching = use_prefetching
        if use_batching is not None:
            ultra_speed_engine.config.use_batching = use_batching
        if optimization_interval is not None:
            ultra_speed_engine.config.optimization_interval = optimization_interval
        
        # Force ultra speed optimization with new parameters
        await force_ultra_speed_optimization()
        
        return BaseResponse(
            message="Ultra speed performance tuned successfully",
            data={
                "ultra_aggressive_optimization": ultra_aggressive_optimization,
                "use_gpu_acceleration": use_gpu_acceleration,
                "use_memory_mapping": use_memory_mapping,
                "use_zero_copy": use_zero_copy,
                "use_vectorization": use_vectorization,
                "use_parallel_processing": use_parallel_processing,
                "use_prefetching": use_prefetching,
                "use_batching": use_batching,
                "optimization_interval": optimization_interval,
                "timestamp": time.time()
            }
        )
        
    except Exception as e:
        logger.error(f"Error tuning ultra speed performance: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to tune ultra speed performance: {e}")


@router.get("/system/ultra-speed-stats", summary="Get ultra speed system statistics")
async def get_ultra_speed_system_stats():
    """Get comprehensive ultra speed system statistics."""
    try:
        # Get all system statistics
        ultra_speed_stats = get_ultra_speed_engine().get_ultra_speed_summary()
        optimization_stats = get_resource_manager().get_performance_summary()
        pool_stats = await get_pool_manager().get_all_stats()
        cache_stats = await get_cache_manager().get_stats()
        metrics_data = get_metrics_collector().get_all_metrics()
        plugin_stats = await get_plugin_stats()
        executor_stats = get_optimized_executor().get_stats()
        
        return {
            "timestamp": time.time(),
            "system_status": "ultra_speed",
            "ultra_speed_optimization": ultra_speed_stats,
            "optimization": optimization_stats,
            "pools": pool_stats,
            "cache": cache_stats,
            "metrics": metrics_data,
            "plugins": plugin_stats,
            "executor": executor_stats,
            "performance_level": "ultra_speed"
        }
        
    except Exception as e:
        logger.error(f"Error getting ultra speed system stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get ultra speed system stats: {e}")


@router.post("/system/ultra-speed-warmup", response_model=BaseResponse, summary="Ultra speed system warmup")
async def ultra_speed_system_warmup():
    """Perform comprehensive ultra speed system warmup."""
    try:
        # Warm up all systems
        tasks = [
            _warmup_ultra_speed_system(),
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
        
        warmup_names = ["ultra_speed", "cache", "pools", "optimization", "plugins", "metrics"]
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed_warmups.append({"system": warmup_names[i], "error": str(result)})
            else:
                successful_warmups.append(warmup_names[i])
        
        return BaseResponse(
            message=f"Ultra speed system warmup completed: {len(successful_warmups)} successful, {len(failed_warmups)} failed",
            data={
                "successful_warmups": successful_warmups,
                "failed_warmups": failed_warmups,
                "timestamp": time.time()
            }
        )
        
    except Exception as e:
        logger.error(f"Ultra speed system warmup failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ultra speed system warmup failed: {e}")


# Background task functions
async def _execute_plugin_hooks_ultra_speed(hook_name: str, data: Dict[str, Any]):
    """Execute plugin hooks with ultra-speed error handling."""
    try:
        results = await execute_plugin_hook(hook_name, data)
        logger.info(f"Ultra-speed plugin hook {hook_name} executed successfully", extra={"results_count": len(results)})
    except Exception as e:
        logger.error(f"Error executing ultra-speed plugin hook {hook_name}: {e}")
        # Don't raise - this is a background task


# Warmup functions
async def _warmup_ultra_speed_system():
    """Warm up ultra speed optimization system."""
    ultra_speed_engine = get_ultra_speed_engine()
    await ultra_speed_engine.start_ultra_speed_optimization()


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
@router.get("/debug/ultra-speed-info", summary="Ultra speed debug information")
async def get_ultra_speed_debug_info():
    """Get ultra speed debug information (only in development)."""
    if not settings.debug:
        raise HTTPException(status_code=404, detail="Not found")
    
    try:
        return {
            "environment": settings.environment,
            "debug": settings.debug,
            "version": settings.app_version,
            "ultra_speed_stats": get_ultra_speed_engine().get_ultra_speed_summary(),
            "optimization_stats": get_resource_manager().get_performance_summary(),
            "pool_stats": await get_pool_manager().get_all_stats(),
            "cache_stats": await get_cache_manager().get_stats(),
            "metrics_summary": get_metrics_collector().get_all_metrics(),
            "plugin_stats": await get_plugin_stats(),
            "executor_stats": get_optimized_executor().get_stats()
        }
        
    except Exception as e:
        logger.error(f"Error getting ultra speed debug info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get ultra speed debug info: {e}")


@router.post("/debug/ultra-speed-reset", response_model=BaseResponse, summary="Ultra speed system reset (debug only)")
async def ultra_speed_reset_system():
    """Reset ultra speed system state (only in development)."""
    if not settings.debug:
        raise HTTPException(status_code=404, detail="Not found")
    
    try:
        # Reset all systems
        tasks = [
            _reset_ultra_speed_system(),
            _reset_cache_system(),
            _reset_pool_system(),
            _reset_optimization_system(),
            _reset_metrics_system()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return BaseResponse(
            message="Ultra speed system reset successfully",
            data={
                "reset_components": ["ultra_speed", "cache", "pools", "optimization", "metrics"],
                "timestamp": time.time()
            }
        )
        
    except Exception as e:
        logger.error(f"Ultra speed system reset failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ultra speed system reset failed: {e}")


# Reset functions
async def _reset_ultra_speed_system():
    """Reset ultra speed optimization system."""
    ultra_speed_engine = get_ultra_speed_engine()
    ultra_speed_engine.ultra_speed_history.clear()


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


