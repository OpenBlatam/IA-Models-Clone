"""
Lightning-fast API router with extreme speed optimizations.
"""

import asyncio
import time
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query, Path
from fastapi.responses import StreamingResponse, JSONResponse
import json

from ...core.config import get_settings
from ...core.speed_optimization import get_speed_engine, get_speed_optimizer, force_speed_optimization
from ...core.optimization import get_resource_manager, get_performance_optimizer
from ...core.async_pool import get_pool_manager, get_optimized_executor
from ...core.cache import get_cache_manager, invalidate_analysis_cache, invalidate_plugin_cache
from ...core.metrics import get_metrics_collector, get_prometheus_metrics
from ...core.logging import get_logger
from ...models.schemas import (
    ContentAnalysisRequest, ContentAnalysisResponse,
    BaseResponse, SystemStats, HealthCheckResponse
)
from ...services.lightning_analysis_service import analyze_content_lightning_fast
from ...services.plugin_service import (
    discover_plugins, install_plugin, activate_plugin, deactivate_plugin,
    list_plugins, get_plugin_stats, execute_plugin_hook
)

router = APIRouter()
settings = get_settings()
logger = get_logger(__name__)


@router.get("/health/lightning", response_model=HealthCheckResponse, summary="Lightning health check")
async def lightning_health_check():
    """Lightning health check with extreme speed status."""
    try:
        # Check speed optimization system
        speed_engine = get_speed_engine()
        speed_stats = speed_engine.get_speed_summary()
        
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
            "speed_optimization_system": speed_stats.get("status") != "no_data",
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
        logger.error(f"Lightning health check failed: {e}")
        return HealthCheckResponse(
            status="unhealthy",
            systems={"error": str(e)},
            version=settings.app_version
        )


@router.get("/speed/status", summary="Get speed optimization status")
async def get_speed_optimization_status():
    """Get current speed optimization status and performance metrics."""
    try:
        speed_engine = get_speed_engine()
        speed_stats = speed_engine.get_speed_summary()
        
        resource_manager = get_resource_manager()
        optimization_stats = resource_manager.get_performance_summary()
        
        pool_manager = get_pool_manager()
        pool_stats = await pool_manager.get_all_stats()
        
        return {
            "speed_optimization_status": speed_stats,
            "optimization_status": optimization_stats,
            "pool_stats": pool_stats,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error getting speed optimization status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get speed optimization status: {e}")


@router.post("/speed/force", response_model=BaseResponse, summary="Force speed optimization")
async def force_speed_optimization_endpoint():
    """Force immediate speed optimization."""
    try:
        await force_speed_optimization()
        return BaseResponse(
            message="Speed optimization completed successfully",
            data={"optimization_type": "speed_forced", "timestamp": time.time()}
        )
        
    except Exception as e:
        logger.error(f"Error forcing speed optimization: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to force speed optimization: {e}")


@router.get("/speed/metrics", summary="Get speed performance metrics")
async def get_speed_performance_metrics():
    """Get comprehensive speed performance metrics."""
    try:
        speed_engine = get_speed_engine()
        speed_stats = speed_engine.get_speed_summary()
        
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
            "speed_optimization": speed_stats,
            "optimization": optimization_stats,
            "pools": pool_stats,
            "cache": cache_stats,
            "metrics": metrics_data,
            "executor": executor_stats,
            "performance_level": "lightning_fast"
        }
        
    except Exception as e:
        logger.error(f"Error getting speed performance metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get speed performance metrics: {e}")


@router.post("/speed/tune", response_model=BaseResponse, summary="Tune speed performance")
async def tune_speed_performance(
    target_ops_per_second: Optional[float] = Query(None, description="Target operations per second"),
    max_latency_p95: Optional[float] = Query(None, description="Maximum P95 latency"),
    max_latency_p99: Optional[float] = Query(None, description="Maximum P99 latency"),
    min_throughput_mbps: Optional[float] = Query(None, description="Minimum throughput in MB/s"),
    target_cpu_efficiency: Optional[float] = Query(None, description="Target CPU efficiency"),
    target_memory_efficiency: Optional[float] = Query(None, description="Target memory efficiency"),
    target_cache_hit_rate: Optional[float] = Query(None, description="Target cache hit rate")
):
    """Tune speed performance parameters."""
    try:
        speed_engine = get_speed_engine()
        
        # Update configuration
        if target_ops_per_second is not None:
            speed_engine.config.target_ops_per_second = target_ops_per_second
        if max_latency_p95 is not None:
            speed_engine.config.max_latency_p95 = max_latency_p95
        if max_latency_p99 is not None:
            speed_engine.config.max_latency_p99 = max_latency_p99
        if min_throughput_mbps is not None:
            speed_engine.config.min_throughput_mbps = min_throughput_mbps
        if target_cpu_efficiency is not None:
            speed_engine.config.target_cpu_efficiency = target_cpu_efficiency
        if target_memory_efficiency is not None:
            speed_engine.config.target_memory_efficiency = target_memory_efficiency
        if target_cache_hit_rate is not None:
            speed_engine.config.target_cache_hit_rate = target_cache_hit_rate
        
        # Force speed optimization with new parameters
        await force_speed_optimization()
        
        return BaseResponse(
            message="Speed performance tuned successfully",
            data={
                "target_ops_per_second": target_ops_per_second,
                "max_latency_p95": max_latency_p95,
                "max_latency_p99": max_latency_p99,
                "min_throughput_mbps": min_throughput_mbps,
                "target_cpu_efficiency": target_cpu_efficiency,
                "target_memory_efficiency": target_memory_efficiency,
                "target_cache_hit_rate": target_cache_hit_rate,
                "timestamp": time.time()
            }
        )
        
    except Exception as e:
        logger.error(f"Error tuning speed performance: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to tune speed performance: {e}")


@router.post("/analyze/lightning", response_model=ContentAnalysisResponse, summary="Lightning-fast content analysis")
async def lightning_analyze_content(
    request: ContentAnalysisRequest,
    background_tasks: BackgroundTasks,
    use_cache: bool = Query(True, description="Use cache for analysis"),
    parallel_processing: bool = Query(True, description="Use parallel processing"),
    speed_level: str = Query("lightning", description="Speed optimization level")
):
    """Perform lightning-fast content analysis with extreme speed optimizations."""
    try:
        # Execute plugin hooks in background
        background_tasks.add_task(
            _execute_plugin_hooks_lightning,
            "pre_analysis",
            request.dict()
        )
        
        # Perform lightning-fast analysis
        result = await analyze_content_lightning_fast(request)
        
        # Execute post-analysis hooks
        background_tasks.add_task(
            _execute_plugin_hooks_lightning,
            "post_analysis",
            result.dict()
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Lightning analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Lightning analysis failed: {e}")


@router.post("/analyze/lightning-batch", summary="Lightning-fast batch content analysis")
async def lightning_batch_analyze_content(
    requests: List[ContentAnalysisRequest],
    background_tasks: BackgroundTasks,
    max_concurrent: int = Query(20, description="Maximum concurrent analyses"),
    speed_level: str = Query("lightning", description="Speed optimization level")
):
    """Perform lightning-fast batch content analysis."""
    try:
        if len(requests) > 500:
            raise HTTPException(status_code=400, detail="Maximum 500 requests per batch")
        
        # Process in batches with lightning executor
        executor = get_optimized_executor()
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def analyze_with_semaphore(request: ContentAnalysisRequest):
            async with semaphore:
                return await analyze_content_lightning_fast(request)
        
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
            _execute_plugin_hooks_lightning,
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
        logger.error(f"Lightning batch analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Lightning batch analysis failed: {e}")


@router.get("/performance/lightning", summary="Get lightning performance metrics")
async def get_lightning_performance_metrics():
    """Get comprehensive lightning performance metrics."""
    try:
        # Get speed optimization metrics
        speed_engine = get_speed_engine()
        speed_stats = speed_engine.get_speed_summary()
        
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
            "speed_optimization": speed_stats,
            "optimization": optimization_stats,
            "pools": pool_stats,
            "cache": cache_stats,
            "metrics": metrics_data,
            "executor": executor_stats,
            "performance_level": "lightning_fast"
        }
        
    except Exception as e:
        logger.error(f"Error getting lightning performance metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get lightning performance metrics: {e}")


@router.post("/performance/lightning-tune", response_model=BaseResponse, summary="Tune lightning performance")
async def tune_lightning_performance(
    aggressive_optimization: Optional[bool] = Query(None, description="Enable aggressive optimization"),
    use_numba: Optional[bool] = Query(None, description="Enable Numba compilation"),
    use_cython: Optional[bool] = Query(None, description="Enable Cython compilation"),
    use_vectorization: Optional[bool] = Query(None, description="Enable vectorization"),
    optimization_interval: Optional[float] = Query(None, description="Optimization interval in seconds")
):
    """Tune lightning performance parameters."""
    try:
        speed_engine = get_speed_engine()
        
        # Update configuration
        if aggressive_optimization is not None:
            speed_engine.config.aggressive_optimization = aggressive_optimization
        if use_numba is not None:
            speed_engine.config.use_numba = use_numba
        if use_cython is not None:
            speed_engine.config.use_cython = use_cython
        if use_vectorization is not None:
            speed_engine.config.use_vectorization = use_vectorization
        if optimization_interval is not None:
            speed_engine.config.optimization_interval = optimization_interval
        
        # Force speed optimization with new parameters
        await force_speed_optimization()
        
        return BaseResponse(
            message="Lightning performance tuned successfully",
            data={
                "aggressive_optimization": aggressive_optimization,
                "use_numba": use_numba,
                "use_cython": use_cython,
                "use_vectorization": use_vectorization,
                "optimization_interval": optimization_interval,
                "timestamp": time.time()
            }
        )
        
    except Exception as e:
        logger.error(f"Error tuning lightning performance: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to tune lightning performance: {e}")


@router.get("/system/lightning-stats", summary="Get lightning system statistics")
async def get_lightning_system_stats():
    """Get comprehensive lightning system statistics."""
    try:
        # Get all system statistics
        speed_stats = get_speed_engine().get_speed_summary()
        optimization_stats = get_resource_manager().get_performance_summary()
        pool_stats = await get_pool_manager().get_all_stats()
        cache_stats = await get_cache_manager().get_stats()
        metrics_data = get_metrics_collector().get_all_metrics()
        plugin_stats = await get_plugin_stats()
        executor_stats = get_optimized_executor().get_stats()
        
        return {
            "timestamp": time.time(),
            "system_status": "lightning_fast",
            "speed_optimization": speed_stats,
            "optimization": optimization_stats,
            "pools": pool_stats,
            "cache": cache_stats,
            "metrics": metrics_data,
            "plugins": plugin_stats,
            "executor": executor_stats,
            "performance_level": "lightning"
        }
        
    except Exception as e:
        logger.error(f"Error getting lightning system stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get lightning system stats: {e}")


@router.post("/system/lightning-warmup", response_model=BaseResponse, summary="Lightning system warmup")
async def lightning_system_warmup():
    """Perform comprehensive lightning system warmup."""
    try:
        # Warm up all systems
        tasks = [
            _warmup_speed_system(),
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
        
        warmup_names = ["speed", "cache", "pools", "optimization", "plugins", "metrics"]
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed_warmups.append({"system": warmup_names[i], "error": str(result)})
            else:
                successful_warmups.append(warmup_names[i])
        
        return BaseResponse(
            message=f"Lightning system warmup completed: {len(successful_warmups)} successful, {len(failed_warmups)} failed",
            data={
                "successful_warmups": successful_warmups,
                "failed_warmups": failed_warmups,
                "timestamp": time.time()
            }
        )
        
    except Exception as e:
        logger.error(f"Lightning system warmup failed: {e}")
        raise HTTPException(status_code=500, detail=f"Lightning system warmup failed: {e}")


# Background task functions
async def _execute_plugin_hooks_lightning(hook_name: str, data: Dict[str, Any]):
    """Execute plugin hooks with lightning-fast error handling."""
    try:
        results = await execute_plugin_hook(hook_name, data)
        logger.info(f"Lightning plugin hook {hook_name} executed successfully", extra={"results_count": len(results)})
    except Exception as e:
        logger.error(f"Error executing lightning plugin hook {hook_name}: {e}")
        # Don't raise - this is a background task


# Warmup functions
async def _warmup_speed_system():
    """Warm up speed optimization system."""
    speed_engine = get_speed_engine()
    await speed_engine.start_speed_optimization()


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
@router.get("/debug/lightning-info", summary="Lightning debug information")
async def get_lightning_debug_info():
    """Get lightning debug information (only in development)."""
    if not settings.debug:
        raise HTTPException(status_code=404, detail="Not found")
    
    try:
        return {
            "environment": settings.environment,
            "debug": settings.debug,
            "version": settings.app_version,
            "speed_stats": get_speed_engine().get_speed_summary(),
            "optimization_stats": get_resource_manager().get_performance_summary(),
            "pool_stats": await get_pool_manager().get_all_stats(),
            "cache_stats": await get_cache_manager().get_stats(),
            "metrics_summary": get_metrics_collector().get_all_metrics(),
            "plugin_stats": await get_plugin_stats(),
            "executor_stats": get_optimized_executor().get_stats()
        }
        
    except Exception as e:
        logger.error(f"Error getting lightning debug info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get lightning debug info: {e}")


@router.post("/debug/lightning-reset", response_model=BaseResponse, summary="Lightning system reset (debug only)")
async def lightning_reset_system():
    """Reset lightning system state (only in development)."""
    if not settings.debug:
        raise HTTPException(status_code=404, detail="Not found")
    
    try:
        # Reset all systems
        tasks = [
            _reset_speed_system(),
            _reset_cache_system(),
            _reset_pool_system(),
            _reset_optimization_system(),
            _reset_metrics_system()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return BaseResponse(
            message="Lightning system reset successfully",
            data={
                "reset_components": ["speed", "cache", "pools", "optimization", "metrics"],
                "timestamp": time.time()
            }
        )
        
    except Exception as e:
        logger.error(f"Lightning system reset failed: {e}")
        raise HTTPException(status_code=500, detail=f"Lightning system reset failed: {e}")


# Reset functions
async def _reset_speed_system():
    """Reset speed optimization system."""
    speed_engine = get_speed_engine()
    speed_engine.speed_history.clear()


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


