"""
Advanced API router with enhanced features.
"""

import asyncio
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query, Path
from fastapi.responses import StreamingResponse, JSONResponse
import json

from ...core.config import get_settings
from ...core.cache import get_cache_manager, invalidate_analysis_cache, invalidate_plugin_cache
from ...core.metrics import get_metrics_collector, get_prometheus_metrics
from ...core.logging import get_logger
from ...models.schemas import (
    ContentAnalysisRequest, ContentAnalysisResponse,
    BaseResponse, SystemStats, HealthCheckResponse
)
from ...services.advanced_analysis_service import analyze_content_advanced
from ...services.plugin_service import (
    discover_plugins, install_plugin, activate_plugin, deactivate_plugin,
    list_plugins, get_plugin_stats, execute_plugin_hook
)

router = APIRouter()
settings = get_settings()
logger = get_logger(__name__)


@router.get("/health/advanced", response_model=HealthCheckResponse, summary="Advanced health check")
async def advanced_health_check():
    """Advanced health check with detailed system status."""
    try:
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
            "cache_system": cache_stats.get("redis_connected", False) or cache_stats.get("memory_cache_size", 0) >= 0,
            "metrics_system": len(metrics_stats) > 0,
            "plugin_system": plugin_stats.get("total_plugins", 0) >= 0,
            "analysis_system": True,  # Always healthy for now
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
        logger.error(f"Health check failed: {e}")
        return HealthCheckResponse(
            status="unhealthy",
            systems={"error": str(e)},
            version=settings.app_version
        )


@router.get("/metrics", summary="Get system metrics")
async def get_system_metrics():
    """Get comprehensive system metrics."""
    try:
        metrics_collector = get_metrics_collector()
        return metrics_collector.get_all_metrics()
        
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {e}")


@router.get("/metrics/prometheus", summary="Get Prometheus metrics")
async def get_prometheus_metrics_endpoint():
    """Get metrics in Prometheus format."""
    try:
        metrics = await get_prometheus_metrics()
        return StreamingResponse(
            iter([metrics]),
            media_type="text/plain; version=0.0.4; charset=utf-8"
        )
        
    except Exception as e:
        logger.error(f"Error getting Prometheus metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get Prometheus metrics: {e}")


@router.get("/cache/stats", summary="Get cache statistics")
async def get_cache_stats():
    """Get cache system statistics."""
    try:
        cache_manager = await get_cache_manager()
        return await cache_manager.get_stats()
        
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get cache stats: {e}")


@router.post("/cache/clear", response_model=BaseResponse, summary="Clear cache")
async def clear_cache(
    cache_type: str = Query("all", description="Type of cache to clear: all, analysis, plugins")
):
    """Clear cache by type."""
    try:
        cache_manager = await get_cache_manager()
        
        if cache_type == "all":
            await cache_manager.clear()
            message = "All cache cleared successfully"
        elif cache_type == "analysis":
            await invalidate_analysis_cache()
            message = "Analysis cache cleared successfully"
        elif cache_type == "plugins":
            await invalidate_plugin_cache()
            message = "Plugin cache cleared successfully"
        else:
            raise HTTPException(status_code=400, detail="Invalid cache type")
        
        return BaseResponse(message=message)
        
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {e}")


@router.post("/analyze/advanced", response_model=ContentAnalysisResponse, summary="Advanced content analysis")
async def advanced_analyze_content(
    request: ContentAnalysisRequest,
    background_tasks: BackgroundTasks,
    use_cache: bool = Query(True, description="Use cache for analysis"),
    parallel_processing: bool = Query(True, description="Use parallel processing")
):
    """Perform advanced content analysis with ML capabilities."""
    try:
        # Execute plugin hooks in background
        background_tasks.add_task(
            _execute_plugin_hooks_advanced,
            "pre_analysis",
            request.dict()
        )
        
        # Perform advanced analysis
        result = await analyze_content_advanced(request)
        
        # Execute post-analysis hooks
        background_tasks.add_task(
            _execute_plugin_hooks_advanced,
            "post_analysis",
            result.dict()
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Advanced analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Advanced analysis failed: {e}")


@router.post("/analyze/batch", summary="Batch content analysis")
async def batch_analyze_content(
    requests: List[ContentAnalysisRequest],
    background_tasks: BackgroundTasks,
    max_concurrent: int = Query(5, description="Maximum concurrent analyses")
):
    """Perform batch content analysis."""
    try:
        if len(requests) > 100:
            raise HTTPException(status_code=400, detail="Maximum 100 requests per batch")
        
        # Process in batches
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def analyze_with_semaphore(request: ContentAnalysisRequest):
            async with semaphore:
                return await analyze_content_advanced(request)
        
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
            _execute_plugin_hooks_advanced,
            "batch_analysis_completed",
            {
                "total_requests": len(requests),
                "successful": len(successful_results),
                "failed": len(failed_results)
            }
        )
        
        return {
            "total_requests": len(requests),
            "successful_analyses": len(successful_results),
            "failed_analyses": len(failed_results),
            "results": successful_results,
            "errors": failed_results
        }
        
    except Exception as e:
        logger.error(f"Batch analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {e}")


@router.get("/plugins/discover", summary="Discover plugins")
async def discover_plugins_endpoint(
    plugin_directory: str = Query("plugins", description="Plugin directory to scan")
):
    """Discover available plugins in directory."""
    try:
        plugins = await discover_plugins(plugin_directory)
        return BaseResponse(
            message=f"Discovered {len(plugins)} plugins",
            data={"plugins": [plugin.dict() for plugin in plugins]}
        )
        
    except Exception as e:
        logger.error(f"Plugin discovery failed: {e}")
        raise HTTPException(status_code=500, detail=f"Plugin discovery failed: {e}")


@router.post("/plugins/batch-install", response_model=BaseResponse, summary="Batch install plugins")
async def batch_install_plugins(
    plugin_names: List[str],
    auto_install_dependencies: bool = Query(True, description="Auto-install dependencies")
):
    """Install multiple plugins in batch."""
    try:
        if len(plugin_names) > 20:
            raise HTTPException(status_code=400, detail="Maximum 20 plugins per batch")
        
        results = []
        for plugin_name in plugin_names:
            try:
                success = await install_plugin(plugin_name, auto_install_dependencies)
                results.append({"plugin": plugin_name, "success": success})
            except Exception as e:
                results.append({"plugin": plugin_name, "success": False, "error": str(e)})
        
        successful = sum(1 for r in results if r["success"])
        failed = len(results) - successful
        
        return BaseResponse(
            message=f"Batch installation completed: {successful} successful, {failed} failed",
            data={"results": results}
        )
        
    except Exception as e:
        logger.error(f"Batch plugin installation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch plugin installation failed: {e}")


@router.post("/plugins/{plugin_name}/hooks/{hook_name}/execute", summary="Execute plugin hook")
async def execute_plugin_hook_endpoint(
    plugin_name: str = Path(..., description="Plugin name"),
    hook_name: str = Path(..., description="Hook name"),
    data: Dict[str, Any] = None
):
    """Execute a specific plugin hook."""
    try:
        # Check if plugin is active
        from ...services.plugin_service import is_plugin_active
        if not await is_plugin_active(plugin_name):
            raise HTTPException(status_code=400, detail=f"Plugin {plugin_name} is not active")
        
        # Execute hook
        results = await execute_plugin_hook(hook_name, data or {})
        
        return BaseResponse(
            message=f"Hook {hook_name} executed for plugin {plugin_name}",
            data={"results": results}
        )
        
    except Exception as e:
        logger.error(f"Plugin hook execution failed: {e}")
        raise HTTPException(status_code=500, detail=f"Plugin hook execution failed: {e}")


@router.get("/analysis/history", summary="Get analysis history")
async def get_analysis_history(
    limit: int = Query(50, description="Number of analyses to return"),
    offset: int = Query(0, description="Number of analyses to skip")
):
    """Get analysis history (cached results)."""
    try:
        cache_manager = await get_cache_manager()
        
        # This would typically query a database in a real implementation
        # For now, we'll return cached analysis results
        history = []
        
        # Get recent analysis results from cache
        # In a real implementation, this would be stored in a database
        return BaseResponse(
            message=f"Retrieved {len(history)} analysis records",
            data={
                "history": history,
                "limit": limit,
                "offset": offset,
                "total": len(history)
            }
        )
        
    except Exception as e:
        logger.error(f"Error getting analysis history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get analysis history: {e}")


@router.get("/system/performance", summary="Get system performance metrics")
async def get_system_performance():
    """Get detailed system performance metrics."""
    try:
        metrics_collector = get_metrics_collector()
        cache_manager = await get_cache_manager()
        
        # Get performance data
        performance_data = {}
        for operation in ["advanced_analysis", "plugin_operations", "cache_operations"]:
            performance_data[operation] = metrics_collector.get_performance_stats(operation)
        
        # Get cache performance
        cache_stats = await cache_manager.get_stats()
        
        # Get error statistics
        error_stats = metrics_collector.get_error_stats()
        
        return {
            "timestamp": metrics_collector.get_all_metrics().get("timestamp"),
            "performance": performance_data,
            "cache": cache_stats,
            "errors": error_stats,
            "system_health": "good" if len(error_stats) < 10 else "degraded"
        }
        
    except Exception as e:
        logger.error(f"Error getting system performance: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get system performance: {e}")


@router.post("/system/warmup", response_model=BaseResponse, summary="Warm up system")
async def warm_up_system():
    """Warm up system caches and preload data."""
    try:
        from ...core.cache import warm_up_cache
        
        # Warm up cache
        await warm_up_cache()
        
        # Discover plugins
        await discover_plugins()
        
        # Preload common data
        await get_plugin_stats()
        
        return BaseResponse(
            message="System warmed up successfully",
            data={"warmed_up_components": ["cache", "plugins", "stats"]}
        )
        
    except Exception as e:
        logger.error(f"System warmup failed: {e}")
        raise HTTPException(status_code=500, detail=f"System warmup failed: {e}")


# Background task functions
async def _execute_plugin_hooks_advanced(hook_name: str, data: Dict[str, Any]):
    """Execute plugin hooks with advanced error handling."""
    try:
        results = await execute_plugin_hook(hook_name, data)
        logger.info(f"Plugin hook {hook_name} executed successfully", extra={"results_count": len(results)})
    except Exception as e:
        logger.error(f"Error executing plugin hook {hook_name}: {e}")
        # Don't raise - this is a background task


# Utility endpoints
@router.get("/debug/info", summary="Debug information")
async def get_debug_info():
    """Get debug information (only in development)."""
    if not settings.debug:
        raise HTTPException(status_code=404, detail="Not found")
    
    try:
        return {
            "environment": settings.environment,
            "debug": settings.debug,
            "version": settings.app_version,
            "cache_stats": await (await get_cache_manager()).get_stats(),
            "metrics_summary": get_metrics_collector().get_all_metrics(),
            "plugin_stats": await get_plugin_stats()
        }
        
    except Exception as e:
        logger.error(f"Error getting debug info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get debug info: {e}")


@router.post("/debug/reset", response_model=BaseResponse, summary="Reset system (debug only)")
async def reset_system():
    """Reset system state (only in development)."""
    if not settings.debug:
        raise HTTPException(status_code=404, detail="Not found")
    
    try:
        # Clear cache
        cache_manager = await get_cache_manager()
        await cache_manager.clear()
        
        # Reset metrics
        metrics_collector = get_metrics_collector()
        metrics_collector.reset_metrics()
        
        return BaseResponse(
            message="System reset successfully",
            data={"reset_components": ["cache", "metrics"]}
        )
        
    except Exception as e:
        logger.error(f"System reset failed: {e}")
        raise HTTPException(status_code=500, detail=f"System reset failed: {e}")


