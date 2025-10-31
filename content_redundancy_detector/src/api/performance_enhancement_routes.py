"""
Performance Enhancement API Routes - Advanced performance optimization endpoints
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..core.performance_enhancement_engine import (
    get_performance_enhancement_engine, PerformanceConfig, 
    PerformanceMetrics, OptimizationResult, MemoryProfile
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/performance-enhancement", tags=["Performance Enhancement"])


# Request/Response Models
class PerformanceConfigRequest(BaseModel):
    """Performance configuration request model"""
    enable_memory_optimization: bool = Field(default=True, description="Enable memory optimization")
    enable_cpu_optimization: bool = Field(default=True, description="Enable CPU optimization")
    enable_cache_optimization: bool = Field(default=True, description="Enable cache optimization")
    enable_database_optimization: bool = Field(default=True, description="Enable database optimization")
    enable_api_optimization: bool = Field(default=True, description="Enable API optimization")
    enable_async_optimization: bool = Field(default=True, description="Enable async optimization")
    memory_threshold_mb: int = Field(default=1024, description="Memory threshold in MB")
    cpu_threshold_percent: float = Field(default=80.0, description="CPU threshold percentage")
    cache_size_limit: int = Field(default=1000, description="Cache size limit")
    max_workers: int = Field(default=4, description="Maximum number of workers")
    enable_profiling: bool = Field(default=True, description="Enable profiling")
    enable_monitoring: bool = Field(default=True, description="Enable monitoring")
    monitoring_interval: float = Field(default=1.0, description="Monitoring interval in seconds")
    enable_auto_optimization: bool = Field(default=True, description="Enable automatic optimization")
    optimization_threshold: float = Field(default=0.8, description="Optimization threshold")


class OptimizationRequest(BaseModel):
    """Optimization request model"""
    optimization_type: str = Field(default="all", description="Type of optimization to perform")
    force_optimization: bool = Field(default=False, description="Force optimization even if not needed")
    include_recommendations: bool = Field(default=True, description="Include optimization recommendations")


class PerformanceMetricsRequest(BaseModel):
    """Performance metrics request model"""
    time_range_minutes: int = Field(default=60, description="Time range in minutes")
    include_details: bool = Field(default=True, description="Include detailed metrics")
    aggregation: str = Field(default="average", description="Aggregation method (average, max, min)")


# Dependency to get performance enhancement engine
async def get_performance_engine():
    """Get performance enhancement engine dependency"""
    engine = await get_performance_enhancement_engine()
    if not engine:
        raise HTTPException(status_code=503, detail="Performance Enhancement Engine not available")
    return engine


# Performance Enhancement Routes
@router.post("/optimize", response_model=Dict[str, Any])
async def optimize_performance(
    request: OptimizationRequest,
    engine: PerformanceEnhancementEngine = Depends(get_performance_engine)
):
    """Optimize system performance"""
    try:
        start_time = time.time()
        
        if request.optimization_type == "all":
            # Optimize all components
            results = await engine.optimize_all()
        elif request.optimization_type == "memory":
            # Optimize memory only
            result = await engine.optimize_memory()
            results = [result]
        elif request.optimization_type == "cpu":
            # Optimize CPU only
            result = await engine.optimize_cpu()
            results = [result]
        else:
            raise HTTPException(status_code=400, detail=f"Unknown optimization type: {request.optimization_type}")
        
        processing_time = (time.time() - start_time) * 1000
        
        # Format results
        formatted_results = []
        total_improvement = 0.0
        
        for result in results:
            formatted_results.append({
                "optimization_id": result.optimization_id,
                "timestamp": result.timestamp.isoformat(),
                "optimization_type": result.optimization_type,
                "improvement_percent": result.improvement_percent,
                "before_metrics": {
                    "cpu_percent": result.before_metrics.cpu_percent,
                    "memory_percent": result.before_metrics.memory_percent,
                    "memory_used_mb": result.before_metrics.memory_used_mb
                },
                "after_metrics": {
                    "cpu_percent": result.after_metrics.cpu_percent,
                    "memory_percent": result.after_metrics.memory_percent,
                    "memory_used_mb": result.after_metrics.memory_used_mb
                },
                "optimization_details": result.optimization_details,
                "recommendations": result.recommendations if request.include_recommendations else []
            })
            total_improvement += result.improvement_percent
        
        return {
            "success": True,
            "optimization_results": formatted_results,
            "total_improvement_percent": total_improvement,
            "optimization_count": len(results),
            "processing_time_ms": processing_time,
            "message": f"Performance optimization completed successfully"
        }
        
    except Exception as e:
        logger.error(f"Error in performance optimization: {e}")
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")


@router.get("/metrics", response_model=Dict[str, Any])
async def get_performance_metrics(
    request: PerformanceMetricsRequest = Depends(),
    engine: PerformanceEnhancementEngine = Depends(get_performance_engine)
):
    """Get performance metrics"""
    try:
        # Get performance metrics
        metrics = await engine.get_performance_metrics()
        
        # Filter by time range
        cutoff_time = datetime.now() - timedelta(minutes=request.time_range_minutes)
        filtered_metrics = [m for m in metrics if m.timestamp >= cutoff_time]
        
        if not filtered_metrics:
            return {
                "success": True,
                "performance_metrics": [],
                "summary": {},
                "message": "No metrics available for the specified time range"
            }
        
        # Aggregate metrics based on request
        if request.aggregation == "average":
            summary = {
                "avg_cpu_percent": sum(m.cpu_percent for m in filtered_metrics) / len(filtered_metrics),
                "avg_memory_percent": sum(m.memory_percent for m in filtered_metrics) / len(filtered_metrics),
                "avg_memory_used_mb": sum(m.memory_used_mb for m in filtered_metrics) / len(filtered_metrics),
                "avg_disk_usage_percent": sum(m.disk_usage_percent for m in filtered_metrics) / len(filtered_metrics),
                "avg_active_threads": sum(m.active_threads for m in filtered_metrics) / len(filtered_metrics)
            }
        elif request.aggregation == "max":
            summary = {
                "max_cpu_percent": max(m.cpu_percent for m in filtered_metrics),
                "max_memory_percent": max(m.memory_percent for m in filtered_metrics),
                "max_memory_used_mb": max(m.memory_used_mb for m in filtered_metrics),
                "max_disk_usage_percent": max(m.disk_usage_percent for m in filtered_metrics),
                "max_active_threads": max(m.active_threads for m in filtered_metrics)
            }
        elif request.aggregation == "min":
            summary = {
                "min_cpu_percent": min(m.cpu_percent for m in filtered_metrics),
                "min_memory_percent": min(m.memory_percent for m in filtered_metrics),
                "min_memory_used_mb": min(m.memory_used_mb for m in filtered_metrics),
                "min_disk_usage_percent": min(m.disk_usage_percent for m in filtered_metrics),
                "min_active_threads": min(m.active_threads for m in filtered_metrics)
            }
        else:
            summary = {}
        
        # Format metrics
        formatted_metrics = []
        for metric in filtered_metrics:
            metric_data = {
                "timestamp": metric.timestamp.isoformat(),
                "cpu_percent": metric.cpu_percent,
                "memory_percent": metric.memory_percent,
                "memory_used_mb": metric.memory_used_mb,
                "memory_available_mb": metric.memory_available_mb,
                "disk_usage_percent": metric.disk_usage_percent,
                "active_threads": metric.active_threads,
                "active_processes": metric.active_processes
            }
            
            if request.include_details:
                metric_data.update({
                    "network_io_bytes": metric.network_io_bytes,
                    "cache_hit_rate": metric.cache_hit_rate,
                    "response_time_ms": metric.response_time_ms,
                    "throughput_requests_per_sec": metric.throughput_requests_per_sec,
                    "error_rate": metric.error_rate,
                    "gc_collections": metric.gc_collections,
                    "gc_time_ms": metric.gc_time_ms
                })
            
            formatted_metrics.append(metric_data)
        
        return {
            "success": True,
            "performance_metrics": formatted_metrics,
            "summary": summary,
            "metrics_count": len(formatted_metrics),
            "time_range_minutes": request.time_range_minutes,
            "aggregation": request.aggregation,
            "message": "Performance metrics retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance metrics: {str(e)}")


@router.get("/memory-profile", response_model=Dict[str, Any])
async def get_memory_profile(
    engine: PerformanceEnhancementEngine = Depends(get_performance_engine)
):
    """Get detailed memory profile"""
    try:
        # Get memory profile
        profile = await engine.get_memory_profile()
        
        return {
            "success": True,
            "memory_profile": {
                "timestamp": profile.timestamp.isoformat(),
                "total_memory_mb": profile.total_memory_mb,
                "used_memory_mb": profile.used_memory_mb,
                "available_memory_mb": profile.available_memory_mb,
                "memory_percent": profile.memory_percent,
                "top_memory_consumers": profile.top_memory_consumers,
                "memory_leaks": profile.memory_leaks,
                "gc_stats": profile.gc_stats
            },
            "message": "Memory profile retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Error getting memory profile: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get memory profile: {str(e)}")


@router.get("/optimization-history", response_model=Dict[str, Any])
async def get_optimization_history(
    limit: int = 50,
    optimization_type: Optional[str] = None,
    engine: PerformanceEnhancementEngine = Depends(get_performance_engine)
):
    """Get optimization history"""
    try:
        # Get optimization history
        history = await engine.get_optimization_history()
        
        # Filter by optimization type if specified
        if optimization_type:
            history = [h for h in history if h.optimization_type == optimization_type]
        
        # Limit results
        history = history[-limit:] if limit > 0 else history
        
        # Format history
        formatted_history = []
        for optimization in history:
            formatted_history.append({
                "optimization_id": optimization.optimization_id,
                "timestamp": optimization.timestamp.isoformat(),
                "optimization_type": optimization.optimization_type,
                "improvement_percent": optimization.improvement_percent,
                "before_metrics": {
                    "cpu_percent": optimization.before_metrics.cpu_percent,
                    "memory_percent": optimization.before_metrics.memory_percent,
                    "memory_used_mb": optimization.before_metrics.memory_used_mb
                },
                "after_metrics": {
                    "cpu_percent": optimization.after_metrics.cpu_percent,
                    "memory_percent": optimization.after_metrics.memory_percent,
                    "memory_used_mb": optimization.after_metrics.memory_used_mb
                },
                "recommendations": optimization.recommendations
            })
        
        return {
            "success": True,
            "optimization_history": formatted_history,
            "total_count": len(formatted_history),
            "message": "Optimization history retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Error getting optimization history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get optimization history: {str(e)}")


@router.get("/performance-summary", response_model=Dict[str, Any])
async def get_performance_summary(
    engine: PerformanceEnhancementEngine = Depends(get_performance_engine)
):
    """Get performance summary"""
    try:
        # Get performance summary
        summary = await engine.get_performance_summary()
        
        return {
            "success": True,
            "performance_summary": summary,
            "message": "Performance summary retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Error getting performance summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance summary: {str(e)}")


@router.post("/configure", response_model=Dict[str, Any])
async def configure_performance(
    request: PerformanceConfigRequest,
    engine: PerformanceEnhancementEngine = Depends(get_performance_engine)
):
    """Configure performance enhancement settings"""
    try:
        # Update configuration
        config = PerformanceConfig(
            enable_memory_optimization=request.enable_memory_optimization,
            enable_cpu_optimization=request.enable_cpu_optimization,
            enable_cache_optimization=request.enable_cache_optimization,
            enable_database_optimization=request.enable_database_optimization,
            enable_api_optimization=request.enable_api_optimization,
            enable_async_optimization=request.enable_async_optimization,
            memory_threshold_mb=request.memory_threshold_mb,
            cpu_threshold_percent=request.cpu_threshold_percent,
            cache_size_limit=request.cache_size_limit,
            max_workers=request.max_workers,
            enable_profiling=request.enable_profiling,
            enable_monitoring=request.enable_monitoring,
            monitoring_interval=request.monitoring_interval,
            enable_auto_optimization=request.enable_auto_optimization,
            optimization_threshold=request.optimization_threshold
        )
        
        # Update engine configuration
        engine.config = config
        
        return {
            "success": True,
            "configuration": {
                "enable_memory_optimization": config.enable_memory_optimization,
                "enable_cpu_optimization": config.enable_cpu_optimization,
                "enable_cache_optimization": config.enable_cache_optimization,
                "enable_database_optimization": config.enable_database_optimization,
                "enable_api_optimization": config.enable_api_optimization,
                "enable_async_optimization": config.enable_async_optimization,
                "memory_threshold_mb": config.memory_threshold_mb,
                "cpu_threshold_percent": config.cpu_threshold_percent,
                "cache_size_limit": config.cache_size_limit,
                "max_workers": config.max_workers,
                "enable_profiling": config.enable_profiling,
                "enable_monitoring": config.enable_monitoring,
                "monitoring_interval": config.monitoring_interval,
                "enable_auto_optimization": config.enable_auto_optimization,
                "optimization_threshold": config.optimization_threshold
            },
            "message": "Performance configuration updated successfully"
        }
        
    except Exception as e:
        logger.error(f"Error configuring performance: {e}")
        raise HTTPException(status_code=500, detail=f"Configuration failed: {str(e)}")


@router.get("/capabilities", response_model=Dict[str, Any])
async def get_performance_capabilities(
    engine: PerformanceEnhancementEngine = Depends(get_performance_engine)
):
    """Get performance enhancement capabilities"""
    try:
        capabilities = {
            "memory_optimization": {
                "garbage_collection": "Automatic garbage collection optimization",
                "memory_pool_optimization": "Memory pool management and optimization",
                "weak_reference_cleanup": "Weak reference cleanup and management",
                "memory_compression": "Memory usage compression techniques",
                "memory_profiling": "Detailed memory profiling and analysis",
                "leak_detection": "Memory leak detection and prevention"
            },
            "cpu_optimization": {
                "thread_pool_optimization": "Dynamic thread pool size optimization",
                "process_pool_optimization": "Process pool management and optimization",
                "cpu_affinity_optimization": "CPU affinity and core utilization",
                "frequency_optimization": "CPU frequency and power management",
                "load_balancing": "Intelligent load balancing across cores",
                "parallel_processing": "Optimized parallel processing strategies"
            },
            "monitoring": {
                "real_time_monitoring": "Real-time performance monitoring",
                "metrics_collection": "Comprehensive metrics collection",
                "performance_profiling": "Detailed performance profiling",
                "system_health_checks": "System health monitoring and alerts",
                "trend_analysis": "Performance trend analysis and reporting",
                "automated_optimization": "Automatic optimization triggers"
            },
            "advanced_features": {
                "auto_optimization": "Automatic performance optimization",
                "threshold_based_optimization": "Threshold-based optimization triggers",
                "performance_prediction": "Performance trend prediction",
                "resource_optimization": "Intelligent resource allocation",
                "bottleneck_detection": "Automatic bottleneck detection",
                "optimization_recommendations": "AI-powered optimization recommendations"
            }
        }
        
        return {
            "success": True,
            "capabilities": capabilities,
            "message": "Performance capabilities retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Error getting performance capabilities: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance capabilities: {str(e)}")


@router.get("/health", response_model=Dict[str, Any])
async def health_check(
    engine: PerformanceEnhancementEngine = Depends(get_performance_engine)
):
    """Performance Enhancement Engine health check"""
    try:
        # Check engine components
        components_status = {
            "memory_optimizer": engine.memory_optimizer is not None,
            "cpu_optimizer": engine.cpu_optimizer is not None,
            "monitoring_active": engine.monitoring_active
        }
        
        # Get current performance metrics
        current_metrics = await engine._collect_performance_metrics()
        
        # Check system health
        system_health = {
            "cpu_usage": current_metrics.cpu_percent,
            "memory_usage": current_metrics.memory_percent,
            "disk_usage": current_metrics.disk_usage_percent,
            "active_threads": current_metrics.active_threads,
            "active_processes": current_metrics.active_processes
        }
        
        # Determine overall health
        all_healthy = all(components_status.values())
        system_healthy = (
            current_metrics.cpu_percent < 90 and
            current_metrics.memory_percent < 90 and
            current_metrics.disk_usage_percent < 90
        )
        
        overall_health = "healthy" if all_healthy and system_healthy else "degraded"
        
        return {
            "status": overall_health,
            "timestamp": datetime.now().isoformat(),
            "components": components_status,
            "system_health": system_health,
            "optimization_count": len(engine.optimization_history),
            "metrics_count": len(engine.performance_history),
            "message": "Performance Enhancement Engine is operational" if overall_health == "healthy" else "Some components may need attention"
        }
        
    except Exception as e:
        logger.error(f"Error in Performance Enhancement health check: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "message": "Performance Enhancement Engine health check failed"
        }
