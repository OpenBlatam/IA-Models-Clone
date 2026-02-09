"""
Optimization Routes - API endpoints for Optimization Engine
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, Field

from ..core.optimization_engine import (
    optimization_engine,
    OptimizationConfig,
    PerformanceMetrics,
    OptimizationResult
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/optimization", tags=["Optimization"])


# Pydantic models
class OptimizationConfigRequest(BaseModel):
    """Request model for optimization configuration"""
    enable_memory_optimization: bool = Field(True, description="Enable memory optimization")
    enable_cpu_optimization: bool = Field(True, description="Enable CPU optimization")
    enable_gpu_optimization: bool = Field(False, description="Enable GPU optimization")
    enable_distributed_computing: bool = Field(False, description="Enable distributed computing")
    enable_async_optimization: bool = Field(True, description="Enable async optimization")
    enable_cache_optimization: bool = Field(True, description="Enable cache optimization")
    enable_database_optimization: bool = Field(True, description="Enable database optimization")
    enable_api_optimization: bool = Field(True, description="Enable API optimization")
    max_memory_usage: float = Field(0.8, description="Maximum memory usage threshold")
    max_cpu_usage: float = Field(0.8, description="Maximum CPU usage threshold")
    cache_size_limit: int = Field(1000, description="Cache size limit")
    cache_ttl: int = Field(3600, description="Cache TTL in seconds")
    batch_size: int = Field(100, description="Batch size for processing")
    max_concurrent_requests: int = Field(100, description="Maximum concurrent requests")
    connection_pool_size: int = Field(20, description="Connection pool size")
    query_timeout: int = Field(30, description="Query timeout in seconds")
    enable_profiling: bool = Field(True, description="Enable performance profiling")
    enable_monitoring: bool = Field(True, description="Enable performance monitoring")
    optimization_interval: int = Field(60, description="Optimization interval in seconds")


class SystemOptimizationRequest(BaseModel):
    """Request model for system optimization"""
    optimization_types: List[str] = Field(
        default=["memory", "cpu", "cache", "database", "api", "async"],
        description="Types of optimization to perform"
    )
    force_optimization: bool = Field(False, description="Force optimization even if not needed")


class PerformanceAnalysisRequest(BaseModel):
    """Request model for performance analysis"""
    analysis_duration: int = Field(60, description="Analysis duration in seconds")
    include_profiling: bool = Field(True, description="Include detailed profiling")
    include_recommendations: bool = Field(True, description="Include optimization recommendations")


# System optimization endpoints
@router.post("/system", response_model=Dict[str, Any])
async def optimize_system(request: SystemOptimizationRequest, background_tasks: BackgroundTasks):
    """Perform comprehensive system optimization"""
    try:
        if not optimization_engine:
            raise HTTPException(status_code=503, detail="Optimization Engine not initialized")
        
        # Perform system optimization
        optimization_results = await optimization_engine.optimize_system()
        
        # Prepare response data
        response_data = {
            "optimization_types": request.optimization_types,
            "total_optimizations": len(optimization_results),
            "optimization_results": {}
        }
        
        # Process optimization results
        total_improvement = 0.0
        successful_optimizations = 0
        
        for opt_type, result in optimization_results.items():
            if result.success:
                successful_optimizations += 1
                total_improvement += result.improvement_percentage
                
                response_data["optimization_results"][opt_type] = {
                    "success": result.success,
                    "improvement_percentage": result.improvement_percentage,
                    "optimization_time_ms": result.optimization_time,
                    "recommendations": result.recommendations,
                    "before_metrics": result.before_metrics,
                    "after_metrics": result.after_metrics
                }
            else:
                response_data["optimization_results"][opt_type] = {
                    "success": result.success,
                    "error_message": result.error_message,
                    "optimization_time_ms": result.optimization_time
                }
        
        # Calculate overall improvement
        avg_improvement = total_improvement / max(successful_optimizations, 1)
        
        response_data["summary"] = {
            "successful_optimizations": successful_optimizations,
            "failed_optimizations": len(optimization_results) - successful_optimizations,
            "average_improvement_percentage": avg_improvement,
            "total_improvement_percentage": total_improvement
        }
        
        return {
            "success": True,
            "data": response_data,
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error optimizing system: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/memory", response_model=Dict[str, Any])
async def optimize_memory():
    """Optimize memory usage"""
    try:
        if not optimization_engine:
            raise HTTPException(status_code=503, detail="Optimization Engine not initialized")
        
        # Perform memory optimization
        result = await optimization_engine.memory_optimizer.optimize_memory()
        
        return {
            "success": result.success,
            "data": {
                "optimization_type": result.optimization_type,
                "improvement_percentage": result.improvement_percentage,
                "optimization_time_ms": result.optimization_time,
                "recommendations": result.recommendations,
                "before_metrics": result.before_metrics,
                "after_metrics": result.after_metrics
            },
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error optimizing memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cpu", response_model=Dict[str, Any])
async def optimize_cpu():
    """Optimize CPU usage"""
    try:
        if not optimization_engine:
            raise HTTPException(status_code=503, detail="Optimization Engine not initialized")
        
        # Perform CPU optimization
        result = await optimization_engine.cpu_optimizer.optimize_cpu()
        
        return {
            "success": result.success,
            "data": {
                "optimization_type": result.optimization_type,
                "improvement_percentage": result.improvement_percentage,
                "optimization_time_ms": result.optimization_time,
                "recommendations": result.recommendations,
                "before_metrics": result.before_metrics,
                "after_metrics": result.after_metrics
            },
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error optimizing CPU: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cache", response_model=Dict[str, Any])
async def optimize_cache():
    """Optimize cache performance"""
    try:
        if not optimization_engine:
            raise HTTPException(status_code=503, detail="Optimization Engine not initialized")
        
        # Perform cache optimization
        result = await optimization_engine.cache_optimizer.optimize_cache()
        
        return {
            "success": result.success,
            "data": {
                "optimization_type": result.optimization_type,
                "improvement_percentage": result.improvement_percentage,
                "optimization_time_ms": result.optimization_time,
                "recommendations": result.recommendations,
                "before_metrics": result.before_metrics,
                "after_metrics": result.after_metrics
            },
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error optimizing cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/database", response_model=Dict[str, Any])
async def optimize_database():
    """Optimize database performance"""
    try:
        if not optimization_engine:
            raise HTTPException(status_code=503, detail="Optimization Engine not initialized")
        
        # Perform database optimization
        result = await optimization_engine.database_optimizer.optimize_database()
        
        return {
            "success": result.success,
            "data": {
                "optimization_type": result.optimization_type,
                "improvement_percentage": result.improvement_percentage,
                "optimization_time_ms": result.optimization_time,
                "recommendations": result.recommendations,
                "before_metrics": result.before_metrics,
                "after_metrics": result.after_metrics
            },
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error optimizing database: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api", response_model=Dict[str, Any])
async def optimize_api():
    """Optimize API performance"""
    try:
        if not optimization_engine:
            raise HTTPException(status_code=503, detail="Optimization Engine not initialized")
        
        # Perform API optimization
        result = await optimization_engine.api_optimizer.optimize_api()
        
        return {
            "success": result.success,
            "data": {
                "optimization_type": result.optimization_type,
                "improvement_percentage": result.improvement_percentage,
                "optimization_time_ms": result.optimization_time,
                "recommendations": result.recommendations,
                "before_metrics": result.before_metrics,
                "after_metrics": result.after_metrics
            },
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error optimizing API: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/async", response_model=Dict[str, Any])
async def optimize_async():
    """Optimize async performance"""
    try:
        if not optimization_engine:
            raise HTTPException(status_code=503, detail="Optimization Engine not initialized")
        
        # Perform async optimization
        result = await optimization_engine.async_optimizer.optimize_async()
        
        return {
            "success": result.success,
            "data": {
                "optimization_type": result.optimization_type,
                "improvement_percentage": result.improvement_percentage,
                "optimization_time_ms": result.optimization_time,
                "recommendations": result.recommendations,
                "before_metrics": result.before_metrics,
                "after_metrics": result.after_metrics
            },
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error optimizing async: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Performance monitoring endpoints
@router.get("/metrics", response_model=Dict[str, Any])
async def get_performance_metrics():
    """Get current performance metrics"""
    try:
        if not optimization_engine:
            raise HTTPException(status_code=503, detail="Optimization Engine not initialized")
        
        # Get performance metrics
        metrics = await optimization_engine.get_performance_metrics()
        
        return {
            "success": True,
            "data": {
                "timestamp": metrics.timestamp,
                "cpu_usage": metrics.cpu_usage,
                "memory_usage": metrics.memory_usage,
                "memory_available_gb": metrics.memory_available,
                "disk_usage": metrics.disk_usage,
                "network_io": metrics.network_io,
                "active_connections": metrics.active_connections,
                "request_count": metrics.request_count,
                "response_time_avg": metrics.response_time_avg,
                "response_time_p95": metrics.response_time_p95,
                "response_time_p99": metrics.response_time_p99,
                "error_rate": metrics.error_rate,
                "cache_hit_rate": metrics.cache_hit_rate,
                "database_query_time": metrics.database_query_time,
                "optimization_score": metrics.optimization_score
            },
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/history", response_model=Dict[str, Any])
async def get_performance_history(limit: int = Query(100, description="Number of metrics to return")):
    """Get performance metrics history"""
    try:
        if not optimization_engine:
            raise HTTPException(status_code=503, detail="Optimization Engine not initialized")
        
        # Get performance history
        history = await optimization_engine.get_performance_history()
        
        # Limit results
        limited_history = history[-limit:] if len(history) > limit else history
        
        # Convert to dict format
        history_data = []
        for metrics in limited_history:
            history_data.append({
                "timestamp": metrics.timestamp,
                "cpu_usage": metrics.cpu_usage,
                "memory_usage": metrics.memory_usage,
                "memory_available_gb": metrics.memory_available,
                "disk_usage": metrics.disk_usage,
                "response_time_avg": metrics.response_time_avg,
                "cache_hit_rate": metrics.cache_hit_rate,
                "optimization_score": metrics.optimization_score
            })
        
        return {
            "success": True,
            "data": {
                "total_metrics": len(history),
                "returned_metrics": len(history_data),
                "history": history_data
            },
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error getting performance history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze", response_model=Dict[str, Any])
async def analyze_performance(request: PerformanceAnalysisRequest, background_tasks: BackgroundTasks):
    """Analyze system performance"""
    try:
        if not optimization_engine:
            raise HTTPException(status_code=503, detail="Optimization Engine not initialized")
        
        # Start performance analysis
        analysis_start_time = datetime.now()
        
        # Collect performance metrics over the analysis duration
        metrics_collection = []
        analysis_duration = request.analysis_duration
        
        for i in range(analysis_duration):
            metrics = await optimization_engine.get_performance_metrics()
            metrics_collection.append(metrics)
            
            if i < analysis_duration - 1:  # Don't sleep on the last iteration
                await asyncio.sleep(1)
        
        # Analyze collected metrics
        analysis_results = {
            "analysis_duration_seconds": analysis_duration,
            "analysis_start_time": analysis_start_time,
            "analysis_end_time": datetime.now(),
            "metrics_collected": len(metrics_collection),
            "performance_summary": {},
            "recommendations": []
        }
        
        if metrics_collection:
            # Calculate summary statistics
            cpu_usage_values = [m.cpu_usage for m in metrics_collection]
            memory_usage_values = [m.memory_usage for m in metrics_collection]
            response_time_values = [m.response_time_avg for m in metrics_collection]
            optimization_scores = [m.optimization_score for m in metrics_collection]
            
            analysis_results["performance_summary"] = {
                "cpu_usage": {
                    "average": statistics.mean(cpu_usage_values),
                    "min": min(cpu_usage_values),
                    "max": max(cpu_usage_values),
                    "std": statistics.stdev(cpu_usage_values) if len(cpu_usage_values) > 1 else 0
                },
                "memory_usage": {
                    "average": statistics.mean(memory_usage_values),
                    "min": min(memory_usage_values),
                    "max": max(memory_usage_values),
                    "std": statistics.stdev(memory_usage_values) if len(memory_usage_values) > 1 else 0
                },
                "response_time": {
                    "average": statistics.mean(response_time_values),
                    "min": min(response_time_values),
                    "max": max(response_time_values),
                    "std": statistics.stdev(response_time_values) if len(response_time_values) > 1 else 0
                },
                "optimization_score": {
                    "average": statistics.mean(optimization_scores),
                    "min": min(optimization_scores),
                    "max": max(optimization_scores),
                    "std": statistics.stdev(optimization_scores) if len(optimization_scores) > 1 else 0
                }
            }
            
            # Generate recommendations
            if request.include_recommendations:
                recommendations = []
                
                avg_cpu = statistics.mean(cpu_usage_values)
                avg_memory = statistics.mean(memory_usage_values)
                avg_response_time = statistics.mean(response_time_values)
                avg_optimization_score = statistics.mean(optimization_scores)
                
                if avg_cpu > 0.8:
                    recommendations.append("High CPU usage detected. Consider optimizing CPU-intensive operations.")
                
                if avg_memory > 0.8:
                    recommendations.append("High memory usage detected. Consider optimizing memory usage.")
                
                if avg_response_time > 2.0:
                    recommendations.append("High response time detected. Consider optimizing API endpoints.")
                
                if avg_optimization_score < 50:
                    recommendations.append("Low optimization score. Consider running system optimization.")
                
                analysis_results["recommendations"] = recommendations
        
        return {
            "success": True,
            "data": analysis_results,
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error analyzing performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Configuration endpoints
@router.get("/config", response_model=Dict[str, Any])
async def get_optimization_config():
    """Get current optimization configuration"""
    try:
        if not optimization_engine:
            raise HTTPException(status_code=503, detail="Optimization Engine not initialized")
        
        config = optimization_engine.config
        
        return {
            "success": True,
            "data": {
                "enable_memory_optimization": config.enable_memory_optimization,
                "enable_cpu_optimization": config.enable_cpu_optimization,
                "enable_gpu_optimization": config.enable_gpu_optimization,
                "enable_distributed_computing": config.enable_distributed_computing,
                "enable_async_optimization": config.enable_async_optimization,
                "enable_cache_optimization": config.enable_cache_optimization,
                "enable_database_optimization": config.enable_database_optimization,
                "enable_api_optimization": config.enable_api_optimization,
                "max_memory_usage": config.max_memory_usage,
                "max_cpu_usage": config.max_cpu_usage,
                "cache_size_limit": config.cache_size_limit,
                "cache_ttl": config.cache_ttl,
                "batch_size": config.batch_size,
                "max_concurrent_requests": config.max_concurrent_requests,
                "connection_pool_size": config.connection_pool_size,
                "query_timeout": config.query_timeout,
                "enable_profiling": config.enable_profiling,
                "enable_monitoring": config.enable_monitoring,
                "optimization_interval": config.optimization_interval
            },
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error getting optimization config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/config", response_model=Dict[str, Any])
async def update_optimization_config(request: OptimizationConfigRequest):
    """Update optimization configuration"""
    try:
        if not optimization_engine:
            raise HTTPException(status_code=503, detail="Optimization Engine not initialized")
        
        # Create new config
        new_config = OptimizationConfig(
            enable_memory_optimization=request.enable_memory_optimization,
            enable_cpu_optimization=request.enable_cpu_optimization,
            enable_gpu_optimization=request.enable_gpu_optimization,
            enable_distributed_computing=request.enable_distributed_computing,
            enable_async_optimization=request.enable_async_optimization,
            enable_cache_optimization=request.enable_cache_optimization,
            enable_database_optimization=request.enable_database_optimization,
            enable_api_optimization=request.enable_api_optimization,
            max_memory_usage=request.max_memory_usage,
            max_cpu_usage=request.max_cpu_usage,
            cache_size_limit=request.cache_size_limit,
            cache_ttl=request.cache_ttl,
            batch_size=request.batch_size,
            max_concurrent_requests=request.max_concurrent_requests,
            connection_pool_size=request.connection_pool_size,
            query_timeout=request.query_timeout,
            enable_profiling=request.enable_profiling,
            enable_monitoring=request.enable_monitoring,
            optimization_interval=request.optimization_interval
        )
        
        # Update engine config
        optimization_engine.config = new_config
        
        return {
            "success": True,
            "message": "Optimization configuration updated successfully",
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error updating optimization config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# History endpoints
@router.get("/history", response_model=Dict[str, Any])
async def get_optimization_history(limit: int = Query(50, description="Number of optimization records to return")):
    """Get optimization history"""
    try:
        if not optimization_engine:
            raise HTTPException(status_code=503, detail="Optimization Engine not initialized")
        
        history = await optimization_engine.get_optimization_history()
        
        # Limit results
        limited_history = history[-limit:] if len(history) > limit else history
        
        return {
            "success": True,
            "data": {
                "total_optimizations": len(history),
                "returned_optimizations": len(limited_history),
                "history": limited_history
            },
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error getting optimization history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Health check
@router.get("/health", response_model=Dict[str, Any])
async def optimization_health_check():
    """Optimization engine health check"""
    try:
        if not optimization_engine:
            return {
                "status": "unhealthy",
                "service": "Optimization Engine",
                "timestamp": datetime.now(),
                "error": "Optimization engine not initialized"
            }
        
        # Test basic functionality
        metrics = await optimization_engine.get_performance_metrics()
        
        # Get history
        history = await optimization_engine.get_optimization_history()
        
        return {
            "status": "healthy",
            "service": "Optimization Engine",
            "timestamp": datetime.now(),
            "performance_metrics": {
                "cpu_usage": metrics.cpu_usage,
                "memory_usage": metrics.memory_usage,
                "optimization_score": metrics.optimization_score
            },
            "optimization_history_count": len(history),
            "config": {
                "enable_memory_optimization": optimization_engine.config.enable_memory_optimization,
                "enable_cpu_optimization": optimization_engine.config.enable_cpu_optimization,
                "enable_cache_optimization": optimization_engine.config.enable_cache_optimization,
                "enable_monitoring": optimization_engine.config.enable_monitoring
            }
        }
    except Exception as e:
        logger.error(f"Optimization engine health check failed: {e}")
        return {
            "status": "unhealthy",
            "service": "Optimization Engine",
            "timestamp": datetime.now(),
            "error": str(e)
        }


# Capabilities
@router.get("/capabilities", response_model=Dict[str, Any])
async def get_optimization_capabilities():
    """Get optimization engine capabilities"""
    return {
        "success": True,
        "data": {
            "memory_optimization": {
                "garbage_collection": "Automatic garbage collection optimization",
                "weak_references": "Weak reference management and cleanup",
                "memory_pool": "Memory pool optimization and management",
                "memory_compression": "Memory compression and cleanup",
                "memory_monitoring": "Real-time memory usage monitoring"
            },
            "cpu_optimization": {
                "cpu_affinity": "CPU affinity optimization for better performance",
                "thread_pool": "Thread pool size optimization",
                "process_pool": "Process pool size optimization",
                "cpu_frequency": "CPU frequency optimization",
                "load_balancing": "CPU load balancing and distribution"
            },
            "cache_optimization": {
                "cache_size": "Cache size optimization and management",
                "cache_ttl": "Cache TTL optimization based on hit rates",
                "cache_eviction": "Intelligent cache eviction policies",
                "cache_compression": "Cache compression for memory efficiency",
                "cache_monitoring": "Cache performance monitoring and analytics"
            },
            "database_optimization": {
                "connection_pool": "Database connection pool optimization",
                "query_optimization": "Database query optimization and analysis",
                "index_optimization": "Database index optimization",
                "transaction_optimization": "Database transaction optimization",
                "query_monitoring": "Database query performance monitoring"
            },
            "api_optimization": {
                "response_caching": "API response caching optimization",
                "request_batching": "Request batching for efficiency",
                "response_compression": "Response compression optimization",
                "rate_limiting": "Intelligent rate limiting optimization",
                "api_monitoring": "API performance monitoring and analytics"
            },
            "async_optimization": {
                "concurrency": "Async concurrency optimization",
                "task_scheduling": "Task scheduling optimization",
                "event_loop": "Event loop optimization",
                "coroutine_optimization": "Coroutine performance optimization",
                "async_monitoring": "Async performance monitoring"
            },
            "performance_profiling": {
                "memory_profiling": "Memory usage profiling and analysis",
                "cpu_profiling": "CPU usage profiling and analysis",
                "performance_metrics": "Comprehensive performance metrics collection",
                "optimization_recommendations": "Intelligent optimization recommendations",
                "performance_monitoring": "Real-time performance monitoring"
            },
            "system_optimization": {
                "comprehensive_optimization": "Comprehensive system-wide optimization",
                "automatic_optimization": "Automatic optimization scheduling",
                "optimization_history": "Optimization history tracking and analysis",
                "performance_analysis": "Performance analysis and reporting",
                "system_monitoring": "System-wide performance monitoring"
            }
        },
        "timestamp": datetime.now()
    }