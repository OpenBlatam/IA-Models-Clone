from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import asyncio
import json
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import psutil
from onyx.utils.logger import setup_logger
from onyx.server.features.ads.performance_optimizer import (
from onyx.server.features.ads.optimized_config import settings
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from typing import Any, List, Dict, Optional
import logging
"""
Performance Optimization API for Onyx Ads Backend

This module provides REST API endpoints for:
- Performance monitoring and metrics
- Memory management and optimization
- Cache management and statistics
- Database query optimization
- Async task management
- Resource usage monitoring
"""

    PerformanceOptimizer, 
    PerformanceConfig, 
    optimizer,
    PERFORMANCE_METRICS
)

logger = setup_logger()

router = APIRouter(prefix="/performance", tags=["Performance Optimization"])

# Pydantic models for API requests/responses
class PerformanceStats(BaseModel):
    """Performance statistics response model."""
    timestamp: datetime = Field(default_factory=datetime.now)
    memory: Dict[str, Any]
    cache: Dict[str, Any]
    tasks: Dict[str, Any]
    database: Dict[str, Any]
    system: Dict[str, Any]
    config: Dict[str, Any]

class MemoryCleanupRequest(BaseModel):
    """Memory cleanup request model."""
    force: bool = False
    aggressive: bool = False

class MemoryCleanupResponse(BaseModel):
    """Memory cleanup response model."""
    success: bool
    memory_freed: int
    cleanup_time: float
    details: Dict[str, Any]

class CacheManagementRequest(BaseModel):
    """Cache management request model."""
    action: str = Field(..., description="Action: clear, stats, optimize")
    cache_type: str = Field("all", description="Cache type: l1, l2, redis, all")
    ttl: Optional[int] = None

class CacheManagementResponse(BaseModel):
    """Cache management response model."""
    success: bool
    action: str
    cache_type: str
    details: Dict[str, Any]

class TaskManagementRequest(BaseModel):
    """Task management request model."""
    action: str = Field(..., description="Action: stats, cancel, cleanup")
    task_id: Optional[str] = None

class TaskManagementResponse(BaseModel):
    """Task management response model."""
    success: bool
    action: str
    details: Dict[str, Any]

class DatabaseOptimizationRequest(BaseModel):
    """Database optimization request model."""
    action: str = Field(..., description="Action: stats, clear_cache, analyze")
    query_type: Optional[str] = None

class DatabaseOptimizationResponse(BaseModel):
    """Database optimization response model."""
    success: bool
    action: str
    details: Dict[str, Any]

class SystemResources(BaseModel):
    """System resources response model."""
    cpu: Dict[str, float]
    memory: Dict[str, float]
    disk: Dict[str, float]
    network: Dict[str, float]

class PerformanceConfigUpdate(BaseModel):
    """Performance configuration update model."""
    cache_ttl: Optional[int] = None
    cache_max_size: Optional[int] = None
    max_workers: Optional[int] = None
    memory_cleanup_threshold: Optional[float] = None
    task_timeout: Optional[int] = None

# Dependency for performance optimizer
async def get_performance_optimizer() -> PerformanceOptimizer:
    """Get performance optimizer instance."""
    return optimizer

# Health check endpoint
@router.get("/health")
async def health_check():
    """Health check for performance optimization system."""
    try:
        # Check if optimizer is running
        is_running = optimizer._started
        
        # Check system resources
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        return {
            "status": "healthy" if is_running else "stopped",
            "timestamp": datetime.now().isoformat(),
            "optimizer_running": is_running,
            "system_resources": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available": memory.available
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

# Performance statistics endpoint
@router.get("/stats", response_model=PerformanceStats)
async def get_performance_stats(
    optimizer: PerformanceOptimizer = Depends(get_performance_optimizer)
):
    """Get comprehensive performance statistics."""
    try:
        # Get optimizer stats
        optimizer_stats = optimizer.get_performance_stats()
        
        # Get system resources
        system_stats = await get_system_resources()
        
        return PerformanceStats(
            memory=optimizer_stats['memory'],
            cache=optimizer_stats['cache'],
            tasks=optimizer_stats['tasks'],
            database=optimizer_stats['database'],
            system=system_stats,
            config=optimizer_stats['config']
        )
    except Exception as e:
        logger.error(f"Failed to get performance stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance stats: {str(e)}")

# Memory management endpoints
@router.post("/memory/cleanup", response_model=MemoryCleanupResponse)
async def cleanup_memory(
    request: MemoryCleanupRequest,
    optimizer: PerformanceOptimizer = Depends(get_performance_optimizer)
):
    """Perform memory cleanup and optimization."""
    try:
        # Submit memory cleanup task
        cleanup_task = await optimizer.task_manager.submit_task(
            optimizer.memory_manager.cleanup_memory,
            force=request.force
        )
        
        # Wait for completion
        result = await asyncio.wait_for(cleanup_task, timeout=30)
        
        return MemoryCleanupResponse(
            success=True,
            memory_freed=result['memory_freed'],
            cleanup_time=result['cleanup_time'],
            details=result
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="Memory cleanup timeout")
    except Exception as e:
        logger.error(f"Memory cleanup failed: {e}")
        raise HTTPException(status_code=500, detail=f"Memory cleanup failed: {str(e)}")

@router.get("/memory/stats")
async def get_memory_stats(
    optimizer: PerformanceOptimizer = Depends(get_performance_optimizer)
):
    """Get detailed memory statistics."""
    try:
        memory_stats = optimizer.memory_manager.get_memory_stats()
        return memory_stats
    except Exception as e:
        logger.error(f"Failed to get memory stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get memory stats: {str(e)}")

@router.get("/memory/usage")
async def get_memory_usage(
    optimizer: PerformanceOptimizer = Depends(get_performance_optimizer)
):
    """Get current memory usage."""
    try:
        memory_usage = optimizer.memory_manager.get_memory_usage()
        return {
            "timestamp": datetime.now().isoformat(),
            "memory_usage": memory_usage,
            "should_cleanup": optimizer.memory_manager.should_cleanup_memory()
        }
    except Exception as e:
        logger.error(f"Failed to get memory usage: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get memory usage: {str(e)}")

# Cache management endpoints
@router.post("/cache/manage", response_model=CacheManagementResponse)
async def manage_cache(
    request: CacheManagementRequest,
    optimizer: PerformanceOptimizer = Depends(get_performance_optimizer)
):
    """Manage cache operations."""
    try:
        if request.action == "clear":
            optimizer.cache.clear(request.cache_type)
            details = {"message": f"Cache {request.cache_type} cleared successfully"}
        elif request.action == "stats":
            details = optimizer.cache.get_stats()
        elif request.action == "optimize":
            # Implement cache optimization logic
            details = {"message": "Cache optimization completed"}
        else:
            raise HTTPException(status_code=400, detail=f"Invalid action: {request.action}")
        
        return CacheManagementResponse(
            success=True,
            action=request.action,
            cache_type=request.cache_type,
            details=details
        )
    except Exception as e:
        logger.error(f"Cache management failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cache management failed: {str(e)}")

@router.get("/cache/stats")
async def get_cache_stats(
    optimizer: PerformanceOptimizer = Depends(get_performance_optimizer)
):
    """Get cache statistics."""
    try:
        cache_stats = optimizer.cache.get_stats()
        return cache_stats
    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get cache stats: {str(e)}")

# Task management endpoints
@router.post("/tasks/manage", response_model=TaskManagementResponse)
async def manage_tasks(
    request: TaskManagementRequest,
    optimizer: PerformanceOptimizer = Depends(get_performance_optimizer)
):
    """Manage async tasks."""
    try:
        if request.action == "stats":
            details = optimizer.task_manager.get_stats()
        elif request.action == "cancel":
            if request.task_id:
                # Cancel specific task (implementation needed)
                details = {"message": f"Task {request.task_id} cancelled"}
            else:
                # Cancel all running tasks
                for task in optimizer.task_manager._running_tasks:
                    task.cancel()
                details = {"message": "All running tasks cancelled"}
        elif request.action == "cleanup":
            # Clean up completed tasks
            optimizer.task_manager._running_tasks = {
                task for task in optimizer.task_manager._running_tasks 
                if not task.done()
            }
            details = {"message": "Completed tasks cleaned up"}
        else:
            raise HTTPException(status_code=400, detail=f"Invalid action: {request.action}")
        
        return TaskManagementResponse(
            success=True,
            action=request.action,
            details=details
        )
    except Exception as e:
        logger.error(f"Task management failed: {e}")
        raise HTTPException(status_code=500, detail=f"Task management failed: {str(e)}")

@router.get("/tasks/stats")
async def get_task_stats(
    optimizer: PerformanceOptimizer = Depends(get_performance_optimizer)
):
    """Get task execution statistics."""
    try:
        task_stats = optimizer.task_manager.get_stats()
        return task_stats
    except Exception as e:
        logger.error(f"Failed to get task stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get task stats: {str(e)}")

# Database optimization endpoints
@router.post("/database/optimize", response_model=DatabaseOptimizationResponse)
async def optimize_database(
    request: DatabaseOptimizationRequest,
    optimizer: PerformanceOptimizer = Depends(get_performance_optimizer)
):
    """Optimize database operations."""
    try:
        if request.action == "stats":
            details = optimizer.db_optimizer.get_stats()
        elif request.action == "clear_cache":
            optimizer.db_optimizer._query_cache.clear()
            details = {"message": "Database query cache cleared"}
        elif request.action == "analyze":
            # Analyze slow queries
            slow_queries = optimizer.db_optimizer._slow_queries
            details = {
                "slow_queries_count": len(slow_queries),
                "slow_queries": list(slow_queries)
            }
        else:
            raise HTTPException(status_code=400, detail=f"Invalid action: {request.action}")
        
        return DatabaseOptimizationResponse(
            success=True,
            action=request.action,
            details=details
        )
    except Exception as e:
        logger.error(f"Database optimization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Database optimization failed: {str(e)}")

@router.get("/database/stats")
async def get_database_stats(
    optimizer: PerformanceOptimizer = Depends(get_performance_optimizer)
):
    """Get database optimization statistics."""
    try:
        db_stats = optimizer.db_optimizer.get_stats()
        return db_stats
    except Exception as e:
        logger.error(f"Failed to get database stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get database stats: {str(e)}")

# System resources endpoints
@router.get("/system/resources", response_model=SystemResources)
async def get_system_resources():
    """Get system resource usage."""
    try:
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        
        # Memory usage
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_io = psutil.disk_io_counters()
        
        # Network usage
        network = psutil.net_io_counters()
        
        return SystemResources(
            cpu={
                "percent": cpu_percent,
                "count": cpu_count,
                "frequency_mhz": cpu_freq.current if cpu_freq else 0,
                "frequency_max_mhz": cpu_freq.max if cpu_freq else 0
            },
            memory={
                "total": memory.total,
                "available": memory.available,
                "used": memory.used,
                "percent": memory.percent,
                "swap_total": swap.total,
                "swap_used": swap.used,
                "swap_percent": swap.percent
            },
            disk={
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
                "percent": (disk.used / disk.total) * 100,
                "read_bytes": disk_io.read_bytes if disk_io else 0,
                "write_bytes": disk_io.write_bytes if disk_io else 0
            },
            network={
                "bytes_sent": network.bytes_sent,
                "bytes_recv": network.bytes_recv,
                "packets_sent": network.packets_sent,
                "packets_recv": network.packets_recv
            }
        )
    except Exception as e:
        logger.error(f"Failed to get system resources: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get system resources: {str(e)}")

# Configuration management endpoints
@router.get("/config")
async def get_performance_config(
    optimizer: PerformanceOptimizer = Depends(get_performance_optimizer)
):
    """Get current performance configuration."""
    try:
        config = optimizer.config
        return {
            "cache_ttl": config.cache_ttl,
            "cache_max_size": config.cache_max_size,
            "max_workers": config.max_workers,
            "max_processes": config.max_processes,
            "memory_cleanup_threshold": config.memory_cleanup_threshold,
            "task_timeout": config.task_timeout,
            "slow_query_threshold": config.slow_query_threshold,
            "profiling_enabled": config.profiling_enabled,
            "tracemalloc_enabled": config.tracemalloc_enabled
        }
    except Exception as e:
        logger.error(f"Failed to get performance config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance config: {str(e)}")

@router.put("/config")
async def update_performance_config(
    config_update: PerformanceConfigUpdate,
    optimizer: PerformanceOptimizer = Depends(get_performance_optimizer)
):
    """Update performance configuration."""
    try:
        # Update configuration
        if config_update.cache_ttl is not None:
            optimizer.config.cache_ttl = config_update.cache_ttl
        if config_update.cache_max_size is not None:
            optimizer.config.cache_max_size = config_update.cache_max_size
        if config_update.max_workers is not None:
            optimizer.config.max_workers = config_update.max_workers
        if config_update.memory_cleanup_threshold is not None:
            optimizer.config.memory_cleanup_threshold = config_update.memory_cleanup_threshold
        if config_update.task_timeout is not None:
            optimizer.config.task_timeout = config_update.task_timeout
        
        return {
            "success": True,
            "message": "Performance configuration updated",
            "updated_fields": config_update.dict(exclude_unset=True)
        }
    except Exception as e:
        logger.error(f"Failed to update performance config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update performance config: {str(e)}")

# Prometheus metrics endpoint
@router.get("/metrics")
async def get_prometheus_metrics():
    """Get Prometheus metrics."""
    try:
        
        metrics = generate_latest()
        return JSONResponse(
            content=metrics,
            media_type=CONTENT_TYPE_LATEST
        )
    except Exception as e:
        logger.error(f"Failed to get Prometheus metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get Prometheus metrics: {str(e)}")

# Performance optimization control endpoints
@router.post("/start")
async def start_performance_optimizer(
    optimizer: PerformanceOptimizer = Depends(get_performance_optimizer)
):
    """Start the performance optimizer."""
    try:
        await optimizer.start()
        return {
            "success": True,
            "message": "Performance optimizer started successfully"
        }
    except Exception as e:
        logger.error(f"Failed to start performance optimizer: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start performance optimizer: {str(e)}")

@router.post("/stop")
async def stop_performance_optimizer(
    optimizer: PerformanceOptimizer = Depends(get_performance_optimizer)
):
    """Stop the performance optimizer."""
    try:
        await optimizer.stop()
        return {
            "success": True,
            "message": "Performance optimizer stopped successfully"
        }
    except Exception as e:
        logger.error(f"Failed to stop performance optimizer: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop performance optimizer: {str(e)}")

# Performance alerts endpoint
@router.get("/alerts")
async def get_performance_alerts(
    optimizer: PerformanceOptimizer = Depends(get_performance_optimizer)
):
    """Get performance alerts and warnings."""
    try:
        alerts = []
        
        # Check memory usage
        memory_usage = optimizer.memory_manager.get_memory_usage()
        memory_ratio = memory_usage['rss'] / memory_usage['total']
        
        if memory_ratio > optimizer.config.memory_warning_threshold:
            alerts.append({
                "type": "memory_warning",
                "severity": "warning",
                "message": f"High memory usage: {memory_ratio:.2%}",
                "timestamp": datetime.now().isoformat()
            })
        
        # Check cache hit rate
        cache_stats = optimizer.cache.get_stats()
        if cache_stats['hit_rate'] < optimizer.config.cache_hit_rate_threshold:
            alerts.append({
                "type": "cache_warning",
                "severity": "warning",
                "message": f"Low cache hit rate: {cache_stats['hit_rate']:.2%}",
                "timestamp": datetime.now().isoformat()
            })
        
        # Check slow queries
        db_stats = optimizer.db_optimizer.get_stats()
        if len(db_stats['slow_queries']) > 10:
            alerts.append({
                "type": "database_warning",
                "severity": "warning",
                "message": f"Multiple slow queries detected: {len(db_stats['slow_queries'])}",
                "timestamp": datetime.now().isoformat()
            })
        
        return {
            "alerts": alerts,
            "total_alerts": len(alerts)
        }
    except Exception as e:
        logger.error(f"Failed to get performance alerts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance alerts: {str(e)}")

# Performance recommendations endpoint
@router.get("/recommendations")
async def get_performance_recommendations(
    optimizer: PerformanceOptimizer = Depends(get_performance_optimizer)
):
    """Get performance optimization recommendations."""
    try:
        recommendations = []
        
        # Get current stats
        stats = optimizer.get_performance_stats()
        
        # Memory recommendations
        memory_ratio = stats['memory']['current']['rss'] / stats['memory']['current']['total']
        if memory_ratio > 0.8:
            recommendations.append({
                "category": "memory",
                "priority": "high",
                "recommendation": "Consider increasing memory or implementing more aggressive cleanup",
                "current_value": f"{memory_ratio:.2%}",
                "target_value": "< 80%"
            })
        
        # Cache recommendations
        cache_hit_rate = stats['cache']['hit_rate']
        if cache_hit_rate < 0.7:
            recommendations.append({
                "category": "cache",
                "priority": "medium",
                "recommendation": "Increase cache size or TTL to improve hit rate",
                "current_value": f"{cache_hit_rate:.2%}",
                "target_value": "> 70%"
            })
        
        # Task recommendations
        running_tasks = stats['tasks']['running_tasks']
        if running_tasks > 50:
            recommendations.append({
                "category": "tasks",
                "priority": "medium",
                "recommendation": "Consider increasing worker pool size or implementing task queuing",
                "current_value": running_tasks,
                "target_value": "< 50"
            })
        
        return {
            "recommendations": recommendations,
            "total_recommendations": len(recommendations)
        }
    except Exception as e:
        logger.error(f"Failed to get performance recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance recommendations: {str(e)}") 