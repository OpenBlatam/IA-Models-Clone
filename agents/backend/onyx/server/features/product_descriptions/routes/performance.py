from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
import logging
from datetime import datetime, timedelta
from ..dependencies.auth import get_authenticated_user, require_permission
from ..routes.base import get_request_context, log_route_access
from ..schemas.base import BaseResponse, ErrorResponse
from ..pydantic_schemas import (
from ..performance_metrics import PerformanceMonitor
from ..caching_manager import CachingManager
from ..async_database_api_operations import AsyncDatabaseManager
from typing import Any, List, Dict, Optional
import asyncio
"""
Performance Router

This module contains routes for performance monitoring, metrics collection,
and optimization endpoints. Provides real-time insights into system performance.
"""


# Import dependencies

# Import schemas
    PerformanceMetricsResponse,
    PerformanceAlertResponse,
    OptimizationRequest,
    OptimizationResponse,
    CacheStatsResponse,
    DatabaseStatsResponse
)

# Import services

# Initialize router
router = APIRouter(prefix="/performance", tags=["performance"])

# Logger
logger = logging.getLogger(__name__)

# Route dependencies
async def get_performance_monitor(
    context: Dict[str, Any] = Depends(get_request_context)
) -> PerformanceMonitor:
    """Get performance monitor from context."""
    return context["performance_monitor"]

async def get_cache_manager(
    context: Dict[str, Any] = Depends(get_request_context)
) -> CachingManager:
    """Get cache manager from context."""
    return context["cache_manager"]

async def get_db_manager(
    context: Dict[str, Any] = Depends(get_request_context)
) -> AsyncDatabaseManager:
    """Get database manager from context."""
    return context["async_io_manager"].db_manager

# Performance Metrics Routes
@router.get("/metrics/current", response_model=PerformanceMetricsResponse)
async def get_current_metrics(
    context: Dict[str, Any] = Depends(get_request_context),
    monitor: PerformanceMonitor = Depends(get_performance_monitor)
):
    """Get current performance metrics."""
    try:
        log_route_access("get_current_metrics")
        
        metrics = await monitor.get_current_metrics()
        
        return PerformanceMetricsResponse(
            status="success",
            message="Current performance metrics retrieved",
            data=metrics
        )
        
    except Exception as e:
        logger.error(f"Error getting current metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve current metrics"
        )

@router.get("/metrics/historical")
async def get_historical_metrics(
    start_time: datetime,
    end_time: datetime,
    interval: str = "1h",
    context: Dict[str, Any] = Depends(get_request_context),
    monitor: PerformanceMonitor = Depends(get_performance_monitor)
):
    """Get historical performance metrics."""
    try:
        log_route_access(
            "get_historical_metrics",
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            interval=interval
        )
        
        # Validate time range
        if end_time <= start_time:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="End time must be after start time"
            )
        
        if end_time - start_time > timedelta(days=30):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Time range cannot exceed 30 days"
            )
        
        metrics = await monitor.get_historical_metrics(
            start_time=start_time,
            end_time=end_time,
            interval=interval
        )
        
        return {
            "status": "success",
            "message": "Historical metrics retrieved successfully",
            "data": metrics
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting historical metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve historical metrics"
        )

@router.get("/metrics/endpoint/{endpoint_name}")
async def get_endpoint_metrics(
    endpoint_name: str,
    time_window: str = "1h",
    context: Dict[str, Any] = Depends(get_request_context),
    monitor: PerformanceMonitor = Depends(get_performance_monitor)
):
    """Get metrics for a specific endpoint."""
    try:
        log_route_access("get_endpoint_metrics", endpoint_name=endpoint_name)
        
        metrics = await monitor.get_endpoint_metrics(
            endpoint_name=endpoint_name,
            time_window=time_window
        )
        
        return {
            "status": "success",
            "message": f"Endpoint metrics for {endpoint_name} retrieved",
            "data": metrics
        }
        
    except Exception as e:
        logger.error(f"Error getting endpoint metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve endpoint metrics"
        )

# Performance Alerts
@router.get("/alerts", response_model=List[PerformanceAlertResponse])
async def get_performance_alerts(
    severity: Optional[str] = None,
    active_only: bool = True,
    context: Dict[str, Any] = Depends(get_request_context),
    monitor: PerformanceMonitor = Depends(get_performance_monitor)
):
    """Get performance alerts."""
    try:
        log_route_access("get_performance_alerts")
        
        alerts = await monitor.get_alerts(
            severity=severity,
            active_only=active_only
        )
        
        return [
            PerformanceAlertResponse(
                status="success",
                message="Performance alerts retrieved",
                data=alert
            ) for alert in alerts
        ]
        
    except Exception as e:
        logger.error(f"Error getting performance alerts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve performance alerts"
        )

@router.post("/alerts/{alert_id}/acknowledge", response_model=BaseResponse)
async def acknowledge_alert(
    alert_id: str,
    context: Dict[str, Any] = Depends(get_request_context),
    monitor: PerformanceMonitor = Depends(get_performance_monitor)
):
    """Acknowledge a performance alert."""
    try:
        log_route_access("acknowledge_alert", alert_id=alert_id)
        
        await monitor.acknowledge_alert(
            alert_id=alert_id,
            user_id=context["user"].id if context["user"] else None
        )
        
        return BaseResponse(
            status="success",
            message="Alert acknowledged successfully"
        )
        
    except Exception as e:
        logger.error(f"Error acknowledging alert: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to acknowledge alert"
        )

# Cache Performance
@router.get("/cache/stats", response_model=CacheStatsResponse)
async def get_cache_stats(
    context: Dict[str, Any] = Depends(get_request_context),
    cache_manager: CachingManager = Depends(get_cache_manager)
):
    """Get cache performance statistics."""
    try:
        log_route_access("get_cache_stats")
        
        stats = await cache_manager.get_stats()
        
        return CacheStatsResponse(
            status="success",
            message="Cache statistics retrieved",
            data=stats
        )
        
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve cache statistics"
        )

@router.post("/cache/optimize", response_model=BaseResponse)
async def optimize_cache(
    context: Dict[str, Any] = Depends(get_request_context),
    cache_manager: CachingManager = Depends(get_cache_manager)
):
    """Optimize cache performance."""
    try:
        log_route_access("optimize_cache")
        
        # Check admin permissions
        if context["user"] and not context["user"].is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin privileges required"
            )
        
        result = await cache_manager.optimize()
        
        return BaseResponse(
            status="success",
            message=f"Cache optimization completed: {result.items_evicted} items evicted"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error optimizing cache: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to optimize cache"
        )

# Database Performance
@router.get("/database/stats", response_model=DatabaseStatsResponse)
async def get_database_stats(
    context: Dict[str, Any] = Depends(get_request_context),
    db_manager: AsyncDatabaseManager = Depends(get_db_manager)
):
    """Get database performance statistics."""
    try:
        log_route_access("get_database_stats")
        
        stats = await db_manager.get_performance_stats()
        
        return DatabaseStatsResponse(
            status="success",
            message="Database statistics retrieved",
            data=stats
        )
        
    except Exception as e:
        logger.error(f"Error getting database stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve database statistics"
        )

@router.post("/database/optimize", response_model=BaseResponse)
async def optimize_database(
    context: Dict[str, Any] = Depends(get_request_context),
    db_manager: AsyncDatabaseManager = Depends(get_db_manager)
):
    """Optimize database performance."""
    try:
        log_route_access("optimize_database")
        
        # Check admin permissions
        if context["user"] and not context["user"].is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin privileges required"
            )
        
        result = await db_manager.optimize_performance()
        
        return BaseResponse(
            status="success",
            message=f"Database optimization completed: {result.queries_optimized} queries optimized"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error optimizing database: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to optimize database"
        )

# System Optimization
@router.post("/optimize", response_model=OptimizationResponse)
async def optimize_system(
    request: OptimizationRequest,
    context: Dict[str, Any] = Depends(get_request_context),
    monitor: PerformanceMonitor = Depends(get_performance_monitor)
):
    """Optimize system performance based on current metrics."""
    try:
        log_route_access("optimize_system")
        
        # Check admin permissions
        if context["user"] and not context["user"].is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin privileges required"
            )
        
        optimization_result = await monitor.optimize_system(
            target_metrics=request.target_metrics,
            optimization_level=request.optimization_level
        )
        
        return OptimizationResponse(
            status="success",
            message="System optimization completed",
            data=optimization_result
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error optimizing system: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to optimize system"
        )

# Performance Monitoring Configuration
@router.get("/config")
async def get_performance_config(
    context: Dict[str, Any] = Depends(get_request_context),
    monitor: PerformanceMonitor = Depends(get_performance_monitor)
):
    """Get performance monitoring configuration."""
    try:
        log_route_access("get_performance_config")
        
        config = await monitor.get_config()
        
        return {
            "status": "success",
            "message": "Performance configuration retrieved",
            "data": config
        }
        
    except Exception as e:
        logger.error(f"Error getting performance config: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve performance configuration"
        )

@router.put("/config")
async def update_performance_config(
    config: Dict[str, Any],
    context: Dict[str, Any] = Depends(get_request_context),
    monitor: PerformanceMonitor = Depends(get_performance_monitor)
):
    """Update performance monitoring configuration."""
    try:
        log_route_access("update_performance_config")
        
        # Check admin permissions
        if context["user"] and not context["user"].is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin privileges required"
            )
        
        updated_config = await monitor.update_config(config)
        
        return {
            "status": "success",
            "message": "Performance configuration updated",
            "data": updated_config
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating performance config: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update performance configuration"
        )

# Performance Health Check
@router.get("/health")
async def get_performance_health(
    context: Dict[str, Any] = Depends(get_request_context),
    monitor: PerformanceMonitor = Depends(get_performance_monitor)
):
    """Get performance health status."""
    try:
        log_route_access("get_performance_health")
        
        health = await monitor.get_health_status()
        
        return {
            "status": "success",
            "message": "Performance health status retrieved",
            "data": health
        }
        
    except Exception as e:
        logger.error(f"Error getting performance health: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve performance health status"
        ) 