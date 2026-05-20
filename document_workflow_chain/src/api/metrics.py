"""
Metrics API - Advanced Implementation
====================================

Advanced metrics API with comprehensive system and application metrics collection.
"""

from __future__ import annotations
import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, status, Query
from pydantic import BaseModel
from datetime import datetime, timedelta

from ..services import metrics_service, MetricType, MetricCategory

# Create router
router = APIRouter()

logger = logging.getLogger(__name__)


# Request/Response models
class MetricRequest(BaseModel):
    """Metric request model"""
    name: str
    value: float
    metric_type: str
    category: str
    labels: Optional[Dict[str, str]] = None


class MetricResponse(BaseModel):
    """Metric response model"""
    name: str
    value: float
    type: str
    category: str
    labels: Dict[str, str]
    timestamp: str
    metadata: Dict[str, Any]


class MetricSummaryResponse(BaseModel):
    """Metric summary response model"""
    name: str
    type: str
    category: str
    count: int
    sum: float
    min: float
    max: float
    avg: float
    first_timestamp: str
    last_timestamp: str


class MetricsStatsResponse(BaseModel):
    """Metrics service statistics response model"""
    is_collecting: bool
    total_metrics_collected: int
    metrics_by_type: Dict[str, int]
    metrics_by_category: Dict[str, int]
    collection_errors: int
    last_collection: Optional[str]
    counters_count: int
    gauges_count: int
    histograms_count: int
    timers_count: int
    summaries_count: int
    collection_interval: int
    retention_period: int
    max_metrics_per_name: int


# Metric recording endpoints
@router.post("/metrics/counter")
async def record_counter_metric(request: MetricRequest):
    """Record counter metric"""
    try:
        category = MetricCategory(request.category)
        
        metrics_service.increment_counter(
            name=request.name,
            value=request.value,
            category=category,
            labels=request.labels
        )
        
        return {
            "message": "Counter metric recorded successfully",
            "metric_name": request.name,
            "value": request.value,
            "status": "success"
        }
    
    except Exception as e:
        logger.error(f"Failed to record counter metric: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to record counter metric: {str(e)}"
        )


@router.post("/metrics/gauge")
async def record_gauge_metric(request: MetricRequest):
    """Record gauge metric"""
    try:
        category = MetricCategory(request.category)
        
        metrics_service.set_gauge(
            name=request.name,
            value=request.value,
            category=category,
            labels=request.labels
        )
        
        return {
            "message": "Gauge metric recorded successfully",
            "metric_name": request.name,
            "value": request.value,
            "status": "success"
        }
    
    except Exception as e:
        logger.error(f"Failed to record gauge metric: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to record gauge metric: {str(e)}"
        )


@router.post("/metrics/histogram")
async def record_histogram_metric(request: MetricRequest):
    """Record histogram metric"""
    try:
        category = MetricCategory(request.category)
        
        metrics_service.record_histogram(
            name=request.name,
            value=request.value,
            category=category,
            labels=request.labels
        )
        
        return {
            "message": "Histogram metric recorded successfully",
            "metric_name": request.name,
            "value": request.value,
            "status": "success"
        }
    
    except Exception as e:
        logger.error(f"Failed to record histogram metric: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to record histogram metric: {str(e)}"
        )


@router.post("/metrics/timer")
async def record_timer_metric(request: MetricRequest):
    """Record timer metric"""
    try:
        category = MetricCategory(request.category)
        
        metrics_service.record_timer(
            name=request.name,
            value=request.value,
            category=category,
            labels=request.labels
        )
        
        return {
            "message": "Timer metric recorded successfully",
            "metric_name": request.name,
            "value": request.value,
            "status": "success"
        }
    
    except Exception as e:
        logger.error(f"Failed to record timer metric: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to record timer metric: {str(e)}"
        )


@router.post("/metrics/summary")
async def record_summary_metric(request: MetricRequest):
    """Record summary metric"""
    try:
        category = MetricCategory(request.category)
        
        metrics_service.record_summary(
            name=request.name,
            value=request.value,
            category=category,
            labels=request.labels
        )
        
        return {
            "message": "Summary metric recorded successfully",
            "metric_name": request.name,
            "value": request.value,
            "status": "success"
        }
    
    except Exception as e:
        logger.error(f"Failed to record summary metric: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to record summary metric: {str(e)}"
        )


# Metric retrieval endpoints
@router.get("/metrics/{metric_name}", response_model=List[MetricResponse])
async def get_metric(
    metric_name: str,
    metric_type: Optional[str] = Query(None, description="Filter by metric type"),
    category: Optional[str] = Query(None, description="Filter by category"),
    limit: int = Query(100, description="Maximum number of metrics to return")
):
    """Get metric data"""
    try:
        # Convert string parameters to enums
        type_enum = MetricType(metric_type) if metric_type else None
        category_enum = MetricCategory(category) if category else None
        
        metrics = metrics_service.get_metric(
            name=metric_name,
            metric_type=type_enum,
            category=category_enum,
            limit=limit
        )
        
        return [MetricResponse(**metric) for metric in metrics]
    
    except Exception as e:
        logger.error(f"Failed to get metric: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get metric: {str(e)}"
        )


@router.get("/metrics/{metric_name}/summary", response_model=MetricSummaryResponse)
async def get_metric_summary(
    metric_name: str,
    metric_type: Optional[str] = Query(None, description="Filter by metric type"),
    category: Optional[str] = Query(None, description="Filter by category"),
    time_range_hours: int = Query(24, description="Time range in hours")
):
    """Get metric summary"""
    try:
        # Convert string parameters to enums
        type_enum = MetricType(metric_type) if metric_type else None
        category_enum = MetricCategory(category) if category else None
        
        # Calculate time range
        time_range = timedelta(hours=time_range_hours)
        
        summary = metrics_service.get_metric_summary(
            name=metric_name,
            metric_type=type_enum,
            category=category_enum,
            time_range=time_range
        )
        
        if not summary:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No metrics found for the specified criteria"
            )
        
        return MetricSummaryResponse(**summary)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get metric summary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get metric summary: {str(e)}"
        )


@router.get("/metrics", response_model=Dict[str, List[MetricResponse]])
async def get_all_metrics(
    metric_type: Optional[str] = Query(None, description="Filter by metric type"),
    category: Optional[str] = Query(None, description="Filter by category"),
    time_range_hours: int = Query(24, description="Time range in hours"),
    limit: int = Query(1000, description="Maximum number of metrics per name")
):
    """Get all metrics with filtering"""
    try:
        # Convert string parameters to enums
        type_enum = MetricType(metric_type) if metric_type else None
        category_enum = MetricCategory(category) if category else None
        
        # Calculate time range
        time_range = timedelta(hours=time_range_hours)
        
        all_metrics = metrics_service.get_all_metrics(
            metric_type=type_enum,
            category=category_enum,
            time_range=time_range,
            limit=limit
        )
        
        # Convert to response format
        result = {}
        for metric_name, metrics in all_metrics.items():
            result[metric_name] = [MetricResponse(**metric) for metric in metrics]
        
        return result
    
    except Exception as e:
        logger.error(f"Failed to get all metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get all metrics: {str(e)}"
        )


# System metrics endpoints
@router.get("/metrics/system/overview")
async def get_system_metrics_overview():
    """Get system metrics overview"""
    try:
        # Get system metrics
        cpu_metrics = metrics_service.get_metric("system.cpu.percent", limit=1)
        memory_metrics = metrics_service.get_metric("system.memory.percent", limit=1)
        disk_metrics = metrics_service.get_metric("system.disk.percent", limit=1)
        
        overview = {
            "cpu": {
                "percent": cpu_metrics[0]["value"] if cpu_metrics else 0,
                "timestamp": cpu_metrics[0]["timestamp"] if cpu_metrics else None
            },
            "memory": {
                "percent": memory_metrics[0]["value"] if memory_metrics else 0,
                "timestamp": memory_metrics[0]["timestamp"] if memory_metrics else None
            },
            "disk": {
                "percent": disk_metrics[0]["value"] if disk_metrics else 0,
                "timestamp": disk_metrics[0]["timestamp"] if disk_metrics else None
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return overview
    
    except Exception as e:
        logger.error(f"Failed to get system metrics overview: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get system metrics overview: {str(e)}"
        )


@router.get("/metrics/application/overview")
async def get_application_metrics_overview():
    """Get application metrics overview"""
    try:
        # Get application metrics
        uptime_metrics = metrics_service.get_metric("application.uptime", limit=1)
        thread_metrics = metrics_service.get_metric("application.threads.active", limit=1)
        
        overview = {
            "uptime": {
                "seconds": uptime_metrics[0]["value"] if uptime_metrics else 0,
                "timestamp": uptime_metrics[0]["timestamp"] if uptime_metrics else None
            },
            "threads": {
                "active": thread_metrics[0]["value"] if thread_metrics else 0,
                "timestamp": thread_metrics[0]["timestamp"] if thread_metrics else None
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return overview
    
    except Exception as e:
        logger.error(f"Failed to get application metrics overview: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get application metrics overview: {str(e)}"
        )


# Service management endpoints
@router.post("/service/start")
async def start_metrics_service():
    """Start metrics service"""
    try:
        await metrics_service.start()
        return {
            "message": "Metrics service started successfully",
            "status": "success"
        }
    
    except Exception as e:
        logger.error(f"Failed to start metrics service: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start metrics service: {str(e)}"
        )


@router.post("/service/stop")
async def stop_metrics_service():
    """Stop metrics service"""
    try:
        await metrics_service.stop()
        return {
            "message": "Metrics service stopped successfully",
            "status": "success"
        }
    
    except Exception as e:
        logger.error(f"Failed to stop metrics service: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stop metrics service: {str(e)}"
        )


@router.get("/service/stats", response_model=MetricsStatsResponse)
async def get_metrics_stats():
    """Get metrics service statistics"""
    try:
        stats = metrics_service.get_metrics_stats()
        return MetricsStatsResponse(**stats)
    
    except Exception as e:
        logger.error(f"Failed to get metrics stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get metrics stats: {str(e)}"
        )


# Health check endpoint
@router.get("/health")
async def metrics_health():
    """Metrics service health check"""
    try:
        stats = metrics_service.get_metrics_stats()
        
        return {
            "service": "metrics_service",
            "status": "healthy" if stats["is_collecting"] else "stopped",
            "is_collecting": stats["is_collecting"],
            "total_metrics": stats["total_metrics_collected"],
            "collection_errors": stats["collection_errors"],
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Metrics service health check failed: {e}")
        return {
            "service": "metrics_service",
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

