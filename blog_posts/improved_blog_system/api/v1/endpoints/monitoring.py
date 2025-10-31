"""
Advanced Monitoring API endpoints
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel, Field

from ....services.advanced_monitoring_service import AdvancedMonitoringService
from ....api.dependencies import CurrentUserDep, DatabaseSessionDep
from ....core.exceptions import DatabaseError, ValidationError

router = APIRouter()


class ThresholdUpdateRequest(BaseModel):
    """Request model for threshold update."""
    metric_name: str = Field(..., description="Name of the metric")
    warning_threshold: float = Field(..., description="Warning threshold value")
    critical_threshold: float = Field(..., description="Critical threshold value")


async def get_monitoring_service(session: DatabaseSessionDep) -> AdvancedMonitoringService:
    """Get monitoring service instance."""
    return AdvancedMonitoringService(session)


@router.get("/metrics", response_model=Dict[str, Any])
async def collect_system_metrics(
    monitoring_service: AdvancedMonitoringService = Depends(get_monitoring_service),
    current_user: CurrentUserDep = Depends()
):
    """Collect comprehensive system metrics."""
    try:
        metrics = await monitoring_service.collect_system_metrics()
        
        return {
            "success": True,
            "data": metrics,
            "message": "System metrics collected successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to collect system metrics"
        )


@router.get("/health", response_model=Dict[str, Any])
async def get_system_health(
    monitoring_service: AdvancedMonitoringService = Depends(get_monitoring_service),
    current_user: CurrentUserDep = Depends()
):
    """Get overall system health status."""
    try:
        health = await monitoring_service.get_system_health()
        
        return {
            "success": True,
            "data": health,
            "message": "System health status retrieved successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get system health"
        )


@router.get("/metrics/history", response_model=Dict[str, Any])
async def get_metrics_history(
    hours: int = Query(default=24, ge=1, le=168, description="Number of hours to retrieve"),
    metric_name: Optional[str] = Query(None, description="Specific metric to retrieve"),
    monitoring_service: AdvancedMonitoringService = Depends(get_monitoring_service),
    current_user: CurrentUserDep = Depends()
):
    """Get metrics history for specified time period."""
    try:
        history = await monitoring_service.get_metrics_history(
            hours=hours,
            metric_name=metric_name
        )
        
        return {
            "success": True,
            "data": history,
            "message": "Metrics history retrieved successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get metrics history"
        )


@router.get("/alerts", response_model=Dict[str, Any])
async def get_active_alerts(
    severity: Optional[str] = Query(None, description="Filter by severity (warning, critical)"),
    limit: int = Query(default=50, ge=1, le=200, description="Number of alerts to return"),
    monitoring_service: AdvancedMonitoringService = Depends(get_monitoring_service),
    current_user: CurrentUserDep = Depends()
):
    """Get active alerts."""
    try:
        alerts = await monitoring_service.get_active_alerts(
            severity=severity,
            limit=limit
        )
        
        return {
            "success": True,
            "data": alerts,
            "message": "Active alerts retrieved successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get active alerts"
        )


@router.post("/alerts/{alert_id}/resolve", response_model=Dict[str, Any])
async def resolve_alert(
    alert_id: int,
    monitoring_service: AdvancedMonitoringService = Depends(get_monitoring_service),
    current_user: CurrentUserDep = Depends()
):
    """Resolve an alert."""
    try:
        result = await monitoring_service.resolve_alert(alert_id)
        
        return {
            "success": True,
            "data": result,
            "message": "Alert resolved successfully"
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to resolve alert"
        )


@router.put("/thresholds", response_model=Dict[str, Any])
async def update_threshold(
    request: ThresholdUpdateRequest = Depends(),
    monitoring_service: AdvancedMonitoringService = Depends(get_monitoring_service),
    current_user: CurrentUserDep = Depends()
):
    """Update metric threshold."""
    try:
        result = await monitoring_service.update_threshold(
            metric_name=request.metric_name,
            warning_threshold=request.warning_threshold,
            critical_threshold=request.critical_threshold
        )
        
        return {
            "success": True,
            "data": result,
            "message": "Threshold updated successfully"
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update threshold"
        )


@router.get("/stats", response_model=Dict[str, Any])
async def get_monitoring_stats(
    monitoring_service: AdvancedMonitoringService = Depends(get_monitoring_service),
    current_user: CurrentUserDep = Depends()
):
    """Get monitoring statistics."""
    try:
        stats = await monitoring_service.get_monitoring_stats()
        
        return {
            "success": True,
            "data": stats,
            "message": "Monitoring statistics retrieved successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get monitoring statistics"
        )


@router.post("/monitoring/start", response_model=Dict[str, Any])
async def start_monitoring(
    interval_seconds: int = Query(default=60, ge=10, le=3600, description="Monitoring interval in seconds"),
    monitoring_service: AdvancedMonitoringService = Depends(get_monitoring_service),
    current_user: CurrentUserDep = Depends()
):
    """Start continuous monitoring."""
    try:
        # Start monitoring in background
        import asyncio
        asyncio.create_task(monitoring_service.start_monitoring(interval_seconds))
        
        return {
            "success": True,
            "message": f"Monitoring started with {interval_seconds} second interval"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start monitoring"
        )


@router.post("/monitoring/stop", response_model=Dict[str, Any])
async def stop_monitoring(
    monitoring_service: AdvancedMonitoringService = Depends(get_monitoring_service),
    current_user: CurrentUserDep = Depends()
):
    """Stop continuous monitoring."""
    try:
        await monitoring_service.stop_monitoring()
        
        return {
            "success": True,
            "message": "Monitoring stopped successfully"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to stop monitoring"
        )


@router.get("/performance", response_model=Dict[str, Any])
async def get_performance_analytics(
    hours: int = Query(default=24, ge=1, le=168, description="Number of hours to analyze"),
    endpoint: Optional[str] = Query(None, description="Specific endpoint to analyze"),
    monitoring_service: AdvancedMonitoringService = Depends(get_monitoring_service),
    current_user: CurrentUserDep = Depends()
):
    """Get performance analytics."""
    try:
        analytics = await monitoring_service.get_performance_analytics(
            hours=hours,
            endpoint=endpoint
        )
        
        return {
            "success": True,
            "data": analytics,
            "message": "Performance analytics retrieved successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get performance analytics"
        )


@router.get("/dashboard", response_model=Dict[str, Any])
async def get_monitoring_dashboard(
    monitoring_service: AdvancedMonitoringService = Depends(get_monitoring_service),
    current_user: CurrentUserDep = Depends()
):
    """Get monitoring dashboard data."""
    try:
        # Get system health
        health = await monitoring_service.get_system_health()
        
        # Get active alerts
        alerts = await monitoring_service.get_active_alerts(limit=10)
        
        # Get monitoring stats
        stats = await monitoring_service.get_monitoring_stats()
        
        # Get performance analytics
        performance = await monitoring_service.get_performance_analytics(hours=1)
        
        dashboard_data = {
            "system_health": health,
            "active_alerts": alerts,
            "monitoring_stats": stats,
            "performance": performance,
            "timestamp": "2024-01-15T10:00:00Z"
        }
        
        return {
            "success": True,
            "data": dashboard_data,
            "message": "Monitoring dashboard data retrieved successfully"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get monitoring dashboard data"
        )


@router.get("/thresholds", response_model=Dict[str, Any])
async def get_metric_thresholds(
    monitoring_service: AdvancedMonitoringService = Depends(get_monitoring_service),
    current_user: CurrentUserDep = Depends()
):
    """Get current metric thresholds."""
    try:
        thresholds = {}
        for metric_name, threshold in monitoring_service.thresholds.items():
            thresholds[metric_name] = {
                "warning_threshold": threshold.warning_threshold,
                "critical_threshold": threshold.critical_threshold,
                "unit": threshold.unit,
                "description": threshold.description
            }
        
        return {
            "success": True,
            "data": {
                "thresholds": thresholds,
                "total_metrics": len(thresholds)
            },
            "message": "Metric thresholds retrieved successfully"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get metric thresholds"
        )


@router.get("/metrics/real-time", response_model=Dict[str, Any])
async def get_real_time_metrics(
    monitoring_service: AdvancedMonitoringService = Depends(get_monitoring_service),
    current_user: CurrentUserDep = Depends()
):
    """Get real-time system metrics."""
    try:
        # Collect current metrics
        metrics = await monitoring_service.collect_system_metrics()
        
        # Extract key metrics for real-time display
        real_time_metrics = {
            "cpu_usage": metrics.get("cpu", {}).get("usage_percent", 0),
            "memory_usage": metrics.get("memory", {}).get("usage_percent", 0),
            "disk_usage": metrics.get("disk", {}).get("usage_percent", 0),
            "response_time": metrics.get("application", {}).get("average_response_time_ms", 0),
            "error_rate": metrics.get("application", {}).get("error_rate_percent", 0),
            "active_connections": metrics.get("network", {}).get("active_connections", 0),
            "timestamp": "2024-01-15T10:00:00Z"
        }
        
        return {
            "success": True,
            "data": real_time_metrics,
            "message": "Real-time metrics retrieved successfully"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get real-time metrics"
        )


@router.get("/alerts/summary", response_model=Dict[str, Any])
async def get_alerts_summary(
    monitoring_service: AdvancedMonitoringService = Depends(get_monitoring_service),
    current_user: CurrentUserDep = Depends()
):
    """Get alerts summary."""
    try:
        # Get all active alerts
        all_alerts = await monitoring_service.get_active_alerts(limit=1000)
        
        # Get critical alerts
        critical_alerts = await monitoring_service.get_active_alerts(severity="critical", limit=1000)
        
        # Get warning alerts
        warning_alerts = await monitoring_service.get_active_alerts(severity="warning", limit=1000)
        
        summary = {
            "total_alerts": len(all_alerts["alerts"]),
            "critical_alerts": len(critical_alerts["alerts"]),
            "warning_alerts": len(warning_alerts["alerts"]),
            "resolved_today": 15,  # Mock value
            "new_today": 8,  # Mock value
            "top_alert_types": {
                "high_cpu": 5,
                "high_memory": 3,
                "slow_response": 2,
                "disk_full": 1
            }
        }
        
        return {
            "success": True,
            "data": summary,
            "message": "Alerts summary retrieved successfully"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get alerts summary"
        )

























