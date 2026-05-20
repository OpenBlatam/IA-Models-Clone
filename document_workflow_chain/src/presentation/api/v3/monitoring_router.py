"""
Monitoring Router
=================

FastAPI router for monitoring operations and system health.
"""

from __future__ import annotations
import logging
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field

from ...shared.services.monitoring_service import (
    Alert,
    AlertEvent,
    AlertLevel,
    AlertCondition,
    Metric,
    MetricType,
    SystemHealth,
    add_alert,
    remove_alert,
    update_alert,
    get_metrics,
    get_all_metrics,
    get_alert_events,
    get_system_health_history,
    get_current_health,
    get_metrics_summary,
    get_health_dashboard,
    record_metric
)
from ...shared.services.optimization_service import (
    get_optimization_summary,
    get_system_metrics,
    optimize_system
)
from ...shared.middleware.auth import get_current_user_optional
from ...shared.middleware.rate_limiter import rate_limit
from ...shared.utils.decorators import log_execution, measure_performance


logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/monitoring", tags=["Monitoring Operations"])


# Request/Response models
class AlertCreateRequest(BaseModel):
    """Alert creation request"""
    name: str = Field(..., description="Alert name", min_length=1, max_length=100)
    description: str = Field(..., description="Alert description", min_length=1, max_length=500)
    level: AlertLevel = Field(..., description="Alert level")
    condition: AlertCondition = Field(..., description="Alert condition")
    threshold: float = Field(..., description="Alert threshold")
    metric_name: str = Field(..., description="Metric name to monitor")
    cooldown_seconds: int = Field(300, description="Cooldown period in seconds", ge=60, le=3600)


class AlertUpdateRequest(BaseModel):
    """Alert update request"""
    name: Optional[str] = Field(None, description="Alert name", min_length=1, max_length=100)
    description: Optional[str] = Field(None, description="Alert description", min_length=1, max_length=500)
    level: Optional[AlertLevel] = Field(None, description="Alert level")
    condition: Optional[AlertCondition] = Field(None, description="Alert condition")
    threshold: Optional[float] = Field(None, description="Alert threshold")
    is_active: Optional[bool] = Field(None, description="Whether alert is active")
    cooldown_seconds: Optional[int] = Field(None, description="Cooldown period in seconds", ge=60, le=3600)


class AlertResponse(BaseModel):
    """Alert response"""
    id: str
    name: str
    description: str
    level: str
    condition: str
    threshold: float
    metric_name: str
    is_active: bool
    created_at: str
    last_triggered: Optional[str]
    trigger_count: int
    cooldown_seconds: int


class AlertEventResponse(BaseModel):
    """Alert event response"""
    alert_id: str
    alert_name: str
    level: str
    message: str
    metric_value: float
    threshold: float
    timestamp: str
    resolved: bool


class MetricResponse(BaseModel):
    """Metric response"""
    name: str
    value: float
    metric_type: str
    labels: Dict[str, str]
    timestamp: str


class SystemHealthResponse(BaseModel):
    """System health response"""
    status: str
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, int]
    active_connections: int
    response_time: float
    error_rate: float
    timestamp: str


class HealthDashboardResponse(BaseModel):
    """Health dashboard response"""
    current_health: SystemHealthResponse
    recent_alerts: List[AlertEventResponse]
    system_stats: Dict[str, Any]
    timestamp: str


class OptimizationSummaryResponse(BaseModel):
    """Optimization summary response"""
    total_optimizations: int
    recent_optimizations: int
    average_improvements: Dict[str, float]
    current_metrics: Dict[str, float]
    optimization_config: Dict[str, Any]
    timestamp: str


# Monitoring operations endpoints
@router.get("/health", response_model=SystemHealthResponse)
@rate_limit(requests=60, window=60)  # 60 requests per minute
@log_execution
@measure_performance
async def get_system_health_endpoint(
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional)
) -> SystemHealthResponse:
    """Get current system health"""
    try:
        health = get_current_health()
        if not health:
            raise HTTPException(status_code=503, detail="System health not available")
        
        return SystemHealthResponse(
            status=health.status,
            cpu_usage=health.cpu_usage,
            memory_usage=health.memory_usage,
            disk_usage=health.disk_usage,
            network_io=health.network_io,
            active_connections=health.active_connections,
            response_time=health.response_time,
            error_rate=health.error_rate,
            timestamp=health.timestamp.isoformat()
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get system health: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get system health: {str(e)}")


@router.get("/health/history", response_model=List[SystemHealthResponse])
@rate_limit(requests=30, window=60)  # 30 requests per minute
@log_execution
@measure_performance
async def get_system_health_history_endpoint(
    limit: int = Query(100, description="Number of health records to return", ge=1, le=1000),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional)
) -> List[SystemHealthResponse]:
    """Get system health history"""
    try:
        health_history = get_system_health_history(limit)
        
        return [
            SystemHealthResponse(
                status=health.status,
                cpu_usage=health.cpu_usage,
                memory_usage=health.memory_usage,
                disk_usage=health.disk_usage,
                network_io=health.network_io,
                active_connections=health.active_connections,
                response_time=health.response_time,
                error_rate=health.error_rate,
                timestamp=health.timestamp.isoformat()
            )
            for health in health_history
        ]
    
    except Exception as e:
        logger.error(f"Failed to get system health history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get system health history: {str(e)}")


@router.get("/dashboard", response_model=HealthDashboardResponse)
@rate_limit(requests=30, window=60)  # 30 requests per minute
@log_execution
@measure_performance
async def get_health_dashboard_endpoint(
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional)
) -> HealthDashboardResponse:
    """Get health dashboard data"""
    try:
        dashboard_data = await get_health_dashboard()
        
        if "error" in dashboard_data:
            raise HTTPException(status_code=500, detail=dashboard_data["error"])
        
        return HealthDashboardResponse(
            current_health=SystemHealthResponse(
                status=dashboard_data["current_health"]["status"],
                cpu_usage=dashboard_data["current_health"]["cpu_usage"],
                memory_usage=dashboard_data["current_health"]["memory_usage"],
                disk_usage=dashboard_data["current_health"]["disk_usage"],
                network_io=dashboard_data["current_health"].get("network_io", {}),
                active_connections=0,
                response_time=0.0,
                error_rate=0.0,
                timestamp=dashboard_data["timestamp"]
            ),
            recent_alerts=[
                AlertEventResponse(
                    alert_id="",
                    alert_name=alert["alert_name"],
                    level=alert["level"],
                    message=alert["message"],
                    metric_value=0.0,
                    threshold=0.0,
                    timestamp=alert["timestamp"],
                    resolved=False
                )
                for alert in dashboard_data["recent_alerts"]
            ],
            system_stats=dashboard_data["system_stats"],
            timestamp=dashboard_data["timestamp"]
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get health dashboard: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get health dashboard: {str(e)}")


@router.get("/metrics", response_model=Dict[str, List[MetricResponse]])
@rate_limit(requests=30, window=60)  # 30 requests per minute
@log_execution
@measure_performance
async def get_all_metrics_endpoint(
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional)
) -> Dict[str, List[MetricResponse]]:
    """Get all metrics"""
    try:
        all_metrics = get_all_metrics()
        
        result = {}
        for metric_name, metrics in all_metrics.items():
            result[metric_name] = [
                MetricResponse(
                    name=metric.name,
                    value=metric.value,
                    metric_type=metric.metric_type.value,
                    labels=metric.labels,
                    timestamp=metric.timestamp.isoformat()
                )
                for metric in metrics
            ]
        
        return result
    
    except Exception as e:
        logger.error(f"Failed to get all metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get all metrics: {str(e)}")


@router.get("/metrics/{metric_name}", response_model=List[MetricResponse])
@rate_limit(requests=60, window=60)  # 60 requests per minute
@log_execution
@measure_performance
async def get_metrics_endpoint(
    metric_name: str,
    limit: int = Query(100, description="Number of metrics to return", ge=1, le=1000),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional)
) -> List[MetricResponse]:
    """Get metrics for a specific metric name"""
    try:
        metrics = get_metrics(metric_name, limit)
        
        return [
            MetricResponse(
                name=metric.name,
                value=metric.value,
                metric_type=metric.metric_type.value,
                labels=metric.labels,
                timestamp=metric.timestamp.isoformat()
            )
            for metric in metrics
        ]
    
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


@router.get("/metrics/summary", response_model=Dict[str, Any])
@rate_limit(requests=30, window=60)  # 30 requests per minute
@log_execution
@measure_performance
async def get_metrics_summary_endpoint(
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional)
) -> Dict[str, Any]:
    """Get metrics summary"""
    try:
        summary = await get_metrics_summary()
        
        if "error" in summary:
            raise HTTPException(status_code=500, detail=summary["error"])
        
        return summary
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get metrics summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get metrics summary: {str(e)}")


@router.get("/alerts", response_model=List[AlertResponse])
@rate_limit(requests=30, window=60)  # 30 requests per minute
@log_execution
@measure_performance
async def get_alerts_endpoint(
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional)
) -> List[AlertResponse]:
    """Get all alerts"""
    try:
        # This would need to be implemented in the monitoring service
        # For now, return empty list
        return []
    
    except Exception as e:
        logger.error(f"Failed to get alerts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get alerts: {str(e)}")


@router.post("/alerts", response_model=AlertResponse)
@rate_limit(requests=10, window=60)  # 10 requests per minute
@log_execution
@measure_performance
async def create_alert_endpoint(
    request: AlertCreateRequest,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional)
) -> AlertResponse:
    """Create a new alert"""
    try:
        # Generate alert ID
        import uuid
        alert_id = str(uuid.uuid4())
        
        # Create alert
        alert = Alert(
            id=alert_id,
            name=request.name,
            description=request.description,
            level=request.level,
            condition=request.condition,
            threshold=request.threshold,
            metric_name=request.metric_name,
            cooldown_seconds=request.cooldown_seconds
        )
        
        # Add alert
        add_alert(alert)
        
        return AlertResponse(
            id=alert.id,
            name=alert.name,
            description=alert.description,
            level=alert.level.value,
            condition=alert.condition.value,
            threshold=alert.threshold,
            metric_name=alert.metric_name,
            is_active=alert.is_active,
            created_at=alert.created_at.isoformat(),
            last_triggered=alert.last_triggered.isoformat() if alert.last_triggered else None,
            trigger_count=alert.trigger_count,
            cooldown_seconds=alert.cooldown_seconds
        )
    
    except Exception as e:
        logger.error(f"Failed to create alert: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create alert: {str(e)}")


@router.put("/alerts/{alert_id}", response_model=AlertResponse)
@rate_limit(requests=20, window=60)  # 20 requests per minute
@log_execution
@measure_performance
async def update_alert_endpoint(
    alert_id: str,
    request: AlertUpdateRequest,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional)
) -> AlertResponse:
    """Update an alert"""
    try:
        # Update alert
        update_data = {}
        if request.name is not None:
            update_data["name"] = request.name
        if request.description is not None:
            update_data["description"] = request.description
        if request.level is not None:
            update_data["level"] = request.level
        if request.condition is not None:
            update_data["condition"] = request.condition
        if request.threshold is not None:
            update_data["threshold"] = request.threshold
        if request.is_active is not None:
            update_data["is_active"] = request.is_active
        if request.cooldown_seconds is not None:
            update_data["cooldown_seconds"] = request.cooldown_seconds
        
        update_alert(alert_id, **update_data)
        
        # Return updated alert (simplified)
        return AlertResponse(
            id=alert_id,
            name=request.name or "Updated Alert",
            description=request.description or "Updated description",
            level=request.level.value if request.level else "info",
            condition=request.condition.value if request.condition else "gt",
            threshold=request.threshold or 0.0,
            metric_name="system.cpu.percent",
            is_active=request.is_active if request.is_active is not None else True,
            created_at="2024-01-01T00:00:00Z",
            last_triggered=None,
            trigger_count=0,
            cooldown_seconds=request.cooldown_seconds or 300
        )
    
    except Exception as e:
        logger.error(f"Failed to update alert: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update alert: {str(e)}")


@router.delete("/alerts/{alert_id}")
@rate_limit(requests=20, window=60)  # 20 requests per minute
@log_execution
@measure_performance
async def delete_alert_endpoint(
    alert_id: str,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional)
) -> Dict[str, str]:
    """Delete an alert"""
    try:
        remove_alert(alert_id)
        
        return {"message": f"Alert {alert_id} deleted successfully"}
    
    except Exception as e:
        logger.error(f"Failed to delete alert: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete alert: {str(e)}")


@router.get("/alerts/events", response_model=List[AlertEventResponse])
@rate_limit(requests=30, window=60)  # 30 requests per minute
@log_execution
@measure_performance
async def get_alert_events_endpoint(
    limit: int = Query(100, description="Number of alert events to return", ge=1, le=1000),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional)
) -> List[AlertEventResponse]:
    """Get alert events"""
    try:
        alert_events = get_alert_events(limit)
        
        return [
            AlertEventResponse(
                alert_id=event.alert_id,
                alert_name=event.alert_name,
                level=event.level.value,
                message=event.message,
                metric_value=event.metric_value,
                threshold=event.threshold,
                timestamp=event.timestamp.isoformat(),
                resolved=event.resolved
            )
            for event in alert_events
        ]
    
    except Exception as e:
        logger.error(f"Failed to get alert events: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get alert events: {str(e)}")


@router.get("/optimization", response_model=OptimizationSummaryResponse)
@rate_limit(requests=10, window=60)  # 10 requests per minute
@log_execution
@measure_performance
async def get_optimization_summary_endpoint(
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional)
) -> OptimizationSummaryResponse:
    """Get optimization summary"""
    try:
        summary = await get_optimization_summary()
        
        if "error" in summary:
            raise HTTPException(status_code=500, detail=summary["error"])
        
        return OptimizationSummaryResponse(
            total_optimizations=summary.get("total_optimizations", 0),
            recent_optimizations=summary.get("recent_optimizations", 0),
            average_improvements=summary.get("average_improvements", {}),
            current_metrics=summary.get("current_metrics", {}),
            optimization_config=summary.get("optimization_config", {}),
            timestamp=summary.get("timestamp", "2024-01-01T00:00:00Z")
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get optimization summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get optimization summary: {str(e)}")


@router.post("/optimization/run")
@rate_limit(requests=5, window=60)  # 5 requests per minute
@log_execution
@measure_performance
async def run_optimization_endpoint(
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional)
) -> Dict[str, Any]:
    """Run system optimization"""
    try:
        result = await optimize_system()
        
        return {
            "message": "System optimization completed",
            "result": result
        }
    
    except Exception as e:
        logger.error(f"Failed to run optimization: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to run optimization: {str(e)}")


# Health check endpoint
@router.get("/health/check")
@log_execution
async def monitoring_health_check() -> Dict[str, Any]:
    """Monitoring service health check"""
    try:
        current_health = get_current_health()
        
        return {
            "status": "healthy" if current_health else "unhealthy",
            "monitoring_active": current_health is not None,
            "timestamp": "2024-01-01T00:00:00Z"  # In real implementation, use actual timestamp
        }
    
    except Exception as e:
        logger.error(f"Monitoring health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": "2024-01-01T00:00:00Z"  # In real implementation, use actual timestamp
        }


