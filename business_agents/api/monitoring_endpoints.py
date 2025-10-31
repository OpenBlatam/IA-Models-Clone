"""
Real-Time Monitoring API Endpoints
==================================

REST API endpoints for real-time monitoring, metrics, and alerting.
"""

from fastapi import APIRouter, HTTPException, Depends, WebSocket, WebSocketDisconnect
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import json
import asyncio

from ..services.real_time_monitoring_service import (
    RealTimeMonitoringService,
    AlertLevel,
    MetricType,
    HealthStatus
)
from ..middleware.auth_middleware import get_current_user, require_permission, User

# Create API router
router = APIRouter(prefix="/monitoring", tags=["Monitoring"])

# Pydantic models
class MetricRequest(BaseModel):
    name: str = Field(..., description="Metric name")
    value: float = Field(..., description="Metric value")
    metric_type: MetricType = Field(..., description="Metric type")
    tags: Optional[Dict[str, str]] = Field(None, description="Metric tags")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metric metadata")

class AlertRuleRequest(BaseModel):
    name: str = Field(..., description="Alert rule name")
    condition: str = Field(..., description="Alert condition (Python expression)")
    level: AlertLevel = Field(..., description="Alert level")
    description: str = Field(..., description="Alert description")

class HealthCheckRequest(BaseModel):
    name: str = Field(..., description="Health check name")
    endpoint: str = Field(..., description="Health check endpoint")
    timeout: int = Field(30, description="Health check timeout")

# Global monitoring service instance
monitoring_service = None

def get_monitoring_service() -> RealTimeMonitoringService:
    """Get global monitoring service instance."""
    global monitoring_service
    if monitoring_service is None:
        monitoring_service = RealTimeMonitoringService({"monitoring_interval": 5})
    return monitoring_service

# API Endpoints

@router.post("/metrics", response_model=Dict[str, str])
async def record_metric(
    request: MetricRequest,
    current_user: User = Depends(require_permission("monitoring:write"))
):
    """Record a metric."""
    
    monitoring_service = get_monitoring_service()
    
    try:
        monitoring_service.record_metric(
            name=request.name,
            value=request.value,
            metric_type=request.metric_type,
            tags=request.tags,
            metadata=request.metadata
        )
        
        return {"message": "Metric recorded successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to record metric: {str(e)}")

@router.get("/metrics", response_model=Dict[str, Any])
async def get_metrics(
    name: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    current_user: User = Depends(require_permission("monitoring:view"))
):
    """Get metrics data."""
    
    monitoring_service = get_monitoring_service()
    
    try:
        if name:
            # Get specific metric
            metrics = monitoring_service.get_metric_history(name, start_time, end_time)
            return {
                "metric_name": name,
                "metrics": [
                    {
                        "value": m.value,
                        "timestamp": m.timestamp.isoformat(),
                        "tags": m.tags,
                        "metadata": m.metadata
                    }
                    for m in metrics
                ],
                "count": len(metrics)
            }
        else:
            # Get all metrics
            all_metrics = monitoring_service.get_all_metrics()
            return {
                "metrics": {
                    name: [
                        {
                            "value": m.value,
                            "timestamp": m.timestamp.isoformat(),
                            "tags": m.tags,
                            "metadata": m.metadata
                        }
                        for m in metrics
                    ]
                    for name, metrics in all_metrics.items()
                },
                "metric_names": list(all_metrics.keys())
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")

@router.get("/metrics/{metric_name}/statistics", response_model=Dict[str, float])
async def get_metric_statistics(
    metric_name: str,
    window_minutes: int = 60,
    current_user: User = Depends(require_permission("monitoring:view"))
):
    """Get statistics for a specific metric."""
    
    monitoring_service = get_monitoring_service()
    
    try:
        stats = monitoring_service.get_metric_statistics(metric_name, window_minutes)
        
        if not stats:
            raise HTTPException(status_code=404, detail=f"Metric {metric_name} not found")
        
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metric statistics: {str(e)}")

@router.get("/alerts", response_model=List[Dict[str, Any]])
async def get_alerts(
    active_only: bool = True,
    current_user: User = Depends(require_permission("monitoring:view"))
):
    """Get alerts."""
    
    monitoring_service = get_monitoring_service()
    
    try:
        if active_only:
            alerts = monitoring_service.get_active_alerts()
        else:
            alerts = monitoring_service.alerts
        
        return [
            {
                "id": alert.id,
                "name": alert.name,
                "description": alert.description,
                "level": alert.level.value,
                "source": alert.source,
                "timestamp": alert.timestamp.isoformat(),
                "resolved": alert.resolved,
                "resolved_at": alert.resolved_at.isoformat() if alert.resolved_at else None,
                "metadata": alert.metadata
            }
            for alert in alerts
        ]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get alerts: {str(e)}")

@router.post("/alerts/{alert_id}/resolve", response_model=Dict[str, str])
async def resolve_alert(
    alert_id: str,
    current_user: User = Depends(require_permission("monitoring:manage"))
):
    """Resolve an alert."""
    
    monitoring_service = get_monitoring_service()
    
    try:
        success = monitoring_service.resolve_alert(alert_id)
        
        if success:
            return {"message": f"Alert {alert_id} resolved successfully"}
        else:
            raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found or already resolved")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to resolve alert: {str(e)}")

@router.post("/alert-rules", response_model=Dict[str, str])
async def add_alert_rule(
    request: AlertRuleRequest,
    current_user: User = Depends(require_permission("monitoring:manage"))
):
    """Add a new alert rule."""
    
    monitoring_service = get_monitoring_service()
    
    try:
        # Create alert rule
        rule = {
            "name": request.name,
            "condition": lambda metrics: eval(request.condition, {"metrics": metrics}),
            "level": request.level,
            "description": request.description
        }
        
        monitoring_service.add_alert_rule(rule)
        
        return {"message": f"Alert rule '{request.name}' added successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add alert rule: {str(e)}")

@router.delete("/alert-rules/{rule_name}", response_model=Dict[str, str])
async def remove_alert_rule(
    rule_name: str,
    current_user: User = Depends(require_permission("monitoring:manage"))
):
    """Remove an alert rule."""
    
    monitoring_service = get_monitoring_service()
    
    try:
        success = monitoring_service.remove_alert_rule(rule_name)
        
        if success:
            return {"message": f"Alert rule '{rule_name}' removed successfully"}
        else:
            raise HTTPException(status_code=404, detail=f"Alert rule '{rule_name}' not found")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to remove alert rule: {str(e)}")

@router.get("/health", response_model=Dict[str, Any])
async def get_health_status(
    current_user: User = Depends(require_permission("monitoring:view"))
):
    """Get system health status."""
    
    monitoring_service = get_monitoring_service()
    
    try:
        health_status = monitoring_service.get_health_status()
        return health_status
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get health status: {str(e)}")

@router.get("/dashboard", response_model=Dict[str, Any])
async def get_dashboard_data(
    current_user: User = Depends(require_permission("monitoring:view"))
):
    """Get real-time dashboard data."""
    
    monitoring_service = get_monitoring_service()
    
    try:
        dashboard_data = await monitoring_service.get_realtime_dashboard_data()
        return dashboard_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dashboard data: {str(e)}")

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time monitoring updates."""
    
    await websocket.accept()
    monitoring_service = get_monitoring_service()
    
    # Add connection to monitoring service
    monitoring_service.add_websocket_connection(websocket)
    
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
            
    except WebSocketDisconnect:
        # Remove connection from monitoring service
        monitoring_service.remove_websocket_connection(websocket)

@router.post("/workflow-metrics", response_model=Dict[str, str])
async def record_workflow_metric(
    workflow_id: str,
    metric_name: str,
    value: float,
    tags: Optional[Dict[str, str]] = None,
    current_user: User = Depends(require_permission("monitoring:write"))
):
    """Record a workflow-specific metric."""
    
    monitoring_service = get_monitoring_service()
    
    try:
        monitoring_service.record_workflow_metric(
            workflow_id=workflow_id,
            metric_name=metric_name,
            value=value,
            tags=tags
        )
        
        return {"message": f"Workflow metric recorded for {workflow_id}"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to record workflow metric: {str(e)}")

@router.post("/agent-metrics", response_model=Dict[str, str])
async def record_agent_metric(
    agent_id: str,
    metric_name: str,
    value: float,
    tags: Optional[Dict[str, str]] = None,
    current_user: User = Depends(require_permission("monitoring:write"))
):
    """Record an agent-specific metric."""
    
    monitoring_service = get_monitoring_service()
    
    try:
        monitoring_service.record_agent_metric(
            agent_id=agent_id,
            metric_name=metric_name,
            value=value,
            tags=tags
        )
        
        return {"message": f"Agent metric recorded for {agent_id}"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to record agent metric: {str(e)}")

@router.post("/api-metrics", response_model=Dict[str, str])
async def record_api_metric(
    endpoint: str,
    method: str,
    response_time: float,
    status_code: int,
    tags: Optional[Dict[str, str]] = None,
    current_user: User = Depends(require_permission("monitoring:write"))
):
    """Record API metrics."""
    
    monitoring_service = get_monitoring_service()
    
    try:
        monitoring_service.record_api_metric(
            endpoint=endpoint,
            method=method,
            response_time=response_time,
            status_code=status_code,
            tags=tags
        )
        
        return {"message": "API metric recorded successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to record API metric: {str(e)}")

@router.get("/performance", response_model=Dict[str, Any])
async def get_performance_metrics(
    current_user: User = Depends(require_permission("monitoring:view"))
):
    """Get current performance metrics."""
    
    monitoring_service = get_monitoring_service()
    
    try:
        current_metrics = monitoring_service._get_current_metrics()
        
        # Calculate performance indicators
        performance = {
            "timestamp": datetime.now().isoformat(),
            "system_metrics": {
                "cpu_usage": current_metrics.get("cpu_usage", 0),
                "memory_usage": current_metrics.get("memory_usage", 0),
                "disk_usage": current_metrics.get("disk_usage", 0),
                "process_count": current_metrics.get("process_count", 0)
            },
            "application_metrics": {
                "error_rate": current_metrics.get("error_rate", 0),
                "response_time": current_metrics.get("response_time", 0),
                "success_rate": current_metrics.get("success_rate", 100),
                "request_rate": current_metrics.get("request_rate", 0)
            },
            "network_metrics": {
                "bytes_sent": current_metrics.get("network_bytes_sent", 0),
                "bytes_received": current_metrics.get("network_bytes_recv", 0)
            }
        }
        
        return performance
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get performance metrics: {str(e)}")

@router.get("/trends", response_model=Dict[str, Any])
async def get_metric_trends(
    metric_names: List[str],
    window_hours: int = 24,
    current_user: User = Depends(require_permission("monitoring:view"))
):
    """Get metric trends over time."""
    
    monitoring_service = get_monitoring_service()
    
    try:
        trends = {}
        
        for metric_name in metric_names:
            # Get metric history
            start_time = datetime.now() - timedelta(hours=window_hours)
            metrics = monitoring_service.get_metric_history(metric_name, start_time)
            
            if metrics:
                trends[metric_name] = {
                    "values": [m.value for m in metrics],
                    "timestamps": [m.timestamp.isoformat() for m in metrics],
                    "statistics": monitoring_service.get_metric_statistics(metric_name, window_hours * 60)
                }
            else:
                trends[metric_name] = {
                    "values": [],
                    "timestamps": [],
                    "statistics": {}
                }
        
        return {
            "trends": trends,
            "window_hours": window_hours,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metric trends: {str(e)}")

@router.get("/alert-levels", response_model=List[Dict[str, str]])
async def get_alert_levels():
    """Get available alert levels."""
    
    alert_levels = [
        {
            "level": level.value,
            "name": level.value.title(),
            "description": get_alert_level_description(level)
        }
        for level in AlertLevel
    ]
    
    return alert_levels

def get_alert_level_description(level: AlertLevel) -> str:
    """Get description for alert level."""
    
    descriptions = {
        AlertLevel.INFO: "Informational message",
        AlertLevel.WARNING: "Warning condition",
        AlertLevel.ERROR: "Error condition",
        AlertLevel.CRITICAL: "Critical condition requiring immediate attention"
    }
    
    return descriptions.get(level, "Alert level")

@router.get("/metric-types", response_model=List[Dict[str, str]])
async def get_metric_types():
    """Get available metric types."""
    
    metric_types = [
        {
            "type": metric_type.value,
            "name": metric_type.value.title(),
            "description": get_metric_type_description(metric_type)
        }
        for metric_type in MetricType
    ]
    
    return metric_types

def get_metric_type_description(metric_type: MetricType) -> str:
    """Get description for metric type."""
    
    descriptions = {
        MetricType.COUNTER: "Monotonically increasing counter",
        MetricType.GAUGE: "Value that can go up or down",
        MetricType.HISTOGRAM: "Distribution of values",
        MetricType.TIMER: "Duration of operations"
    }
    
    return descriptions.get(metric_type, "Metric type")

@router.get("/health-statuses", response_model=List[Dict[str, str]])
async def get_health_statuses():
    """Get available health statuses."""
    
    health_statuses = [
        {
            "status": status.value,
            "name": status.value.title(),
            "description": get_health_status_description(status)
        }
        for status in HealthStatus
    ]
    
    return health_statuses

def get_health_status_description(status: HealthStatus) -> str:
    """Get description for health status."""
    
    descriptions = {
        HealthStatus.HEALTHY: "Component is functioning normally",
        HealthStatus.DEGRADED: "Component is functioning but with reduced performance",
        HealthStatus.UNHEALTHY: "Component is not functioning properly",
        HealthStatus.CRITICAL: "Component is in a critical state"
    }
    
    return descriptions.get(status, "Health status")





























