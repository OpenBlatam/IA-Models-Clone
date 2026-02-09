"""
Alerts API for intelligent maintenance alerts and notifications.
Refactored to use BaseRouter for reduced duplication.
"""

from fastapi import Depends, Query
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime

from .base_router import BaseRouter
from ..utils.file_helpers import get_iso_timestamp, get_timestamp_id, create_resource
from ..core.ml_predictor import MLPredictor
from ..config.maintenance_config import MaintenanceConfig, MLConfig

# Create base router instance
base = BaseRouter(
    prefix="/api/alerts",
    tags=["Alerts"],
    require_authentication=True,
    require_rate_limit=False
)

router = base.router


class AlertRule(BaseModel):
    """Alert rule configuration."""
    name: str = Field(..., description="Alert rule name")
    robot_type: Optional[str] = Field(None, description="Filter by robot type")
    condition: str = Field(..., description="Condition expression (e.g., 'temperature > 80')")
    severity: str = Field("warning", description="Severity: info, warning, error, critical")
    enabled: bool = Field(True, description="Whether the rule is enabled")


class AlertRequest(BaseModel):
    """Request to create an alert."""
    robot_type: str = Field(..., description="Robot type")
    sensor_data: Dict[str, Any] = Field(..., description="Sensor data to analyze")
    message: Optional[str] = Field(None, description="Custom alert message")


@router.post("/create")
@base.timed_endpoint("create_alert")
async def create_alert(
    request: AlertRequest,
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """
    Create a new alert based on sensor data analysis.
    """
    base.log_request("create_alert", robot_type=request.robot_type)
    
    ml_config = MLConfig()
    predictor = MLPredictor(ml_config)
    
    # Analyze sensor data
    analysis = await predictor.analyze_sensor_data(request.sensor_data)
    
    # Determine alert level
    alert_level = "info"
    if analysis.get("health_score", 0) < -0.5:
        alert_level = "critical"
    elif analysis.get("health_score", 0) < -0.2:
        alert_level = "error"
    elif analysis.get("health_score", 0) < 0:
        alert_level = "warning"
    
    # Check for anomalies
    if analysis.get("anomalies"):
        alert_level = max(alert_level, "warning", key=lambda x: ["info", "warning", "error", "critical"].index(x))
    
    alert = {
        "id": get_timestamp_id("alert_"),
        "robot_type": request.robot_type,
        "level": alert_level,
        "message": request.message or f"Alert for {request.robot_type}",
        "sensor_data": request.sensor_data,
        "analysis": analysis,
        "timestamp": get_iso_timestamp(),
        "acknowledged": False
    }
    
    return base.success(alert)


@router.get("/list")
@base.timed_endpoint("list_alerts")
async def list_alerts(
    robot_type: Optional[str] = Query(None, description="Filter by robot type"),
    level: Optional[str] = Query(None, description="Filter by alert level"),
    acknowledged: Optional[bool] = Query(None, description="Filter by acknowledged status"),
    limit: int = Query(50, ge=1, le=500, description="Maximum results"),
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """
    List alerts with optional filters.
    """
    base.log_request("list_alerts", robot_type=robot_type, level=level)
    
    # In a real implementation, this would query the database
    # For now, return a placeholder structure
    alerts = []
    
    return base.success({
        "alerts": alerts,
        "total": len(alerts),
        "filters": {
            "robot_type": robot_type,
            "level": level,
            "acknowledged": acknowledged
        }
    })


@router.post("/rules")
@base.timed_endpoint("create_alert_rule")
async def create_alert_rule(
    rule: AlertRule,
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """
    Create a new alert rule.
    """
    base.log_request("create_alert_rule", rule_name=rule.name)
    
    # In a real implementation, this would save to database
    rule_data = create_resource(
        {
            "name": rule.name,
            "robot_type": rule.robot_type,
            "condition": rule.condition,
            "severity": rule.severity,
            "enabled": rule.enabled
        },
        id_prefix="rule_",
        include_timestamps=True
    )
    # Only include created_at, not updated_at for rules
    rule_data.pop("updated_at", None)
    
    return base.success(rule_data)


@router.get("/rules")
@base.timed_endpoint("list_alert_rules")
async def list_alert_rules(
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """
    List all alert rules.
    """
    base.log_request("list_alert_rules")
    
    # Placeholder - would query database
    rules = []
    
    return base.success({
        "rules": rules,
        "count": len(rules)
    })


@router.post("/{alert_id}/acknowledge")
@base.timed_endpoint("acknowledge_alert")
async def acknowledge_alert(
    alert_id: str,
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """
    Acknowledge an alert.
    """
    base.log_request("acknowledge_alert", alert_id=alert_id)
    
    # In a real implementation, this would update the database
    return base.success(None, message=f"Alert {alert_id} acknowledged")




