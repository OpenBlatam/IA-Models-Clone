"""
Rutas para Sistema de Alertas
==============================

Endpoints para gestión de alertas.
"""

import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.alerting_system import (
    get_alerting_system,
    AlertingSystem,
    AlertSeverity,
    AlertCondition
)
from ..utils.metrics import get_performance_monitor

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/alerts",
    tags=["Alerting System"]
)


class CreateAlertRuleRequest(BaseModel):
    """Request para crear regla de alerta"""
    name: str = Field(..., description="Nombre de la regla")
    description: str = Field(..., description="Descripción")
    metric: str = Field(..., description="Nombre de la métrica")
    condition: str = Field(..., description="Condición: gt, lt, eq, ne, contains")
    threshold: Any = Field(..., description="Umbral")
    severity: str = Field(..., description="Severidad: info, warning, error, critical")
    cooldown_minutes: int = Field(60, description="Minutos de cooldown")


@router.get("/rules")
async def get_alert_rules(
    alerting: AlertingSystem = Depends(get_alerting_system)
):
    """Obtener todas las reglas de alerta"""
    return {"rules": alerting.get_rules()}


@router.post("/rules")
async def create_alert_rule(
    request: CreateAlertRuleRequest,
    alerting: AlertingSystem = Depends(get_alerting_system)
):
    """Crear nueva regla de alerta"""
    try:
        condition = AlertCondition(request.condition)
        severity = AlertSeverity(request.severity)
        
        rule = alerting.add_rule(
            request.name,
            request.description,
            request.metric,
            condition,
            request.threshold,
            severity,
            request.cooldown_minutes
        )
        
        return {
            "status": "created",
            "rule": {
                "name": rule.name,
                "description": rule.description,
                "metric": rule.metric
            }
        }
    except Exception as e:
        logger.error(f"Error creando regla de alerta: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/check")
async def check_alerts(
    alerting: AlertingSystem = Depends(get_alerting_system),
    monitor = Depends(lambda: get_performance_monitor())
):
    """Verificar alertas basadas en métricas actuales"""
    try:
        # Obtener métricas actuales
        if monitor:
            metrics = monitor.get_all_metrics()
        else:
            metrics = {}
        
        # Verificar alertas
        alerts = alerting.check_alerts(metrics)
        
        return {
            "alerts_count": len(alerts),
            "alerts": [
                {
                    "rule_name": a.rule_name,
                    "severity": a.severity.value,
                    "message": a.message,
                    "metric_value": a.metric_value,
                    "threshold": a.threshold,
                    "timestamp": a.timestamp
                }
                for a in alerts
            ]
        }
    except Exception as e:
        logger.error(f"Error verificando alertas: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
async def get_alert_history(
    severity: Optional[str] = None,
    limit: int = 100,
    alerting: AlertingSystem = Depends(get_alerting_system)
):
    """Obtener historial de alertas"""
    try:
        severity_enum = AlertSeverity(severity) if severity else None
        history = alerting.get_alert_history(severity_enum, limit)
        
        return {
            "total": len(history),
            "alerts": [
                {
                    "rule_name": a.rule_name,
                    "severity": a.severity.value,
                    "message": a.message,
                    "timestamp": a.timestamp
                }
                for a in history
            ]
        }
    except Exception as e:
        logger.error(f"Error obteniendo historial: {e}")
        raise HTTPException(status_code=500, detail=str(e))
















