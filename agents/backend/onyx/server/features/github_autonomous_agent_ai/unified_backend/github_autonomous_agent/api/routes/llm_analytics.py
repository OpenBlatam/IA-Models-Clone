"""
LLM Analytics Routes - Rutas para analytics y métricas.

Incluye:
- Métricas en tiempo real
- Estadísticas y tendencias
- Alertas y notificaciones
- Reportes personalizados
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from datetime import datetime, timedelta

from api.utils import handle_api_errors
from config.di_setup import get_service
from config.logging_config import get_logger
from core.services.llm_service import LLMService
from core.services.llm import (
    get_llm_analytics,
    MetricType,
    AlertRule
)
from core.constants import ErrorMessages

logger = get_logger(__name__)

router = APIRouter(
    prefix="/llm/analytics",
    tags=["LLM Analytics"],
    responses={
        400: {"description": "Bad request"},
        404: {"description": "Not found"}
    }
)


def get_llm_service() -> Optional[LLMService]:
    """Obtener servicio LLM del DI container."""
    try:
        return get_service("llm_service")
    except (ValueError, Exception):
        return None


class CreateAlertRequest(BaseModel):
    """Request para crear regla de alerta."""
    name: str = Field(..., description="Nombre de la alerta")
    metric_type: str = Field(..., description="Tipo de métrica")
    threshold: float = Field(..., description="Umbral de alerta")
    comparison: str = Field("gt", description="Comparación (gt, lt, eq)")
    window_minutes: int = Field(5, description="Ventana de tiempo en minutos")
    notification_channels: Optional[List[str]] = Field(None, description="Canales de notificación")


@router.get("/metrics/{metric_type}")
@handle_api_errors
async def get_metrics(
    metric_type: str,
    start_time: Optional[str] = Query(None, description="Tiempo de inicio (ISO format)"),
    end_time: Optional[str] = Query(None, description="Tiempo de fin (ISO format)"),
    aggregation: Optional[str] = Query(None, description="Agregación (avg, sum, min, max, count)"),
    window_minutes: int = Query(60, description="Ventana de tiempo en minutos")
):
    """
    Obtener métricas con filtros.
    
    Args:
        metric_type: Tipo de métrica
        start_time: Tiempo de inicio (opcional)
        end_time: Tiempo de fin (opcional)
        aggregation: Tipo de agregación (opcional)
        window_minutes: Ventana de tiempo en minutos
        
    Returns:
        Métricas filtradas
    """
    try:
        metric_enum = MetricType(metric_type)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de métrica inválido: {metric_type}"
        )
    
    analytics = get_llm_analytics()
    
    start = None
    end = None
    
    if start_time:
        start = datetime.fromisoformat(start_time)
    if end_time:
        end = datetime.fromisoformat(end_time)
    
    if not start:
        start = datetime.now() - timedelta(minutes=window_minutes)
    if not end:
        end = datetime.now()
    
    points = analytics.get_metrics(
        metric_enum,
        start_time=start,
        end_time=end,
        aggregation=aggregation
    )
    
    return {
        "success": True,
        "metric_type": metric_type,
        "points": [p.to_dict() for p in points],
        "count": len(points),
        "start_time": start.isoformat(),
        "end_time": end.isoformat()
    }


@router.get("/statistics/{metric_type}")
@handle_api_errors
async def get_statistics(
    metric_type: str,
    window_minutes: int = Query(60, description="Ventana de tiempo en minutos")
):
    """
    Obtener estadísticas de una métrica.
    
    Args:
        metric_type: Tipo de métrica
        window_minutes: Ventana de tiempo en minutos
        
    Returns:
        Estadísticas (min, max, avg, sum, count, percentiles)
    """
    try:
        metric_enum = MetricType(metric_type)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de métrica inválido: {metric_type}"
        )
    
    analytics = get_llm_analytics()
    stats = analytics.get_statistics(metric_enum, window_minutes=window_minutes)
    
    return {
        "success": True,
        "metric_type": metric_type,
        "window_minutes": window_minutes,
        "statistics": stats
    }


@router.get("/trends/{metric_type}")
@handle_api_errors
async def get_trends(
    metric_type: str,
    periods: int = Query(7, description="Número de períodos"),
    period_hours: int = Query(24, description="Horas por período")
):
    """
    Obtener tendencias de una métrica.
    
    Args:
        metric_type: Tipo de métrica
        periods: Número de períodos
        period_hours: Horas por período
        
    Returns:
        Tendencias con comparación período a período
    """
    try:
        metric_enum = MetricType(metric_type)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de métrica inválido: {metric_type}"
        )
    
    analytics = get_llm_analytics()
    trends = analytics.get_trends(metric_enum, periods=periods, period_hours=period_hours)
    
    return {
        "success": True,
        "trends": trends
    }


@router.get("/alerts")
@handle_api_errors
async def get_alerts():
    """
    Obtener alertas activas.
    
    Returns:
        Lista de alertas activas
    """
    analytics = get_llm_analytics()
    alerts = analytics.get_active_alerts()
    
    return {
        "success": True,
        "alerts": [a.to_dict() for a in alerts],
        "count": len(alerts)
    }


@router.post("/alerts/create")
@handle_api_errors
async def create_alert(request: CreateAlertRequest):
    """
    Crear regla de alerta.
    
    Args:
        request: Request con configuración de alerta
        
    Returns:
        ID de la alerta creada
    """
    try:
        metric_enum = MetricType(request.metric_type)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de métrica inválido: {request.metric_type}"
        )
    
    if request.comparison not in ["gt", "lt", "eq"]:
        raise HTTPException(
            status_code=400,
            detail="Comparación debe ser 'gt', 'lt' o 'eq'"
        )
    
    analytics = get_llm_analytics()
    
    alert_id = analytics.create_alert_rule(
        name=request.name,
        metric_type=metric_enum,
        threshold=request.threshold,
        comparison=request.comparison,
        window_minutes=request.window_minutes,
        notification_channels=request.notification_channels
    )
    
    return {
        "success": True,
        "alert_id": alert_id,
        "message": "Regla de alerta creada exitosamente"
    }


@router.get("/dashboard")
@handle_api_errors
async def get_dashboard(
    llm_service: Optional[LLMService] = Depends(get_llm_service),
    window_minutes: int = Query(60, description="Ventana de tiempo en minutos")
):
    """
    Obtener datos completos para dashboard.
    
    Args:
        window_minutes: Ventana de tiempo en minutos
        
    Returns:
        Datos consolidados para dashboard
    """
    if not llm_service:
        raise HTTPException(
            status_code=503,
            detail=ErrorMessages.LLM_SERVICE_UNAVAILABLE
        )
    
    analytics = get_llm_analytics()
    
    # Obtener estadísticas de todas las métricas principales
    metrics_stats = {}
    for metric_type in MetricType:
        stats = analytics.get_statistics(metric_type, window_minutes=window_minutes)
        metrics_stats[metric_type.value] = stats
    
    # Obtener alertas activas
    alerts = analytics.get_active_alerts()
    
    # Obtener tendencias
    trends = {}
    for metric_type in [MetricType.LATENCY, MetricType.COST, MetricType.ERROR_RATE]:
        trends[metric_type.value] = analytics.get_trends(metric_type, periods=7, period_hours=24)
    
    # Obtener stats del servicio
    service_stats = llm_service.get_stats()
    
    return {
        "success": True,
        "window_minutes": window_minutes,
        "metrics": metrics_stats,
        "alerts": {
            "active": len(alerts),
            "list": [a.to_dict() for a in alerts]
        },
        "trends": trends,
        "service_stats": service_stats
    }


@router.post("/record")
@handle_api_errors
async def record_metric(
    metric_type: str,
    value: float,
    tags: Optional[Dict[str, str]] = None
):
    """
    Registrar una métrica manualmente.
    
    Args:
        metric_type: Tipo de métrica
        value: Valor de la métrica
        tags: Tags adicionales (opcional)
        
    Returns:
        Confirmación de registro
    """
    try:
        metric_enum = MetricType(metric_type)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de métrica inválido: {metric_type}"
        )
    
    analytics = get_llm_analytics()
    analytics.record_metric(metric_enum, value, tags)
    
    return {
        "success": True,
        "message": "Métrica registrada exitosamente"
    }



