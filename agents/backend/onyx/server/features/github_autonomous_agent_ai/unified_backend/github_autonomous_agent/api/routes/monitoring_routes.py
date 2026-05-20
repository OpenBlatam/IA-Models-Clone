"""
Monitoring Routes - Rutas para monitoreo y métricas avanzadas.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional, List
from pydantic import BaseModel, Field

from api.utils import handle_api_errors
from core.services import MonitoringService, AlertRule, AlertSeverity
from core.services import PerformanceProfiler
from core.services import CacheWarmingService
from config.logging_config import get_logger
from config.di_setup import get_service

router = APIRouter()
logger = get_logger(__name__)


class AlertRuleRequest(BaseModel):
    """Request para crear una regla de alerta."""
    metric_name: str = Field(..., description="Nombre de la métrica")
    condition: str = Field(..., description="Condición (gt, lt, eq, ne)")
    threshold: float = Field(..., description="Umbral")
    severity: str = Field(default="warning", description="Severidad")
    message: Optional[str] = Field(None, description="Mensaje personalizado")


@router.get("/metrics")
@handle_api_errors
async def get_metrics(
    name: Optional[str] = Query(None, description="Nombre de métrica específica"),
    window: Optional[int] = Query(None, ge=1, le=10000, description="Ventana de tiempo")
):
    """
    Obtener métricas.
    
    Args:
        name: Nombre de métrica específica
        window: Ventana de tiempo
        
    Returns:
        Métricas
    """
    try:
        monitoring_service: MonitoringService = get_service("monitoring_service")
    except Exception:
        raise HTTPException(status_code=503, detail="Monitoring service no disponible")
    
    if name:
        metrics = monitoring_service.get_metric(name, window)
        return {"metric": name, "data": metrics}
    else:
        return monitoring_service.get_current_metrics()


@router.get("/metrics/stats")
@handle_api_errors
async def get_monitoring_stats():
    """
    Obtener estadísticas del servicio de monitoreo.
    
    Returns:
        Estadísticas
    """
    try:
        monitoring_service: MonitoringService = get_service("monitoring_service")
    except Exception:
        raise HTTPException(status_code=503, detail="Monitoring service no disponible")
    
    return monitoring_service.get_stats()


@router.post("/alerts/rules")
@handle_api_errors
async def create_alert_rule(
    request: AlertRuleRequest
):
    """
    Crear regla de alerta.
    
    Args:
        metric_name: Nombre de la métrica
        condition: Condición (gt, lt, eq, ne)
        threshold: Umbral
        severity: Severidad
        message: Mensaje personalizado
        
    Returns:
        Regla creada
    """
    try:
        monitoring_service: MonitoringService = get_service("monitoring_service")
    except Exception:
        raise HTTPException(status_code=503, detail="Monitoring service no disponible")
    
    # Crear función de condición
    if request.condition == "gt":
        condition_func = lambda v: v > request.threshold
    elif request.condition == "lt":
        condition_func = lambda v: v < request.threshold
    elif request.condition == "eq":
        condition_func = lambda v: v == request.threshold
    elif request.condition == "ne":
        condition_func = lambda v: v != request.threshold
    else:
        raise HTTPException(status_code=400, detail=f"Condición inválida: {request.condition}")
    
    try:
        severity_enum = AlertSeverity(request.severity)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Severidad inválida: {request.severity}"
        )
    
    rule = AlertRule(
        name=f"{request.metric_name}_{request.condition}_{request.threshold}",
        metric_name=request.metric_name,
        condition=condition_func,
        severity=severity_enum,
        message=request.message or f"{request.metric_name} {request.condition} {request.threshold}"
    )
    
    monitoring_service.add_alert_rule(rule)
    
    return {
        "name": rule.name,
        "metric_name": rule.metric_name,
        "severity": rule.severity.value,
        "message": rule.message
    }


@router.get("/performance/stats")
@handle_api_errors
async def get_performance_stats():
    """
    Obtener estadísticas de performance profiling.
    
    Returns:
        Estadísticas de performance
    """
    try:
        profiler: PerformanceProfiler = get_service("performance_profiler")
    except Exception:
        raise HTTPException(status_code=503, detail="Performance profiler no disponible")
    
    return profiler.get_stats()


@router.get("/performance/slowest")
@handle_api_errors
async def get_slowest_operations(
    limit: int = Query(10, ge=1, le=100)
):
    """
    Obtener operaciones más lentas.
    
    Args:
        limit: Número máximo de operaciones
        
    Returns:
        Lista de operaciones más lentas
    """
    try:
        profiler: PerformanceProfiler = get_service("performance_profiler")
    except Exception:
        raise HTTPException(status_code=503, detail="Performance profiler no disponible")
    
    return profiler.get_slowest_operations(limit)


@router.get("/cache-warming/stats")
@handle_api_errors
async def get_cache_warming_stats():
    """
    Obtener estadísticas de cache warming.
    
    Returns:
        Estadísticas
    """
    try:
        cache_warming: CacheWarmingService = get_service("cache_warming_service")
    except Exception:
        raise HTTPException(status_code=503, detail="Cache warming service no disponible")
    
    return cache_warming.get_stats()


@router.post("/cache-warming/warm")
@handle_api_errors
async def warm_cache(
    strategy_name: Optional[str] = Query(None, description="Estrategia específica")
):
    """
    Calentar cache manualmente.
    
    Args:
        strategy_name: Nombre de estrategia específica (opcional)
        
    Returns:
        Resultado del warming
    """
    try:
        cache_warming: CacheWarmingService = get_service("cache_warming_service")
    except Exception:
        raise HTTPException(status_code=503, detail="Cache warming service no disponible")
    
    return await cache_warming.warm_cache(strategy_name)



