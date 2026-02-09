"""
Monitoring API Routes
=====================

Endpoints para monitoreo y métricas.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Optional

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


def get_monitoring_service():
    """Dependency para obtener MonitoringService"""
    from ...services.monitoring_service import MonitoringService
    return MonitoringService()


@router.get("/health", response_model=dict)
async def health_check(monitoring = Depends(get_monitoring_service)):
    """Health check del sistema"""
    try:
        return monitoring.get_health_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics", response_model=dict)
async def get_metrics(
    hours: int = Query(24, ge=1, le=168),
    monitoring = Depends(get_monitoring_service)
):
    """Obtener métricas del sistema"""
    try:
        return monitoring.get_metrics_summary(hours)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/counters", response_model=dict)
async def get_counters(monitoring = Depends(get_monitoring_service)):
    """Obtener contadores"""
    try:
        return dict(monitoring.counters)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/timings/{operation}", response_model=dict)
async def get_timing_stats(
    operation: str,
    monitoring = Depends(get_monitoring_service)
):
    """Obtener estadísticas de timing de una operación"""
    try:
        return monitoring.get_timing_stats(operation)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reset")
async def reset_metrics(monitoring = Depends(get_monitoring_service)):
    """Resetear todas las métricas"""
    try:
        monitoring.reset()
        return {"status": "reset", "message": "Métricas reseteadas"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




