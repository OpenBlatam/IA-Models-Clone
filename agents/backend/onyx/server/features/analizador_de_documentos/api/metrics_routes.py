"""
Rutas de Métricas y Monitoring
================================

Endpoints para obtener métricas de rendimiento y estadísticas del sistema.
"""

import logging
from fastapi import APIRouter, Depends
from typing import Dict, Any

from ..utils.metrics import get_performance_monitor

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/metrics",
    tags=["Metrics"]
)


def get_monitor():
    """Dependency para obtener monitor de rendimiento"""
    try:
        return get_performance_monitor()
    except:
        return None


@router.get("/")
async def get_metrics(monitor = Depends(get_monitor)) -> Dict[str, Any]:
    """Obtener todas las métricas del sistema"""
    if monitor is None:
        return {"error": "Monitoring no disponible"}
    
    return monitor.get_all_stats()


@router.get("/performance")
async def get_performance_stats(monitor = Depends(get_monitor)) -> Dict[str, Any]:
    """Obtener estadísticas de rendimiento"""
    if monitor is None:
        return {"error": "Monitoring no disponible"}
    
    return monitor.get_performance_stats()


@router.get("/health")
async def health_check_detailed(monitor = Depends(get_monitor)) -> Dict[str, Any]:
    """Health check detallado con métricas"""
    if monitor is None:
        return {
            "status": "degraded",
            "message": "Monitoring no disponible"
        }
    
    stats = monitor.get_performance_stats()
    
    # Determinar estado de salud
    if stats.get("success_rate", 0) > 0.95:
        status = "healthy"
    elif stats.get("success_rate", 0) > 0.80:
        status = "degraded"
    else:
        status = "unhealthy"
    
    return {
        "status": status,
        "metrics": stats
    }
















