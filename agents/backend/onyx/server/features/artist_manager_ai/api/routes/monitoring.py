"""
Monitoring API Routes
====================

Endpoints para monitoreo y métricas.
"""

import os
from fastapi import APIRouter, HTTPException
from typing import Dict, Any

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


def get_services():
    """Dependency para obtener servicios."""
    from ...utils.monitoring import SystemMonitor, PerformanceMonitor
    from ...utils.metrics import MetricsCollector
    
    return {
        "system_monitor": SystemMonitor(),
        "performance_monitor": PerformanceMonitor(),
        "metrics_collector": MetricsCollector()
    }


@router.get("/system", response_model=Dict[str, Any])
async def get_system_metrics():
    """Obtener métricas del sistema."""
    try:
        services = get_services()
        system_metrics = services["system_monitor"].get_system_metrics()
        process_metrics = services["system_monitor"].get_process_metrics()
        
        return {
            "system": system_metrics,
            "process": process_metrics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance", response_model=Dict[str, Any])
async def get_performance_stats():
    """Obtener estadísticas de rendimiento."""
    try:
        services = get_services()
        stats = services["performance_monitor"].get_all_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics", response_model=Dict[str, Any])
async def get_metrics_summary(hours: int = 24):
    """Obtener resumen de métricas."""
    try:
        services = get_services()
        summary = services["metrics_collector"].get_metrics_summary(hours=hours)
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




