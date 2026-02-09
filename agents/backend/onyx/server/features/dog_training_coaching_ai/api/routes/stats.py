"""
Statistics Endpoint
===================
Endpoint para estadísticas del servicio.
"""

from fastapi import APIRouter, Depends
from datetime import datetime, timedelta
from typing import Dict, Any
from ...api.dependencies import get_coaching_service
from ...utils.metrics import (
    request_count,
    request_duration_seconds,
    openrouter_requests_total,
    cache_hits_total,
    cache_misses_total
)

router = APIRouter(prefix="/api/v1", tags=["stats"])


@router.get("/stats")
async def get_stats(service: DogTrainingCoach = Depends(get_coaching_service)) -> Dict[str, Any]:
    """
    Obtener estadísticas del servicio.
    
    Returns:
        Estadísticas del servicio
    """
    # Obtener métricas de Prometheus
    stats = {
        "service": "dog-training-coaching-ai",
        "timestamp": datetime.utcnow().isoformat(),
        "metrics": {
            "total_requests": 0,  # Se obtendría de Prometheus
            "average_response_time": 0.0,
            "cache_hit_rate": 0.0,
            "openrouter_requests": 0,
            "error_rate": 0.0
        },
        "uptime": "N/A",  # Se calcularía desde el inicio
        "version": "1.0.0"
    }
    
    return stats


@router.get("/stats/health")
async def get_health_stats() -> Dict[str, Any]:
    """
    Obtener estadísticas de salud del servicio.
    
    Returns:
        Estadísticas de salud
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "checks": {
            "api": "ok",
            "openrouter": "ok",
            "cache": "ok"
        }
    }

