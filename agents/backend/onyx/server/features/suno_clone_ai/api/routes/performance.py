"""
Endpoints para métricas de rendimiento

Proporciona endpoints para monitorear el rendimiento de la API.
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, Query, Response

from ..utils.performance_monitor import get_performance_stats, clear_performance_stats
from ..utils.response_cache import get_cache_stats, clear_response_cache

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/performance",
    tags=["performance"]
)


@router.get("/stats")
async def get_performance_stats_endpoint(
    operation: Optional[str] = Query(None, description="Nombre de operación específica"),
    response: Optional[Response] = None
) -> Dict[str, Any]:
    """
    Obtiene estadísticas de rendimiento de la API.
    
    Incluye métricas de tiempo de ejecución para diferentes operaciones.
    """
    stats = get_performance_stats(operation)
    
    # Headers de cache corto
    if response:
        response.headers["Cache-Control"] = "public, max-age=30"
    
    return {
        "performance_stats": stats,
        "cache_stats": get_cache_stats()
    }


@router.post("/stats/clear")
async def clear_performance_stats_endpoint() -> Dict[str, str]:
    """
    Limpia todas las estadísticas de rendimiento.
    
    Útil para reiniciar métricas después de cambios o para testing.
    """
    clear_performance_stats()
    clear_response_cache()
    
    logger.info("Performance stats and cache cleared")
    
    return {
        "message": "Performance stats and cache cleared successfully"
    }

