"""
Resources API Endpoints
=======================

Endpoints para información de recursos del sistema.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import logging

try:
    from ..core.resource_manager import get_resource_manager
except ImportError:
    def get_resource_manager():
        return None
from ..core.quality import check_performance_quality, check_system_health_quality

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/resources", tags=["resources"])


@router.get("/")
async def get_resources() -> Dict[str, Any]:
    """Obtener información de recursos del sistema."""
    try:
        manager = get_resource_manager()
        return manager.get_system_resources()
    except Exception as e:
        logger.error(f"Error getting resources: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cpu")
async def get_cpu_usage() -> Dict[str, Any]:
    """Obtener uso de CPU."""
    try:
        manager = get_resource_manager()
        return {
            "usage_percent": manager.get_cpu_usage(),
            "count": manager.get_system_resources()["cpu"]["count"]
        }
    except Exception as e:
        logger.error(f"Error getting CPU usage: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/memory")
async def get_memory_usage() -> Dict[str, Any]:
    """Obtener uso de memoria."""
    try:
        manager = get_resource_manager()
        return manager.get_memory_usage()
    except Exception as e:
        logger.error(f"Error getting memory usage: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/quality")
async def get_quality_report() -> Dict[str, Any]:
    """Obtener reporte de calidad del sistema."""
    try:
        performance_quality = check_performance_quality()
        health_quality = check_system_health_quality()
        
        return {
            "performance": performance_quality,
            "health": health_quality,
            "overall": {
                "performance_score": performance_quality.get("overall_quality", 0.0),
                "health_score": health_quality.get("overall_quality", 0.0)
            }
        }
    except Exception as e:
        logger.error(f"Error getting quality report: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))






