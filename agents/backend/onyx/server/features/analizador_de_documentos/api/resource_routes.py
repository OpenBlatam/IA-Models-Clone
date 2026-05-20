"""
Rutas para Optimización de Recursos
=====================================

Endpoints para gestión y optimización de recursos.
"""

import logging
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import PlainTextResponse

from ..core.resource_optimizer import get_resource_optimizer, ResourceOptimizer

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/resources",
    tags=["Resource Optimization"]
)


@router.get("/metrics")
async def get_current_metrics(
    optimizer: ResourceOptimizer = Depends(get_resource_optimizer)
):
    """Obtener métricas actuales de recursos"""
    try:
        metrics = optimizer.get_current_metrics()
        return {
            "timestamp": metrics.timestamp,
            "cpu_percent": metrics.cpu_percent,
            "memory_percent": metrics.memory_percent,
            "memory_used_mb": metrics.memory_used_mb,
            "memory_available_mb": metrics.memory_available_mb,
            "disk_usage_percent": metrics.disk_usage_percent,
            "network_sent_mb": metrics.network_sent_mb,
            "network_recv_mb": metrics.network_recv_mb
        }
    except Exception as e:
        logger.error(f"Error obteniendo métricas: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/optimize/memory")
async def optimize_memory(
    optimizer: ResourceOptimizer = Depends(get_resource_optimizer)
):
    """Optimizar memoria"""
    try:
        result = optimizer.optimize_memory()
        return result
    except Exception as e:
        logger.error(f"Error optimizando memoria: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recommendations")
async def get_optimization_recommendations(
    optimizer: ResourceOptimizer = Depends(get_resource_optimizer)
):
    """Obtener recomendaciones de optimización"""
    try:
        recommendations = optimizer.get_optimization_recommendations()
        return {"recommendations": recommendations}
    except Exception as e:
        logger.error(f"Error obteniendo recomendaciones: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/summary")
async def get_resource_summary(
    optimizer: ResourceOptimizer = Depends(get_resource_optimizer)
):
    """Obtener resumen de recursos"""
    try:
        summary = optimizer.get_resource_summary()
        return summary
    except Exception as e:
        logger.error(f"Error obteniendo resumen: {e}")
        raise HTTPException(status_code=500, detail=str(e))
















