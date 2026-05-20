"""
Rutas para Performance Profiler
=================================

Endpoints para profiling y análisis de rendimiento.
"""

import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import PlainTextResponse

from ..core.performance_profiler import get_performance_profiler, PerformanceProfiler

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/profiler",
    tags=["Performance Profiler"]
)


@router.get("/stats")
async def get_statistics(
    function_name: Optional[str] = None,
    profiler: PerformanceProfiler = Depends(get_performance_profiler)
):
    """Obtener estadísticas de rendimiento"""
    try:
        stats = profiler.get_statistics(function_name)
        return {"statistics": stats}
    except Exception as e:
        logger.error(f"Error obteniendo estadísticas: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/bottlenecks")
async def get_bottlenecks(
    threshold_ms: float = 1000.0,
    profiler: PerformanceProfiler = Depends(get_performance_profiler)
):
    """Detectar cuellos de botella"""
    try:
        bottlenecks = profiler.get_bottlenecks(threshold_ms)
        return {"bottlenecks": bottlenecks}
    except Exception as e:
        logger.error(f"Error detectando cuellos de botella: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/report", response_class=PlainTextResponse)
async def get_report(
    profiler: PerformanceProfiler = Depends(get_performance_profiler)
):
    """Generar reporte de rendimiento"""
    try:
        report = profiler.generate_report()
        return report
    except Exception as e:
        logger.error(f"Error generando reporte: {e}")
        raise HTTPException(status_code=500, detail=str(e))
















