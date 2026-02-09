"""
Debug Endpoints - Endpoints de debugging
========================================

Endpoints para debugging y troubleshooting.
"""

import logging
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Request, Query
from datetime import datetime, timedelta

from .debug_logger import get_debug_logger
from .error_tracker import get_error_tracker
from .profiler import get_profiler
from ..optimizations.memory_optimizations import MemoryOptimizer
from ..infrastructure.cache import get_cache_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/debug", tags=["debug"])


@router.get("/health")
async def debug_health():
    """Health check detallado para debugging"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "logger": "available",
            "error_tracker": "available",
            "profiler": "available"
        }
    }


@router.get("/errors")
async def get_errors(
    limit: int = Query(50, ge=1, le=1000),
    error_type: Optional[str] = None
):
    """Obtiene errores recientes"""
    tracker = get_error_tracker()
    
    if error_type:
        errors = tracker.get_errors_by_type(error_type)
    else:
        errors = tracker.get_recent_errors(limit)
    
    return {
        "total": len(errors),
        "errors": errors[-limit:]
    }


@router.get("/errors/stats")
async def get_error_stats():
    """Obtiene estadísticas de errores"""
    tracker = get_error_tracker()
    return tracker.get_error_stats()


@router.get("/errors/{error_key}")
async def get_error_group(error_key: str):
    """Obtiene grupo de errores similares"""
    tracker = get_error_tracker()
    errors = tracker.get_error_group(error_key)
    return {
        "error_key": error_key,
        "count": len(errors),
        "errors": errors
    }


@router.delete("/errors")
async def clear_errors():
    """Limpia historial de errores"""
    tracker = get_error_tracker()
    tracker.clear_errors()
    return {"message": "Errors cleared"}


@router.get("/memory")
async def get_memory_info():
    """Obtiene información de memoria"""
    optimizer = MemoryOptimizer()
    usage = optimizer.get_memory_usage()
    
    return {
        "memory_usage": usage,
        "timestamp": datetime.now().isoformat()
    }


@router.post("/memory/optimize")
async def optimize_memory():
    """Optimiza memoria"""
    optimizer = MemoryOptimizer()
    optimizer.optimize_memory()
    return {"message": "Memory optimized"}


@router.get("/cache/stats")
async def get_cache_stats():
    """Obtiene estadísticas de cache"""
    cache = get_cache_service()
    if hasattr(cache, "get_stats"):
        return cache.get_stats()
    return {"message": "Cache stats not available"}


@router.get("/cache/keys")
async def get_cache_keys(pattern: str = "*"):
    """Obtiene claves de cache (solo para debugging)"""
    # Esto es solo para desarrollo, no usar en producción
    return {"message": "Cache keys inspection not implemented for security"}


@router.get("/profiler/stats")
async def get_profiler_stats():
    """Obtiene estadísticas del profiler"""
    profiler = get_profiler()
    return profiler.get_stats()


@router.get("/logs")
async def get_recent_logs(limit: int = Query(100, ge=1, le=1000)):
    """Obtiene logs recientes (solo si están en memoria)"""
    # Esto requiere implementación de log buffer
    return {"message": "Log buffer not implemented"}


@router.get("/services")
async def get_services_status():
    """Obtiene estado de todos los servicios"""
    return {
        "cache": "available" if get_cache_service() else "unavailable",
        "logger": "available",
        "error_tracker": "available",
        "profiler": "available"
    }


@router.get("/config")
async def get_debug_config():
    """Obtiene configuración de debugging"""
    return {
        "debug_enabled": True,
        "log_level": "DEBUG",
        "error_tracking": True,
        "profiling": True
    }


def setup_debug_endpoints(app):
    """Configura endpoints de debugging en la app"""
    app.include_router(router)
    logger.info("Debug endpoints configured")















