"""
API de health checks avanzados
"""

import logging
import time
import psutil
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from config.settings import settings
from utils.prometheus_metrics import (
    music_generation_active,
    active_websocket_connections,
    music_generation_queue_size
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/health",
    tags=["health"]
)


class HealthStatus(BaseModel):
    """Modelo de respuesta de health check"""
    status: str
    timestamp: float
    version: str
    uptime_seconds: float


class DetailedHealthStatus(HealthStatus):
    """Modelo de respuesta de health check detallado"""
    services: Dict[str, Any]
    system: Dict[str, Any]
    metrics: Dict[str, Any]


# Tiempo de inicio de la aplicación
_start_time = time.time()


def get_uptime() -> float:
    """Obtiene el tiempo de actividad en segundos"""
    return time.time() - _start_time


@router.get("", response_model=HealthStatus)
async def health_check() -> HealthStatus:
    """
    Health check básico
    
    Returns:
        HealthStatus con estado básico de la aplicación
    """
    return HealthStatus(
        status="healthy",
        timestamp=time.time(),
        version=getattr(settings, 'app_version', '1.0.0'),
        uptime_seconds=get_uptime()
    )


@router.get("/detailed", response_model=DetailedHealthStatus)
async def detailed_health_check() -> DetailedHealthStatus:
    """
    Health check detallado con información de servicios y sistema
    
    Returns:
        DetailedHealthStatus con información completa
    """
    try:
        # Información de servicios
        services = {}
        
        # Verificar servicios críticos
        try:
            from services.song_service import get_song_service
            song_service = get_song_service()
            services["song_service"] = {
                "status": "healthy" if song_service else "unavailable",
                "type": type(song_service).__name__
            }
        except Exception as e:
            services["song_service"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Verificar caché
        try:
            from utils.cache_manager import MusicCache
            cache = MusicCache()
            services["cache"] = {
                "status": "healthy",
                "cache_dir": getattr(cache, 'cache_dir', 'unknown')
            }
        except Exception as e:
            services["cache"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Información del sistema
        system_info = {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory": {
                "total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                "available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
                "used_percent": psutil.virtual_memory().percent
            },
            "disk": {
                "total_gb": round(psutil.disk_usage('/').total / (1024**3), 2),
                "used_gb": round(psutil.disk_usage('/').used / (1024**3), 2),
                "free_percent": round(psutil.disk_usage('/').free / psutil.disk_usage('/').total * 100, 2)
            }
        }
        
        # Métricas actuales
        metrics = {
            "active_generations": music_generation_active._value.get(),
            "websocket_connections": active_websocket_connections._value.get(),
            "generation_queue_size": music_generation_queue_size._value.get()
        }
        
        # Determinar estado general
        overall_status = "healthy"
        if any(s.get("status") == "unhealthy" for s in services.values()):
            overall_status = "degraded"
        if system_info["memory"]["used_percent"] > 90:
            overall_status = "degraded"
        if system_info["disk"]["free_percent"] < 10:
            overall_status = "degraded"
        
        return DetailedHealthStatus(
            status=overall_status,
            timestamp=time.time(),
            version=getattr(settings, 'app_version', '1.0.0'),
            uptime_seconds=get_uptime(),
            services=services,
            system=system_info,
            metrics=metrics
        )
        
    except Exception as e:
        logger.error(f"Error in detailed health check: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Health check failed: {str(e)}"
        )


@router.get("/ready")
async def readiness_check() -> Dict[str, Any]:
    """
    Readiness check - verifica si la aplicación está lista para recibir tráfico
    
    Returns:
        Dict con estado de readiness
    """
    try:
        # Verificar servicios críticos
        from services.song_service import get_song_service
        song_service = get_song_service()
        
        if not song_service:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Song service not available"
            )
        
        return {
            "ready": True,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.warning(f"Readiness check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Not ready: {str(e)}"
        )


@router.get("/live")
async def liveness_check() -> Dict[str, Any]:
    """
    Liveness check - verifica si la aplicación está viva
    
    Returns:
        Dict con estado de liveness
    """
    return {
        "alive": True,
        "timestamp": time.time(),
        "uptime_seconds": get_uptime()
    }

