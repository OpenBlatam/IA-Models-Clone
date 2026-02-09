"""
Endpoints para health checks
"""

import logging
import time
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, status

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/health",
    tags=["health"]
)


@router.get("")
async def health_check() -> Dict[str, Any]:
    """
    Health check del servicio.
    
    Verifica que los servicios principales estén disponibles.
    """
    health_status = {
        "status": "healthy",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "services": {}
    }
    
    # Verificar servicios
    try:
        from ...core.music_generator import get_music_generator
        music_generator = get_music_generator()
        health_status["services"]["music_generator"] = "available"
    except Exception as e:
        health_status["services"]["music_generator"] = f"error: {str(e)}"
        health_status["status"] = "degraded"
    
    try:
        from ...core.cache_manager import get_cache_manager
        cache_manager = get_cache_manager()
        health_status["services"]["cache_manager"] = "available"
    except Exception as e:
        health_status["services"]["cache_manager"] = f"error: {str(e)}"
        health_status["status"] = "degraded"
    
    try:
        from ...services.song_service import SongService
        song_service = SongService()
        health_status["services"]["song_service"] = "available"
    except Exception as e:
        health_status["services"]["song_service"] = f"error: {str(e)}"
        health_status["status"] = "unhealthy"
    
    return health_status


@router.get("/ready")
async def readiness_check() -> Dict[str, Any]:
    """
    Readiness check - verifica si el servicio está listo para recibir tráfico.
    """
    try:
        from ...services.song_service import SongService
        
        # Verificar que el servicio principal esté disponible
        song_service = SongService()
        
        return {
            "ready": True,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
        }
    except Exception as e:
        logger.error(f"Readiness check failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service not ready: {str(e)}"
        )


@router.get("/live")
async def liveness_check() -> Dict[str, Any]:
    """
    Liveness check - verifica que el servicio esté vivo.
    """
    return {
        "alive": True,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
    }

