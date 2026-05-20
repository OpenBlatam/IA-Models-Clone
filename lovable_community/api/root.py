"""
Root and info endpoints for Lovable Community API

Provides basic API information and root endpoint.
"""

import logging
from fastapi import APIRouter

from ..config import settings

logger = logging.getLogger(__name__)

router = APIRouter(
    tags=["info"]
)


@router.get(
    "/",
    summary="Root endpoint",
    description="Endpoint raíz que devuelve información básica de la API"
)
async def root():
    """
    Root endpoint of the API.
    
    Returns:
        Dictionary with basic API information
    """
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "docs": "/docs",
        "redoc": "/redoc"
    }


@router.get(
    "/info",
    summary="API Information",
    description="Obtiene información detallada sobre la API"
)
async def api_info():
    """
    Detailed API information.
    
    Returns:
        Dictionary with complete API information
    """
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "description": "API para comunidad de chats estilo Lovable con sistema de ranking, remixes y votación",
        "features": [
            "Publicar chats",
            "Remixar chats",
            "Sistema de votación",
            "Ranking inteligente",
            "Búsqueda avanzada",
            "Analytics",
            "Perfiles de usuario",
            "Chats trending",
            "Operaciones en lote"
        ],
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "health": "/health",
            "info": "/info"
        },
        "database": {
            "type": "sqlite" if "sqlite" in settings.database_url else "other",
            "echo": settings.database_echo
        },
        "settings": {
            "debug": settings.debug,
            "cors_enabled": True,
            "rate_limit_enabled": settings.rate_limit_enabled
        }
    }

__all__ = ["router"]



