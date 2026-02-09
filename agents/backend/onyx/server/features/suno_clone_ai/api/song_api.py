"""
Router principal para la API de canciones Suno Clone AI

Este módulo actúa como punto de entrada principal que importa y registra
todos los routers modulares especializados siguiendo el patrón de arquitectura
modular y separación de responsabilidades.

Características Principales:
============================

- Arquitectura Modular: Cada funcionalidad en su propio módulo
- Optimización: Caching, lazy loading, async operations
- Validación Robusta: Validación de UUIDs, tipos y rangos
- Manejo de Errores: Errores específicos y mensajes claros
- Documentación: Docstrings completos con ejemplos
- Escalabilidad: Diseñado para crecer sin acoplamiento

Versión de API: v1.0.0
Última actualización: 2024
"""

import logging
from typing import Dict, Any
from fastapi import APIRouter

from .router_registry import get_registry, register_core_routes, register_optional_routes

logger = logging.getLogger(__name__)

# Router principal
router = APIRouter(
    prefix="/suno",
    tags=["suno"],
    responses={
        400: {"description": "Bad request - Parámetros inválidos"},
        401: {"description": "Unauthorized - Autenticación requerida"},
        403: {"description": "Forbidden - Sin permisos"},
        404: {"description": "Not found - Recurso no encontrado"},
        409: {"description": "Conflict - Conflicto de estado"},
        422: {"description": "Unprocessable entity - Validación fallida"},
        429: {"description": "Too many requests - Rate limit excedido"},
        500: {"description": "Internal server error - Error del servidor"},
        503: {"description": "Service unavailable - Servicio no disponible"},
        507: {"description": "Insufficient storage - Almacenamiento insuficiente"}
    }
)


@router.get("", include_in_schema=True)
async def api_info() -> Dict[str, Any]:
    """
    Información general de la API Suno Clone AI.
    
    Proporciona información sobre versiones, endpoints disponibles,
    y estado del servicio.
    
    Returns:
        Diccionario con información de la API
    """
    try:
        from ..config.settings import settings
        
        return {
            "name": "Suno Clone AI API",
            "version": "1.0.0",
            "description": "API completa para generación y gestión de música con IA",
            "status": "active",
            "endpoints": {
                "generation": "/suno/generate",
                "songs": "/suno/songs",
                "search": "/suno/search",
                "health": "/suno/health",
                "metrics": "/suno/metrics",
                "models": "/suno/models",
                "performance": "/suno/performance"
            },
            "features": [
                "Generación de música con IA",
                "Procesamiento y edición de audio",
                "Sistema de búsqueda avanzada",
                "Recomendaciones inteligentes",
                "Gestión de playlists",
                "Sistema de ratings y favoritos",
                "Compartición segura",
                "Exportación de metadatos",
                "Estadísticas y métricas"
            ],
            "documentation": "/docs",
            "openapi_schema": "/openapi.json",
            "settings": {
                "default_model": settings.music_model,
                "max_audio_length": settings.max_audio_length,
                "default_duration": settings.default_duration
            }
        }
    except Exception as e:
        logger.error(f"Error getting API info: {e}", exc_info=True)
        return {
            "name": "Suno Clone AI API",
            "version": "1.0.0",
            "status": "error",
            "error": str(e)
        }


@router.get("/version")
async def get_api_version() -> Dict[str, Any]:
    """
    Obtiene la versión actual de la API.
    
    Returns:
        Información de versión
    """
    return {
        "version": "1.0.0",
        "api_version": "v1",
        "build_date": "2024",
        "compatibility": {
            "min_client_version": "1.0.0",
            "recommended_client_version": "1.0.0"
        }
    }


# Initialize router registry and register all routes
_registry = get_registry()
register_core_routes(_registry)
register_optional_routes(_registry)
_registry.apply_to(router)

logger.info("Suno Clone AI API router initialized with all sub-routers")

__all__ = ["router"]
