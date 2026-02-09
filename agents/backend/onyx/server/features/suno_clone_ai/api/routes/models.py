"""
Endpoints para gestión de modelos y caché

Este módulo proporciona información sobre:
- Modelos de generación disponibles
- Información detallada de cada modelo
- Estadísticas del caché
- Operaciones de limpieza de caché

Características:
- Información completa de modelos
- Estadísticas de caché en tiempo real
- Operaciones de mantenimiento
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, status, Depends

from ...config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/models",
    tags=["models"],
    responses={
        404: {"description": "Not found - Modelo no encontrado"},
        500: {"description": "Internal server error"}
    }
)


@router.get(
    "",
    summary="Listar modelos",
    description="Lista todos los modelos de generación de música disponibles"
)
async def list_models() -> Dict[str, Any]:
    """
    Lista los modelos de generación de música disponibles.
    
    Returns:
        Lista de modelos con información básica y modelo actual
    
    Example:
        ```
        GET /suno/models
        ```
    """
    try:
        models_list = [
            {
                "id": "facebook/musicgen-small",
                "name": "MusicGen Small",
                "description": "Modelo pequeño y rápido",
                "size": "~300MB",
                "speed": "Fast",
                "quality": "Good"
            },
            {
                "id": "facebook/musicgen-medium",
                "name": "MusicGen Medium",
                "description": "Modelo balanceado (por defecto)",
                "size": "~1.5GB",
                "speed": "Medium",
                "quality": "Very Good"
            },
            {
                "id": "facebook/musicgen-large",
                "name": "MusicGen Large",
                "description": "Modelo grande con mejor calidad",
                "size": "~3GB",
                "speed": "Slow",
                "quality": "Excellent"
            }
        ]
        
        return {
            "models": models_list,
            "current": settings.music_model,
            "total_models": len(models_list)
        }
    except Exception as e:
        logger.error(f"Error listing models: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing models: {str(e)}"
        )


@router.get(
    "/{model_id}",
    summary="Información de modelo",
    description="Obtiene información detallada de un modelo específico"
)
async def get_model_info(model_id: str) -> Dict[str, Any]:
    """
    Obtiene información detallada de un modelo específico.
    
    Args:
        model_id: ID del modelo
    
    Returns:
        Información completa del modelo incluyendo especificaciones
    
    Raises:
        HTTPException 404: Si el modelo no existe
    
    Example:
        ```
        GET /suno/models/facebook/musicgen-medium
        ```
    """
    try:
        models = {
            "facebook/musicgen-small": {
                "id": "facebook/musicgen-small",
                "name": "MusicGen Small",
                "description": "Modelo pequeño y rápido",
                "size": "~300MB",
                "speed": "Fast",
                "quality": "Good",
                "recommended_for": "Desarrollo y pruebas rápidas",
                "parameters": "~300M",
                "latency": "Low",
                "memory_usage": "Low"
            },
            "facebook/musicgen-medium": {
                "id": "facebook/musicgen-medium",
                "name": "MusicGen Medium",
                "description": "Modelo balanceado",
                "size": "~1.5GB",
                "speed": "Medium",
                "quality": "Very Good",
                "recommended_for": "Producción general (por defecto)",
                "parameters": "~1.5B",
                "latency": "Medium",
                "memory_usage": "Medium"
            },
            "facebook/musicgen-large": {
                "id": "facebook/musicgen-large",
                "name": "MusicGen Large",
                "description": "Modelo grande con mejor calidad",
                "size": "~3GB",
                "speed": "Slow",
                "quality": "Excellent",
                "recommended_for": "Producción de alta calidad",
                "parameters": "~3B",
                "latency": "High",
                "memory_usage": "High"
            }
        }
        
        if model_id not in models:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model {model_id} not found"
            )
        
        model_info = models[model_id].copy()
        model_info["is_current"] = (model_id == settings.music_model)
        
        return model_info
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model info: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting model info: {str(e)}"
        )


@router.get(
    "/cache/stats",
    summary="Estadísticas de caché",
    description="Obtiene estadísticas detalladas del caché"
)
async def get_cache_stats() -> Dict[str, Any]:
    """
    Obtiene estadísticas detalladas del caché.
    
    Returns:
        Estadísticas del caché incluyendo hit rate, tamaño, etc.
    
    Example:
        ```
        GET /suno/models/cache/stats
        ```
    """
    try:
        from datetime import datetime
        from ...core.cache_manager import get_cache_manager
        
        cache_mgr = get_cache_manager()
        stats = cache_mgr.stats()
        
        # Agregar información adicional
        return {
            **stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting cache stats: {str(e)}"
        )


@router.delete(
    "/cache/clear",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Limpiar caché",
    description="Limpia el caché de canciones generadas"
)
async def clear_cache() -> None:
    """
    Limpia el caché de canciones generadas.
    
    Raises:
        HTTPException 500: Si hay error al limpiar el caché
    
    Example:
        ```
        DELETE /suno/models/cache/clear
        ```
    """
    try:
        from ...core.cache_manager import get_cache_manager
        
        cache_manager = get_cache_manager()
        cache_manager.clear()
        logger.info("Cache cleared successfully")
    except Exception as e:
        logger.error(f"Error clearing cache: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error clearing cache: {str(e)}"
        )

