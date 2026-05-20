"""
Health Routes - Rutas de salud y estado
"""

import logging
from datetime import datetime
from fastapi import APIRouter

from ...core.engine import get_export_engine

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
async def health_check():
    """
    Verificar salud del sistema.
    
    Returns:
        Estado de salud del sistema
    """
    try:
        engine = get_export_engine()
        
        # Verificar que el motor esté inicializado
        is_initialized = engine._initialized
        
        # Obtener estadísticas básicas
        stats = await engine.get_export_statistics()
        
        return {
            "status": "healthy" if is_initialized else "initializing",
            "timestamp": datetime.now().isoformat(),
            "version": "2.0.0",
            "initialized": is_initialized,
            "statistics": {
                "total_tasks": stats.get("total_tasks", 0),
                "active_tasks": stats.get("active_tasks", 0),
                "completed_tasks": stats.get("completed_tasks", 0),
                "failed_tasks": stats.get("failed_tasks", 0)
            }
        }
        
    except Exception as e:
        logger.error(f"Error en health check: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "version": "2.0.0",
            "error": str(e)
        }


@router.get("/")
async def root():
    """
    Endpoint raíz con información del sistema.
    
    Returns:
        Información básica del sistema
    """
    try:
        engine = get_export_engine()
        formats = engine.list_supported_formats()
        
        return {
            "name": "Export IA API",
            "version": "2.0.0",
            "description": "API para exportación de documentos con IA",
            "status": "running",
            "timestamp": datetime.now().isoformat(),
            "supported_formats": [fmt["format"] for fmt in formats],
            "endpoints": {
                "health": "/api/v1/health",
                "export": "/api/v1/export",
                "formats": "/api/v1/formats",
                "templates": "/api/v1/templates/{doc_type}",
                "docs": "/docs",
                "redoc": "/redoc"
            }
        }
        
    except Exception as e:
        logger.error(f"Error en endpoint raíz: {e}")
        return {
            "name": "Export IA API",
            "version": "2.0.0",
            "status": "error",
            "error": str(e)
        }




