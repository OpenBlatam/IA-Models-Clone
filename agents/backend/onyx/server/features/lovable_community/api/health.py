"""
Health check endpoint para la comunidad Lovable

Proporciona información sobre el estado de la aplicación.
"""

import logging
from datetime import datetime
from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session
from sqlalchemy import text

from ..dependencies import get_db
from ..config import settings

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/health",
    tags=["health"]
)


@router.get(
    "",
    status_code=status.HTTP_200_OK,
    summary="Health check",
    description="Verifica el estado de la aplicación y la base de datos"
)
async def health_check(db: Session = Depends(get_db)):
    """
    Health check endpoint (optimizado).
    
    Verifica:
    - Estado de la aplicación
    - Conexión a la base de datos
    - Versión de la aplicación
    
    Returns:
        Diccionario con información de salud
        
    Note:
        Returns 200 even if database is disconnected (for monitoring purposes).
        Use /ready endpoint for readiness checks.
    """
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": settings.app_version,
        "app_name": settings.app_name,
        "checks": {
            "database": "unknown"
        }
    }
    
    # Verificar conexión a base de datos
    try:
        result = db.execute(text("SELECT 1"))
        result.fetchone()  # Consume result to verify connection
        health_status["checks"]["database"] = "connected"
    except Exception as e:
        logger.error(
            "Database health check failed",
            error=str(e),
            error_type=type(e).__name__
        )
        health_status["status"] = "unhealthy"
        health_status["checks"]["database"] = "disconnected"
        health_status["error"] = str(e) if settings.debug else "Database connection failed"
    
    return health_status


@router.get(
    "/ready",
    summary="Readiness check",
    description="Verifica si la aplicación está lista para recibir tráfico"
)
async def readiness_check(db: Session = Depends(get_db)):
    """
    Readiness check endpoint.
    
    Verifica que todos los servicios estén listos.
    Retorna 503 si algún servicio no está listo.
    
    Returns:
        Diccionario con estado de readiness
        
    Raises:
        HTTPException: 503 if not ready
    """
    from fastapi import HTTPException
    
    ready = True
    checks = {}
    errors = {}
    
    # Verificar base de datos
    try:
        result = db.execute(text("SELECT 1"))
        result.fetchone()  # Consume result to verify connection
        checks["database"] = "ready"
    except Exception as e:
        logger.error(
            "Database readiness check failed",
            error=str(e),
            error_type=type(e).__name__
        )
        checks["database"] = "not_ready"
        errors["database"] = str(e) if settings.debug else "Database connection failed"
        ready = False
    
    if not ready:
        response = {
            "ready": False,
            "checks": checks,
            "timestamp": datetime.utcnow().isoformat()
        }
        if errors:
            response["errors"] = errors
        
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=response
        )
    
    return {
        "ready": True,
        "checks": checks,
        "timestamp": datetime.utcnow().isoformat()
    }

