"""
Health Routes
=============

Endpoints de health check mejorados con validación de dependencias.
"""

import logging
from fastapi import APIRouter
from typing import Dict, Any

from ...core.health_check import HealthChecker

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/health", tags=["health"])

health_checker = HealthChecker()


@router.get("/")
async def health_check() -> Dict[str, Any]:
    """
    Health check general con validación de dependencias.
    
    Returns:
        Estado de salud del sistema con checks de dependencias
    """
    return await health_checker.get_health_status()


@router.get("/ready")
async def readiness_check() -> Dict[str, Any]:
    """
    Readiness check - verifica si el servicio está listo para recibir tráfico.
    
    Returns:
        Estado de readiness
    """
    health = await health_checker.get_health_status()
    
    # Service is ready if database is healthy
    db_status = health["checks"].get("database", {}).get("status", "unknown")
    
    return {
        "status": "ready" if db_status == "healthy" else "not_ready",
        "database": db_status,
    }


@router.get("/live")
async def liveness_check() -> Dict[str, Any]:
    """
    Liveness check - verifica si el servicio está vivo.
    
    Returns:
        Estado de liveness
    """
    return {
        "status": "alive",
    }

