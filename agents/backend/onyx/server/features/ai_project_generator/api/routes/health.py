"""
Health Routes - Endpoints de health check
=========================================

Endpoints para verificar el estado de salud del sistema.
"""

import logging
from fastapi import APIRouter
from typing import Dict, Any

from ...core.robustness import get_health_checker, get_dependency_validator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/health", tags=["health"])


@router.get("")
async def health_check() -> Dict[str, Any]:
    """
    Health check general del sistema.
    
    Returns:
        Estado de salud del sistema
    """
    checker = get_health_checker()
    return await checker.check_all()


@router.get("/dependencies")
async def dependencies_check() -> Dict[str, Any]:
    """
    Health check de dependencias.
    
    Returns:
        Estado de todas las dependencias
    """
    validator = get_dependency_validator()
    return await validator.validate_all()


@router.get("/liveness")
async def liveness() -> Dict[str, str]:
    """
    Liveness probe - verifica que la aplicación está viva.
    
    Returns:
        Estado de liveness
    """
    return {"status": "alive"}


@router.get("/readiness")
async def readiness() -> Dict[str, Any]:
    """
    Readiness probe - verifica que la aplicación está lista para recibir tráfico.
    
    Returns:
        Estado de readiness
    """
    validator = get_dependency_validator()
    results = await validator.validate_all()
    
    if results["required_failed"]:
        return {
            "status": "not_ready",
            "reason": "Required dependencies unavailable",
            "details": results
        }
    
    return {
        "status": "ready",
        "details": results
    }















