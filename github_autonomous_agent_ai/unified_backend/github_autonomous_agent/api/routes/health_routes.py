"""
Health Routes - Rutas para health checks.
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Optional

from api.utils import handle_api_errors
from core.health.health_checker import HealthChecker
from config.logging_config import get_logger
from config.di_setup import get_service

router = APIRouter()
logger = get_logger(__name__)


def get_health_checker() -> HealthChecker:
    """Obtener health checker."""
    try:
        return get_service("health_checker")
    except Exception:
        raise HTTPException(status_code=503, detail="Health checker no disponible")


@router.get("/")
@handle_api_errors
async def health_check(
    health_checker: HealthChecker = Depends(get_health_checker)
):
    """
    Health check general.
    
    Returns:
        Estado de salud del sistema
    """
    results = health_checker.run_all_checks()
    
    # Retornar código de estado apropiado
    status_code = 200
    if results["status"] == "unhealthy":
        status_code = 503
    elif results["status"] == "degraded":
        status_code = 200  # Degraded pero aún funcional
    
    return results


@router.get("/{check_name}")
@handle_api_errors
async def specific_health_check(
    check_name: str,
    health_checker: HealthChecker = Depends(get_health_checker)
):
    """
    Health check específico.
    
    Args:
        check_name: Nombre del check
        
    Returns:
        Resultado del check específico
    """
    result = health_checker.run_check(check_name)
    
    status_code = 200
    if result.status.value == "unhealthy":
        status_code = 503
    elif result.status.value == "degraded":
        status_code = 200
    
    return {
        "name": result.name,
        "status": result.status.value,
        "message": result.message,
        "details": result.details,
        "timestamp": result.timestamp.isoformat()
    }



