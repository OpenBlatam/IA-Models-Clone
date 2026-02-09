"""
Monitoring Endpoint
===================
Endpoint para monitoreo y observabilidad.
"""

from fastapi import APIRouter
from typing import Dict, Any
from datetime import datetime
from ...utils.monitoring import PerformanceMonitor, get_system_info

router = APIRouter(prefix="/api/v1/monitoring", tags=["monitoring"])

# Instancia global del monitor
performance_monitor = PerformanceMonitor()


@router.get("/performance")
async def get_performance_stats() -> Dict[str, Any]:
    """
    Obtener estadísticas de rendimiento.
    
    Returns:
        Estadísticas de rendimiento
    """
    return {
        "success": True,
        "stats": performance_monitor.get_stats(),
        "timestamp": datetime.now().isoformat()
    }


@router.get("/system")
async def get_system_monitoring() -> Dict[str, Any]:
    """
    Obtener información del sistema.
    
    Returns:
        Información del sistema
    """
    return {
        "success": True,
        "system": get_system_info(),
        "timestamp": datetime.now().isoformat()
    }


@router.get("/health/detailed")
async def get_detailed_health() -> Dict[str, Any]:
    """
    Obtener health check detallado.
    
    Returns:
        Health check detallado
    """
    from ...config.app_config import get_config
    
    config = get_config()
    stats = performance_monitor.get_stats()
    
    return {
        "status": "healthy",
        "service": config.app_name,
        "version": config.app_version,
        "performance": stats,
        "system": get_system_info(),
        "timestamp": datetime.now().isoformat()
    }

