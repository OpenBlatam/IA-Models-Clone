"""
Rutas para Monitor de Salud Avanzado
=====================================

Endpoints para monitoreo de salud avanzado.
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.health_monitor import (
    get_health_monitor,
    AdvancedHealthMonitor,
    HealthStatus
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/health-advanced",
    tags=["Advanced Health"]
)


class PerformHealthCheckRequest(BaseModel):
    """Request para realizar health check"""
    component: str = Field(..., description="Componente")
    status: str = Field(..., description="Estado (healthy, degraded, unhealthy, critical)")
    message: str = Field(..., description="Mensaje")
    metrics: Optional[Dict[str, Any]] = Field(None, description="Métricas")


@router.post("/check")
async def perform_health_check(
    request: PerformHealthCheckRequest,
    monitor: AdvancedHealthMonitor = Depends(get_health_monitor)
):
    """Realizar health check"""
    try:
        status = HealthStatus(request.status)
        monitor.perform_health_check(
            request.component,
            status,
            request.message,
            request.metrics
        )
        
        return {"status": "checked", "component": request.component}
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Estado inválido: {request.status}")
    except Exception as e:
        logger.error(f"Error realizando health check: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/overall")
async def get_overall_health(
    monitor: AdvancedHealthMonitor = Depends(get_health_monitor)
):
    """Obtener salud general del sistema"""
    health = monitor.get_overall_health()
    return health


@router.get("/component/{component}")
async def get_component_health(
    component: str,
    monitor: AdvancedHealthMonitor = Depends(get_health_monitor)
):
    """Obtener salud de componente específico"""
    check = monitor.get_component_health(component)
    
    if not check:
        raise HTTPException(status_code=404, detail="Componente no encontrado")
    
    return {
        "component": check.component,
        "status": check.status.value,
        "message": check.message,
        "timestamp": check.timestamp,
        "metrics": check.metrics
    }


@router.get("/history")
async def get_health_history(
    component: Optional[str] = None,
    hours: int = 24,
    monitor: AdvancedHealthMonitor = Depends(get_health_monitor)
):
    """Obtener historial de salud"""
    history = monitor.get_health_history(component, hours)
    return {"history": history}















