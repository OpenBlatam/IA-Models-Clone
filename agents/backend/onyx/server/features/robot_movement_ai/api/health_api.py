"""
Health API Endpoints
====================

Endpoints para health monitor y graceful shutdown.
Optimizado siguiendo mejores prácticas de FastAPI.
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from ..core.health_monitor import get_health_monitor
from ..core.graceful_shutdown import get_graceful_shutdown_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/health", tags=["health"])


class HealthCheckItem(BaseModel):
    """Modelo para un health check individual."""
    check_id: str
    name: str
    enabled: bool
    interval: float
    last_status: str
    last_check: Optional[datetime] = None


class HealthCheckResponse(BaseModel):
    """Respuesta para listado de health checks."""
    checks: List[HealthCheckItem]
    count: int


class ShutdownHandlerItem(BaseModel):
    """Modelo para un handler de shutdown."""
    handler_id: str
    name: str
    priority: int
    enabled: bool
    timeout: float


class ShutdownHandlersResponse(BaseModel):
    """Respuesta para listado de shutdown handlers."""
    handlers: List[ShutdownHandlerItem]
    count: int


class ShutdownRequest(BaseModel):
    """Request model para shutdown graceful."""
    timeout: float = Field(default=60.0, gt=0.0, le=300.0, description="Timeout en segundos")


class HealthResponse(BaseModel):
    """Respuesta de health check."""
    status: str
    timestamp: datetime
    checks: Dict[str, Any]
    metrics: Dict[str, Any]


async def _get_health_monitor():
    """Obtener health monitor con validación."""
    monitor = get_health_monitor()
    if monitor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Health monitor not available"
        )
    return monitor


async def _get_shutdown_manager():
    """Obtener shutdown manager con validación."""
    manager = get_graceful_shutdown_manager()
    if manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Shutdown manager not available"
        )
    return manager


@router.get("/", response_model=HealthResponse)
async def get_health() -> HealthResponse:
    """
    Obtener reporte de salud del sistema.
    
    Returns:
        HealthResponse con el estado del sistema
    """
    monitor = await _get_health_monitor()
    report = await monitor.get_health_report()
    
    return HealthResponse(
        status=report.overall_status.value,
        timestamp=report.timestamp,
        checks=report.checks,
        metrics=report.metrics
    )


@router.get("/checks", response_model=HealthCheckResponse)
async def list_health_checks() -> HealthCheckResponse:
    """
    Listar todos los health checks configurados.
    
    Returns:
        HealthCheckResponse con la lista de checks
    """
    monitor = await _get_health_monitor()
    checks = monitor.list_checks()
    
    check_items = [
        HealthCheckItem(
            check_id=c.check_id,
            name=c.name,
            enabled=c.enabled,
            interval=c.interval,
            last_status=c.last_status.value,
            last_check=c.last_check
        )
        for c in checks
    ]
    
    return HealthCheckResponse(checks=check_items, count=len(check_items))


@router.post("/shutdown")
async def graceful_shutdown(request: ShutdownRequest) -> Dict[str, Any]:
    """
    Iniciar shutdown graceful del sistema.
    
    Args:
        request: ShutdownRequest con timeout configurado
        
    Returns:
        Dict con el resultado del shutdown
    """
    manager = await _get_shutdown_manager()
    result = await manager.shutdown(timeout=request.timeout)
    return result


@router.get("/shutdown/handlers", response_model=ShutdownHandlersResponse)
async def list_shutdown_handlers() -> ShutdownHandlersResponse:
    """
    Listar todos los handlers de shutdown configurados.
    
    Returns:
        ShutdownHandlersResponse con la lista de handlers
    """
    manager = await _get_shutdown_manager()
    handlers = manager.list_handlers()
    
    handler_items = [
        ShutdownHandlerItem(
            handler_id=h.handler_id,
            name=h.name,
            priority=h.priority,
            enabled=h.enabled,
            timeout=h.timeout
        )
        for h in handlers
    ]
    
    return ShutdownHandlersResponse(handlers=handler_items, count=len(handler_items))
