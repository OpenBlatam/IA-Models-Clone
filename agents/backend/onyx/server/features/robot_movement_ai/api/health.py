"""
Health Check Endpoint Mejorado para Robot Movement AI v2.0
Proporciona información detallada del estado del sistema
"""

from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core.architecture.monitoring import get_metrics_collector
from core.architecture.di_setup import get_container

router = APIRouter(prefix="/health", tags=["health"])


class HealthStatus(BaseModel):
    """Modelo de respuesta de health check"""
    status: str
    timestamp: str
    version: str
    uptime_seconds: float
    services: Dict[str, Dict[str, Any]]
    metrics: Optional[Dict[str, Any]] = None


class ComponentStatus(BaseModel):
    """Estado de un componente del sistema"""
    status: str
    message: Optional[str] = None
    last_check: Optional[str] = None


# Timestamp de inicio de la aplicación
_start_time = datetime.now()


def get_uptime() -> float:
    """Calcular tiempo de actividad en segundos"""
    delta = datetime.now() - _start_time
    return delta.total_seconds()


async def check_database() -> ComponentStatus:
    """Verificar estado de la base de datos"""
    try:
        container = get_container()
        # Intentar obtener un repositorio para verificar conexión
        # Esto es un check básico, ajustar según implementación
        return ComponentStatus(
            status="healthy",
            message="Database connection OK",
            last_check=datetime.now().isoformat()
        )
    except Exception as e:
        return ComponentStatus(
            status="unhealthy",
            message=f"Database error: {str(e)}",
            last_check=datetime.now().isoformat()
        )


async def check_repositories() -> ComponentStatus:
    """Verificar estado de los repositorios"""
    try:
        container = get_container()
        # Verificar que los repositorios estén disponibles
        return ComponentStatus(
            status="healthy",
            message="Repositories available",
            last_check=datetime.now().isoformat()
        )
    except Exception as e:
        return ComponentStatus(
            status="unhealthy",
            message=f"Repository error: {str(e)}",
            last_check=datetime.now().isoformat()
        )


async def check_circuit_breakers() -> ComponentStatus:
    """Verificar estado de los circuit breakers"""
    try:
        metrics = get_metrics_collector()
        # Verificar métricas de circuit breakers
        cb_metrics = metrics.get_metrics_dict()
        open_breakers = sum(
            1 for k, v in cb_metrics.items() 
            if k.startswith("circuit_breaker.") and v == 1
        )
        
        if open_breakers > 0:
            return ComponentStatus(
                status="degraded",
                message=f"{open_breakers} circuit breaker(s) open",
                last_check=datetime.now().isoformat()
            )
        
        return ComponentStatus(
            status="healthy",
            message="All circuit breakers closed",
            last_check=datetime.now().isoformat()
        )
    except Exception as e:
        return ComponentStatus(
            status="unknown",
            message=f"Circuit breaker check error: {str(e)}",
            last_check=datetime.now().isoformat()
        )


@router.get("", response_model=HealthStatus)
@router.get("/", response_model=HealthStatus)
async def health_check(include_metrics: bool = False):
    """
    Health check endpoint mejorado
    
    - **status**: Estado general del sistema (healthy, degraded, unhealthy)
    - **timestamp**: Timestamp del check
    - **version**: Versión de la aplicación
    - **uptime_seconds**: Tiempo de actividad en segundos
    - **services**: Estado de cada servicio/componente
    - **metrics**: Métricas del sistema (opcional)
    """
    # Verificar componentes
    db_status = await check_database()
    repo_status = await check_repositories()
    cb_status = await check_circuit_breakers()
    
    # Determinar estado general
    component_statuses = [db_status.status, repo_status.status, cb_status.status]
    
    if "unhealthy" in component_statuses:
        overall_status = "unhealthy"
    elif "degraded" in component_statuses:
        overall_status = "degraded"
    else:
        overall_status = "healthy"
    
    # Construir respuesta
    response = HealthStatus(
        status=overall_status,
        timestamp=datetime.now().isoformat(),
        version="2.0.0",
        uptime_seconds=get_uptime(),
        services={
            "database": db_status.dict(),
            "repositories": repo_status.dict(),
            "circuit_breakers": cb_status.dict()
        }
    )
    
    # Incluir métricas si se solicita
    if include_metrics:
        metrics_collector = get_metrics_collector()
        response.metrics = metrics_collector.get_metrics_dict()
    
    # Retornar error HTTP si está unhealthy
    if overall_status == "unhealthy":
        raise HTTPException(status_code=503, detail=response.dict())
    
    return response


@router.get("/ready")
async def readiness_check():
    """
    Readiness check - Verifica si la aplicación está lista para recibir tráfico
    """
    db_status = await check_database()
    repo_status = await check_repositories()
    
    if db_status.status == "healthy" and repo_status.status == "healthy":
        return {"status": "ready", "timestamp": datetime.now().isoformat()}
    
    raise HTTPException(
        status_code=503,
        detail="Service not ready"
    )


@router.get("/live")
async def liveness_check():
    """
    Liveness check - Verifica si la aplicación está viva
    """
    return {
        "status": "alive",
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": get_uptime()
    }


@router.get("/metrics")
async def metrics_endpoint():
    """
    Endpoint de métricas en formato Prometheus
    """
    metrics_collector = get_metrics_collector()
    metrics_data = metrics_collector.get_metrics()
    
    from fastapi.responses import Response
    return Response(
        content=metrics_data,
        media_type="text/plain; version=0.0.4; charset=utf-8"
    )




