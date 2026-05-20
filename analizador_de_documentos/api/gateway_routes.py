"""
Rutas para API Gateway
======================

Endpoints para gestión de API Gateway.
"""

import logging
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.api_gateway import (
    get_api_gateway,
    APIGateway,
    RoutingStrategy
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/gateway",
    tags=["API Gateway"]
)


class RegisterServiceRequest(BaseModel):
    """Request para registrar servicio"""
    service_name: str = Field(..., description="Nombre del servicio")
    service_id: str = Field(..., description="ID del servicio")
    url: str = Field(..., description="URL del servicio")
    weight: int = Field(1, description="Peso para routing")
    strategy: str = Field("round_robin", description="Estrategia de routing")


@router.post("/services")
async def register_service(
    request: RegisterServiceRequest,
    gateway: APIGateway = Depends(get_api_gateway)
):
    """Registrar servicio en gateway"""
    try:
        strategy = RoutingStrategy(request.strategy)
        gateway.register_service(
            request.service_name,
            request.service_id,
            request.url,
            request.weight,
            strategy
        )
        
        return {"status": "registered", "service_name": request.service_name}
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Estrategia inválida: {request.strategy}")
    except Exception as e:
        logger.error(f"Error registrando servicio: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/services/{service_name}/endpoint")
async def get_endpoint(
    service_name: str,
    gateway: APIGateway = Depends(get_api_gateway)
):
    """Obtener endpoint según estrategia de routing"""
    endpoint = gateway.get_endpoint(service_name)
    
    if not endpoint:
        raise HTTPException(status_code=404, detail="Servicio no encontrado o sin endpoints activos")
    
    return {
        "service_id": endpoint.service_id,
        "url": endpoint.url,
        "weight": endpoint.weight
    }


@router.get("/services/{service_name}/health")
async def get_service_health(
    service_name: str,
    gateway: APIGateway = Depends(get_api_gateway)
):
    """Obtener salud del servicio"""
    health = gateway.get_service_health(service_name)
    
    if health.get("status") == "unknown":
        raise HTTPException(status_code=404, detail="Servicio no encontrado")
    
    return health


@router.post("/services/{service_name}/endpoints/{service_id}/used")
async def mark_endpoint_used(
    service_name: str,
    service_id: str,
    gateway: APIGateway = Depends(get_api_gateway)
):
    """Marcar endpoint como usado"""
    gateway.mark_endpoint_used(service_name, service_id)
    return {"status": "marked"}


@router.post("/services/{service_name}/endpoints/{service_id}/failed")
async def mark_endpoint_failed(
    service_name: str,
    service_id: str,
    gateway: APIGateway = Depends(get_api_gateway)
):
    """Marcar endpoint como fallido"""
    gateway.mark_endpoint_failed(service_name, service_id)
    return {"status": "marked"}
















