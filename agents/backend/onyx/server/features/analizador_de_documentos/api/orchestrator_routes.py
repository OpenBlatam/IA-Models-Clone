"""
Rutas para Orquestador de Servicios
=====================================

Endpoints para orquestación de servicios.
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.service_orchestrator import (
    get_service_orchestrator,
    ServiceOrchestrator,
    ServiceStatus
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/orchestrator",
    tags=["Service Orchestrator"]
)


class RegisterServiceRequest(BaseModel):
    """Request para registrar servicio"""
    service_id: str = Field(..., description="ID del servicio")
    name: str = Field(..., description="Nombre del servicio")
    url: str = Field(..., description="URL del servicio")
    dependencies: Optional[List[str]] = Field(None, description="Dependencias")


@router.post("/services")
async def register_service(
    request: RegisterServiceRequest,
    orchestrator: ServiceOrchestrator = Depends(get_service_orchestrator)
):
    """Registrar servicio"""
    try:
        service = orchestrator.register_service(
            request.service_id,
            request.name,
            request.url,
            request.dependencies
        )
        
        return {
            "status": "registered",
            "service_id": service.service_id,
            "name": service.name,
            "status": service.status.value
        }
    except Exception as e:
        logger.error(f"Error registrando servicio: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/services/{service_id}/start")
async def start_service(
    service_id: str,
    orchestrator: ServiceOrchestrator = Depends(get_service_orchestrator)
):
    """Iniciar servicio"""
    success = orchestrator.start_service(service_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Servicio no encontrado")
    
    return {"status": "started", "service_id": service_id}


@router.post("/services/{service_id}/stop")
async def stop_service(
    service_id: str,
    orchestrator: ServiceOrchestrator = Depends(get_service_orchestrator)
):
    """Detener servicio"""
    success = orchestrator.stop_service(service_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Servicio no encontrado")
    
    return {"status": "stopped", "service_id": service_id}


@router.get("/services")
async def get_all_services(
    orchestrator: ServiceOrchestrator = Depends(get_service_orchestrator)
):
    """Obtener estado de todos los servicios"""
    status = orchestrator.get_all_services_status()
    return {"services": status}


@router.get("/services/{service_id}")
async def get_service_status(
    service_id: str,
    orchestrator: ServiceOrchestrator = Depends(get_service_orchestrator)
):
    """Obtener estado de servicio específico"""
    status = orchestrator.get_service_status(service_id)
    
    if not status:
        raise HTTPException(status_code=404, detail="Servicio no encontrado")
    
    return status














