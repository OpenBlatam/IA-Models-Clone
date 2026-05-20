"""
Rutas para Integración Cloud
=============================

Endpoints para integración con servicios cloud.
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.cloud_integration import (
    get_cloud_integration,
    CloudIntegration,
    CloudProvider
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/cloud",
    tags=["Cloud Integration"]
)


class RegisterServiceRequest(BaseModel):
    """Request para registrar servicio cloud"""
    service_id: str = Field(..., description="ID del servicio")
    provider: str = Field(..., description="Proveedor (aws, azure, gcp, custom)")
    service_type: str = Field(..., description="Tipo de servicio")
    config: Dict[str, Any] = Field(..., description="Configuración")


class SyncRequest(BaseModel):
    """Request para sincronizar"""
    service_id: str = Field(..., description="ID del servicio")
    data: Any = Field(..., description="Datos a sincronizar")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadatos")


@router.post("/services")
async def register_service(
    request: RegisterServiceRequest,
    integration: CloudIntegration = Depends(get_cloud_integration)
):
    """Registrar servicio cloud"""
    try:
        provider = CloudProvider(request.provider)
        service = integration.register_service(
            request.service_id,
            provider,
            request.service_type,
            request.config
        )
        
        return {
            "status": "registered",
            "service_id": service.service_id,
            "provider": service.provider.value
        }
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Proveedor inválido: {request.provider}")
    except Exception as e:
        logger.error(f"Error registrando servicio: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sync")
async def sync_to_cloud(
    request: SyncRequest,
    integration: CloudIntegration = Depends(get_cloud_integration)
):
    """Sincronizar datos a cloud"""
    try:
        success = integration.sync_to_cloud(
            request.service_id,
            request.data,
            request.metadata
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Servicio no encontrado")
        
        return {"status": "synced", "service_id": request.service_id}
    except Exception as e:
        logger.error(f"Error sincronizando: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/services")
async def list_services(
    integration: CloudIntegration = Depends(get_cloud_integration)
):
    """Listar servicios cloud"""
    services = integration.list_services()
    return {"services": services}


@router.get("/sync/history")
async def get_sync_history(
    service_id: Optional[str] = None,
    limit: int = 100,
    integration: CloudIntegration = Depends(get_cloud_integration)
):
    """Obtener historial de sincronización"""
    history = integration.get_sync_history(service_id, limit)
    return {"history": history}
















