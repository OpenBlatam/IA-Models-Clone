"""
Rutas para Multi-Tenancy
=========================

Endpoints para gestión de tenants.
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Header
from pydantic import BaseModel, Field

from ..core.multi_tenancy import get_multi_tenancy_manager, MultiTenancyManager

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/tenants",
    tags=["Multi-Tenancy"]
)


class CreateTenantRequest(BaseModel):
    """Request para crear tenant"""
    tenant_id: str = Field(..., description="ID único del tenant")
    name: str = Field(..., description="Nombre del tenant")
    config: Optional[Dict[str, Any]] = Field(None, description="Configuración")


@router.post("/")
async def create_tenant(
    request: CreateTenantRequest,
    manager: MultiTenancyManager = Depends(get_multi_tenancy_manager)
):
    """Crear nuevo tenant"""
    try:
        tenant = manager.register_tenant(
            request.tenant_id,
            request.name,
            request.config
        )
        
        return {
            "status": "created",
            "tenant": {
                "tenant_id": tenant.tenant_id,
                "name": tenant.name,
                "created_at": tenant.created_at
            }
        }
    except Exception as e:
        logger.error(f"Error creando tenant: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/")
async def list_tenants(
    manager: MultiTenancyManager = Depends(get_multi_tenancy_manager)
):
    """Listar todos los tenants"""
    return {"tenants": manager.list_tenants()}


@router.get("/{tenant_id}/stats")
async def get_tenant_stats(
    tenant_id: str,
    manager: MultiTenancyManager = Depends(get_multi_tenancy_manager)
):
    """Obtener estadísticas del tenant"""
    tenant = manager.get_tenant(tenant_id)
    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant no encontrado")
    
    stats = manager.get_tenant_stats(tenant_id)
    return {
        "tenant_id": tenant_id,
        "stats": stats
    }


@router.get("/{tenant_id}/config")
async def get_tenant_config(
    tenant_id: str,
    manager: MultiTenancyManager = Depends(get_multi_tenancy_manager)
):
    """Obtener configuración del tenant"""
    config = manager.get_tenant_config(tenant_id)
    if not config and not manager.get_tenant(tenant_id):
        raise HTTPException(status_code=404, detail="Tenant no encontrado")
    
    return {
        "tenant_id": tenant_id,
        "config": config
    }
















