"""
Integration Manager endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.integration_manager import IntegrationManagerService, IntegrationType

router = APIRouter()
integration_service = IntegrationManagerService()


@router.post("/create/{user_id}")
async def create_integration(
    user_id: str,
    integration_type: str,
    credentials: Dict[str, str],
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Crear nueva integración"""
    try:
        type_enum = IntegrationType(integration_type)
        integration = integration_service.create_integration(
            user_id, type_enum, credentials, config
        )
        return {
            "id": integration.id,
            "type": integration.integration_type.value,
            "status": integration.status.value,
            "error_message": integration.error_message,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sync/{integration_id}")
async def sync_integration(integration_id: str) -> Dict[str, Any]:
    """Sincronizar datos de integración"""
    try:
        result = integration_service.sync_integration(integration_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/user/{user_id}")
async def get_user_integrations(user_id: str) -> Dict[str, Any]:
    """Obtener integraciones del usuario"""
    try:
        integrations = integration_service.get_user_integrations(user_id)
        return {
            "user_id": user_id,
            "integrations": integrations,
            "total": len(integrations),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/deactivate/{integration_id}")
async def deactivate_integration(integration_id: str) -> Dict[str, Any]:
    """Desactivar integración"""
    try:
        success = integration_service.deactivate_integration(integration_id)
        return {"success": success, "integration_id": integration_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




