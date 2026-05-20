"""
Model Registry endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.model_registry import ModelRegistryService, ModelStatus

router = APIRouter()
registry_service = ModelRegistryService()


@router.post("/register")
async def register_model(
    model_id: str,
    name: str,
    version: str,
    model_type: str,
    status: str = "development",
    description: str = ""
) -> Dict[str, Any]:
    """Registrar nuevo modelo"""
    try:
        status_enum = ModelStatus(status)
        metadata = registry_service.register_model(
            model_id, name, version, model_type, status_enum, description
        )
        
        return {
            "model_id": metadata.model_id,
            "name": metadata.name,
            "version": metadata.version,
            "status": metadata.status.value,
            "created_at": metadata.created_at.isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list")
async def list_models(
    status: Optional[str] = None,
    model_type: Optional[str] = None
) -> Dict[str, Any]:
    """Listar modelos"""
    try:
        status_enum = ModelStatus(status) if status else None
        models = registry_service.list_models(status_enum, model_type)
        
        return {
            "models": [
                {
                    "model_id": m.model_id,
                    "name": m.name,
                    "version": m.version,
                    "status": m.status.value,
                    "model_type": m.model_type,
                }
                for m in models
            ],
            "total": len(models),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/promote/{model_id}")
async def promote_to_production(model_id: str) -> Dict[str, Any]:
    """Promover modelo a producción"""
    try:
        success = registry_service.promote_to_production(model_id)
        return {
            "model_id": model_id,
            "promoted": success,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




