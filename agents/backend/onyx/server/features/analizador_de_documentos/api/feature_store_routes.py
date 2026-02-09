"""
Rutas para Feature Store
==========================

Endpoints para feature store.
"""

import logging
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.feature_store import (
    get_feature_store,
    FeatureStore,
    FeatureType
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/feature-store",
    tags=["Feature Store"]
)


class RegisterFeatureRequest(BaseModel):
    """Request para registrar feature"""
    name: str = Field(..., description="Nombre")
    feature_type: str = Field(..., description="Tipo")
    description: str = Field(..., description="Descripción")
    schema: Dict[str, Any] = Field(..., description="Esquema")


@router.post("/features")
async def register_feature(
    request: RegisterFeatureRequest,
    system: FeatureStore = Depends(get_feature_store)
):
    """Registrar feature"""
    try:
        feature_type = FeatureType(request.feature_type)
        feature = system.register_feature(
            request.name,
            feature_type,
            request.description,
            request.schema
        )
        
        return {
            "feature_id": feature.feature_id,
            "name": feature.name,
            "feature_type": feature.feature_type.value,
            "description": feature.description
        }
    except Exception as e:
        logger.error(f"Error registrando feature: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/featuresets")
async def create_feature_set(
    name: str = Field(..., description="Nombre"),
    feature_ids: List[str] = Field(..., description="IDs de features"),
    system: FeatureStore = Depends(get_feature_store)
):
    """Crear conjunto de features"""
    try:
        feature_set = system.create_feature_set(name, feature_ids)
        
        return {
            "featureset_id": feature_set.featureset_id,
            "name": feature_set.name,
            "features": feature_set.features,
            "version": feature_set.version
        }
    except Exception as e:
        logger.error(f"Error creando feature set: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/features/{feature_id}/lineage")
async def get_lineage(
    feature_id: str,
    system: FeatureStore = Depends(get_feature_store)
):
    """Obtener lineage de feature"""
    try:
        lineage = system.get_feature_lineage(feature_id)
        
        return lineage
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error obteniendo lineage: {e}")
        raise HTTPException(status_code=500, detail=str(e))


