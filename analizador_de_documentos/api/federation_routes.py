"""
Rutas para Model Federation
=============================

Endpoints para federación de modelos.
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.model_federation import (
    get_model_federation,
    ModelFederation,
    FederationStrategy
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/model-federation",
    tags=["Model Federation"]
)


class CreateFederationRequest(BaseModel):
    """Request para crear federación"""
    model_ids: List[str] = Field(..., description="IDs de modelos")
    strategy: str = Field("weighted_average", description="Estrategia")
    weights: Optional[List[float]] = Field(None, description="Pesos")


@router.post("/federations")
async def create_federation(
    request: CreateFederationRequest,
    system: ModelFederation = Depends(get_model_federation)
):
    """Crear federación de modelos"""
    try:
        strategy = FederationStrategy(request.strategy)
        federation = system.create_federation(
            request.model_ids,
            strategy,
            request.weights
        )
        
        return {
            "federation_id": federation.federation_id,
            "model_ids": federation.model_ids,
            "strategy": federation.strategy.value,
            "weights": federation.weights,
            "status": federation.status
        }
    except Exception as e:
        logger.error(f"Error creando federación: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/federations/{federation_id}/predict")
async def federated_predict(
    federation_id: str,
    input_data: Dict[str, Any] = Field(..., description="Datos de entrada"),
    system: ModelFederation = Depends(get_model_federation)
):
    """Predecir con modelos federados"""
    try:
        prediction = system.federated_predict(federation_id, input_data)
        
        return prediction
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error prediciendo: {e}")
        raise HTTPException(status_code=500, detail=str(e))


