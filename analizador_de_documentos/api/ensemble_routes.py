"""
Rutas para Model Ensembling
=============================

Endpoints para ensamblado de modelos.
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.model_ensembling import (
    get_model_ensembling,
    ModelEnsembling,
    EnsembleMethod
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/model-ensembling",
    tags=["Model Ensembling"]
)


class CreateEnsembleRequest(BaseModel):
    """Request para crear ensamblado"""
    base_models: List[str] = Field(..., description="IDs de modelos base")
    method: str = Field("voting", description="Método")
    weights: Optional[List[float]] = Field(None, description="Pesos")


@router.post("/ensembles")
async def create_ensemble(
    request: CreateEnsembleRequest,
    system: ModelEnsembling = Depends(get_model_ensembling)
):
    """Crear ensamblado de modelos"""
    try:
        method = EnsembleMethod(request.method)
        ensemble = system.create_ensemble(
            request.base_models,
            method,
            request.weights
        )
        
        return {
            "ensemble_id": ensemble.ensemble_id,
            "base_models": ensemble.base_models,
            "method": ensemble.method.value,
            "weights": ensemble.weights
        }
    except Exception as e:
        logger.error(f"Error creando ensamblado: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ensembles/{ensemble_id}/train")
async def train_ensemble(
    ensemble_id: str,
    training_data: List[Dict[str, Any]] = Field(..., description="Datos de entrenamiento"),
    validation_data: Optional[List[Dict[str, Any]]] = Field(None, description="Datos de validación"),
    system: ModelEnsembling = Depends(get_model_ensembling)
):
    """Entrenar ensamblado"""
    try:
        result = system.train_ensemble(ensemble_id, training_data, validation_data)
        
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error entrenando: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ensembles/{ensemble_id}/predict")
async def predict_ensemble(
    ensemble_id: str,
    input_data: Dict[str, Any] = Field(..., description="Datos de entrada"),
    system: ModelEnsembling = Depends(get_model_ensembling)
):
    """Predecir con ensamblado"""
    try:
        prediction = system.predict_ensemble(ensemble_id, input_data)
        
        return prediction
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error prediciendo: {e}")
        raise HTTPException(status_code=500, detail=str(e))

