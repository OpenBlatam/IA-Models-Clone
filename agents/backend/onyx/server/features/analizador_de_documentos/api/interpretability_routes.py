"""
Rutas para Model Interpretability
===================================

Endpoints para interpretabilidad de modelos.
"""

import logging
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.model_interpretability import (
    get_model_interpretability,
    ModelInterpretability,
    InterpretabilityMethod
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/model-interpretability",
    tags=["Model Interpretability"]
)


class InterpretModelRequest(BaseModel):
    """Request para interpretar modelo"""
    method: str = Field("shap", description="Método")
    data: List[Dict[str, Any]] = Field(None, description="Datos")


@router.post("/models/{model_id}/interpret")
async def interpret_model(
    model_id: str,
    request: InterpretModelRequest,
    system: ModelInterpretability = Depends(get_model_interpretability)
):
    """Interpretar modelo"""
    try:
        method = InterpretabilityMethod(request.method)
        interpretation = system.interpret_model(model_id, method, request.data)
        
        return {
            "interpretation_id": interpretation.interpretation_id,
            "model_id": interpretation.model_id,
            "method": interpretation.method.value,
            "feature_importance": interpretation.feature_importance,
            "global_importance": interpretation.global_importance,
            "local_explanations": interpretation.local_explanations
        }
    except Exception as e:
        logger.error(f"Error interpretando modelo: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/{model_id}/explain-prediction")
async def explain_prediction(
    model_id: str,
    input_data: Dict[str, Any] = Field(..., description="Datos de entrada"),
    method: str = Field("shap", description="Método"),
    system: ModelInterpretability = Depends(get_model_interpretability)
):
    """Explicar predicción"""
    try:
        interp_method = InterpretabilityMethod(method)
        explanation = system.explain_prediction(model_id, input_data, interp_method)
        
        return explanation
    except Exception as e:
        logger.error(f"Error explicando predicción: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/{model_id}/feature-interactions")
async def analyze_feature_interactions(
    model_id: str,
    features: List[str] = Field(..., description="Características"),
    system: ModelInterpretability = Depends(get_model_interpretability)
):
    """Analizar interacciones entre características"""
    try:
        interactions = system.analyze_feature_interactions(model_id, features)
        
        return interactions
    except Exception as e:
        logger.error(f"Error analizando interacciones: {e}")
        raise HTTPException(status_code=500, detail=str(e))


