"""
Rutas para Explainable AI
===========================

Endpoints para explainable AI.
"""

import logging
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.explainable_ai import (
    get_explainable_ai,
    ExplainableAI,
    ExplanationMethod
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/xai",
    tags=["Explainable AI"]
)


class ExplainPredictionRequest(BaseModel):
    """Request para explicar predicción"""
    prediction: Any = Field(..., description="Predicción")
    input_features: Dict[str, Any] = Field(..., description="Características de entrada")
    method: str = Field("feature_importance", description="Método de explicación")
    model_id: str = Field(None, description="ID del modelo")


@router.post("/explain")
async def explain_prediction(
    request: ExplainPredictionRequest,
    xai: ExplainableAI = Depends(get_explainable_ai)
):
    """Explicar predicción"""
    try:
        method = ExplanationMethod(request.method)
        explanation = xai.explain_prediction(
            request.prediction,
            request.input_features,
            method,
            request.model_id
        )
        
        return {
            "explanation_id": explanation.explanation_id,
            "prediction": explanation.prediction,
            "confidence": explanation.confidence,
            "features": explanation.features,
            "method": explanation.method.value,
            "reasoning": explanation.reasoning
        }
    except Exception as e:
        logger.error(f"Error explicando predicción: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/model-behavior")
async def explain_model_behavior(
    model_id: str = Field(..., description="ID del modelo"),
    sample_inputs: List[Dict[str, Any]] = Field(..., description="Muestras de entrada"),
    xai: ExplainableAI = Depends(get_explainable_ai)
):
    """Explicar comportamiento del modelo"""
    try:
        analysis = xai.explain_model_behavior(model_id, sample_inputs)
        
        return analysis
    except Exception as e:
        logger.error(f"Error analizando comportamiento: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/explanations/{explanation_id}")
async def get_explanation(
    explanation_id: str,
    xai: ExplainableAI = Depends(get_explainable_ai)
):
    """Obtener explicación"""
    explanation = xai.get_explanation(explanation_id)
    
    if not explanation:
        raise HTTPException(status_code=404, detail="Explicación no encontrada")
    
    return {
        "explanation_id": explanation.explanation_id,
        "prediction": explanation.prediction,
        "confidence": explanation.confidence,
        "features": explanation.features,
        "reasoning": explanation.reasoning
    }



