"""
Rutas para Advanced Model Interpretability
============================================

Endpoints para interpretabilidad avanzada.
"""

import logging
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.advanced_model_interpretability import (
    get_advanced_model_interpretability,
    AdvancedModelInterpretability,
    InterpretabilityMethod
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/advanced-model-interpretability",
    tags=["Advanced Model Interpretability"]
)


class InterpretRequest(BaseModel):
    """Request para interpretar"""
    prediction: Any = Field(..., description="Predicción")
    input_data: Dict[str, Any] = Field(..., description="Datos de entrada")
    method: str = Field("shap", description="Método")


@router.post("/models/{model_id}/interpret")
async def interpret_prediction(
    model_id: str,
    request: InterpretRequest,
    system: AdvancedModelInterpretability = Depends(get_advanced_model_interpretability)
):
    """Interpretar predicción"""
    try:
        method = InterpretabilityMethod(request.method)
        interpretation = system.interpret_prediction(
            model_id,
            request.prediction,
            request.input_data,
            method
        )
        
        return {
            "interpretation_id": interpretation.interpretation_id,
            "model_id": interpretation.model_id,
            "method": interpretation.method.value,
            "feature_importance": interpretation.feature_importance,
            "explanation": interpretation.explanation,
            "confidence": interpretation.confidence
        }
    except Exception as e:
        logger.error(f"Error interpretando: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/{model_id}/global-interpretation")
async def generate_global_interpretation(
    model_id: str,
    training_data: List[Dict[str, Any]] = Field(..., description="Datos de entrenamiento"),
    system: AdvancedModelInterpretability = Depends(get_advanced_model_interpretability)
):
    """Generar interpretación global"""
    try:
        global_interp = system.generate_global_interpretation(model_id, training_data)
        
        return {
            "interpretation_id": global_interp.interpretation_id,
            "model_id": global_interp.model_id,
            "overall_importance": global_interp.overall_importance,
            "decision_boundary": global_interp.decision_boundary,
            "feature_interactions": global_interp.feature_interactions
        }
    except Exception as e:
        logger.error(f"Error generando interpretación global: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/{model_id}/counterfactual")
async def generate_counterfactual(
    model_id: str,
    original_input: Dict[str, Any] = Field(..., description="Input original"),
    target_class: Any = Field(..., description="Clase objetivo"),
    system: AdvancedModelInterpretability = Depends(get_advanced_model_interpretability)
):
    """Generar ejemplo contrafactual"""
    try:
        counterfactual = system.generate_counterfactual(model_id, original_input, target_class)
        
        return counterfactual
    except Exception as e:
        logger.error(f"Error generando contrafactual: {e}")
        raise HTTPException(status_code=500, detail=str(e))


