"""
Rutas para Model Evaluation
=============================

Endpoints para evaluación de modelos.
"""

import logging
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.model_evaluation import (
    get_model_evaluation,
    ModelEvaluation,
    EvaluationMetric
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/model-evaluation",
    tags=["Model Evaluation"]
)


class EvaluateModelRequest(BaseModel):
    """Request para evaluar modelo"""
    test_data: List[Dict[str, Any]] = Field(..., description="Datos de prueba")
    metrics: List[str] = Field(..., description="Métricas")


@router.post("/models/{model_id}/evaluate")
async def evaluate_model(
    model_id: str,
    request: EvaluateModelRequest,
    system: ModelEvaluation = Depends(get_model_evaluation)
):
    """Evaluar modelo"""
    try:
        metrics = [EvaluationMetric(m) for m in request.metrics]
        result = system.evaluate_model(model_id, request.test_data, metrics)
        
        return {
            "evaluation_id": result.evaluation_id,
            "model_id": result.model_id,
            "metrics": result.metrics,
            "confusion_matrix": result.confusion_matrix,
            "performance_report": result.performance_report
        }
    except Exception as e:
        logger.error(f"Error evaluando modelo: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare")
async def compare_models(
    model_ids: List[str] = Field(..., description="IDs de modelos"),
    test_data: List[Dict[str, Any]] = Field(..., description="Datos de prueba"),
    system: ModelEvaluation = Depends(get_model_evaluation)
):
    """Comparar múltiples modelos"""
    try:
        comparison = system.compare_models(model_ids, test_data)
        
        return comparison
    except Exception as e:
        logger.error(f"Error comparando modelos: {e}")
        raise HTTPException(status_code=500, detail=str(e))


