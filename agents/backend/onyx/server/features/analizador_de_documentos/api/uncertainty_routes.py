"""
Rutas para Uncertainty Analysis
=================================

Endpoints para análisis de incertidumbre.
"""

import logging
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.uncertainty_analysis import (
    get_uncertainty_analysis,
    UncertaintyAnalysis
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/uncertainty-analysis",
    tags=["Uncertainty Analysis"]
)


class EstimateUncertaintyRequest(BaseModel):
    """Request para estimar incertidumbre"""
    prediction: Any = Field(..., description="Predicción")
    input_data: Dict[str, Any] = Field(..., description="Datos de entrada")


@router.post("/models/{model_id}/estimate")
async def estimate_uncertainty(
    model_id: str,
    request: EstimateUncertaintyRequest,
    system: UncertaintyAnalysis = Depends(get_uncertainty_analysis)
):
    """Estimar incertidumbre de predicción"""
    try:
        estimate = system.estimate_uncertainty(
            request.prediction,
            model_id,
            request.input_data
        )
        
        return {
            "estimate_id": estimate.estimate_id,
            "prediction": estimate.prediction,
            "aleatoric_uncertainty": estimate.aleatoric_uncertainty,
            "epistemic_uncertainty": estimate.epistemic_uncertainty,
            "total_uncertainty": estimate.total_uncertainty,
            "confidence": estimate.confidence
        }
    except Exception as e:
        logger.error(f"Error estimando incertidumbre: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/{model_id}/analyze-calibration")
async def analyze_calibration(
    model_id: str,
    predictions: List[Dict[str, Any]] = Field(..., description="Predicciones"),
    system: UncertaintyAnalysis = Depends(get_uncertainty_analysis)
):
    """Analizar calibración del modelo"""
    try:
        calibration = system.analyze_calibration(model_id, predictions)
        
        return calibration
    except Exception as e:
        logger.error(f"Error analizando calibración: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/{model_id}/detect-ood")
async def detect_out_of_distribution(
    model_id: str,
    input_data: Dict[str, Any] = Field(..., description="Datos de entrada"),
    system: UncertaintyAnalysis = Depends(get_uncertainty_analysis)
):
    """Detectar datos fuera de distribución"""
    try:
        detection = system.detect_out_of_distribution(input_data, model_id)
        
        return detection
    except Exception as e:
        logger.error(f"Error detectando OOD: {e}")
        raise HTTPException(status_code=500, detail=str(e))


