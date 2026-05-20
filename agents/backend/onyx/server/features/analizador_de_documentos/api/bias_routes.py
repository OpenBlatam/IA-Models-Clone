"""
Rutas para Bias Detection
===========================

Endpoints para detección de sesgos.
"""

import logging
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.bias_detection import (
    get_bias_detection,
    BiasDetection
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/bias-detection",
    tags=["Bias Detection"]
)


class DetectBiasRequest(BaseModel):
    """Request para detectar sesgos"""
    test_data: List[Dict[str, Any]] = Field(..., description="Datos de prueba")
    protected_attributes: List[str] = Field(..., description="Atributos protegidos")


@router.post("/models/{model_id}/detect-bias")
async def detect_bias(
    model_id: str,
    request: DetectBiasRequest,
    system: BiasDetection = Depends(get_bias_detection)
):
    """Detectar sesgos en modelo"""
    try:
        report = system.detect_bias(
            model_id,
            request.test_data,
            request.protected_attributes
        )
        
        return {
            "report_id": report.report_id,
            "model_id": report.model_id,
            "bias_types": [bt.value for bt in report.bias_types],
            "bias_scores": report.bias_scores,
            "fairness_metrics": report.fairness_metrics,
            "recommendations": report.recommendations
        }
    except Exception as e:
        logger.error(f"Error detectando sesgos: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/{model_id}/fairness-metrics")
async def calculate_fairness(
    model_id: str,
    predictions: List[Dict[str, Any]] = Field(..., description="Predicciones"),
    protected_attributes: List[str] = Field(..., description="Atributos protegidos"),
    system: BiasDetection = Depends(get_bias_detection)
):
    """Calcular métricas de equidad"""
    try:
        metrics = system.calculate_fairness_metrics(
            model_id,
            predictions,
            protected_attributes
        )
        
        return metrics
    except Exception as e:
        logger.error(f"Error calculando equidad: {e}")
        raise HTTPException(status_code=500, detail=str(e))


