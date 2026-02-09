"""
Rutas para Advanced Model Monitoring
======================================

Endpoints para monitoreo avanzado.
"""

import logging
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.advanced_model_monitoring import (
    get_advanced_model_monitoring,
    AdvancedModelMonitoring,
    MetricType
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/advanced-model-monitoring",
    tags=["Advanced Model Monitoring"]
)


class SetupMonitoringRequest(BaseModel):
    """Request para configurar monitoreo"""
    metrics: List[str] = Field(..., description="Métricas")
    thresholds: Dict[str, float] = Field(..., description="Umbrales")


@router.post("/models/{model_id}/setup")
async def setup_monitoring(
    model_id: str,
    request: SetupMonitoringRequest,
    system: AdvancedModelMonitoring = Depends(get_advanced_model_monitoring)
):
    """Configurar monitoreo"""
    try:
        metrics = [MetricType(m) for m in request.metrics]
        config = system.setup_monitoring(model_id, metrics, request.thresholds)
        
        return config
    except Exception as e:
        logger.error(f"Error configurando monitoreo: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/{model_id}/metrics")
async def record_metrics(
    model_id: str,
    metrics: Dict[str, float] = Field(..., description="Métricas"),
    system: AdvancedModelMonitoring = Depends(get_advanced_model_monitoring)
):
    """Registrar métricas"""
    try:
        model_metrics = system.record_metrics(model_id, metrics)
        
        return {
            "model_id": model_metrics.model_id,
            "metrics": model_metrics.metrics,
            "timestamp": model_metrics.timestamp
        }
    except Exception as e:
        logger.error(f"Error registrando métricas: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/{model_id}/detect-drift")
async def detect_drift(
    model_id: str,
    current_data: List[Dict[str, Any]] = Field(..., description="Datos actuales"),
    reference_data: List[Dict[str, Any]] = Field(..., description="Datos de referencia"),
    system: AdvancedModelMonitoring = Depends(get_advanced_model_monitoring)
):
    """Detectar drift"""
    try:
        drift_analysis = system.detect_drift(model_id, current_data, reference_data)
        
        return drift_analysis
    except Exception as e:
        logger.error(f"Error detectando drift: {e}")
        raise HTTPException(status_code=500, detail=str(e))


