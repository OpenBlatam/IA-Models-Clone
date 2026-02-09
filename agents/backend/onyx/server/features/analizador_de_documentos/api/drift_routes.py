"""
Rutas para Concept Drift Detection
===================================

Endpoints para detección de concept drift.
"""

import logging
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.concept_drift_detection import (
    get_concept_drift,
    ConceptDriftDetection,
    DriftDetectionMethod
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/concept-drift",
    tags=["Concept Drift Detection"]
)


class MonitorRequest(BaseModel):
    """Request para monitorear"""
    predictions: List[Dict[str, Any]] = Field(..., description="Predicciones")
    method: str = Field("adwin", description="Método")


@router.post("/models/{model_id}/monitor")
async def monitor_model(
    model_id: str,
    request: MonitorRequest,
    system: ConceptDriftDetection = Depends(get_concept_drift)
):
    """Monitorear modelo para detectar drift"""
    try:
        method = DriftDetectionMethod(request.method)
        result = system.monitor_model(model_id, request.predictions, method)
        
        return result
    except Exception as e:
        logger.error(f"Error monitoreando: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/drift-events/{event_id}/recommendations")
async def get_recommendations(
    event_id: str,
    system: ConceptDriftDetection = Depends(get_concept_drift)
):
    """Obtener recomendaciones para manejar drift"""
    try:
        recommendations = system.get_drift_recommendations(event_id)
        
        return {"event_id": event_id, "recommendations": recommendations}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error obteniendo recomendaciones: {e}")
        raise HTTPException(status_code=500, detail=str(e))


