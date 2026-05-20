"""
Rutas para Advanced Anomaly Detection
=======================================

Endpoints para detección avanzada de anomalías.
"""

import logging
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.advanced_anomaly_detection import (
    get_anomaly_detection,
    AdvancedAnomalyDetection,
    AnomalyDetectionMethod
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/anomaly-detection",
    tags=["Advanced Anomaly Detection"]
)


class CreateDetectorRequest(BaseModel):
    """Request para crear detector"""
    method: str = Field("isolation_forest", description="Método")
    parameters: Dict[str, Any] = Field(None, description="Parámetros")


class DetectAnomaliesRequest(BaseModel):
    """Request para detectar anomalías"""
    data: List[Dict[str, Any]] = Field(..., description="Datos")
    threshold: float = Field(0.5, description="Umbral")


@router.post("/detectors/{detector_id}")
async def create_detector(
    detector_id: str,
    request: CreateDetectorRequest,
    system: AdvancedAnomalyDetection = Depends(get_anomaly_detection)
):
    """Crear detector de anomalías"""
    try:
        method = AnomalyDetectionMethod(request.method)
        detector = system.create_detector(detector_id, method, request.parameters)
        
        return detector
    except Exception as e:
        logger.error(f"Error creando detector: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/detectors/{detector_id}/detect")
async def detect_anomalies(
    detector_id: str,
    request: DetectAnomaliesRequest,
    system: AdvancedAnomalyDetection = Depends(get_anomaly_detection)
):
    """Detectar anomalías"""
    try:
        anomalies = system.detect_anomalies(
            detector_id,
            request.data,
            request.threshold
        )
        
        return {
            "anomalies": [
                {
                    "anomaly_id": a.anomaly_id,
                    "anomaly_score": a.anomaly_score,
                    "severity": a.severity,
                    "method": a.method.value
                }
                for a in anomalies
            ],
            "count": len(anomalies)
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error detectando anomalías: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/anomalies/{anomaly_id}/explain")
async def explain_anomaly(
    anomaly_id: str,
    system: AdvancedAnomalyDetection = Depends(get_anomaly_detection)
):
    """Explicar anomalía"""
    try:
        explanation = system.explain_anomaly(anomaly_id)
        
        return explanation
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error explicando anomalía: {e}")
        raise HTTPException(status_code=500, detail=str(e))
