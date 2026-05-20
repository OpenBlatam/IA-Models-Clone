"""
Model Monitoring endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.model_monitoring import ModelMonitoringService

router = APIRouter()
monitoring_service = ModelMonitoringService()


@router.post("/record")
async def record_prediction(
    model_id: str,
    prediction: Any,
    confidence: Optional[float] = None,
    latency_ms: Optional[float] = None
) -> Dict[str, Any]:
    """Registrar predicción"""
    try:
        monitoring_service.record_prediction(
            model_id, prediction, confidence, latency_ms
        )
        
        return {
            "model_id": model_id,
            "recorded": True,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/{model_id}")
async def get_metrics(model_id: str) -> Dict[str, Any]:
    """Obtener métricas del modelo"""
    try:
        metrics = monitoring_service.get_metrics(model_id)
        if not metrics:
            return {"error": "Model not found"}
        
        return {
            "model_id": metrics.model_id,
            "total_predictions": metrics.total_predictions,
            "avg_latency_ms": metrics.avg_latency_ms,
            "p95_latency_ms": metrics.p95_latency_ms,
            "avg_confidence": metrics.avg_confidence,
            "predictions_per_hour": metrics.predictions_per_hour,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/drift/{model_id}")
async def detect_drift(
    model_id: str,
    window_hours: int = 24
) -> Dict[str, Any]:
    """Detectar drift en modelo"""
    try:
        drift_info = monitoring_service.detect_drift(model_id, window_hours)
        return drift_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




