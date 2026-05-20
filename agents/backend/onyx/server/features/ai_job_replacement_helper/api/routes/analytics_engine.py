"""
Analytics Engine endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.analytics_engine import AnalyticsEngineService

router = APIRouter()
analytics_service = AnalyticsEngineService()


@router.post("/track")
async def track_event(
    event_type: str,
    user_id: str,
    properties: Dict[str, Any]
) -> Dict[str, Any]:
    """Rastrear evento"""
    try:
        event = analytics_service.track_event(event_type, user_id, properties)
        return {
            "event_type": event.event_type,
            "user_id": event.user_id,
            "timestamp": event.timestamp.isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/behavior/{user_id}")
async def analyze_user_behavior(
    user_id: str,
    days: int = 30
) -> Dict[str, Any]:
    """Analizar comportamiento del usuario"""
    try:
        analysis = analytics_service.analyze_user_behavior(user_id, days)
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/predict/{user_id}")
async def predict_success_probability(
    user_id: str,
    target_outcome: str
) -> Dict[str, Any]:
    """Predecir probabilidad de éxito"""
    try:
        prediction = analytics_service.predict_success_probability(user_id, target_outcome)
        return {
            "model_type": prediction.model_type,
            "accuracy": prediction.accuracy,
            "predictions": prediction.predictions,
            "factors": prediction.factors,
            "confidence": prediction.confidence,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/funnel/{user_id}")
async def get_funnel_analysis(
    user_id: str,
    funnel_steps: List[str]
) -> Dict[str, Any]:
    """Análisis de embudo"""
    try:
        analysis = analytics_service.get_funnel_analysis(user_id, funnel_steps)
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




