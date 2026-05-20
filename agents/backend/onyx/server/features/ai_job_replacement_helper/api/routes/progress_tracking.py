"""
Progress Tracking endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.progress_tracking import ProgressTrackingService

router = APIRouter()
progress_service = ProgressTrackingService()


@router.post("/milestone/{user_id}")
async def create_milestone(
    user_id: str,
    title: str,
    description: str,
    target_value: float
) -> Dict[str, Any]:
    """Crear hito de progreso"""
    try:
        milestone = progress_service.create_milestone(
            user_id, title, description, target_value
        )
        return {
            "id": milestone.id,
            "title": milestone.title,
            "target_value": milestone.target_value,
            "current_value": milestone.current_value,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/timeline/{user_id}")
async def get_progress_timeline(
    user_id: str,
    days: int = 30
) -> Dict[str, Any]:
    """Obtener timeline de progreso"""
    try:
        timeline = progress_service.get_progress_timeline(user_id, days)
        return {
            "snapshots": [
                {
                    "timestamp": s.timestamp.isoformat(),
                    "metrics": s.metrics,
                    "achievements": s.achievements,
                }
                for s in timeline
            ],
            "total": len(timeline),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/growth/{user_id}")
async def get_growth_rate(
    user_id: str,
    metric_name: str,
    days: int = 7
) -> Dict[str, Any]:
    """Calcular tasa de crecimiento"""
    try:
        growth_rate = progress_service.calculate_growth_rate(user_id, metric_name, days)
        return {
            "metric": metric_name,
            "growth_rate": growth_rate,
            "period_days": days,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




