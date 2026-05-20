"""
Dashboard endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.analytics import AnalyticsService

router = APIRouter()
analytics_service = AnalyticsService()


@router.get("/{user_id}")
async def get_dashboard(user_id: str) -> Dict[str, Any]:
    """Obtener datos completos del dashboard"""
    try:
        dashboard_data = analytics_service.get_dashboard_data(user_id)
        return dashboard_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/{user_id}")
async def get_metrics(user_id: str) -> Dict[str, Any]:
    """Obtener métricas del usuario"""
    try:
        metrics = analytics_service.get_user_metrics(user_id)
        return {
            "user_id": metrics.user_id,
            "total_points": metrics.total_points,
            "current_level": metrics.current_level,
            "current_streak": metrics.current_streak,
            "jobs_applied": metrics.jobs_applied,
            "jobs_saved": metrics.jobs_saved,
            "steps_completed": metrics.steps_completed,
            "skills_learned": metrics.skills_learned,
            "challenges_completed": metrics.challenges_completed,
            "badges_earned": metrics.badges_earned,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/activity/{user_id}")
async def get_activity_stats(
    user_id: str,
    days: int = 30
) -> Dict[str, Any]:
    """Obtener estadísticas de actividad"""
    try:
        stats = analytics_service.get_activity_stats(user_id, days)
        return {
            "stats": [
                {
                    "date": stat.date,
                    "actions_count": stat.actions_count,
                    "points_earned": stat.points_earned,
                }
                for stat in stats
            ],
            "total_days": len(stats),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




