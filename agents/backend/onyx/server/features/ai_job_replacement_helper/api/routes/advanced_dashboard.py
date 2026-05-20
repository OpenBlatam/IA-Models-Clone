"""
Advanced Dashboard endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.advanced_dashboard import AdvancedDashboardService

router = APIRouter()
dashboard_service = AdvancedDashboardService()


@router.post("/create/{user_id}")
async def create_dashboard(
    user_id: str,
    name: str,
    layout: str = "grid"
) -> Dict[str, Any]:
    """Crear nuevo dashboard"""
    try:
        dashboard = dashboard_service.create_dashboard(user_id, name, layout)
        return {
            "id": dashboard.id,
            "user_id": dashboard.user_id,
            "name": dashboard.name,
            "layout": dashboard.layout,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/{user_id}")
async def get_user_metrics(user_id: str) -> Dict[str, Any]:
    """Obtener métricas del usuario"""
    try:
        metrics = dashboard_service.get_user_metrics(user_id)
        return {
            "total_applications": metrics.total_applications,
            "applications_this_week": metrics.applications_this_week,
            "interview_rate": metrics.interview_rate,
            "offer_rate": metrics.offer_rate,
            "average_response_time": metrics.average_response_time,
            "skills_learned": metrics.skills_learned,
            "assessments_completed": metrics.assessments_completed,
            "network_growth": metrics.network_growth,
            "portfolio_views": metrics.portfolio_views,
            "job_alerts_active": metrics.job_alerts_active,
            "streak_days": metrics.streak_days,
            "current_level": metrics.current_level,
            "total_points": metrics.total_points,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/insights/{user_id}")
async def generate_insights(user_id: str) -> Dict[str, Any]:
    """Generar insights del usuario"""
    try:
        insights = dashboard_service.generate_insights(user_id)
        return {
            "user_id": user_id,
            "insights": insights,
            "total": len(insights),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trend/{user_id}")
async def get_trend_data(
    user_id: str,
    metric: str,
    days: int = 30
) -> Dict[str, Any]:
    """Obtener datos de tendencia"""
    try:
        trend = dashboard_service.get_trend_data(user_id, metric, days)
        return trend
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




