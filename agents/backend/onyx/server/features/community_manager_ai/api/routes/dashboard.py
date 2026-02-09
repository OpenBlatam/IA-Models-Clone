"""
Dashboard API Routes
====================

Endpoints para datos del dashboard.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Optional

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


def get_community_manager():
    """Dependency para obtener CommunityManager"""
    from ...core.community_manager import CommunityManager
    return CommunityManager()


def get_dashboard_service():
    """Dependency para obtener DashboardService"""
    from ...services.dashboard_service import DashboardService
    return DashboardService()


def get_analytics_service():
    """Dependency para obtener AnalyticsService"""
    from ...services.analytics_service import AnalyticsService
    return AnalyticsService()


@router.get("/overview", response_model=dict)
async def get_overview(
    days: int = Query(7, ge=1, le=365),
    manager = Depends(get_community_manager),
    dashboard_service = Depends(get_dashboard_service)
):
    """Obtener estadísticas generales"""
    try:
        stats = dashboard_service.get_overview_stats(manager, days)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/engagement", response_model=dict)
async def get_engagement_summary(
    days: int = Query(7, ge=1, le=365),
    dashboard_service = Depends(get_dashboard_service),
    analytics_service = Depends(get_analytics_service)
):
    """Obtener resumen de engagement"""
    try:
        summary = dashboard_service.get_engagement_summary(analytics_service, days)
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/upcoming-posts", response_model=list)
async def get_upcoming_posts(
    limit: int = Query(10, ge=1, le=100),
    manager = Depends(get_community_manager),
    dashboard_service = Depends(get_dashboard_service)
):
    """Obtener próximos posts programados"""
    try:
        posts = dashboard_service.get_upcoming_posts(manager, limit)
        return posts
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recent-activity", response_model=list)
async def get_recent_activity(
    limit: int = Query(20, ge=1, le=100),
    manager = Depends(get_community_manager),
    dashboard_service = Depends(get_dashboard_service)
):
    """Obtener actividad reciente"""
    try:
        activities = dashboard_service.get_recent_activity(manager, limit)
        return activities
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




