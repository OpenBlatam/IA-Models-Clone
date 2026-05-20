"""
Analytics Routes
API endpoints for analytics and reporting
"""

import logging
from typing import Optional, Dict, Any

from fastapi import APIRouter, Depends, Query

from ..models import User
from ..dependencies import get_analytics_service, get_current_user
from ..error_handlers import handle_route_errors
from ...services.analytics_service import AnalyticsService

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/dashboard")
@handle_route_errors
async def get_analytics_dashboard(
    time_period: str = Query("7d", description="Time period"),
    analytics_service: AnalyticsService = Depends(get_analytics_service),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get analytics dashboard data"""
    dashboard_data = await analytics_service.get_dashboard_data(
        user_id=current_user.id,
        time_period=time_period
    )
    return dashboard_data

@router.get("/content-performance")
@handle_route_errors
async def get_content_performance(
    content_id: Optional[str] = Query(None, description="Content ID"),
    time_period: str = Query("30d", description="Time period"),
    analytics_service: AnalyticsService = Depends(get_analytics_service),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get content performance analytics"""
    performance_data = await analytics_service.get_content_performance(
        user_id=current_user.id,
        content_id=content_id,
        time_period=time_period
    )
    return performance_data

@router.get("/collaboration-stats")
@handle_route_errors
async def get_collaboration_stats(
    time_period: str = Query("30d", description="Time period"),
    analytics_service: AnalyticsService = Depends(get_analytics_service),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get collaboration statistics"""
    stats = await analytics_service.get_collaboration_stats(
        user_id=current_user.id,
        time_period=time_period
    )
    return stats







