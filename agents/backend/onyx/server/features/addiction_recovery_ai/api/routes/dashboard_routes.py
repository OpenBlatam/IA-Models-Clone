"""
Dashboard routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

try:
    from services.dashboard_service import DashboardService
    from core.progress_tracker import ProgressTracker
    from services.analytics_service import AnalyticsService
except ImportError:
    from ...services.dashboard_service import DashboardService
    from ...core.progress_tracker import ProgressTracker
    from ...services.analytics_service import AnalyticsService

router = APIRouter()

dashboard = DashboardService()
tracker = ProgressTracker()
analytics = AnalyticsService()


@router.get("/dashboard/{user_id}")
async def get_dashboard(user_id: str):
    """Obtiene datos completos del dashboard"""
    try:
        progress = tracker.get_progress(user_id, None, [])
        stats = tracker.get_stats(user_id, [])
        analytics_data = analytics.generate_comprehensive_analytics(user_id, [])
        
        dashboard_data = dashboard.get_dashboard_data(
            user_id=user_id,
            entries=[],
            progress_data=progress,
            analytics_data=analytics_data
        )
        
        return JSONResponse(content=dashboard_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo dashboard: {str(e)}")



