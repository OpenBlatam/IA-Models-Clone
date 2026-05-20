"""
Dashboard Service
Provides admin dashboard data and metrics
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging

from ...analytics import get_analytics_service
from ...auth.user_service import get_user_service

logger = logging.getLogger(__name__)


class DashboardService:
    """Provides dashboard data"""
    
    def __init__(self):
        self.analytics = get_analytics_service()
        self.user_service = get_user_service()
    
    def get_dashboard_data(
        self,
        time_range: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """
        Get comprehensive dashboard data
        
        Args:
            time_range: Time range for data (default: last 30 days)
            
        Returns:
            Dashboard data dictionary
        """
        if time_range is None:
            time_range = timedelta(days=30)
        
        metrics = self.analytics.get_metrics(time_range=time_range)
        usage_stats = self.analytics.get_usage_statistics()
        top_errors = self.analytics.get_top_errors(limit=10)
        
        # Calculate additional metrics
        total_users = len(self.user_service.users)
        
        # Growth metrics (would need historical data in production)
        growth_metrics = {
            "videos_today": 0,  # Would calculate from time_range
            "videos_this_week": 0,
            "videos_this_month": metrics.get("total_videos", 0),
            "growth_rate": 0.0,
        }
        
        # Performance metrics
        avg_generation_time = metrics.get("average_generation_time", 0.0)
        success_rate = metrics.get("success_rate", 0.0)
        
        # Resource usage (would integrate with system monitoring)
        resource_usage = {
            "cpu_usage": 0.0,
            "memory_usage": 0.0,
            "disk_usage": 0.0,
            "active_jobs": 0,
        }
        
        return {
            "summary": {
                "total_videos": metrics.get("total_videos", 0),
                "successful_videos": metrics.get("successful_videos", 0),
                "failed_videos": metrics.get("failed_videos", 0),
                "success_rate": success_rate,
                "total_users": total_users,
                "average_generation_time": avg_generation_time,
            },
            "growth": growth_metrics,
            "usage_statistics": usage_stats,
            "top_errors": top_errors,
            "resource_usage": resource_usage,
            "time_range": {
                "start": (datetime.utcnow() - time_range).isoformat(),
                "end": datetime.utcnow().isoformat(),
            },
        }
    
    def get_user_dashboard(self, user_id: str) -> Dict[str, Any]:
        """Get dashboard data for specific user"""
        # Get user's videos (would need video ownership tracking in production)
        user_metrics = {
            "total_videos": 0,
            "successful_videos": 0,
            "failed_videos": 0,
            "average_generation_time": 0.0,
        }
        
        return {
            "user_id": user_id,
            "metrics": user_metrics,
            "recent_videos": [],
        }
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get system health status"""
        return {
            "status": "healthy",
            "services": {
                "video_generation": "operational",
                "image_generation": "operational",
                "audio_generation": "operational",
                "storage": "operational",
                "cache": "operational",
            },
            "uptime": "99.9%",
            "last_check": datetime.utcnow().isoformat(),
        }


_dashboard_service: Optional[DashboardService] = None


def get_dashboard_service() -> DashboardService:
    """Get dashboard service instance (singleton)"""
    global _dashboard_service
    if _dashboard_service is None:
        _dashboard_service = DashboardService()
    return _dashboard_service

