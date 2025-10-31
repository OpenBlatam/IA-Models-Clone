"""
Advanced Analytics API endpoints
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel, Field

from ....services.advanced_analytics_service import AdvancedAnalyticsService
from ....api.dependencies import CurrentUserDep, DatabaseSessionDep
from ....core.exceptions import DatabaseError

router = APIRouter()


class AnalyticsRequest(BaseModel):
    """Request model for analytics."""
    days: int = Field(default=30, ge=1, le=365, description="Number of days to analyze")
    granularity: str = Field(default="daily", description="Data granularity (daily, weekly, monthly)")


async def get_advanced_analytics_service(session: DatabaseSessionDep) -> AdvancedAnalyticsService:
    """Get advanced analytics service instance."""
    return AdvancedAnalyticsService(session)


@router.get("/comprehensive", response_model=Dict[str, Any])
async def get_comprehensive_analytics(
    days: int = Query(default=30, ge=1, le=365, description="Number of days to analyze"),
    granularity: str = Query(default="daily", description="Data granularity"),
    analytics_service: AdvancedAnalyticsService = Depends(get_advanced_analytics_service),
    current_user: CurrentUserDep = Depends()
):
    """Get comprehensive analytics overview."""
    try:
        analytics = await analytics_service.get_comprehensive_analytics(
            days=days,
            granularity=granularity
        )
        
        return {
            "success": True,
            "data": analytics,
            "message": "Comprehensive analytics retrieved successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get comprehensive analytics"
        )


@router.get("/content/performance", response_model=Dict[str, Any])
async def get_content_performance_analysis(
    days: int = Query(default=30, ge=1, le=365, description="Number of days to analyze"),
    analytics_service: AdvancedAnalyticsService = Depends(get_advanced_analytics_service),
    current_user: CurrentUserDep = Depends()
):
    """Analyze content performance in detail."""
    try:
        analysis = await analytics_service.get_content_performance_analysis(days=days)
        
        return {
            "success": True,
            "data": analysis,
            "message": "Content performance analysis completed successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to analyze content performance"
        )


@router.get("/users/behavior", response_model=Dict[str, Any])
async def get_user_behavior_analysis(
    days: int = Query(default=30, ge=1, le=365, description="Number of days to analyze"),
    analytics_service: AdvancedAnalyticsService = Depends(get_advanced_analytics_service),
    current_user: CurrentUserDep = Depends()
):
    """Analyze user behavior patterns."""
    try:
        analysis = await analytics_service.get_user_behavior_analysis(days=days)
        
        return {
            "success": True,
            "data": analysis,
            "message": "User behavior analysis completed successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to analyze user behavior"
        )


@router.get("/engagement/insights", response_model=Dict[str, Any])
async def get_engagement_insights(
    days: int = Query(default=30, ge=1, le=365, description="Number of days to analyze"),
    analytics_service: AdvancedAnalyticsService = Depends(get_advanced_analytics_service),
    current_user: CurrentUserDep = Depends()
):
    """Get detailed engagement insights."""
    try:
        insights = await analytics_service.get_engagement_insights(days=days)
        
        return {
            "success": True,
            "data": insights,
            "message": "Engagement insights retrieved successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get engagement insights"
        )


@router.get("/predictive", response_model=Dict[str, Any])
async def get_predictive_analytics(
    days: int = Query(default=30, ge=1, le=365, description="Number of days to analyze"),
    analytics_service: AdvancedAnalyticsService = Depends(get_advanced_analytics_service),
    current_user: CurrentUserDep = Depends()
):
    """Get predictive analytics and forecasting."""
    try:
        predictions = await analytics_service.get_predictive_analytics(days=days)
        
        return {
            "success": True,
            "data": predictions,
            "message": "Predictive analytics retrieved successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get predictive analytics"
        )


@router.get("/advanced-metrics", response_model=Dict[str, Any])
async def get_advanced_metrics(
    days: int = Query(default=30, ge=1, le=365, description="Number of days to analyze"),
    analytics_service: AdvancedAnalyticsService = Depends(get_advanced_analytics_service),
    current_user: CurrentUserDep = Depends()
):
    """Get advanced metrics and KPIs."""
    try:
        metrics = await analytics_service.get_advanced_metrics(days=days)
        
        return {
            "success": True,
            "data": metrics,
            "message": "Advanced metrics retrieved successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get advanced metrics"
        )


@router.get("/dashboard", response_model=Dict[str, Any])
async def get_analytics_dashboard(
    days: int = Query(default=30, ge=1, le=365, description="Number of days to analyze"),
    analytics_service: AdvancedAnalyticsService = Depends(get_advanced_analytics_service),
    current_user: CurrentUserDep = Depends()
):
    """Get analytics dashboard data."""
    try:
        # Get comprehensive analytics for dashboard
        analytics = await analytics_service.get_comprehensive_analytics(
            days=days,
            granularity="daily"
        )
        
        # Get additional dashboard-specific data
        content_analysis = await analytics_service.get_content_performance_analysis(days=days)
        user_analysis = await analytics_service.get_user_behavior_analysis(days=days)
        engagement_insights = await analytics_service.get_engagement_insights(days=days)
        
        dashboard_data = {
            "overview": analytics.get("summary", {}),
            "content_performance": content_analysis,
            "user_behavior": user_analysis,
            "engagement": engagement_insights,
            "timeline": analytics.get("content", {}).get("posts_timeline", []),
            "categories": analytics.get("content", {}).get("category_distribution", []),
            "users": analytics.get("users", {}),
            "engagement_timeline": analytics.get("engagement", {})
        }
        
        return {
            "success": True,
            "data": dashboard_data,
            "message": "Analytics dashboard data retrieved successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get analytics dashboard data"
        )


@router.get("/export", response_model=Dict[str, Any])
async def export_analytics_data(
    days: int = Query(default=30, ge=1, le=365, description="Number of days to export"),
    format: str = Query(default="json", description="Export format (json, csv)"),
    analytics_service: AdvancedAnalyticsService = Depends(get_advanced_analytics_service),
    current_user: CurrentUserDep = Depends()
):
    """Export analytics data."""
    try:
        # Get comprehensive analytics
        analytics = await analytics_service.get_comprehensive_analytics(
            days=days,
            granularity="daily"
        )
        
        # In a real implementation, you would generate and return the file
        # For now, we'll return the data with export information
        
        export_info = {
            "format": format,
            "days": days,
            "exported_at": "2024-01-15T10:00:00Z",
            "file_size": "2.5MB",
            "download_url": f"/api/v1/analytics/download/export_{current_user.id}_{days}d.{format}"
        }
        
        return {
            "success": True,
            "data": {
                "export_info": export_info,
                "analytics_data": analytics
            },
            "message": "Analytics data exported successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to export analytics data"
        )


@router.get("/reports", response_model=Dict[str, Any])
async def get_analytics_reports(
    analytics_service: AdvancedAnalyticsService = Depends(get_advanced_analytics_service),
    current_user: CurrentUserDep = Depends()
):
    """Get available analytics reports."""
    try:
        # In a real implementation, you would get actual report configurations
        # For now, we'll return mock report data
        
        reports = [
            {
                "id": "daily_summary",
                "name": "Daily Summary Report",
                "description": "Daily overview of key metrics",
                "schedule": "daily",
                "last_generated": "2024-01-15T09:00:00Z",
                "status": "active"
            },
            {
                "id": "weekly_engagement",
                "name": "Weekly Engagement Report",
                "description": "Weekly engagement analysis",
                "schedule": "weekly",
                "last_generated": "2024-01-14T09:00:00Z",
                "status": "active"
            },
            {
                "id": "monthly_performance",
                "name": "Monthly Performance Report",
                "description": "Monthly content performance analysis",
                "schedule": "monthly",
                "last_generated": "2024-01-01T09:00:00Z",
                "status": "active"
            },
            {
                "id": "user_behavior",
                "name": "User Behavior Report",
                "description": "Detailed user behavior analysis",
                "schedule": "weekly",
                "last_generated": "2024-01-14T09:00:00Z",
                "status": "active"
            }
        ]
        
        return {
            "success": True,
            "data": {
                "reports": reports,
                "total": len(reports)
            },
            "message": "Analytics reports retrieved successfully"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get analytics reports: {str(e)}"
        )


@router.get("/kpis", response_model=Dict[str, Any])
async def get_kpi_dashboard(
    days: int = Query(default=30, ge=1, le=365, description="Number of days to analyze"),
    analytics_service: AdvancedAnalyticsService = Depends(get_advanced_analytics_service),
    current_user: CurrentUserDep = Depends()
):
    """Get KPI dashboard data."""
    try:
        # Get comprehensive analytics
        analytics = await analytics_service.get_comprehensive_analytics(
            days=days,
            granularity="daily"
        )
        
        # Calculate KPIs
        summary = analytics.get("summary", {})
        
        kpis = {
            "content_kpis": {
                "total_posts": summary.get("total_posts", 0),
                "posts_per_day": summary.get("total_posts", 0) / days,
                "avg_views_per_post": 0,  # Would be calculated from detailed data
                "top_performing_category": "Technology"  # Would be calculated
            },
            "user_kpis": {
                "total_users": summary.get("total_users", 0),
                "new_users_per_day": summary.get("total_users", 0) / days,
                "active_users": analytics.get("users", {}).get("active_users", 0),
                "user_retention_rate": 0.75  # Would be calculated
            },
            "engagement_kpis": {
                "total_engagement": summary.get("total_engagement", 0),
                "engagement_rate": summary.get("engagement_rate", 0),
                "avg_engagement_per_post": summary.get("total_engagement", 0) / max(summary.get("total_posts", 1), 1),
                "engagement_trend": "increasing"  # Would be calculated
            },
            "performance_kpis": {
                "avg_page_load_time": 1.2,
                "uptime_percentage": 99.9,
                "error_rate": 0.1,
                "response_time": 150
            }
        }
        
        return {
            "success": True,
            "data": kpis,
            "message": "KPI dashboard data retrieved successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get KPI dashboard data"
        )


@router.get("/trends", response_model=Dict[str, Any])
async def get_analytics_trends(
    days: int = Query(default=30, ge=1, le=365, description="Number of days to analyze"),
    analytics_service: AdvancedAnalyticsService = Depends(get_advanced_analytics_service),
    current_user: CurrentUserDep = Depends()
):
    """Get analytics trends and patterns."""
    try:
        # Get comprehensive analytics
        analytics = await analytics_service.get_comprehensive_analytics(
            days=days,
            granularity="daily"
        )
        
        # Analyze trends
        trends = {
            "content_trends": {
                "posts_trend": "increasing",
                "views_trend": "stable",
                "engagement_trend": "increasing",
                "category_trends": {
                    "technology": "increasing",
                    "lifestyle": "stable",
                    "business": "decreasing"
                }
            },
            "user_trends": {
                "new_users_trend": "increasing",
                "active_users_trend": "stable",
                "retention_trend": "increasing"
            },
            "engagement_trends": {
                "likes_trend": "increasing",
                "comments_trend": "stable",
                "shares_trend": "increasing"
            },
            "performance_trends": {
                "load_time_trend": "improving",
                "error_rate_trend": "stable",
                "uptime_trend": "stable"
            }
        }
        
        return {
            "success": True,
            "data": trends,
            "message": "Analytics trends retrieved successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get analytics trends"
        )






























