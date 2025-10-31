from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging
from ..core.database import get_session
from ..core.auth import get_current_user, get_current_admin_user
from ..models.schemas import (
from ..services.analytics_service import (
from ..utils.helpers import generate_request_id, format_error_message
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Analytics routes for HeyGen AI API
Provides data analytics and reporting endpoints with Pydantic models and type hints.
"""


    AnalyticsInput,
    AnalyticsOutput,
    DateRangeInput,
    UserAnalyticsOutput,
    VideoAnalyticsOutput
)
    get_user_analytics,
    get_video_analytics,
    get_system_analytics,
    get_usage_statistics,
    get_performance_metrics,
    get_revenue_analytics
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/user", response_model=UserAnalyticsOutput)
async def get_user_analytics_endpoint(
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    current_user: Dict[str, Any] = Depends(get_current_user),
    session: AsyncSession = Depends(get_session)
) -> UserAnalyticsOutput:
    """Get user-specific analytics"""
    try:
        # Default to last 30 days if no date range provided
        if not date_from:
            date_from = datetime.utcnow() - timedelta(days=30)
        if not date_to:
            date_to = datetime.utcnow()
        
        analytics = await get_user_analytics(
            session, 
            current_user["user_id"], 
            date_from, 
            date_to
        )
        
        return UserAnalyticsOutput(**analytics)
    except Exception as e:
        logger.error(f"Error getting user analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get user analytics"
        )


@router.get("/video", response_model=VideoAnalyticsOutput)
async def get_video_analytics_endpoint(
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    current_user: Dict[str, Any] = Depends(get_current_user),
    session: AsyncSession = Depends(get_session)
) -> VideoAnalyticsOutput:
    """Get video generation analytics"""
    try:
        # Default to last 30 days if no date range provided
        if not date_from:
            date_from = datetime.utcnow() - timedelta(days=30)
        if not date_to:
            date_to = datetime.utcnow()
        
        analytics = await get_video_analytics(
            session, 
            current_user["user_id"], 
            date_from, 
            date_to
        )
        
        return VideoAnalyticsOutput(**analytics)
    except Exception as e:
        logger.error(f"Error getting video analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get video analytics"
        )


@router.get("/usage")
async def get_usage_statistics_endpoint(
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    current_user: Dict[str, Any] = Depends(get_current_user),
    session: AsyncSession = Depends(get_session)
) -> Dict[str, Any]:
    """Get usage statistics for current user"""
    try:
        # Default to last 30 days if no date range provided
        if not date_from:
            date_from = datetime.utcnow() - timedelta(days=30)
        if not date_to:
            date_to = datetime.utcnow()
        
        usage_stats = await get_usage_statistics(
            session, 
            current_user["user_id"], 
            date_from, 
            date_to
        )
        
        return usage_stats
    except Exception as e:
        logger.error(f"Error getting usage statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get usage statistics"
        )


@router.get("/performance")
async def get_performance_metrics_endpoint(
    current_user: Dict[str, Any] = Depends(get_current_user),
    session: AsyncSession = Depends(get_session)
) -> Dict[str, Any]:
    """Get performance metrics for current user"""
    try:
        performance_metrics = await get_performance_metrics(
            session, 
            current_user["user_id"]
        )
        
        return performance_metrics
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get performance metrics"
        )


# Admin-only analytics endpoints
@router.get("/system")
async def get_system_analytics_endpoint(
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    current_admin: Dict[str, Any] = Depends(get_current_admin_user),
    session: AsyncSession = Depends(get_session)
) -> Dict[str, Any]:
    """Get system-wide analytics (admin only)"""
    try:
        # Default to last 30 days if no date range provided
        if not date_from:
            date_from = datetime.utcnow() - timedelta(days=30)
        if not date_to:
            date_to = datetime.utcnow()
        
        system_analytics = await get_system_analytics(session, date_from, date_to)
        
        return system_analytics
    except Exception as e:
        logger.error(f"Error getting system analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get system analytics"
        )


@router.get("/revenue")
async def get_revenue_analytics_endpoint(
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    current_admin: Dict[str, Any] = Depends(get_current_admin_user),
    session: AsyncSession = Depends(get_session)
) -> Dict[str, Any]:
    """Get revenue analytics (admin only)"""
    try:
        # Default to last 30 days if no date range provided
        if not date_from:
            date_from = datetime.utcnow() - timedelta(days=30)
        if not date_to:
            date_to = datetime.utcnow()
        
        revenue_analytics = await get_revenue_analytics(session, date_from, date_to)
        
        return revenue_analytics
    except Exception as e:
        logger.error(f"Error getting revenue analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get revenue analytics"
        )


@router.get("/export")
async def export_analytics_endpoint(
    analytics_type: str,
    format: str = "json",
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    current_admin: Dict[str, Any] = Depends(get_current_admin_user),
    session: AsyncSession = Depends(get_session)
) -> Dict[str, Any]:
    """Export analytics data (admin only)"""
    try:
        # Default to last 30 days if no date range provided
        if not date_from:
            date_from = datetime.utcnow() - timedelta(days=30)
        if not date_to:
            date_to = datetime.utcnow()
        
        # Validate analytics type
        valid_types = ["user", "video", "system", "revenue", "performance"]
        if analytics_type not in valid_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid analytics type. Must be one of: {', '.join(valid_types)}"
            )
        
        # Validate format
        valid_formats = ["json", "csv", "xlsx"]
        if format not in valid_formats:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid format. Must be one of: {', '.join(valid_formats)}"
            )
        
        # In a real implementation, you would generate and return the export file
        export_data = {
            "analytics_type": analytics_type,
            "format": format,
            "date_from": date_from.isoformat(),
            "date_to": date_to.isoformat(),
            "export_url": f"/exports/{analytics_type}_{format}_{date_from.date()}_{date_to.date()}.{format}",
            "generated_at": datetime.utcnow().isoformat()
        }
        
        return export_data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to export analytics"
        )


# Named exports
__all__ = ["router"] 