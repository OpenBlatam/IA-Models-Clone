from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any, List
import logging
from ..core.database import get_session
from ..core.auth import get_current_admin_user
from ..models.schemas import (
from ..services.admin_service import (
from ..utils.helpers import generate_request_id, format_error_message
        from ..routers.health_routes import get_detailed_health_status
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Admin routes for HeyGen AI API
Provides administrative endpoints for system management with Pydantic models and type hints.
"""


    SystemSettingsInput,
    SystemSettingsOutput,
    RateLimitInput,
    RateLimitOutput
)
    get_system_settings,
    update_system_settings,
    get_system_statistics,
    get_user_analytics,
    get_video_analytics,
    update_rate_limits,
    get_rate_limits,
    clear_cache,
    backup_database
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/dashboard")
async def admin_dashboard(
    current_admin: Dict[str, Any] = Depends(get_current_admin_user),
    session: AsyncSession = Depends(get_session)
) -> Dict[str, Any]:
    """Get admin dashboard data"""
    try:
        system_stats = await get_system_statistics(session)
        user_analytics = await get_user_analytics(session)
        video_analytics = await get_video_analytics(session)
        
        return {
            "system_statistics": system_stats,
            "user_analytics": user_analytics,
            "video_analytics": video_analytics,
            "timestamp": "2024-01-01T00:00:00Z"
        }
    except Exception as e:
        logger.error(f"Error getting admin dashboard: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get admin dashboard"
        )


@router.get("/settings", response_model=SystemSettingsOutput)
async def get_system_settings_endpoint(
    current_admin: Dict[str, Any] = Depends(get_current_admin_user)
) -> SystemSettingsOutput:
    """Get system settings"""
    try:
        settings = await get_system_settings()
        return SystemSettingsOutput(**settings)
    except Exception as e:
        logger.error(f"Error getting system settings: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get system settings"
        )


@router.put("/settings", response_model=SystemSettingsOutput)
async def update_system_settings_endpoint(
    settings_data: SystemSettingsInput,
    current_admin: Dict[str, Any] = Depends(get_current_admin_user)
) -> SystemSettingsOutput:
    """Update system settings"""
    try:
        updated_settings = await update_system_settings(settings_data)
        return SystemSettingsOutput(**updated_settings)
    except Exception as e:
        logger.error(f"Error updating system settings: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update system settings"
        )


@router.get("/rate-limits", response_model=RateLimitOutput)
async def get_rate_limits_endpoint(
    current_admin: Dict[str, Any] = Depends(get_current_admin_user)
) -> RateLimitOutput:
    """Get current rate limits"""
    try:
        rate_limits = await get_rate_limits()
        return RateLimitOutput(**rate_limits)
    except Exception as e:
        logger.error(f"Error getting rate limits: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get rate limits"
        )


@router.put("/rate-limits", response_model=RateLimitOutput)
async def update_rate_limits_endpoint(
    rate_limit_data: RateLimitInput,
    current_admin: Dict[str, Any] = Depends(get_current_admin_user)
) -> RateLimitOutput:
    """Update rate limits"""
    try:
        updated_rate_limits = await update_rate_limits(rate_limit_data)
        return RateLimitOutput(**updated_rate_limits)
    except Exception as e:
        logger.error(f"Error updating rate limits: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update rate limits"
        )


@router.post("/cache/clear")
async def clear_cache_endpoint(
    current_admin: Dict[str, Any] = Depends(get_current_admin_user)
) -> Dict[str, Any]:
    """Clear system cache"""
    try:
        result = await clear_cache()
        return {
            "message": "Cache cleared successfully",
            "cleared_items": result.get("cleared_items", 0)
        }
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear cache"
        )


@router.post("/backup")
async def backup_database_endpoint(
    current_admin: Dict[str, Any] = Depends(get_current_admin_user)
) -> Dict[str, Any]:
    """Create database backup"""
    try:
        backup_info = await backup_database()
        return {
            "message": "Database backup created successfully",
            "backup_file": backup_info.get("backup_file"),
            "backup_size": backup_info.get("backup_size"),
            "timestamp": backup_info.get("timestamp")
        }
    except Exception as e:
        logger.error(f"Error creating database backup: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create database backup"
        )


@router.get("/logs")
async def get_system_logs(
    level: str = "INFO",
    limit: int = 100,
    current_admin: Dict[str, Any] = Depends(get_current_admin_user)
) -> Dict[str, Any]:
    """Get system logs"""
    try:
        # In a real implementation, you would read from log files
        logs = [
            {
                "timestamp": "2024-01-01T00:00:00Z",
                "level": "INFO",
                "message": "System log entry",
                "module": "admin"
            }
        ]
        
        return {
            "logs": logs,
            "total_count": len(logs),
            "level": level,
            "limit": limit
        }
    except Exception as e:
        logger.error(f"Error getting system logs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get system logs"
        )


@router.get("/health/detailed")
async def detailed_health_check(
    current_admin: Dict[str, Any] = Depends(get_current_admin_user),
    session: AsyncSession = Depends(get_session)
) -> Dict[str, Any]:
    """Get detailed system health information"""
    try:
        health_status = await get_detailed_health_status(session)
        return health_status
    except Exception as e:
        logger.error(f"Error getting detailed health status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get detailed health status"
        )


# Named exports
__all__ = ["router"] 