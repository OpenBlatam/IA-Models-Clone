"""
🚀 AI VIDEO API MODULE
======================

Modular FastAPI implementation for the AI Video system.

This module contains:
- models: Pydantic data models
- video_service: Business logic service
- dependencies: FastAPI dependencies
- routers: API endpoints
- background_tasks: Background task utilities
- middleware: Middleware stack
- main: Main application
"""

from typing import Any, List, Dict, Optional
import logging
import asyncio

from .models import (
    VideoStatus,
    VideoQuality,
    ProcessingPriority,
    VideoData,
    VideoResponse,
    BatchVideoRequest,
    BatchVideoResponse,
    ErrorResponse,
    PaginationParams,
    VideoListResponse,
)
from .video_service import VideoService, video_service
from .dependencies import (
    get_video_service,
    get_current_user,
    validate_video_id,
    validate_pagination_params,
    check_rate_limit,
    log_request,
)
from .routers import (
    video_router,
    analytics_router,
    health_router,
)
from .background_tasks import (
    cleanup_temp_files,
    cleanup_old_videos,
    process_video_async,
    generate_thumbnail_async,
    send_processing_notification,
    send_batch_completion_notification,
    update_analytics,
    aggregate_daily_metrics,
    health_check_task,
    database_cleanup_task,
    create_background_task,
    schedule_periodic_task,
)
from .middleware import (
    create_middleware_stack,
    SecurityMiddleware,
    LoggingMiddleware,
    add_custom_headers,
    get_request_info,
)
from .main import app


__all__ = [
    # Models
    "VideoStatus", "VideoQuality", "ProcessingPriority",
    "VideoData", "VideoResponse", "BatchVideoRequest", "BatchVideoResponse",
    "ErrorResponse", "PaginationParams", "VideoListResponse",
    
    # Services
    "VideoService", "video_service",
    
    # Dependencies
    "get_video_service", "get_current_user",
    "validate_video_id", "validate_pagination_params",
    "check_rate_limit", "log_request",
    
    # Routers
    "video_router", "analytics_router", "health_router",
    
    # Background Tasks
    "cleanup_temp_files", "cleanup_old_videos",
    "process_video_async", "generate_thumbnail_async",
    "send_processing_notification", "send_batch_completion_notification",
    "update_analytics", "aggregate_daily_metrics",
    "health_check_task", "database_cleanup_task",
    "create_background_task", "schedule_periodic_task",
    
    # Middleware
    "create_middleware_stack", "SecurityMiddleware", "LoggingMiddleware",
    "add_custom_headers", "get_request_info",
    
    # Main App
    "app"
] 