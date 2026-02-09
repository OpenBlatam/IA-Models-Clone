from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

from typing import List, Optional, Dict, Any, Tuple
from fastapi import APIRouter, Depends, HTTPException, status, Query, Path
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
import structlog
from datetime import datetime
from ..schemas.functional_models import (
from ..services.functional_services import (
from ..core.dependencies import get_db, get_current_active_user
from ..core.database import get_user_repository, get_video_repository
        from ..core.database import get_model_usage_repository
        from ..core.database import get_api_key_repository
from typing import Any, List, Dict, Optional
import logging
import asyncio
#!/usr/bin/env python3
"""
Functional API Endpoints for HeyGen AI API
Pure functions and functional programming patterns for API endpoints.
"""


    UserCreate, UserUpdate, UserResponse, UserSummary,
    VideoCreate, VideoUpdate, VideoResponse, VideoSummary,
    ModelUsageCreate, ModelUsageResponse,
    APIKeyCreate, APIKeyResponse,
    AnalyticsRequest, AnalyticsResponse,
    VideoStatus, VideoQuality, ModelType
)
    validate_user_data, create_user_dict, update_user_dict,
    transform_user_to_response, transform_user_to_summary,
    calculate_user_stats,
    validate_video_data, create_video_dict, update_video_status,
    transform_video_to_response, transform_video_to_summary,
    filter_videos_by_status, sort_videos_by_date,
    validate_model_usage_data, create_model_usage_dict,
    transform_model_usage_to_response,
    validate_api_key_data, create_api_key_dict,
    check_api_key_permissions, is_api_key_expired,
    transform_api_key_to_response,
    validate_analytics_request, calculate_analytics_metrics,
    transform_analytics_to_response,
    process_user_registration, process_video_creation,
    process_analytics_request,
    pipe, compose, map_with_error_handling
)

logger = structlog.get_logger()

router = APIRouter()

# =============================================================================
# Functional Response Helpers
# =============================================================================

def create_success_response(data: Any, message: str = "Success") -> Dict[str, Any]:
    """Functional success response creator."""
    return {
        "success": True,
        "message": message,
        "data": data
    }

def create_error_response(message: str, error_code: str = "ERROR") -> Dict[str, Any]:
    """Functional error response creator."""
    return {
        "success": False,
        "message": message,
        "error_code": error_code
    }

def handle_validation_error(error: str) -> JSONResponse:
    """Functional validation error handler."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=create_error_response(error, "VALIDATION_ERROR")
    )

def handle_not_found_error(resource: str) -> JSONResponse:
    """Functional not found error handler."""
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content=create_error_response(f"{resource} not found", "NOT_FOUND")
    )

# =============================================================================
# User Endpoints (Functional)
# =============================================================================

@router.post("/users/", response_model=Dict[str, Any])
async def create_user(
    user_create: UserCreate,
    db: AsyncSession = Depends(get_db)
) -> JSONResponse:
    """Functional user creation endpoint."""
    try:
        # Process user registration using functional pipeline
        user_data, error = process_user_registration(user_create.dict())
        if error:
            return handle_validation_error(error)
        
        # Get user repository
        user_repo = get_user_repository(db)
        
        # Create user in database
        user = await user_repo.create(**user_data)
        
        # Transform to response
        response_data = transform_user_to_response(user.to_dict())
        
        logger.info("User created successfully", user_id=user.id, username=user.username)
        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content=create_success_response(response_data.dict(), "User created successfully")
        )
        
    except Exception as e:
        logger.error("Error creating user", error=str(e))
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=create_error_response("Failed to create user", "INTERNAL_ERROR")
        )

@router.get("/users/", response_model=Dict[str, Any])
async def get_users(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    active_only: bool = Query(True),
    db: AsyncSession = Depends(get_db)
) -> JSONResponse:
    """Functional users listing endpoint."""
    try:
        user_repo = get_user_repository(db)
        
        # Get users with functional filtering
        users = await user_repo.get_all(limit=limit, offset=skip)
        
        # Transform to response using functional composition
        transform_pipeline = compose(
            lambda users: [user.to_dict() for user in users],
            lambda users: filter_active_items(users) if active_only else users,
            lambda users: [transform_user_to_summary(user).dict() for user in users]
        )
        
        response_data = transform_pipeline(users)
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=create_success_response(response_data, f"Retrieved {len(response_data)} users")
        )
        
    except Exception as e:
        logger.error("Error retrieving users", error=str(e))
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=create_error_response("Failed to retrieve users", "INTERNAL_ERROR")
        )

@router.get("/users/{user_id}", response_model=Dict[str, Any])
async def get_user(
    user_id: int = Path(..., gt=0),
    db: AsyncSession = Depends(get_db)
) -> JSONResponse:
    """Functional user retrieval endpoint."""
    try:
        user_repo = get_user_repository(db)
        user = await user_repo.get_by_id(user_id)
        
        if not user:
            return handle_not_found_error("User")
        
        # Transform to response
        response_data = transform_user_to_response(user.to_dict())
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=create_success_response(response_data.dict())
        )
        
    except Exception as e:
        logger.error("Error retrieving user", user_id=user_id, error=str(e))
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=create_error_response("Failed to retrieve user", "INTERNAL_ERROR")
        )

@router.put("/users/{user_id}", response_model=Dict[str, Any])
async def update_user(
    user_update: UserUpdate,
    user_id: int = Path(..., gt=0),
    db: AsyncSession = Depends(get_db)
) -> JSONResponse:
    """Functional user update endpoint."""
    try:
        user_repo = get_user_repository(db)
        
        # Get existing user
        existing_user = await user_repo.get_by_id(user_id)
        if not existing_user:
            return handle_not_found_error("User")
        
        # Update user using functional approach
        update_data = {k: v for k, v in user_update.dict().items() if v is not None}
        updated_user_dict = update_user_dict(existing_user.to_dict(), update_data)
        
        # Update in database
        updated_user = await user_repo.update(user_id, **update_data)
        
        # Transform to response
        response_data = transform_user_to_response(updated_user.to_dict())
        
        logger.info("User updated successfully", user_id=user_id)
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=create_success_response(response_data.dict(), "User updated successfully")
        )
        
    except Exception as e:
        logger.error("Error updating user", user_id=user_id, error=str(e))
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=create_error_response("Failed to update user", "INTERNAL_ERROR")
        )

@router.get("/users/{user_id}/stats", response_model=Dict[str, Any])
async def get_user_stats(
    user_id: int = Path(..., gt=0),
    db: AsyncSession = Depends(get_db)
) -> JSONResponse:
    """Functional user statistics endpoint."""
    try:
        user_repo = get_user_repository(db)
        video_repo = get_video_repository(db)
        
        # Get user and videos
        user = await user_repo.get_by_id(user_id)
        if not user:
            return handle_not_found_error("User")
        
        videos = await video_repo.get_user_videos(user_id)
        
        # Calculate stats using functional approach
        stats = calculate_user_stats(user.to_dict(), [v.to_dict() for v in videos])
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=create_success_response(stats, "User statistics retrieved")
        )
        
    except Exception as e:
        logger.error("Error retrieving user stats", user_id=user_id, error=str(e))
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=create_error_response("Failed to retrieve user statistics", "INTERNAL_ERROR")
        )

# =============================================================================
# Video Endpoints (Functional)
# =============================================================================

@router.post("/videos/", response_model=Dict[str, Any])
async def create_video(
    video_create: VideoCreate,
    current_user = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> JSONResponse:
    """Functional video creation endpoint."""
    try:
        # Process video creation using functional pipeline
        video_data, error = process_video_creation(video_create.dict(), current_user.id)
        if error:
            return handle_validation_error(error)
        
        # Get video repository
        video_repo = get_video_repository(db)
        
        # Create video in database
        video = await video_repo.create(**video_data)
        
        # Transform to response
        response_data = transform_video_to_response(video.to_dict())
        
        logger.info("Video created successfully", video_id=video.id, user_id=current_user.id)
        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content=create_success_response(response_data.dict(), "Video created successfully")
        )
        
    except Exception as e:
        logger.error("Error creating video", error=str(e))
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=create_error_response("Failed to create video", "INTERNAL_ERROR")
        )

@router.get("/videos/", response_model=Dict[str, Any])
async def get_videos(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    status_filter: Optional[VideoStatus] = Query(None),
    quality_filter: Optional[VideoQuality] = Query(None),
    current_user = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> JSONResponse:
    """Functional videos listing endpoint."""
    try:
        video_repo = get_video_repository(db)
        
        # Get user videos
        videos = await video_repo.get_user_videos(current_user.id, limit=limit, offset=skip)
        
        # Transform to response using functional composition
        def filter_videos(videos) -> Any:
            video_dicts = [v.to_dict() for v in videos]
            
            # Apply filters
            if status_filter:
                video_dicts = filter_videos_by_status(video_dicts, status_filter)
            if quality_filter:
                video_dicts = [v for v in video_dicts if v.get('quality') == quality_filter]
            
            # Sort by date
            video_dicts = sort_videos_by_date(video_dicts)
            
            # Transform to summaries
            return [transform_video_to_summary(v).dict() for v in video_dicts]
        
        response_data = filter_videos(videos)
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=create_success_response(response_data, f"Retrieved {len(response_data)} videos")
        )
        
    except Exception as e:
        logger.error("Error retrieving videos", error=str(e))
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=create_error_response("Failed to retrieve videos", "INTERNAL_ERROR")
        )

@router.get("/videos/{video_id}", response_model=Dict[str, Any])
async def get_video(
    video_id: int = Path(..., gt=0),
    current_user = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> JSONResponse:
    """Functional video retrieval endpoint."""
    try:
        video_repo = get_video_repository(db)
        video = await video_repo.get_by_id(video_id)
        
        if not video or video.user_id != current_user.id:
            return handle_not_found_error("Video")
        
        # Transform to response
        response_data = transform_video_to_response(video.to_dict())
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=create_success_response(response_data.dict())
        )
        
    except Exception as e:
        logger.error("Error retrieving video", video_id=video_id, error=str(e))
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=create_error_response("Failed to retrieve video", "INTERNAL_ERROR")
        )

@router.put("/videos/{video_id}/status", response_model=Dict[str, Any])
async def update_video_status(
    status_update: Dict[str, Any],
    video_id: int = Path(..., gt=0),
    current_user = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> JSONResponse:
    """Functional video status update endpoint."""
    try:
        video_repo = get_video_repository(db)
        
        # Get existing video
        video = await video_repo.get_by_id(video_id)
        if not video or video.user_id != current_user.id:
            return handle_not_found_error("Video")
        
        # Update status using functional approach
        new_status = status_update.get('status')
        if not new_status or new_status not in [s.value for s in VideoStatus]:
            return handle_validation_error("Invalid status")
        
        # Update video status
        updated_video_dict = update_video_status(
            video.to_dict(), 
            VideoStatus(new_status),
            **{k: v for k, v in status_update.items() if k != 'status'}
        )
        
        # Update in database
        updated_video = await video_repo.update(video_id, **status_update)
        
        # Transform to response
        response_data = transform_video_to_response(updated_video.to_dict())
        
        logger.info("Video status updated", video_id=video_id, status=new_status)
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=create_success_response(response_data.dict(), "Video status updated")
        )
        
    except Exception as e:
        logger.error("Error updating video status", video_id=video_id, error=str(e))
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=create_error_response("Failed to update video status", "INTERNAL_ERROR")
        )

# =============================================================================
# Model Usage Endpoints (Functional)
# =============================================================================

@router.post("/model-usage/", response_model=Dict[str, Any])
async def log_model_usage(
    usage_create: ModelUsageCreate,
    current_user = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> JSONResponse:
    """Functional model usage logging endpoint."""
    try:
        # Validate usage data
        is_valid, error = validate_model_usage_data(usage_create.dict())
        if not is_valid:
            return handle_validation_error(error)
        
        # Create usage dictionary
        usage_data = create_model_usage_dict(
            usage_create.dict(), 
            current_user.id, 
            usage_create.video_id
        )
        
        # Get repository and create usage record
        usage_repo = get_model_usage_repository(db)
        usage = await usage_repo.create(**usage_data)
        
        # Transform to response
        response_data = transform_model_usage_to_response(usage.to_dict())
        
        logger.info("Model usage logged", usage_id=usage.id, model_type=usage.model_type)
        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content=create_success_response(response_data.dict(), "Model usage logged")
        )
        
    except Exception as e:
        logger.error("Error logging model usage", error=str(e))
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=create_error_response("Failed to log model usage", "INTERNAL_ERROR")
        )

# =============================================================================
# API Key Endpoints (Functional)
# =============================================================================

@router.post("/api-keys/", response_model=Dict[str, Any])
async def create_api_key(
    key_create: APIKeyCreate,
    current_user = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> JSONResponse:
    """Functional API key creation endpoint."""
    try:
        # Validate API key data
        is_valid, error = validate_api_key_data(key_create.dict())
        if not is_valid:
            return handle_validation_error(error)
        
        # Create API key dictionary
        key_data, api_key = create_api_key_dict(key_create.dict(), current_user.id)
        
        # Get repository and create API key
        key_repo = get_api_key_repository(db)
        key_record = await key_repo.create(**key_data)
        
        # Transform to response
        response_data = transform_api_key_to_response(key_record.to_dict())
        response_data_dict = response_data.dict()
        response_data_dict['api_key'] = api_key  # Include the actual key for display
        
        logger.info("API key created", key_id=key_record.id, user_id=current_user.id)
        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content=create_success_response(response_data_dict, "API key created successfully")
        )
        
    except Exception as e:
        logger.error("Error creating API key", error=str(e))
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=create_error_response("Failed to create API key", "INTERNAL_ERROR")
        )

# =============================================================================
# Analytics Endpoints (Functional)
# =============================================================================

@router.post("/analytics/", response_model=Dict[str, Any])
async def get_analytics(
    analytics_request: AnalyticsRequest,
    current_user = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> JSONResponse:
    """Functional analytics endpoint."""
    try:
        # Get repositories
        user_repo = get_user_repository(db)
        video_repo = get_video_repository(db)
        
        # Get data for analytics
        users = await user_repo.get_all()
        videos = await video_repo.get_all()
        
        # Process analytics using functional pipeline
        response_data, error = process_analytics_request(
            analytics_request.dict(),
            [v.to_dict() for v in videos],
            [u.to_dict() for u in users]
        )
        
        if error:
            return handle_validation_error(error)
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=create_success_response(response_data.dict(), "Analytics retrieved successfully")
        )
        
    except Exception as e:
        logger.error("Error retrieving analytics", error=str(e))
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=create_error_response("Failed to retrieve analytics", "INTERNAL_ERROR")
        )

# =============================================================================
# Health and Status Endpoints (Functional)
# =============================================================================

@router.get("/health", response_model=Dict[str, Any])
async def health_check() -> JSONResponse:
    """Functional health check endpoint."""
    try:
        health_data = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "services": {
                "api": "healthy",
                "database": "healthy",
                "cache": "healthy"
            }
        }
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=create_success_response(health_data, "Service is healthy")
        )
        
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=create_error_response("Service is unhealthy", "SERVICE_UNAVAILABLE")
        )

@router.get("/status", response_model=Dict[str, Any])
async def get_status(
    current_user = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> JSONResponse:
    """Functional status endpoint."""
    try:
        # Get user repository
        user_repo = get_user_repository(db)
        video_repo = get_video_repository(db)
        
        # Get user stats
        user_videos = await video_repo.get_user_videos(current_user.id)
        
        # Calculate status using functional approach
        status_data = {
            "user": {
                "id": current_user.id,
                "username": current_user.username,
                "email": current_user.email,
                "is_active": current_user.is_active
            },
            "videos": {
                "total": len(user_videos),
                "completed": len([v for v in user_videos if v.status == VideoStatus.COMPLETED]),
                "processing": len([v for v in user_videos if v.status == VideoStatus.PROCESSING]),
                "failed": len([v for v in user_videos if v.status == VideoStatus.FAILED])
            },
            "last_activity": current_user.updated_at.isoformat() if current_user.updated_at else None
        }
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=create_success_response(status_data, "Status retrieved successfully")
        )
        
    except Exception as e:
        logger.error("Error retrieving status", error=str(e))
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=create_error_response("Failed to retrieve status", "INTERNAL_ERROR")
        )

# =============================================================================
# Export router
# =============================================================================

__all__ = ['router'] 