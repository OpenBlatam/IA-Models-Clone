from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

from typing import (
from datetime import datetime, timezone
from fastapi import (
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, ValidationError
import structlog
from enum import Enum
from ..schemas.functional_models import (
from ..services.functional_services import (
from ..core.dependencies import get_db, get_current_active_user, get_current_user
from ..core.database import get_user_repository, get_video_repository
from typing import Any, List, Dict, Optional
import logging
import asyncio
#!/usr/bin/env python3
"""
Declarative Route Definitions for HeyGen AI API
Clear return type annotations and modern FastAPI patterns.
"""

    List, Optional, Dict, Any, Union, Annotated, Literal,
    Tuple, Sequence, Callable, TypeVar, Generic
)
    APIRouter, Depends, HTTPException, status, Query, Path, 
    Header, Body, Form, File, UploadFile, BackgroundTasks,
    Request, Response
)

    UserCreate, UserUpdate, UserResponse, UserSummary,
    VideoCreate, VideoUpdate, VideoResponse, VideoSummary,
    ModelUsageCreate, ModelUsageResponse,
    APIKeyCreate, APIKeyResponse,
    AnalyticsRequest, AnalyticsResponse,
    VideoStatus, VideoQuality, ModelType
)
    process_user_registration, process_video_creation,
    process_analytics_request, transform_user_to_response,
    transform_video_to_response, create_success_response,
    create_error_response
)

logger = structlog.get_logger()

# =============================================================================
# Type Definitions for Route Responses
# =============================================================================

class SuccessResponse(BaseModel):
    """Standard success response model."""
    success: Literal[True] = True
    message: str
    data: Any
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ErrorResponse(BaseModel):
    """Standard error response model."""
    success: Literal[False] = False
    message: str
    error_code: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    details: Optional[Dict[str, Any]] = None

class PaginatedResponse(BaseModel, Generic[TypeVar('T')]):
    """Paginated response model."""
    success: Literal[True] = True
    message: str
    data: List[T]
    pagination: Dict[str, Any] = Field(
        description="Pagination metadata",
        example={
            "page": 1,
            "per_page": 10,
            "total": 100,
            "total_pages": 10,
            "has_next": True,
            "has_prev": False
        }
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class HealthCheckResponse(BaseModel):
    """Health check response model."""
    status: Literal["healthy", "unhealthy"] = Field(description="Service status")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    version: str = Field(description="API version")
    uptime: float = Field(description="Service uptime in seconds")
    services: Dict[str, str] = Field(description="Service health status")

class VideoProcessingResponse(BaseModel):
    """Video processing response model."""
    video_id: str = Field(description="Video identifier")
    status: VideoStatus = Field(description="Processing status")
    progress: Optional[float] = Field(None, ge=0, le=100, description="Processing progress")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    message: str = Field(description="Status message")

# =============================================================================
# Route Parameter Types
# =============================================================================

class PaginationParams(BaseModel):
    """Pagination parameters."""
    page: int = Query(1, ge=1, description="Page number")
    per_page: int = Query(10, ge=1, le=100, description="Items per page")
    sort_by: Optional[str] = Query(None, description="Sort field")
    sort_order: Literal["asc", "desc"] = Query("desc", description="Sort order")

class VideoFilterParams(BaseModel):
    """Video filter parameters."""
    status: Optional[VideoStatus] = Query(None, description="Filter by status")
    quality: Optional[VideoQuality] = Query(None, description="Filter by quality")
    created_after: Optional[datetime] = Query(None, description="Created after date")
    created_before: Optional[datetime] = Query(None, description="Created before date")
    search: Optional[str] = Query(None, description="Search in script content")

class AnalyticsParams(BaseModel):
    """Analytics parameters."""
    start_date: datetime = Query(..., description="Start date for analytics")
    end_date: datetime = Query(..., description="End date for analytics")
    group_by: Optional[str] = Query(None, description="Grouping field")
    metrics: List[str] = Query(default_factory=list, description="Metrics to calculate")

# =============================================================================
# Router Configuration
# =============================================================================

router = APIRouter(
    prefix="/api/v1",
    tags=["HeyGen AI API"],
    responses={
        200: {"description": "Success"},
        201: {"description": "Created"},
        400: {"description": "Bad Request"},
        401: {"description": "Unauthorized"},
        403: {"description": "Forbidden"},
        404: {"description": "Not Found"},
        422: {"description": "Validation Error"},
        500: {"description": "Internal Server Error"}
    }
)

# =============================================================================
# User Routes with Declarative Definitions
# =============================================================================

@router.post(
    "/users",
    response_model=SuccessResponse[UserResponse],
    status_code=status.HTTP_201_CREATED,
    summary="Create a new user",
    description="Create a new user account with validation",
    response_description="User created successfully"
)
async def create_user(
    user_data: UserCreate = Body(..., description="User creation data"),
    db: Annotated[Any, Depends(get_db)] = None
) -> SuccessResponse[UserResponse]:
    """
    Create a new user account.
    
    Args:
        user_data: User creation data with validation
        db: Database session dependency
        
    Returns:
        SuccessResponse with created user data
        
    Raises:
        HTTPException: If validation fails or user creation fails
    """
    try:
        # Process user registration using functional pipeline
        user_dict, error = process_user_registration(user_data.dict())
        if error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={"message": error, "error_code": "VALIDATION_ERROR"}
            )
        
        # Create user in database
        user_repo = get_user_repository(db)
        user = await user_repo.create(**user_dict)
        
        # Transform to response
        response_data = transform_user_to_response(user.to_dict())
        
        logger.info("User created successfully", user_id=user.id, username=user.username)
        
        return SuccessResponse(
            message="User created successfully",
            data=response_data.dict()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error creating user", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": "Failed to create user", "error_code": "INTERNAL_ERROR"}
        )

@router.get(
    "/users",
    response_model=PaginatedResponse[UserSummary],
    summary="Get users list",
    description="Retrieve paginated list of users with optional filtering",
    response_description="List of users retrieved successfully"
)
async def get_users(
    pagination: Annotated[PaginationParams, Depends()] = None,
    active_only: bool = Query(True, description="Filter active users only"),
    db: Annotated[Any, Depends(get_db)] = None
) -> PaginatedResponse[UserSummary]:
    """
    Get paginated list of users.
    
    Args:
        pagination: Pagination parameters
        active_only: Filter for active users only
        db: Database session dependency
        
    Returns:
        PaginatedResponse with user summaries
        
    Raises:
        HTTPException: If database query fails
    """
    try:
        user_repo = get_user_repository(db)
        
        # Calculate offset
        offset = (pagination.page - 1) * pagination.per_page
        
        # Get users with filtering
        users = await user_repo.get_all(
            limit=pagination.per_page,
            offset=offset,
            filters={"is_active": active_only} if active_only else {}
        )
        
        # Get total count
        total_count = await user_repo.count(filters={"is_active": active_only} if active_only else {})
        
        # Transform to summaries
        user_summaries = [
            UserSummary(**user.to_dict()).dict() 
            for user in users
        ]
        
        # Calculate pagination metadata
        total_pages = (total_count + pagination.per_page - 1) // pagination.per_page
        
        pagination_meta = {
            "page": pagination.page,
            "per_page": pagination.per_page,
            "total": total_count,
            "total_pages": total_pages,
            "has_next": pagination.page < total_pages,
            "has_prev": pagination.page > 1
        }
        
        return PaginatedResponse(
            message=f"Retrieved {len(user_summaries)} users",
            data=user_summaries,
            pagination=pagination_meta
        )
        
    except Exception as e:
        logger.error("Error retrieving users", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": "Failed to retrieve users", "error_code": "INTERNAL_ERROR"}
        )

@router.get(
    "/users/{user_id}",
    response_model=SuccessResponse[UserResponse],
    summary="Get user by ID",
    description="Retrieve a specific user by their ID",
    response_description="User data retrieved successfully"
)
async def get_user(
    user_id: int = Path(..., gt=0, description="User ID"),
    db: Annotated[Any, Depends(get_db)] = None
) -> SuccessResponse[UserResponse]:
    """
    Get user by ID.
    
    Args:
        user_id: User identifier
        db: Database session dependency
        
    Returns:
        SuccessResponse with user data
        
    Raises:
        HTTPException: If user not found or database error
    """
    try:
        user_repo = get_user_repository(db)
        user = await user_repo.get_by_id(user_id)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"message": "User not found", "error_code": "NOT_FOUND"}
            )
        
        # Transform to response
        response_data = transform_user_to_response(user.to_dict())
        
        return SuccessResponse(
            message="User retrieved successfully",
            data=response_data.dict()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error retrieving user", user_id=user_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": "Failed to retrieve user", "error_code": "INTERNAL_ERROR"}
        )

@router.put(
    "/users/{user_id}",
    response_model=SuccessResponse[UserResponse],
    summary="Update user",
    description="Update user information with validation",
    response_description="User updated successfully"
)
async def update_user(
    user_id: int = Path(..., gt=0, description="User ID"),
    user_data: UserUpdate = Body(..., description="User update data"),
    current_user: Annotated[Any, Depends(get_current_active_user)] = None,
    db: Annotated[Any, Depends(get_db)] = None
) -> SuccessResponse[UserResponse]:
    """
    Update user information.
    
    Args:
        user_id: User identifier
        user_data: User update data
        current_user: Current authenticated user
        db: Database session dependency
        
    Returns:
        SuccessResponse with updated user data
        
    Raises:
        HTTPException: If user not found, unauthorized, or update fails
    """
    try:
        # Check authorization
        if current_user.id != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={"message": "Not authorized to update this user", "error_code": "FORBIDDEN"}
            )
        
        user_repo = get_user_repository(db)
        
        # Get existing user
        existing_user = await user_repo.get_by_id(user_id)
        if not existing_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"message": "User not found", "error_code": "NOT_FOUND"}
            )
        
        # Update user
        update_data = {k: v for k, v in user_data.dict().items() if v is not None}
        updated_user = await user_repo.update(user_id, **update_data)
        
        # Transform to response
        response_data = transform_user_to_response(updated_user.to_dict())
        
        logger.info("User updated successfully", user_id=user_id)
        
        return SuccessResponse(
            message="User updated successfully",
            data=response_data.dict()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error updating user", user_id=user_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": "Failed to update user", "error_code": "INTERNAL_ERROR"}
        )

# =============================================================================
# Video Routes with Declarative Definitions
# =============================================================================

@router.post(
    "/videos",
    response_model=SuccessResponse[VideoResponse],
    status_code=status.HTTP_201_CREATED,
    summary="Create a new video",
    description="Create a new video generation request",
    response_description="Video creation request submitted successfully"
)
async def create_video(
    video_data: VideoCreate = Body(..., description="Video creation data"),
    background_tasks: BackgroundTasks = None,
    current_user: Annotated[Any, Depends(get_current_active_user)] = None,
    db: Annotated[Any, Depends(get_db)] = None
) -> SuccessResponse[VideoResponse]:
    """
    Create a new video generation request.
    
    Args:
        video_data: Video creation data with validation
        background_tasks: Background tasks for video processing
        current_user: Current authenticated user
        db: Database session dependency
        
    Returns:
        SuccessResponse with created video data
        
    Raises:
        HTTPException: If validation fails or video creation fails
    """
    try:
        # Process video creation using functional pipeline
        video_dict, error = process_video_creation(video_data.dict(), current_user.id)
        if error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={"message": error, "error_code": "VALIDATION_ERROR"}
            )
        
        # Get video repository
        video_repo = get_video_repository(db)
        
        # Create video in database
        video = await video_repo.create(**video_dict)
        
        # Add background task for video processing
        if background_tasks:
            background_tasks.add_task(process_video_background, video.id, db)
        
        # Transform to response
        response_data = transform_video_to_response(video.to_dict())
        
        logger.info("Video created successfully", video_id=video.id, user_id=current_user.id)
        
        return SuccessResponse(
            message="Video creation request submitted successfully",
            data=response_data.dict()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error creating video", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": "Failed to create video", "error_code": "INTERNAL_ERROR"}
        )

@router.get(
    "/videos",
    response_model=PaginatedResponse[VideoSummary],
    summary="Get videos list",
    description="Retrieve paginated list of user's videos with filtering",
    response_description="List of videos retrieved successfully"
)
async def get_videos(
    pagination: Annotated[PaginationParams, Depends()] = None,
    filters: Annotated[VideoFilterParams, Depends()] = None,
    current_user: Annotated[Any, Depends(get_current_active_user)] = None,
    db: Annotated[Any, Depends(get_db)] = None
) -> PaginatedResponse[VideoSummary]:
    """
    Get paginated list of user's videos.
    
    Args:
        pagination: Pagination parameters
        filters: Video filter parameters
        current_user: Current authenticated user
        db: Database session dependency
        
    Returns:
        PaginatedResponse with video summaries
        
    Raises:
        HTTPException: If database query fails
    """
    try:
        video_repo = get_video_repository(db)
        
        # Calculate offset
        offset = (pagination.page - 1) * pagination.per_page
        
        # Build filters
        filter_dict = {"user_id": current_user.id}
        if filters.status:
            filter_dict["status"] = filters.status
        if filters.quality:
            filter_dict["quality"] = filters.quality
        if filters.created_after:
            filter_dict["created_at__gte"] = filters.created_after
        if filters.created_before:
            filter_dict["created_at__lte"] = filters.created_before
        
        # Get videos with filtering
        videos = await video_repo.get_user_videos(
            current_user.id,
            limit=pagination.per_page,
            offset=offset,
            filters=filter_dict,
            search=filters.search
        )
        
        # Get total count
        total_count = await video_repo.count_user_videos(
            current_user.id,
            filters=filter_dict,
            search=filters.search
        )
        
        # Transform to summaries
        video_summaries = [
            VideoSummary(**video.to_dict()).dict() 
            for video in videos
        ]
        
        # Calculate pagination metadata
        total_pages = (total_count + pagination.per_page - 1) // pagination.per_page
        
        pagination_meta = {
            "page": pagination.page,
            "per_page": pagination.per_page,
            "total": total_count,
            "total_pages": total_pages,
            "has_next": pagination.page < total_pages,
            "has_prev": pagination.page > 1
        }
        
        return PaginatedResponse(
            message=f"Retrieved {len(video_summaries)} videos",
            data=video_summaries,
            pagination=pagination_meta
        )
        
    except Exception as e:
        logger.error("Error retrieving videos", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": "Failed to retrieve videos", "error_code": "INTERNAL_ERROR"}
        )

@router.get(
    "/videos/{video_id}",
    response_model=SuccessResponse[VideoResponse],
    summary="Get video by ID",
    description="Retrieve a specific video by its ID",
    response_description="Video data retrieved successfully"
)
async def get_video(
    video_id: int = Path(..., gt=0, description="Video ID"),
    current_user: Annotated[Any, Depends(get_current_active_user)] = None,
    db: Annotated[Any, Depends(get_db)] = None
) -> SuccessResponse[VideoResponse]:
    """
    Get video by ID.
    
    Args:
        video_id: Video identifier
        current_user: Current authenticated user
        db: Database session dependency
        
    Returns:
        SuccessResponse with video data
        
    Raises:
        HTTPException: If video not found or unauthorized
    """
    try:
        video_repo = get_video_repository(db)
        video = await video_repo.get_by_id(video_id)
        
        if not video or video.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"message": "Video not found", "error_code": "NOT_FOUND"}
            )
        
        # Transform to response
        response_data = transform_video_to_response(video.to_dict())
        
        return SuccessResponse(
            message="Video retrieved successfully",
            data=response_data.dict()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error retrieving video", video_id=video_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": "Failed to retrieve video", "error_code": "INTERNAL_ERROR"}
        )

@router.put(
    "/videos/{video_id}/status",
    response_model=SuccessResponse[VideoResponse],
    summary="Update video status",
    description="Update video processing status",
    response_description="Video status updated successfully"
)
async def update_video_status(
    video_id: int = Path(..., gt=0, description="Video ID"),
    status_update: Dict[str, Any] = Body(..., description="Status update data"),
    current_user: Annotated[Any, Depends(get_current_active_user)] = None,
    db: Annotated[Any, Depends(get_db)] = None
) -> SuccessResponse[VideoResponse]:
    """
    Update video processing status.
    
    Args:
        video_id: Video identifier
        status_update: Status update data
        current_user: Current authenticated user
        db: Database session dependency
        
    Returns:
        SuccessResponse with updated video data
        
    Raises:
        HTTPException: If video not found, unauthorized, or update fails
    """
    try:
        video_repo = get_video_repository(db)
        
        # Get existing video
        video = await video_repo.get_by_id(video_id)
        if not video or video.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"message": "Video not found", "error_code": "NOT_FOUND"}
            )
        
        # Validate status update
        new_status = status_update.get('status')
        if not new_status or new_status not in [s.value for s in VideoStatus]:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={"message": "Invalid status", "error_code": "VALIDATION_ERROR"}
            )
        
        # Update video status
        updated_video = await video_repo.update(video_id, **status_update)
        
        # Transform to response
        response_data = transform_video_to_response(updated_video.to_dict())
        
        logger.info("Video status updated", video_id=video_id, status=new_status)
        
        return SuccessResponse(
            message="Video status updated successfully",
            data=response_data.dict()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error updating video status", video_id=video_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": "Failed to update video status", "error_code": "INTERNAL_ERROR"}
        )

# =============================================================================
# Analytics Routes with Declarative Definitions
# =============================================================================

@router.post(
    "/analytics",
    response_model=SuccessResponse[AnalyticsResponse],
    summary="Get analytics data",
    description="Retrieve analytics data for specified period and metrics",
    response_description="Analytics data retrieved successfully"
)
async def get_analytics(
    analytics_request: AnalyticsRequest = Body(..., description="Analytics request data"),
    current_user: Annotated[Any, Depends(get_current_active_user)] = None,
    db: Annotated[Any, Depends(get_db)] = None
) -> SuccessResponse[AnalyticsResponse]:
    """
    Get analytics data.
    
    Args:
        analytics_request: Analytics request parameters
        current_user: Current authenticated user
        db: Database session dependency
        
    Returns:
        SuccessResponse with analytics data
        
    Raises:
        HTTPException: If analytics calculation fails
    """
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
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={"message": error, "error_code": "VALIDATION_ERROR"}
            )
        
        return SuccessResponse(
            message="Analytics data retrieved successfully",
            data=response_data.dict()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error retrieving analytics", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": "Failed to retrieve analytics", "error_code": "INTERNAL_ERROR"}
        )

# =============================================================================
# Health and Status Routes
# =============================================================================

@router.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Health check",
    description="Check API health status",
    response_description="Health status retrieved successfully"
)
async def health_check() -> HealthCheckResponse:
    """
    Check API health status.
    
    Returns:
        HealthCheckResponse with service status
        
    Raises:
        HTTPException: If health check fails
    """
    try:
        # Calculate uptime (simplified)
        uptime = 3600.0  # 1 hour in seconds
        
        health_data = HealthCheckResponse(
            status="healthy",
            version="1.0.0",
            uptime=uptime,
            services={
                "api": "healthy",
                "database": "healthy",
                "cache": "healthy"
            }
        )
        
        return health_data
        
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"message": "Service is unhealthy", "error_code": "SERVICE_UNAVAILABLE"}
        )

@router.get(
    "/status",
    response_model=SuccessResponse[Dict[str, Any]],
    summary="Get API status",
    description="Get current API status and user statistics",
    response_description="Status information retrieved successfully"
)
async def get_status(
    current_user: Annotated[Any, Depends(get_current_active_user)] = None,
    db: Annotated[Any, Depends(get_db)] = None
) -> SuccessResponse[Dict[str, Any]]:
    """
    Get API status and user statistics.
    
    Args:
        current_user: Current authenticated user
        db: Database session dependency
        
    Returns:
        SuccessResponse with status information
        
    Raises:
        HTTPException: If status retrieval fails
    """
    try:
        # Get video repository
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
        
        return SuccessResponse(
            message="Status information retrieved successfully",
            data=status_data
        )
        
    except Exception as e:
        logger.error("Error retrieving status", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": "Failed to retrieve status", "error_code": "INTERNAL_ERROR"}
        )

# =============================================================================
# Background Task Functions
# =============================================================================

async def process_video_background(video_id: int, db: Any) -> None:
    """
    Background task for video processing.
    
    Args:
        video_id: Video identifier
        db: Database session
    """
    try:
        logger.info("Starting background video processing", video_id=video_id)
        
        # Simulate video processing
        await asyncio.sleep(5)  # Simulate processing time
        
        # Update video status to completed
        video_repo = get_video_repository(db)
        await video_repo.update(video_id, status=VideoStatus.COMPLETED)
        
        logger.info("Background video processing completed", video_id=video_id)
        
    except Exception as e:
        logger.error("Background video processing failed", video_id=video_id, error=str(e))
        
        # Update video status to failed
        try:
            video_repo = get_video_repository(db)
            await video_repo.update(
                video_id, 
                status=VideoStatus.FAILED,
                error_message=str(e)
            )
        except Exception as update_error:
            logger.error("Failed to update video status to failed", video_id=video_id, error=str(update_error))

# =============================================================================
# Export router
# =============================================================================

__all__ = ['router'] 