from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException, status, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
import logging
import time
import asyncio
from typing import Dict, Any, Optional, List
import traceback
from ..core.roro import (
from ..core.database import get_session
from ..core.auth import get_current_user
from ..core.error_handling import (
from ..models.schemas import (
from ..services.video_service import (
from ..utils.helpers import (
from ..utils.validators import (
        from ..services.video_service import delete_video_record
        import os
        from ..core.database import get_session
    import random
        from ..core.database import get_session
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Enhanced Video routes using RORO pattern
Provides video generation and management endpoints with early error handling and edge case validation.
"""


    VideoGenerationRequest,
    VideoGenerationResponse,
    VideoStatusRequest,
    VideoStatusResponse,
    UserVideosRequest,
    UserVideosResponse,
    create_success_response,
    create_error_response,
    validate_roro_request
)
    handle_errors,
    error_factory,
    ErrorCategory,
    VideoProcessingError,
    VideoGenerationError,
    VideoRenderingError,
    VoiceSynthesisError,
    TemplateProcessingError,
    FileProcessingError,
    QuotaExceededError,
    ResourceNotFoundError,
    ValidationError,
    RateLimitError,
    TimeoutError,
    ConcurrencyError,
    ResourceExhaustionError,
    CircuitBreaker,
    RetryHandler,
    ErrorLogger,
    UserFriendlyMessageGenerator
)
    VideoGenerationInput,
    VideoStatusInput,
    UserVideosInput,
    VideoGenerationOutput,
    VideoStatusOutput,
    UserVideosOutput,
    QualityLevel,
    VideoStatus,
    ProcessingSettings
)
    create_video_record,
    get_video_status,
    update_video_status,
    get_user_videos,
    get_video_statistics
)
    generate_video_id,
    calculate_estimated_duration,
    parse_quality_settings,
    calculate_progress,
    generate_thumbnail_url,
    validate_video_id_format
)
    validate_video_generation_request,
    validate_video_id,
    validate_script_content,
    validate_voice_id,
    validate_language_code,
    validate_quality_settings,
    validate_video_duration,
    validate_pagination_parameters,
    validate_business_logic_constraints
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/videos", tags=["videos"])

# Circuit breakers for external services
video_processing_circuit_breaker = CircuitBreaker(
    service_name="video_processing",
    failure_threshold=3,
    recovery_timeout=30
)

# Retry handler for transient failures
retry_handler = RetryHandler(
    max_retries=3,
    base_delay=1.0,
    max_delay=10.0,
    exponential_backoff=True
)


async def _validate_request_context(request: Request, user_id: str) -> None:
    """Validate request context at the beginning of functions"""
    # Early validation - check if request has required headers
    if not request.headers.get("user-agent"):
        raise error_factory.validation_error(
            message="User-Agent header is required",
            field="user-agent",
            context={"operation": "request_validation", "user_id": user_id}
        )
    
    # Early validation - check request size
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > 1024 * 1024:  # 1MB limit
        raise error_factory.validation_error(
            message="Request payload too large",
            field="content-length",
            value=content_length,
            context={"operation": "request_validation", "user_id": user_id}
        )


def _validate_user_permissions(user_id: str, operation: str) -> None:
    """Validate user permissions at the beginning of functions"""
    # Early validation - check if user_id is valid
    if not user_id or not isinstance(user_id, str):
        raise error_factory.validation_error(
            message="Invalid user ID",
            field="user_id",
            value=user_id,
            context={"operation": operation}
        )
    
    # Early validation - check user_id format
    if len(user_id) < 3 or len(user_id) > 50:
        raise error_factory.validation_error(
            message="User ID length invalid",
            field="user_id",
            value=user_id,
            context={"operation": operation}
        )


async def _validate_rate_limits(user_id: str, operation: str) -> None:
    """Validate rate limits at the beginning of functions"""
    # Early validation - check rate limits
    # In production, this would check against Redis or database
    # For now, we'll simulate rate limit checking
    current_time = time.time()
    rate_limit_key = f"rate_limit:{user_id}:{operation}"
    
    # Simulate rate limit check
    if operation == "video_generation":
        # Allow max 10 video generations per hour
        hourly_limit = 10
        # In production: check actual usage from cache/database
        pass
    
    if operation == "api_request":
        # Allow max 1000 API requests per minute
        minute_limit = 1000
        # In production: check actual usage from cache/database
        pass


def _validate_input_data_types(data: Dict[str, Any], expected_types: Dict[str, type]) -> None:
    """Validate input data types at the beginning of functions"""
    for field, expected_type in expected_types.items():
        if field not in data:
            continue
        
        if isinstance(data[field], expected_type):
            continue
        
        raise error_factory.validation_error(
            message=f"Field '{field}' must be of type {expected_type.__name__}",
            field=field,
            value=data[field],
            context={"operation": "type_validation"}
        )


@router.post("/generate", response_model=VideoGenerationResponse)
@handle_errors(
    category=ErrorCategory.VIDEO_PROCESSING,
    operation="generate_video",
    retry_on_failure=True,
    max_retries=2,
    circuit_breaker=video_processing_circuit_breaker
)
async def generate_video_roro(
    request_data: Dict[str, Any],
    background_tasks: BackgroundTasks,
    request: Request,
    user_id: str = Depends(get_current_user),
    session: AsyncSession = Depends(get_session)
) -> VideoGenerationResponse:
    """Generate video using RORO pattern with comprehensive early validation"""
    
    # EARLY VALIDATION - Request context
    _validate_request_context(request, user_id)
    
    # EARLY VALIDATION - User permissions
    _validate_user_permissions(user_id, "video_generation")
    
    # EARLY VALIDATION - Rate limits
    await _validate_rate_limits(user_id, "video_generation")
    
    # EARLY VALIDATION - Input data types
    expected_types = {
        "script": str,
        "voice_id": str,
        "language": str,
        "quality": str,
        "duration": (int, type(None)),
        "custom_settings": (dict, type(None))
    }
    _validate_input_data_types(request_data, expected_types)
    
    # EARLY VALIDATION - RORO request format
    is_valid: bool
    roro_request: VideoGenerationRequest
    validation_errors: List[str]
    is_valid, roro_request, validation_errors = validate_roro_request(
        request_data, VideoGenerationRequest
    )
    
    if not is_valid:
        raise error_factory.validation_error(
            message="Invalid RORO request format",
            validation_errors=validation_errors,
            context={"operation": "generate_video", "user_id": user_id}
        )
    
    # EARLY VALIDATION - Video generation request data
    try:
        validate_video_generation_request(request_data)
    except ValidationError as e:
        # Re-raise with additional context
        raise error_factory.validation_error(
            message=e.message,
            field=e.details.get('field'),
            value=e.details.get('value'),
            validation_errors=e.details.get('validation_errors', []),
            context={"operation": "generate_video", "user_id": user_id}
        )
    
    # EARLY VALIDATION - Business logic constraints
    try:
        constraints = {
            'daily_video_limit': 10,
            'subscription_status': 'active'  # In production, get from user profile
        }
        is_valid, errors = await validate_business_logic_constraints(
            user_id, "video_generation", constraints
        )
        if not is_valid:
            raise error_factory.validation_error(
                message="Business logic validation failed",
                validation_errors=errors,
                context={"operation": "generate_video", "user_id": user_id}
            )
    except ConcurrencyError as e:
        # Handle concurrent validation conflicts
        raise error_factory.concurrency_error(
            message="Video generation already in progress",
            resource=f"video_generation_{user_id}",
            conflict_type="concurrent_generation"
        )
    
    # Generate video ID
    video_id: str = generate_video_id(user_id)
    
    # Calculate estimated duration
    estimated_duration: int = calculate_estimated_duration(roro_request.quality)
    
    # Create video record with error handling
    try:
        video_data: Dict[str, Any] = prepare_video_data_for_creation(roro_request, video_id, user_id, estimated_duration)
        video_record: Dict[str, Any] = await create_video_record(session, video_data)
    except Exception as e:
        logger.error(f"Failed to create video record: {e}", exc_info=True)
        raise error_factory.database_error(
            message="Failed to create video record",
            operation="create_video_record",
            context={"video_id": video_id, "user_id": user_id}
        )
    
    # Add background task for video processing
    background_tasks.add_task(
        process_video_background,
        video_id,
        request_data,
        user_id
    )
    
    # Create response
    response_data: Dict[str, Any] = create_generation_response_data(video_id, roro_request, estimated_duration)
    
    # Happy path - return success response
    return create_success_response(
        roro_request,
        "Video generation started successfully",
        response_data
    )


@router.post("/status", response_model=VideoStatusResponse)
@handle_errors(category=ErrorCategory.VIDEO_PROCESSING, operation="get_video_status")
async def get_video_status_roro(
    request_data: Dict[str, Any],
    request: Request,
    user_id: str = Depends(get_current_user),
    session: AsyncSession = Depends(get_session)
) -> VideoStatusResponse:
    """Get video status using RORO pattern with early validation"""
    
    # EARLY VALIDATION - Request context
    _validate_request_context(request, user_id)
    
    # EARLY VALIDATION - User permissions
    _validate_user_permissions(user_id, "get_video_status")
    
    # EARLY VALIDATION - Input data types
    expected_types = {
        "video_id": str
    }
    _validate_input_data_types(request_data, expected_types)
    
    # EARLY VALIDATION - RORO request format
    is_valid: bool
    roro_request: VideoStatusRequest
    validation_errors: List[str]
    is_valid, roro_request, validation_errors = validate_roro_request(
        request_data, VideoStatusRequest
    )
    
    if not is_valid:
        raise error_factory.validation_error(
            message="Invalid RORO request format",
            validation_errors=validation_errors,
            context={"operation": "get_video_status", "user_id": user_id}
        )
    
    # EARLY VALIDATION - Video ID format
    if not validate_video_id(roro_request.video_id):
        raise error_factory.validation_error(
            message="Invalid video ID format",
            field="video_id",
            value=roro_request.video_id,
            context={"operation": "get_video_status", "user_id": user_id}
        )
    
    # Get video status from service with error handling
    try:
        status_result: Dict[str, Any] = await get_video_status(session, roro_request.video_id)
    except Exception as e:
        logger.error(f"Failed to retrieve video status: {e}", exc_info=True)
        raise error_factory.database_error(
            message="Failed to retrieve video status",
            operation="get_video_status",
            context={"video_id": roro_request.video_id, "user_id": user_id}
        )
    
    if not status_result["is_found"]:
        raise error_factory.resource_not_found_error(
            message="Video not found",
            resource_type="video",
            resource_id=roro_request.video_id,
            context={"operation": "get_video_status", "user_id": user_id}
        )
    
    # Create response data
    response_data: Dict[str, Any] = create_status_response_data(status_result, roro_request.video_id)
    
    # Happy path - return success response
    return create_success_response(
        roro_request,
        "Video status retrieved successfully",
        response_data
    )


@router.post("/list", response_model=UserVideosResponse)
@handle_errors(category=ErrorCategory.VIDEO_PROCESSING, operation="list_user_videos")
async def list_user_videos_roro(
    request_data: Dict[str, Any],
    request: Request,
    user_id: str = Depends(get_current_user),
    session: AsyncSession = Depends(get_session)
) -> UserVideosResponse:
    """List user videos using RORO pattern with early validation"""
    
    # EARLY VALIDATION - Request context
    _validate_request_context(request, user_id)
    
    # EARLY VALIDATION - User permissions
    _validate_user_permissions(user_id, "list_user_videos")
    
    # EARLY VALIDATION - Input data types
    expected_types = {
        "limit": int,
        "offset": int,
        "status_filter": (str, type(None)),
        "date_from": (str, type(None)),
        "date_to": (str, type(None))
    }
    _validate_input_data_types(request_data, expected_types)
    
    # EARLY VALIDATION - RORO request format
    is_valid: bool
    roro_request: UserVideosRequest
    validation_errors: List[str]
    is_valid, roro_request, validation_errors = validate_roro_request(
        request_data, UserVideosRequest
    )
    
    if not is_valid:
        raise error_factory.validation_error(
            message="Invalid RORO request format",
            validation_errors=validation_errors,
            context={"operation": "list_user_videos", "user_id": user_id}
        )
    
    # EARLY VALIDATION - Pagination parameters
    is_valid, errors = validate_pagination_parameters(roro_request.limit, roro_request.offset)
    if not is_valid:
        raise error_factory.validation_error(
            message="Invalid pagination parameters",
            validation_errors=errors,
            context={"operation": "list_user_videos", "user_id": user_id}
        )
    
    # Get user videos from service with error handling
    try:
        videos: List[Dict[str, Any]] = await get_user_videos(
            session,
            user_id,
            roro_request.limit,
            roro_request.offset
        )
    except Exception as e:
        logger.error(f"Failed to retrieve user videos: {e}", exc_info=True)
        raise error_factory.database_error(
            message="Failed to retrieve user videos",
            operation="get_user_videos",
            context={"user_id": user_id}
        )
    
    # Get total count for pagination
    total_count: int = len(videos)  # In production, get from separate query
    
    # Create response data
    response_data: Dict[str, Any] = create_user_videos_response_data(videos, total_count, roro_request)
    
    # Happy path - return success response
    return create_success_response(
        roro_request,
        "User videos retrieved successfully",
        response_data
    )


@router.get("/{video_id}/download")
@handle_errors(category=ErrorCategory.VIDEO_PROCESSING, operation="download_video")
async def download_video(
    video_id: str,
    request: Request,
    user_id: str = Depends(get_current_user),
    session: AsyncSession = Depends(get_session)
) -> StreamingResponse:
    """Download generated video file with early validation"""
    
    # EARLY VALIDATION - Request context
    _validate_request_context(request, user_id)
    
    # EARLY VALIDATION - User permissions
    _validate_user_permissions(user_id, "download_video")
    
    # EARLY VALIDATION - Video ID format
    if not validate_video_id(video_id):
        raise error_factory.validation_error(
            message="Invalid video ID format",
            field="video_id",
            value=video_id,
            context={"operation": "download_video", "user_id": user_id}
        )
    
    # EARLY VALIDATION - Video ID length
    if len(video_id) < 10 or len(video_id) > 100:
        raise error_factory.validation_error(
            message="Video ID length invalid",
            field="video_id",
            value=video_id,
            context={"operation": "download_video", "user_id": user_id}
        )
    
    # Get video status with error handling
    try:
        status_result: Dict[str, Any] = await get_video_status(session, video_id)
    except Exception as e:
        logger.error(f"Failed to retrieve video status for download: {e}", exc_info=True)
        raise error_factory.database_error(
            message="Failed to retrieve video status",
            operation="get_video_status",
            context={"video_id": video_id, "user_id": user_id}
        )
    
    if not status_result["is_found"]:
        raise error_factory.resource_not_found_error(
            message="Video not found",
            resource_type="video",
            resource_id=video_id,
            context={"operation": "download_video", "user_id": user_id}
        )
    
    if not status_result["is_completed"]:
        raise error_factory.video_processing_error(
            message="Video is not ready for download",
            video_id=video_id,
            processing_stage="download",
            context={"operation": "download_video", "user_id": user_id}
        )
    
    # Stream video file with error handling
    def iterfile() -> Any:
        try:
            with open(status_result["file_path"], "rb") as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                yield from f
        except FileNotFoundError:
            raise error_factory.video_processing_error(
                message="Video file not found on disk",
                video_id=video_id,
                processing_stage="file_access",
                context={"operation": "download_video", "user_id": user_id}
            )
        except PermissionError:
            raise error_factory.video_processing_error(
                message="Permission denied accessing video file",
                video_id=video_id,
                processing_stage="file_access",
                context={"operation": "download_video", "user_id": user_id}
            )
        except Exception as e:
            raise error_factory.video_processing_error(
                message=f"Error accessing video file: {str(e)}",
                video_id=video_id,
                processing_stage="file_access",
                context={"operation": "download_video", "user_id": user_id}
            )
    
    # Happy path - return streaming response
    return StreamingResponse(
        iterfile(),
        media_type="video/mp4",
        headers={"Content-Disposition": f"attachment; filename={video_id}.mp4"}
    )


@router.delete("/{video_id}")
@handle_errors(category=ErrorCategory.VIDEO_PROCESSING, operation="delete_video")
async def delete_video_roro(
    video_id: str,
    request: Request,
    user_id: str = Depends(get_current_user),
    session: AsyncSession = Depends(get_session)
) -> Dict[str, Any]:
    """Delete video using RORO pattern with early validation"""
    
    # EARLY VALIDATION - Request context
    _validate_request_context(request, user_id)
    
    # EARLY VALIDATION - User permissions
    _validate_user_permissions(user_id, "delete_video")
    
    # EARLY VALIDATION - Video ID format
    if not validate_video_id(video_id):
        raise error_factory.validation_error(
            message="Invalid video ID format",
            field="video_id",
            value=video_id,
            context={"operation": "delete_video", "user_id": user_id}
        )
    
    # EARLY VALIDATION - Video ID length
    if len(video_id) < 10 or len(video_id) > 100:
        raise error_factory.validation_error(
            message="Video ID length invalid",
            field="video_id",
            value=video_id,
            context={"operation": "delete_video", "user_id": user_id}
        )
    
    # Create request object for response
    roro_request: VideoStatusRequest = create_delete_request_object(video_id, user_id)
    
    # Delete video with error handling
    try:
        success: bool = await delete_video_record(session, video_id)
    except Exception as e:
        logger.error(f"Failed to delete video: {e}", exc_info=True)
        raise error_factory.database_error(
            message="Failed to delete video",
            operation="delete_video_record",
            context={"video_id": video_id, "user_id": user_id}
        )
    
    if not success:
        raise error_factory.resource_not_found_error(
            message="Video not found or could not be deleted",
            resource_type="video",
            resource_id=video_id,
            context={"operation": "delete_video", "user_id": user_id}
        )
    
    # Happy path - return success response
    return create_success_response(
        roro_request,
        "Video deleted successfully"
    )


# Pure functions with comprehensive type hints and early validation
def prepare_video_data_for_creation(
    request: VideoGenerationRequest,
    video_id: str,
    user_id: str,
    estimated_duration: int
) -> Dict[str, Any]:
    """Prepare video data for creation with early validation"""
    # Early validation - check if all required parameters are provided
    if not video_id or not user_id:
        raise error_factory.validation_error(
            message="Video ID and user ID are required",
            context={"operation": "prepare_video_data"}
        )
    
    if not isinstance(estimated_duration, int) or estimated_duration <= 0:
        raise error_factory.validation_error(
            message="Estimated duration must be a positive integer",
            field="estimated_duration",
            value=estimated_duration,
            context={"operation": "prepare_video_data"}
        )
    
    # Happy path - return prepared data
    return {
        "video_id": video_id,
        "user_id": user_id,
        "script": request.script,
        "voice_id": request.voice_id,
        "language": request.language,
        "quality": request.quality,
        "status": VideoStatus.PROCESSING.value,
        "estimated_duration": estimated_duration
    }


def create_generation_response_data(
    video_id: str,
    request: VideoGenerationRequest,
    estimated_duration: int
) -> Dict[str, Any]:
    """Create generation response data with early validation"""
    # Early validation - check if all required parameters are provided
    if not video_id:
        raise error_factory.validation_error(
            message="Video ID is required",
            context={"operation": "create_generation_response"}
        )
    
    if not isinstance(estimated_duration, int) or estimated_duration <= 0:
        raise error_factory.validation_error(
            message="Estimated duration must be a positive integer",
            field="estimated_duration",
            value=estimated_duration,
            context={"operation": "create_generation_response"}
        )
    
    # Happy path - return response data
    return {
        "video_id": video_id,
        "status": VideoStatus.PROCESSING.value,
        "processing_time": 0.0,
        "estimated_completion": time.time() + estimated_duration,
        "metadata": {
            "script_length": len(request.script),
            "quality": request.quality,
            "voice_id": request.voice_id,
            "language": request.language
        }
    }


def create_status_response_data(status_result: Dict[str, Any], video_id: str) -> Dict[str, Any]:
    """Create status response data with early validation"""
    # Early validation - check if status_result is dict
    if not isinstance(status_result, dict):
        raise error_factory.validation_error(
            message="Status result must be a dictionary",
            context={"operation": "create_status_response"}
        )
    
    if not video_id:
        raise error_factory.validation_error(
            message="Video ID is required",
            context={"operation": "create_status_response"}
        )
    
    # Happy path - return status response data
    return {
        "video_id": video_id,
        "status": status_result["status"],
        "progress": calculate_progress_from_status(status_result),
        "file_size": get_file_size_from_status(status_result),
        "duration": get_video_duration_from_status(status_result),
        "thumbnail_url": generate_thumbnail_url(video_id)
    }


def create_user_videos_response_data(
    videos: List[Dict[str, Any]],
    total_count: int,
    request: UserVideosRequest
) -> Dict[str, Any]:
    """Create user videos response data with early validation"""
    # Early validation - check if videos is list
    if not isinstance(videos, list):
        raise error_factory.validation_error(
            message="Videos must be a list",
            context={"operation": "create_user_videos_response"}
        )
    
    if not isinstance(total_count, int) or total_count < 0:
        raise error_factory.validation_error(
            message="Total count must be a non-negative integer",
            field="total_count",
            value=total_count,
            context={"operation": "create_user_videos_response"}
        )
    
    # Happy path - return user videos response data
    return {
        "videos": videos,
        "total_count": total_count,
        "has_more": len(videos) == request.limit,
        "pagination": {
            "limit": request.limit,
            "offset": request.offset,
            "total": total_count
        }
    }


async def create_delete_request_object(video_id: str, user_id: str) -> VideoStatusRequest:
    """Create delete request object with early validation"""
    # Early validation - check if parameters are provided
    if not video_id or not user_id:
        raise error_factory.validation_error(
            message="Video ID and user ID are required",
            context={"operation": "create_delete_request"}
        )
    
    # Happy path - return delete request object
    return VideoStatusRequest(
        request_id=f"delete_{video_id}",
        video_id=video_id,
        user_id=user_id
    )


def calculate_progress_from_status(status_result: Dict[str, Any]) -> float:
    """Calculate progress from status result with early validation"""
    # Early validation - check if status_result is dict
    if not isinstance(status_result, dict):
        raise error_factory.validation_error(
            message="Status result must be a dictionary",
            context={"operation": "calculate_progress"}
        )
    
    if status_result["is_completed"]:
        return 100.0
    
    if status_result["is_failed"]:
        return 0.0
    
    # Happy path - estimate progress based on processing time
    processing_time: float = status_result.get("processing_time", 0.0)
    estimated_duration: float = 60.0  # Default 60 seconds
    return calculate_progress(processing_time, estimated_duration)


def get_file_size_from_status(status_result: Dict[str, Any]) -> Optional[int]:
    """Get file size from status result with early validation"""
    # Early validation - check if status_result is dict
    if not isinstance(status_result, dict):
        raise error_factory.validation_error(
            message="Status result must be a dictionary",
            context={"operation": "get_file_size"}
        )
    
    if not status_result["has_output_file"]:
        return None
    
    # Happy path - get file size
    try:
        return os.path.getsize(status_result["file_path"])
    except Exception:
        return None


def get_video_duration_from_status(status_result: Dict[str, Any]) -> Optional[float]:
    """Get video duration from status result with early validation"""
    # Early validation - check if status_result is dict
    if not isinstance(status_result, dict):
        raise error_factory.validation_error(
            message="Status result must be a dictionary",
            context={"operation": "get_video_duration"}
        )
    
    # Happy path - in production, extract from video metadata
    return None


# Async functions with enhanced error handling and early validation
async def process_video_background(
    video_id: str,
    request_data: Dict[str, Any],
    user_id: str
) -> None:
    """Background task for video processing with enhanced error handling and early validation"""
    
    # EARLY VALIDATION - Check parameters
    if not video_id or not user_id:
        ErrorLogger.log_error(
            error=error_factory.validation_error(
                message="Invalid parameters for video processing",
                context={"operation": "process_video_background"}
            ),
            context={"video_id": video_id, "user_id": user_id},
            user_id=user_id,
            operation="process_video_background"
        )
        return
    
    # EARLY VALIDATION - Check if video_id is string
    if not isinstance(video_id, str):
        ErrorLogger.log_error(
            error=error_factory.validation_error(
                message="Video ID must be a string",
                field="video_id",
                value=video_id,
                context={"operation": "process_video_background"}
            ),
            context={"video_id": video_id, "user_id": user_id},
            user_id=user_id,
            operation="process_video_background"
        )
        return
    
    # Log operation start
    ErrorLogger.log_user_action(
        action="video_processing_started",
        user_id=user_id,
        details={"video_id": video_id, "request_data_keys": list(request_data.keys())}
    )
    
    try:
        logger.info(f"Starting video processing for {video_id}")
        
        # Simulate video processing with timeout handling
        try:
            await asyncio.wait_for(
                simulate_video_processing(video_id),
                timeout=300.0  # 5 minute timeout
            )
        except asyncio.TimeoutError:
            timeout_error = error_factory.timeout_error(
                message="Video processing timeout",
                video_id=video_id,
                timeout_duration=300.0,
                operation="video_processing"
            )
            ErrorLogger.log_error(
                error=timeout_error,
                context={"video_id": video_id, "user_id": user_id},
                user_id=user_id,
                operation="process_video_background"
            )
            await update_video_status_failed(video_id, "Processing timeout")
            return
        
        # Update status to completed
        async with get_session() as session:
            await update_video_status(
                session,
                video_id,
                VideoStatus.COMPLETED.value,
                processing_time=5.0,
                file_path=f"/outputs/videos/{video_id}.mp4"
            )
        
        # Log successful completion
        ErrorLogger.log_user_action(
            action="video_processing_completed",
            user_id=user_id,
            details={"video_id": video_id, "processing_time": 5.0},
            success=True
        )
        logger.info(f"Video processing completed for {video_id}")
        
    except Exception as e:
        # Log error with full context
        ErrorLogger.log_error(
            error=e,
            context={
                "video_id": video_id,
                "user_id": user_id,
                "operation": "process_video_background",
                "request_data_keys": list(request_data.keys())
            },
            user_id=user_id,
            operation="process_video_background"
        )
        
        # Log failed operation
        ErrorLogger.log_user_action(
            action="video_processing_failed",
            user_id=user_id,
            details={"video_id": video_id, "error": str(e)},
            success=False
        )
        
        # Update status to failed
        try:
            await update_video_status_failed(video_id, str(e))
        except Exception as update_error:
            ErrorLogger.log_error(
                error=update_error,
                context={"video_id": video_id, "user_id": user_id, "original_error": str(e)},
                user_id=user_id,
                operation="update_video_status_failed"
            )


async def simulate_video_processing(video_id: str) -> None:
    """Simulate video processing with error simulation"""
    # Simulate processing time
    await asyncio.sleep(5)
    
    # Simulate random failures (for testing error handling)
    if random.random() < 0.1:  # 10% chance of failure
        raise VideoProcessingError(
            message="Simulated processing failure",
            video_id=video_id,
            processing_stage="simulation"
        )
    
    # Happy path - processing completed successfully


async def update_video_status_failed(video_id: str, error_message: str) -> None:
    """Update video status to failed with error handling"""
    try:
        async with get_session() as session:
            await update_video_status(
                session,
                video_id,
                VideoStatus.FAILED.value,
                processing_time=0.0,
                error_message=error_message
            )
    except Exception as e:
        logger.error(f"Failed to update video status to failed: {e}", exc_info=True)


# Named exports
__all__ = [
    "router",
    "generate_video_roro",
    "get_video_status_roro",
    "list_user_videos_roro",
    "download_video",
    "delete_video_roro"
] 