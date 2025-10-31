from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

from typing import List, Optional
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from fastapi.security import HTTPBearer
from ..dependencies.auth import get_current_user, require_permissions
from ..dependencies.rate_limit import check_rate_limit
from ..dependencies.validation import validate_request_id
from ..schemas.video_schemas import (
from ..services.video_service import (
from ..utils.response import create_error_response, create_success_response
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Video Router - Modern FastAPI Implementation
===========================================

Functional router implementation with:
- Pure functions with type hints
- Early error returns
- RORO pattern
- Async optimization
- Dependency injection
"""



    APIResponse,
    BatchVideoRequest,
    BatchVideoResponse,
    VideoLogsResponse,
    VideoRequest,
    VideoResponse,
)
    cancel_video_processing,
    create_video_async,
    get_batch_video_status,
    get_video_logs,
    get_video_status,
    retry_video_processing,
)


router = APIRouter()
security = HTTPBearer()


@router.post(
    "/videos",
    response_model=APIResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Create video generation request",
    description="Submit a video generation request for processing",
)
async def create_video(
    request: VideoRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
    _rate_limit: None = Depends(check_rate_limit),
    _permissions: None = Depends(require_permissions(["video:create"])),
) -> APIResponse:
    """
    Create a new video generation request.
    
    Validates input, checks permissions, and queues video for processing.
    """
    # Early return for validation errors handled by Pydantic
    
    try:
        # Create video processing task
        video_response = await create_video_async(
            request=request,
            user_id=current_user["sub"],
            background_tasks=background_tasks,
        )
        
        return create_success_response(
            data=video_response,
            message="Video generation request created successfully",
        )
        
    except ValueError as e:
        return create_error_response(
            message=str(e),
            status_code=status.HTTP_400_BAD_REQUEST,
        )
    
    except PermissionError as e:
        return create_error_response(
            message=str(e),
            status_code=status.HTTP_403_FORBIDDEN,
        )
    
    except Exception as e:
        return create_error_response(
            message="Internal server error occurred",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@router.get(
    "/videos/{request_id}",
    response_model=APIResponse,
    summary="Get video status",
    description="Retrieve the current status of a video generation request",
)
async def get_video(
    request_id: str = Depends(validate_request_id),
    current_user: dict = Depends(get_current_user),
    _permissions: None = Depends(require_permissions(["video:read"])),
) -> APIResponse:
    """
    Get video processing status and details.
    
    Returns current status, progress, and download URLs when available.
    """
    try:
        video_response = await get_video_status(
            request_id=request_id,
            user_id=current_user["sub"],
        )
        
        if not video_response:
            return create_error_response(
                message="Video not found",
                status_code=status.HTTP_404_NOT_FOUND,
            )
        
        return create_success_response(data=video_response)
        
    except PermissionError as e:
        return create_error_response(
            message=str(e),
            status_code=status.HTTP_403_FORBIDDEN,
        )
    
    except Exception as e:
        return create_error_response(
            message="Internal server error occurred",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@router.get(
    "/videos/{request_id}/logs",
    response_model=APIResponse,
    summary="Get video processing logs",
    description="Retrieve processing logs for a video generation request",
)
async def get_video_processing_logs(
    request_id: str = Depends(validate_request_id),
    skip: int = Query(0, ge=0, description="Number of logs to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Number of logs to return"),
    current_user: dict = Depends(get_current_user),
    _permissions: None = Depends(require_permissions(["video:read"])),
) -> APIResponse:
    """
    Get processing logs for video generation.
    
    Supports pagination with skip/limit parameters.
    """
    try:
        logs_response = await get_video_logs(
            request_id=request_id,
            user_id=current_user["sub"],
            skip=skip,
            limit=limit,
        )
        
        if not logs_response:
            return create_error_response(
                message="Video logs not found",
                status_code=status.HTTP_404_NOT_FOUND,
            )
        
        return create_success_response(data=logs_response)
        
    except PermissionError as e:
        return create_error_response(
            message=str(e),
            status_code=status.HTTP_403_FORBIDDEN,
        )
    
    except Exception as e:
        return create_error_response(
            message="Internal server error occurred",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@router.post(
    "/videos/{request_id}/cancel",
    response_model=APIResponse,
    summary="Cancel video processing",
    description="Cancel an in-progress video generation request",
)
async def cancel_video(
    request_id: str = Depends(validate_request_id),
    current_user: dict = Depends(get_current_user),
    _permissions: None = Depends(require_permissions(["video:cancel"])),
) -> APIResponse:
    """
    Cancel video processing.
    
    Only works for requests that are currently processing.
    """
    try:
        success = await cancel_video_processing(
            request_id=request_id,
            user_id=current_user["sub"],
        )
        
        if not success:
            return create_error_response(
                message="Cannot cancel video - not in progress or not found",
                status_code=status.HTTP_400_BAD_REQUEST,
            )
        
        return create_success_response(
            message="Video processing cancelled successfully"
        )
        
    except PermissionError as e:
        return create_error_response(
            message=str(e),
            status_code=status.HTTP_403_FORBIDDEN,
        )
    
    except Exception as e:
        return create_error_response(
            message="Internal server error occurred",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@router.post(
    "/videos/{request_id}/retry",
    response_model=APIResponse,
    summary="Retry failed video processing",
    description="Retry a failed video generation request",
)
async def retry_video(
    request_id: str = Depends(validate_request_id),
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
    _permissions: None = Depends(require_permissions(["video:retry"])),
) -> APIResponse:
    """
    Retry failed video processing.
    
    Only works for requests that have failed.
    """
    try:
        success = await retry_video_processing(
            request_id=request_id,
            user_id=current_user["sub"],
            background_tasks=background_tasks,
        )
        
        if not success:
            return create_error_response(
                message="Cannot retry video - not failed or not found",
                status_code=status.HTTP_400_BAD_REQUEST,
            )
        
        return create_success_response(
            message="Video processing retry initiated successfully"
        )
        
    except PermissionError as e:
        return create_error_response(
            message=str(e),
            status_code=status.HTTP_403_FORBIDDEN,
        )
    
    except Exception as e:
        return create_error_response(
            message="Internal server error occurred",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@router.post(
    "/videos/batch",
    response_model=APIResponse,
    summary="Get batch video status",
    description="Retrieve status for multiple video requests in a single call",
)
async def get_batch_videos(
    request: BatchVideoRequest,
    current_user: dict = Depends(get_current_user),
    _permissions: None = Depends(require_permissions(["video:read"])),
) -> APIResponse:
    """
    Get status for multiple videos in batch.
    
    Efficient for checking status of many videos at once.
    """
    try:
        batch_response = await get_batch_video_status(
            request_ids=request.request_ids,
            user_id=current_user["sub"],
        )
        
        return create_success_response(data=batch_response)
        
    except PermissionError as e:
        return create_error_response(
            message=str(e),
            status_code=status.HTTP_403_FORBIDDEN,
        )
    
    except Exception as e:
        return create_error_response(
            message="Internal server error occurred",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        ) 