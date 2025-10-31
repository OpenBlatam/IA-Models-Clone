from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
from datetime import datetime
from typing import Dict, List, Optional
from uuid import uuid4
from fastapi import BackgroundTasks
from ..schemas.video_schemas import (
from ..utils.cache import get_cache_client
from ..utils.metrics import record_metric, track_processing_time
from ..utils.validation import validate_user_access
    import random
from typing import Any, List, Dict, Optional
import logging
"""
Video Service - Pure Business Logic
===================================

Clean service layer with:
- Pure functions
- Async optimization  
- Type safety
- Error handling
- Performance optimization
"""



    BatchVideoResponse,
    ProcessingStatus,
    VideoLogsResponse,
    VideoRequest,
    VideoResponse,
)


async def create_video_async(
    request: VideoRequest,
    user_id: str,
    background_tasks: BackgroundTasks,
) -> VideoResponse:
    """
    Create video generation request asynchronously.
    
    Args:
        request: Video generation request
        user_id: User identifier
        background_tasks: FastAPI background tasks
        
    Returns:
        VideoResponse with initial status
        
    Raises:
        ValueError: If request validation fails
        PermissionError: If user lacks permissions
    """
    # Validate user access
    if not await validate_user_access(user_id, "video:create"):
        raise PermissionError("User lacks video creation permissions")
    
    # Generate unique request ID
    request_id = f"vid_{uuid4().hex[:12]}"
    
    # Create initial response
    video_response = VideoResponse(
        request_id=request_id,
        status=ProcessingStatus.PENDING,
        quality=request.quality,
        format=request.format,
        metadata={
            "user_id": user_id,
            "original_request": request.model_dump(),
        },
    )
    
    # Cache initial status
    cache = await get_cache_client()
    await cache.set(
        f"video_status:{request_id}",
        video_response.model_dump_json(),
        expire=3600,  # 1 hour
    )
    
    # Queue background processing
    background_tasks.add_task(
        process_video_background,
        request_id=request_id,
        request=request,
        user_id=user_id,
    )
    
    # Record metrics
    await record_metric("video_requests_created", 1, {"user_id": user_id})
    
    return video_response


async def get_video_status(
    request_id: str,
    user_id: str,
) -> Optional[VideoResponse]:
    """
    Get video processing status.
    
    Args:
        request_id: Video request identifier
        user_id: User identifier
        
    Returns:
        VideoResponse if found, None otherwise
        
    Raises:
        PermissionError: If user lacks access to this video
    """
    # Get from cache first
    cache = await get_cache_client()
    cached_status = await cache.get(f"video_status:{request_id}")
    
    if not cached_status:
        return None
    
    video_response = VideoResponse.model_validate_json(cached_status)
    
    # Validate user access
    if not await validate_user_access(user_id, "video:read", request_id):
        if video_response.metadata.get("user_id") != user_id:
            raise PermissionError("User lacks access to this video")
    
    return video_response


async def get_video_logs(
    request_id: str,
    user_id: str,
    skip: int = 0,
    limit: int = 100,
) -> Optional[VideoLogsResponse]:
    """
    Get video processing logs with pagination.
    
    Args:
        request_id: Video request identifier
        user_id: User identifier
        skip: Number of logs to skip
        limit: Number of logs to return
        
    Returns:
        VideoLogsResponse if found, None otherwise
        
    Raises:
        PermissionError: If user lacks access to this video
    """
    # Validate access first
    video_status = await get_video_status(request_id, user_id)
    if not video_status:
        return None
    
    # Get logs from cache
    cache = await get_cache_client()
    logs_data = await cache.get(f"video_logs:{request_id}")
    
    if not logs_data:
        return VideoLogsResponse(
            request_id=request_id,
            logs=[],
            total_count=0,
            has_more=False,
        )
    
    # Parse and paginate logs
    all_logs = VideoLogsResponse.model_validate_json(logs_data)
    paginated_logs = all_logs.logs[skip:skip + limit]
    
    return VideoLogsResponse(
        request_id=request_id,
        logs=paginated_logs,
        total_count=len(all_logs.logs),
        has_more=skip + limit < len(all_logs.logs),
    )


async def cancel_video_processing(
    request_id: str,
    user_id: str,
) -> bool:
    """
    Cancel video processing if in progress.
    
    Args:
        request_id: Video request identifier
        user_id: User identifier
        
    Returns:
        True if cancelled successfully, False otherwise
        
    Raises:
        PermissionError: If user lacks access to this video
    """
    video_status = await get_video_status(request_id, user_id)
    
    if not video_status:
        return False
    
    # Can only cancel pending or processing videos
    if video_status.status not in [ProcessingStatus.PENDING, ProcessingStatus.PROCESSING]:
        return False
    
    # Update status to cancelled
    video_status.status = ProcessingStatus.CANCELLED
    video_status.updated_at = datetime.utcnow()
    
    # Update cache
    cache = await get_cache_client()
    await cache.set(
        f"video_status:{request_id}",
        video_status.model_dump_json(),
        expire=3600,
    )
    
    # Record metrics
    await record_metric("video_requests_cancelled", 1, {"user_id": user_id})
    
    return True


async def retry_video_processing(
    request_id: str,
    user_id: str,
    background_tasks: BackgroundTasks,
) -> bool:
    """
    Retry failed video processing.
    
    Args:
        request_id: Video request identifier
        user_id: User identifier
        background_tasks: FastAPI background tasks
        
    Returns:
        True if retry initiated successfully, False otherwise
        
    Raises:
        PermissionError: If user lacks access to this video
    """
    video_status = await get_video_status(request_id, user_id)
    
    if not video_status:
        return False
    
    # Can only retry failed videos
    if video_status.status != ProcessingStatus.FAILED:
        return False
    
    # Reset status to pending
    video_status.status = ProcessingStatus.PENDING
    video_status.updated_at = datetime.utcnow()
    video_status.error_message = None
    
    # Update cache
    cache = await get_cache_client()
    await cache.set(
        f"video_status:{request_id}",
        video_status.model_dump_json(),
        expire=3600,
    )
    
    # Queue retry processing
    original_request = VideoRequest.model_validate(
        video_status.metadata["original_request"]
    )
    
    background_tasks.add_task(
        process_video_background,
        request_id=request_id,
        request=original_request,
        user_id=user_id,
    )
    
    # Record metrics
    await record_metric("video_requests_retried", 1, {"user_id": user_id})
    
    return True


async def get_batch_video_status(
    request_ids: List[str],
    user_id: str,
) -> BatchVideoResponse:
    """
    Get status for multiple videos efficiently.
    
    Args:
        request_ids: List of video request identifiers
        user_id: User identifier
        
    Returns:
        BatchVideoResponse with all results
    """
    # Use asyncio.gather for concurrent requests
    tasks = [
        get_video_status(request_id, user_id)
        for request_id in request_ids
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    video_responses: Dict[str, VideoResponse] = {}
    success_count = 0
    error_count = 0
    
    for request_id, result in zip(request_ids, results):
        if isinstance(result, Exception):
            error_count += 1
            continue
        
        if result is not None:
            video_responses[request_id] = result
            success_count += 1
        else:
            error_count += 1
    
    return BatchVideoResponse(
        results=video_responses,
        success_count=success_count,
        error_count=error_count,
    )


@track_processing_time("video_processing")
async def process_video_background(
    request_id: str,
    request: VideoRequest,
    user_id: str,
) -> None:
    """
    Background task for video processing.
    
    Args:
        request_id: Video request identifier
        request: Original video request
        user_id: User identifier
    """
    cache = await get_cache_client()
    
    try:
        # Update status to processing
        await update_video_status(
            request_id=request_id,
            status=ProcessingStatus.PROCESSING,
            cache=cache,
        )
        
        # Simulate video processing (replace with actual implementation)
        await simulate_video_processing(request_id, request)
        
        # Update status to completed
        await update_video_status(
            request_id=request_id,
            status=ProcessingStatus.COMPLETED,
            cache=cache,
            output_url=f"https://videos.example.com/{request_id}.mp4",
            processing_time=45.5,  # Example processing time
        )
        
        # Record success metrics
        await record_metric("video_requests_completed", 1, {"user_id": user_id})
        
    except Exception as e:
        # Update status to failed
        await update_video_status(
            request_id=request_id,
            status=ProcessingStatus.FAILED,
            cache=cache,
            error_message=str(e),
        )
        
        # Record failure metrics
        await record_metric("video_requests_failed", 1, {"user_id": user_id})


async def update_video_status(
    request_id: str,
    status: ProcessingStatus,
    cache,
    output_url: Optional[str] = None,
    processing_time: Optional[float] = None,
    error_message: Optional[str] = None,
) -> None:
    """Update video status in cache."""
    cached_status = await cache.get(f"video_status:{request_id}")
    
    if cached_status:
        video_response = VideoResponse.model_validate_json(cached_status)
        video_response.status = status
        video_response.updated_at = datetime.utcnow()
        
        if output_url:
            video_response.output_url = output_url
        if processing_time:
            video_response.processing_time = processing_time
        if error_message:
            video_response.error_message = error_message
        
        await cache.set(
            f"video_status:{request_id}",
            video_response.model_dump_json(),
            expire=3600,
        )


async def simulate_video_processing(
    request_id: str,
    request: VideoRequest,
) -> None:
    """
    Simulate video processing (replace with actual implementation).
    
    Args:
        request_id: Video request identifier
        request: Video generation request
    """
    # Simulate processing time based on video duration
    processing_time = request.duration * 0.5  # Simulated processing factor
    await asyncio.sleep(min(processing_time, 10))  # Cap at 10 seconds for demo
    
    # Simulate occasional failures for testing
    if random.random() < 0.1:  # 10% failure rate
        raise Exception("Simulated processing failure") 