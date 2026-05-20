from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

from typing import Dict, Any, List, Optional, Union
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, insert, update, delete
import logging
from datetime import datetime
from ..core.roro import (
from ..models.video import Video
from ..models.schemas import (
from ..utils.helpers import (
from ..utils.validators import validate_video_id
        import os
    from sqlalchemy import select
    from sqlalchemy import select
        import os
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Video service module using RORO pattern
Provides video-related operations with comprehensive type hints and Pydantic models.
"""


    VideoGenerationRequest,
    VideoGenerationResponse,
    VideoStatusRequest,
    VideoStatusResponse,
    UserVideosRequest,
    UserVideosResponse,
    create_success_response,
    create_error_response
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
    generate_video_id, 
    create_output_directory,
    calculate_progress,
    generate_thumbnail_url,
    validate_video_id_format
)

logger = logging.getLogger(__name__)


async def create_video_record_roro(
    session: AsyncSession,
    request: VideoGenerationRequest
) -> VideoGenerationResponse:
    """Create new video record using RORO pattern (async database operation)"""
    
    try:
        # Generate video ID (pure function)
        video_id: str = generate_video_id(request.user_id)
        
        # Prepare video data (pure function)
        video_data: Dict[str, Any] = prepare_video_data(request, video_id)
        
        # Create video record (async database operation)
        result = await session.execute(
            insert(Video).values(**video_data).returning(Video)
        )
        await session.commit()
        
        video: Video = result.scalar_one()
        
        # Create response data (pure function)
        response_data: Dict[str, Any] = create_video_response_data(video_id, request)
        
        return create_success_response(
            request,
            "Video record created successfully",
            response_data
        )
        
    except Exception as e:
        logger.error(f"Error creating video record: {e}")
        return create_error_response(
            request,
            "Failed to create video record",
            "DATABASE_ERROR",
            "DatabaseError",
            details={"error": str(e)}
        )


async def get_video_status_roro(
    session: AsyncSession,
    request: VideoStatusRequest
) -> VideoStatusResponse:
    """Get video processing status using RORO pattern (async database operation)"""
    
    try:
        # Validate video ID format (pure function)
        if not validate_video_id_format(request.video_id):
            return create_error_response(
                request,
                "Invalid video ID format",
                "VALIDATION_ERROR",
                "ValidationError"
            )
        
        # Get video from database (async database operation)
        result = await session.execute(
            select(Video).where(Video.video_id == request.video_id)
        )
        video: Optional[Video] = result.scalar_one_or_none()
        
        if not video:
            return create_error_response(
                request,
                "Video not found",
                "NOT_FOUND",
                "NotFoundError"
            )
        
        # Create response data (pure function)
        response_data: Dict[str, Any] = create_status_response_data(video, request.video_id)
        
        return create_success_response(
            request,
            "Video status retrieved successfully",
            response_data
        )
        
    except Exception as e:
        logger.error(f"Error getting video status: {e}")
        return create_error_response(
            request,
            "Failed to get video status",
            "DATABASE_ERROR",
            "DatabaseError",
            details={"error": str(e)}
        )


async def update_video_status_roro(
    session: AsyncSession,
    video_id: str,
    status: VideoStatus,
    request: VideoStatusRequest,
    **kwargs: Any
) -> VideoStatusResponse:
    """Update video processing status using RORO pattern (async database operation)"""
    
    try:
        # Validate video ID format (pure function)
        if not validate_video_id_format(video_id):
            return create_error_response(
                request,
                "Invalid video ID format",
                "VALIDATION_ERROR",
                "ValidationError"
            )
        
        # Prepare update data (pure function)
        update_data: Dict[str, Any] = prepare_status_update_data(status, **kwargs)
        
        # Update video record (async database operation)
        result = await session.execute(
            update(Video)
            .where(Video.video_id == video_id)
            .values(**update_data)
        )
        await session.commit()
        
        if result.rowcount == 0:
            return create_error_response(
                request,
                "Video not found",
                "NOT_FOUND",
                "NotFoundError"
            )
        
        # Get updated video status (async database operation)
        return await get_video_status_roro(session, request)
        
    except Exception as e:
        logger.error(f"Error updating video status: {e}")
        return create_error_response(
            request,
            "Failed to update video status",
            "DATABASE_ERROR",
            "DatabaseError",
            details={"error": str(e)}
        )


async def delete_video_record_roro(
    session: AsyncSession,
    video_id: str,
    request: VideoStatusRequest
) -> VideoStatusResponse:
    """Delete video record using RORO pattern (async database operation)"""
    
    try:
        # Validate video ID format (pure function)
        if not validate_video_id_format(video_id):
            return create_error_response(
                request,
                "Invalid video ID format",
                "VALIDATION_ERROR",
                "ValidationError"
            )
        
        # Get video record first (async database operation)
        result = await session.execute(
            select(Video).where(Video.video_id == video_id)
        )
        video: Optional[Video] = result.scalar_one_or_none()
        
        if not video:
            return create_error_response(
                request,
                "Video not found",
                "NOT_FOUND",
                "NotFoundError"
            )
        
        # Delete associated file if exists (sync file operation)
        delete_video_file(video.file_path)
        
        # Delete database record (async database operation)
        result = await session.execute(
            delete(Video).where(Video.video_id == video_id)
        )
        await session.commit()
        
        return create_success_response(
            request,
            "Video deleted successfully"
        )
        
    except Exception as e:
        logger.error(f"Error deleting video: {e}")
        return create_error_response(
            request,
            "Failed to delete video",
            "DATABASE_ERROR",
            "DatabaseError",
            details={"error": str(e)}
        )


async def get_user_videos_roro(
    session: AsyncSession,
    request: UserVideosRequest
) -> UserVideosResponse:
    """Get videos for specific user using RORO pattern (async database operation)"""
    
    try:
        # Build query with filters (pure function)
        query = build_user_videos_query(request)
        
        # Execute query (async database operation)
        result = await session.execute(query)
        videos: List[Video] = result.scalars().all()
        
        # Convert to dictionaries (pure function)
        video_list: List[Dict[str, Any]] = [video.to_dict() for video in videos]
        
        # Get total count for pagination (async database operation)
        count_query = build_count_query(request)
        count_result = await session.execute(count_query)
        total_count: int = len(count_result.scalars().all())
        
        # Create response data (pure function)
        response_data: Dict[str, Any] = create_user_videos_response_data(video_list, total_count, request)
        
        return create_success_response(
            request,
            "User videos retrieved successfully",
            response_data
        )
        
    except Exception as e:
        logger.error(f"Error getting user videos: {e}")
        return create_error_response(
            request,
            "Failed to retrieve user videos",
            "DATABASE_ERROR",
            "DatabaseError",
            details={"error": str(e)}
        )


async def get_video_statistics_roro(
    session: AsyncSession,
    user_id: str,
    request: UserVideosRequest
) -> UserVideosResponse:
    """Get video statistics for user using RORO pattern (async database operation)"""
    
    try:
        # Get video counts (async database operations)
        total_videos: int = await get_total_videos_count(session, user_id)
        completed_videos: int = await get_completed_videos_count(session, user_id)
        failed_videos: int = await get_failed_videos_count(session, user_id)
        processing_videos: int = await get_processing_videos_count(session, user_id)
        
        # Calculate success rate (pure function)
        success_rate: float = calculate_success_rate(completed_videos, total_videos)
        
        # Create response data (pure function)
        response_data: Dict[str, Any] = create_statistics_response_data(
            total_videos, completed_videos, failed_videos, processing_videos, success_rate
        )
        
        return create_success_response(
            request,
            "Video statistics retrieved successfully",
            response_data
        )
        
    except Exception as e:
        logger.error(f"Error getting video statistics: {e}")
        return create_error_response(
            request,
            "Failed to retrieve video statistics",
            "DATABASE_ERROR",
            "DatabaseError",
            details={"error": str(e)}
        )


# Pure functions with comprehensive type hints (def)
def prepare_video_data(request: VideoGenerationRequest, video_id: str) -> Dict[str, Any]:
    """Prepare video data for database insertion (pure function)"""
    return {
        "video_id": video_id,
        "user_id": request.user_id,
        "script": request.script,
        "voice_id": request.voice_id,
        "language": request.language,
        "quality": request.quality,
        "status": VideoStatus.PROCESSING.value,
        "created_at": datetime.utcnow()
    }


def create_video_response_data(video_id: str, request: VideoGenerationRequest) -> Dict[str, Any]:
    """Create video response data (pure function)"""
    return {
        "video_id": video_id,
        "status": VideoStatus.PROCESSING.value,
        "processing_time": 0.0,
        "metadata": {
            "script_length": len(request.script),
            "quality": request.quality,
            "voice_id": request.voice_id,
            "language": request.language
        }
    }


def create_status_response_data(video: Video, video_id: str) -> Dict[str, Any]:
    """Create status response data (pure function)"""
    return {
        "video_id": video_id,
        "status": video.status,
        "progress": calculate_video_progress(video),
        "file_size": get_video_file_size(video),
        "duration": get_video_duration(video),
        "thumbnail_url": generate_thumbnail_url(video_id)
    }


def prepare_status_update_data(status: VideoStatus, **kwargs: Any) -> Dict[str, Any]:
    """Prepare status update data (pure function)"""
    update_data: Dict[str, Any] = {
        "status": status.value,
        "updated_at": datetime.utcnow()
    }
    update_data.update(kwargs)
    return update_data


def delete_video_file(file_path: Optional[str]) -> None:
    """Delete video file if exists (sync file operation)"""
    if not file_path:
        return
    
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        logger.warning(f"Failed to delete video file {file_path}: {e}")


def build_user_videos_query(request: UserVideosRequest):
    """Build user videos query (pure function)"""
    
    query = select(Video).where(Video.user_id == request.user_id)
    
    # Apply status filter
    if request.status_filter:
        query = query.where(Video.status == request.status_filter.value)
    
    # Apply date filters
    if request.date_from:
        query = query.where(Video.created_at >= request.date_from)
    if request.date_to:
        query = query.where(Video.created_at <= request.date_to)
    
    # Apply ordering and pagination
    query = query.order_by(Video.created_at.desc())
    query = query.limit(request.limit).offset(request.offset)
    
    return query


def build_count_query(request: UserVideosRequest):
    """Build count query (pure function)"""
    
    query = select(Video).where(Video.user_id == request.user_id)
    
    if request.status_filter:
        query = query.where(Video.status == request.status_filter.value)
    if request.date_from:
        query = query.where(Video.created_at >= request.date_from)
    if request.date_to:
        query = query.where(Video.created_at <= request.date_to)
    
    return query


def create_user_videos_response_data(video_list: List[Dict[str, Any]], total_count: int, request: UserVideosRequest) -> Dict[str, Any]:
    """Create user videos response data (pure function)"""
    return {
        "videos": video_list,
        "total_count": total_count,
        "has_more": len(video_list) == request.limit,
        "pagination": {
            "limit": request.limit,
            "offset": request.offset,
            "total": total_count
        }
    }


def calculate_video_progress(video: Video) -> float:
    """Calculate video processing progress (pure function)"""
    if video.status == VideoStatus.COMPLETED.value:
        return 100.0
    elif video.status == VideoStatus.FAILED.value:
        return 0.0
    else:
        # Estimate progress based on processing time
        processing_time: float = video.processing_time or 0.0
        estimated_duration: float = 60.0  # Default 60 seconds
        return calculate_progress(processing_time, estimated_duration)


def get_video_file_size(video: Video) -> Optional[int]:
    """Get video file size if available (pure function)"""
    if not video.file_path:
        return None
    
    try:
        return os.path.getsize(video.file_path)
    except Exception:
        return None


def get_video_duration(video: Video) -> Optional[float]:
    """Get video duration if available (pure function)"""
    return video.duration


def calculate_success_rate(completed_videos: int, total_videos: int) -> float:
    """Calculate success rate (pure function)"""
    return (completed_videos / total_videos * 100) if total_videos > 0 else 0.0


def create_statistics_response_data(
    total_videos: int,
    completed_videos: int,
    failed_videos: int,
    processing_videos: int,
    success_rate: float
) -> Dict[str, Any]:
    """Create statistics response data (pure function)"""
    return {
        "total_videos": total_videos,
        "completed_videos": completed_videos,
        "failed_videos": failed_videos,
        "processing_videos": processing_videos,
        "success_rate": round(success_rate, 2)
    }


# Async database helper functions with type hints
async def get_total_videos_count(session: AsyncSession, user_id: str) -> int:
    """Get total videos count (async database operation)"""
    result = await session.execute(
        select(Video).where(Video.user_id == user_id)
    )
    return len(result.scalars().all())


async def get_completed_videos_count(session: AsyncSession, user_id: str) -> int:
    """Get completed videos count (async database operation)"""
    result = await session.execute(
        select(Video).where(
            Video.user_id == user_id,
            Video.status == VideoStatus.COMPLETED.value
        )
    )
    return len(result.scalars().all())


async def get_failed_videos_count(session: AsyncSession, user_id: str) -> int:
    """Get failed videos count (async database operation)"""
    result = await session.execute(
        select(Video).where(
            Video.user_id == user_id,
            Video.status == VideoStatus.FAILED.value
        )
    )
    return len(result.scalars().all())


async def get_processing_videos_count(session: AsyncSession, user_id: str) -> int:
    """Get processing videos count (async database operation)"""
    result = await session.execute(
        select(Video).where(
            Video.user_id == user_id,
            Video.status == VideoStatus.PROCESSING.value
        )
    )
    return len(result.scalars().all())


# Legacy functions for backward compatibility with type hints
async def create_video_record(
    session: AsyncSession,
    video_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Legacy function for creating video record (async database operation)"""
    result = await session.execute(
        insert(Video).values(**video_data).returning(Video)
    )
    await session.commit()
    video: Video = result.scalar_one()
    return video.to_dict()


async def get_video_status(
    session: AsyncSession,
    video_id: str
) -> Dict[str, Any]:
    """Legacy function for getting video status (async database operation)"""
    if not validate_video_id(video_id):
        return {"is_found": False, "error": "Invalid video ID format"}
    
    result = await session.execute(
        select(Video).where(Video.video_id == video_id)
    )
    video: Optional[Video] = result.scalar_one_or_none()
    
    if not video:
        return {"is_found": False, "error": "Video not found"}
    
    return {
        "is_found": True,
        "video_id": video.video_id,
        "status": video.status,
        "is_processing": video.status == VideoStatus.PROCESSING.value,
        "is_completed": video.status == VideoStatus.COMPLETED.value,
        "is_failed": video.status == VideoStatus.FAILED.value,
        "has_output_file": video.file_path is not None,
        "has_processing_time": video.processing_time is not None,
        "processing_time": video.processing_time,
        "file_path": video.file_path,
        "created_at": video.created_at.isoformat() if video.created_at else None,
        "updated_at": video.updated_at.isoformat() if video.updated_at else None
    }


async def update_video_status(
    session: AsyncSession,
    video_id: str,
    status: str,
    **kwargs: Any
) -> bool:
    """Legacy function for updating video status (async database operation)"""
    if not validate_video_id(video_id):
        return False
    
    update_data: Dict[str, Any] = {
        "status": status,
        "updated_at": datetime.utcnow()
    }
    update_data.update(kwargs)
    
    result = await session.execute(
        update(Video)
        .where(Video.video_id == video_id)
        .values(**update_data)
    )
    await session.commit()
    
    return result.rowcount > 0


async def delete_video_record(
    session: AsyncSession,
    video_id: str
) -> bool:
    """Legacy function for deleting video record (async database operation)"""
    if not validate_video_id(video_id):
        return False
    
    result = await session.execute(
        select(Video).where(Video.video_id == video_id)
    )
    video: Optional[Video] = result.scalar_one_or_none()
    
    if not video:
        return False
    
    if video.file_path:
        delete_video_file(video.file_path)
    
    result = await session.execute(
        delete(Video).where(Video.video_id == video_id)
    )
    await session.commit()
    
    return result.rowcount > 0


async def get_user_videos(
    session: AsyncSession,
    user_id: str,
    limit: int = 50,
    offset: int = 0
) -> List[Dict[str, Any]]:
    """Legacy function for getting user videos (async database operation)"""
    result = await session.execute(
        select(Video)
        .where(Video.user_id == user_id)
        .order_by(Video.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    
    videos: List[Video] = result.scalars().all()
    return [video.to_dict() for video in videos]


async def get_video_statistics(
    session: AsyncSession,
    user_id: str
) -> Dict[str, Any]:
    """Legacy function for getting video statistics (async database operation)"""
    total_videos: int = await get_total_videos_count(session, user_id)
    completed_videos: int = await get_completed_videos_count(session, user_id)
    failed_videos: int = await get_failed_videos_count(session, user_id)
    processing_videos: int = await get_processing_videos_count(session, user_id)
    
    success_rate: float = calculate_success_rate(completed_videos, total_videos)
    
    return {
        "total_videos": total_videos,
        "completed_videos": completed_videos,
        "failed_videos": failed_videos,
        "processing_videos": processing_videos,
        "success_rate": round(success_rate, 2)
    }


# Named exports
__all__ = [
    "create_video_record_roro",
    "get_video_status_roro",
    "update_video_status_roro",
    "delete_video_record_roro",
    "get_user_videos_roro",
    "get_video_statistics_roro",
    # Legacy functions
    "create_video_record",
    "get_video_status",
    "update_video_status",
    "delete_video_record",
    "get_user_videos",
    "get_video_statistics"
] 