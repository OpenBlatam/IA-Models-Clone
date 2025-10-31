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

import asyncio
import time
from typing import List, Optional
from uuid import uuid4
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, Query, Path
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi.responses import StreamingResponse, JSONResponse
from ..dependencies import (
from ..core.error_handler import error_handler, ErrorContext
from ..core.exceptions import AIVideoError, ValidationError as AIVideoValidationError
import logging
        import uuid
        import redis
        import json
        import os
            import uuid
        import redis
        import json
        import redis
        import json
        import redis
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
API Routes - AI Video System

Declarative route definitions with clear return type annotations,
functional components, and Pydantic models for FastAPI.
"""



# Import dependencies and models
    VideoRequest, VideoResponse, SystemStatus, ErrorResponse,
    get_db_session, get_performance_optimizer, get_error_context,
    validate_video_request, generate_video_id, format_processing_time,
    create_video_record, update_video_status, get_video_record,
    app_state, get_current_user, check_rate_limit, check_quota, increment_quota,
    get_cached_result, get_health_status, get_metrics
)

# Import core components

# Setup logging
logger = logging.getLogger(__name__)


# Create routers
video_router = APIRouter()
system_router = APIRouter()
api_router = APIRouter(prefix="/api/v1")
health_router = APIRouter(prefix="/health")
admin_router = APIRouter(prefix="/admin")


# Video Routes
@video_router.post("/generate", response_model=VideoResponse)
async def generate_video(
    request: VideoRequest,
    background_tasks: BackgroundTasks,
    db_session: AsyncSession = Depends(get_db_session),
    optimizer: PerformanceOptimizer = Depends(get_performance_optimizer),
    error_context: ErrorContext = Depends(get_error_context)
) -> VideoResponse:
    """
    Generate video from prompt.
    
    Args:
        request: Video generation request
        background_tasks: FastAPI background tasks
        db_session: Database session
        optimizer: Performance optimizer
        error_context: Error context
    
    Returns:
        VideoResponse: Video generation response
    """
    # Early return for validation errors
    try:
        validated_request = validate_video_request(request)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    # Generate video ID
    video_id = generate_video_id()
    
    # Create video record
    try:
        video_record = await create_video_record(
            db_session,
            video_id,
            validated_request.prompt,
            validated_request.model_dump()
        )
    except Exception as e:
        error = error_handler.handle_error(e, error_context)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create video record: {error.message}"
        )
    
    # Add background task for video generation
    background_tasks.add_task(
        process_video_generation,
        video_id,
        validated_request,
        db_session,
        optimizer,
        error_context
    )
    
    # Return immediate response
    return VideoResponse(
        video_id=video_id,
        status="pending",
        metadata=validated_request.model_dump()
    )


@video_router.get("/{video_id}", response_model=VideoResponse)
async def get_video_status(
    video_id: str,
    db_session: AsyncSession = Depends(get_db_session),
    error_context: ErrorContext = Depends(get_error_context)
) -> VideoResponse:
    """
    Get video generation status.
    
    Args:
        video_id: Video identifier
        db_session: Database session
        error_context: Error context
    
    Returns:
        VideoResponse: Video status response
    """
    # Get video record
    video_record = await get_video_record(db_session, video_id)
    
    # Early return for not found
    if not video_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video with ID {video_id} not found"
        )
    
    # Convert to response model
    return VideoResponse(
        video_id=video_record.id,
        status=video_record.status,
        video_url=f"/api/v1/videos/{video_id}/download" if video_record.video_path else None,
        thumbnail_url=f"/api/v1/videos/{video_id}/thumbnail" if video_record.thumbnail_path else None,
        processing_time=video_record.processing_time,
        error_message=video_record.error_message,
        metadata={}  # Parse from video_record.metadata if needed
    )


@video_router.get("/{video_id}/download")
async def download_video(
    video_id: str,
    db_session: AsyncSession = Depends(get_db_session),
    error_context: ErrorContext = Depends(get_error_context)
) -> dict:
    """
    Download generated video.
    
    Args:
        video_id: Video identifier
        db_session: Database session
        error_context: Error context
    
    Returns:
        dict: Download information
    """
    # Get video record
    video_record = await get_video_record(db_session, video_id)
    
    # Early return for not found
    if not video_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video with ID {video_id} not found"
        )
    
    # Early return for not completed
    if video_record.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Video is not ready for download. Status: {video_record.status}"
        )
    
    # Return download information
    return {
        "video_id": video_id,
        "download_url": f"/files/videos/{video_id}.mp4",
        "file_size": "2.5MB",  # Calculate actual file size
        "format": "MP4"
    }


@video_router.get("/{video_id}/thumbnail")
async def get_video_thumbnail(
    video_id: str,
    db_session: AsyncSession = Depends(get_db_session),
    error_context: ErrorContext = Depends(get_error_context)
) -> dict:
    """
    Get video thumbnail.
    
    Args:
        video_id: Video identifier
        db_session: Database session
        error_context: Error context
    
    Returns:
        dict: Thumbnail information
    """
    # Get video record
    video_record = await get_video_record(db_session, video_id)
    
    # Early return for not found
    if not video_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video with ID {video_id} not found"
        )
    
    # Early return for not completed
    if video_record.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Video thumbnail not available. Status: {video_record.status}"
        )
    
    # Return thumbnail information
    return {
        "video_id": video_id,
        "thumbnail_url": f"/files/thumbnails/{video_id}.jpg",
        "format": "JPEG"
    }


@video_router.get("/", response_model=List[VideoResponse])
async def list_videos(
    limit: int = 10,
    offset: int = 0,
    status_filter: Optional[str] = None,
    db_session: AsyncSession = Depends(get_db_session),
    error_context: ErrorContext = Depends(get_error_context)
) -> List[VideoResponse]:
    """
    List videos with pagination and filtering.
    
    Args:
        limit: Maximum number of videos to return
        offset: Number of videos to skip
        status_filter: Filter by status
        db_session: Database session
        error_context: Error context
    
    Returns:
        List[VideoResponse]: List of video responses
    """
    # Validate parameters
    if limit < 1 or limit > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Limit must be between 1 and 100"
        )
    
    if offset < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Offset must be non-negative"
        )
    
    # Build query
    query = "SELECT * FROM videos"
    params = {}
    
    if status_filter:
        query += " WHERE status = :status"
        params["status"] = status_filter
    
    query += " ORDER BY created_at DESC LIMIT :limit OFFSET :offset"
    params.update({"limit": limit, "offset": offset})
    
    # Execute query
    try:
        result = await db_session.execute(text(query), params)
        videos = result.fetchall()
    except Exception as e:
        error = error_handler.handle_error(e, error_context)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch videos: {error.message}"
        )
    
    # Convert to response models
    return [
        VideoResponse(
            video_id=video.id,
            status=video.status,
            video_url=f"/api/v1/videos/{video.id}/download" if video.video_path else None,
            thumbnail_url=f"/api/v1/videos/{video.id}/thumbnail" if video.thumbnail_path else None,
            processing_time=video.processing_time,
            error_message=video.error_message,
            metadata={}
        )
        for video in videos
    ]


# System Routes
@system_router.get("/status", response_model=SystemStatus)
async def get_system_status(
    error_context: ErrorContext = Depends(get_error_context)
) -> SystemStatus:
    """
    Get system status and health information.
    
    Args:
        error_context: Error context
    
    Returns:
        SystemStatus: System status information
    """
    try:
        # Get performance optimizer stats
        optimizer_stats = {}
        if app_state.performance_optimizer:
            optimizer_stats = await app_state.performance_optimizer.get_optimization_stats()
        
        # Get GPU utilization
        gpu_utilization = None
        if optimizer_stats.get("memory_info"):
            # Calculate GPU utilization from memory info
            gpu_utilization = 75.5  # Example value
        
        # Get memory usage
        memory_usage = None
        if optimizer_stats.get("memory_info"):
            # Calculate memory usage
            memory_usage = 45.2  # Example value
        
        return SystemStatus(
            status="healthy",
            version="1.0.0",
            uptime=app_state.get_uptime(),
            active_requests=app_state.active_requests,
            gpu_utilization=gpu_utilization,
            memory_usage=memory_usage
        )
        
    except Exception as e:
        error = error_handler.handle_error(e, error_context)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get system status: {error.message}"
        )


@system_router.get("/health")
def health_check() -> dict:
    """
    Simple health check endpoint.
    
    Returns:
        dict: Health status
    """
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.0.0"
    }


@system_router.get("/metrics")
async def get_system_metrics(
    error_context: ErrorContext = Depends(get_error_context)
) -> dict:
    """
    Get detailed system metrics.
    
    Args:
        error_context: Error context
    
    Returns:
        dict: System metrics
    """
    try:
        # Get performance optimizer metrics
        optimizer_metrics = {}
        if app_state.performance_optimizer:
            optimizer_metrics = await app_state.performance_optimizer.get_optimization_stats()
        
        # Get error statistics
        error_stats = error_handler.get_error_stats()
        
        return {
            "system": {
                "uptime": app_state.get_uptime(),
                "active_requests": app_state.active_requests,
                "version": "1.0.0"
            },
            "performance": optimizer_metrics,
            "errors": error_stats,
            "timestamp": time.time()
        }
        
    except Exception as e:
        error = error_handler.handle_error(e, error_context)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get metrics: {error.message}"
        )


# Background Task Functions
async def process_video_generation(
    video_id: str,
    request: VideoRequest,
    db_session: AsyncSession,
    optimizer: PerformanceOptimizer,
    error_context: ErrorContext
) -> None:
    """
    Background task for video generation.
    
    Args:
        video_id: Video identifier
        request: Video generation request
        db_session: Database session
        optimizer: Performance optimizer
        error_context: Error context
    """
    start_time = time.time()
    
    try:
        # Update status to processing
        await update_video_status(db_session, video_id, "processing")
        
        # Increment active requests
        app_state.increment_requests()
        
        # Generate video using optimizer
        video_data = await optimizer.optimize_video_generation(
            request.prompt,
            num_inference_steps=request.num_steps,
            width=request.width,
            height=request.height,
            seed=request.seed
        )
        
        # Save video file
        video_path = f"/files/videos/{video_id}.mp4"
        # await save_video_file(video_data, video_path)
        
        # Generate thumbnail
        thumbnail_path = f"/files/thumbnails/{video_id}.jpg"
        # await generate_thumbnail(video_data, thumbnail_path)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Update status to completed
        await update_video_status(
            db_session,
            video_id,
            "completed",
            video_path=video_path,
            thumbnail_path=thumbnail_path,
            processing_time=processing_time
        )
        
        logger.info(f"Video {video_id} generated successfully in {format_processing_time(processing_time)}")
        
    except Exception as e:
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Update status to failed
        await update_video_status(
            db_session,
            video_id,
            "failed",
            error_message=str(e),
            processing_time=processing_time
        )
        
        # Log error
        error = error_handler.handle_error(e, error_context)
        logger.error(f"Video {video_id} generation failed: {error.message}")
        
    finally:
        # Decrement active requests
        app_state.decrement_requests()


# Video Generation Routes
@api_router.post("/generate", response_model=VideoGenerationResponse)
async def generate_video(
    request: VideoGenerationRequest,
    background_tasks: BackgroundTasks,
    user: dict = Depends(get_current_user),
    rate_limit: bool = Depends(check_rate_limit),
    quota: bool = Depends(check_quota)
):
    """Generate video from text prompt."""
    try:
        # Check cache first
        cache_key = f"video:{hash(request.json())}"
        cached_result = await get_cached_result(cache_key)
        
        if cached_result:
            logger.info(f"Returning cached result for user {user['user_id']}")
            return VideoGenerationResponse(
                job_id=cached_result["job_id"],
                status="completed",
                message="Cached result returned",
                estimated_time=0
            )
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Estimate processing time based on parameters
        estimated_time = request.num_inference_steps * 2  # Rough estimate
        
        # Add to background tasks
        background_tasks.add_task(process_video_generation, request, job_id, user['user_id'])
        
        # Increment quota
        await increment_quota(user)
        
        logger.info(f"Video generation job {job_id} queued for user {user['user_id']}")
        
        return VideoGenerationResponse(
            job_id=job_id,
            status="queued",
            message="Video generation job queued successfully",
            estimated_time=estimated_time
        )
        
    except Exception as e:
        logger.error(f"Error in video generation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error queuing video generation: {str(e)}"
        )

@api_router.get("/job/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str = Path(..., description="Job identifier"),
    user: dict = Depends(get_current_user)
):
    """Get job status and progress."""
    try:
        
        redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        job_data = redis_client.get(f"job:{job_id}")
        
        if not job_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found"
            )
        
        job = json.loads(job_data)
        
        return JobStatusResponse(
            job_id=job_id,
            status=job["status"],
            progress=job["progress"],
            result_url=job.get("result_url"),
            error_message=job.get("error_message"),
            created_at=datetime.fromisoformat(job["created_at"]),
            updated_at=datetime.fromisoformat(job["updated_at"])
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting job status: {str(e)}"
        )

@api_router.get("/video/{video_id}")
async def download_video(
    video_id: str = Path(..., description="Video identifier"),
    user: dict = Depends(get_current_user)
):
    """Download generated video."""
    try:
        video_path = f"generated_videos/{video_id}.mp4"
        
        if not os.path.exists(video_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Video not found"
            )
        
        return StreamingResponse(
            open(video_path, "rb"),
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            media_type="video/mp4",
            headers={"Content-Disposition": f"attachment; filename={video_id}.mp4"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving video: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error serving video: {str(e)}"
        )

@api_router.post("/batch", response_model=BatchGenerationResponse)
async def batch_generate_videos(
    batch_request: BatchGenerationRequest,
    background_tasks: BackgroundTasks,
    user: dict = Depends(get_current_user),
    rate_limit: bool = Depends(check_rate_limit),
    quota: bool = Depends(check_quota)
):
    """Generate multiple videos in batch."""
    try:
        job_ids = []
        
        for request in batch_request.requests:
            job_id = str(uuid.uuid4())
            job_ids.append(job_id)
            
            # Add to background tasks
            background_tasks.add_task(process_video_generation, request, job_id, user['user_id'])
        
        # Generate batch ID
        batch_id = str(uuid.uuid4())
        
        logger.info(f"Batch video generation queued: {len(job_ids)} jobs for user {user['user_id']}")
        
        return BatchGenerationResponse(
            batch_id=batch_id,
            job_ids=job_ids,
            total_jobs=len(job_ids),
            message=f"Batch video generation queued successfully"
        )
        
    except Exception as e:
        logger.error(f"Error in batch video generation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error queuing batch video generation: {str(e)}"
        )

@api_router.get("/jobs", response_model=List[JobStatusResponse])
async def list_user_jobs(
    user: dict = Depends(get_current_user),
    status_filter: Optional[str] = Query(None, description="Filter by job status"),
    limit: int = Query(10, ge=1, le=100, description="Number of jobs to return"),
    offset: int = Query(0, ge=0, description="Number of jobs to skip")
):
    """List user's video generation jobs."""
    try:
        
        redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        
        # Get user's jobs
        user_jobs = []
        for key in redis_client.keys("job:*"):
            job_data = redis_client.get(key)
            if job_data:
                job = json.loads(job_data)
                if job.get("user_id") == user['user_id']:
                    if status_filter is None or job["status"] == status_filter:
                        user_jobs.append({
                            "job_id": key.split(":")[1],
                            **job
                        })
        
        # Sort by creation time (newest first)
        user_jobs.sort(key=lambda x: x["created_at"], reverse=True)
        
        # Apply pagination
        paginated_jobs = user_jobs[offset:offset + limit]
        
        return [
            JobStatusResponse(
                job_id=job["job_id"],
                status=job["status"],
                progress=job["progress"],
                result_url=job.get("result_url"),
                error_message=job.get("error_message"),
                created_at=datetime.fromisoformat(job["created_at"]),
                updated_at=datetime.fromisoformat(job["updated_at"])
            )
            for job in paginated_jobs
        ]
        
    except Exception as e:
        logger.error(f"Error listing user jobs: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing user jobs: {str(e)}"
        )

@api_router.delete("/job/{job_id}")
async def cancel_job(
    job_id: str = Path(..., description="Job identifier"),
    user: dict = Depends(get_current_user)
):
    """Cancel a video generation job."""
    try:
        
        redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        job_data = redis_client.get(f"job:{job_id}")
        
        if not job_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found"
            )
        
        job = json.loads(job_data)
        
        # Check if user owns the job
        if job.get("user_id") != user['user_id']:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to cancel this job"
            )
        
        # Check if job can be cancelled
        if job["status"] in ["completed", "failed", "cancelled"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Job cannot be cancelled"
            )
        
        # Update job status
        job["status"] = "cancelled"
        job["updated_at"] = datetime.now().isoformat()
        redis_client.setex(f"job:{job_id}", 3600, json.dumps(job))
        
        logger.info(f"Job {job_id} cancelled by user {user['user_id']}")
        
        return {"message": "Job cancelled successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling job: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error cancelling job: {str(e)}"
        )

# Health Check Routes
@health_router.get("/", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint."""
    return await get_health_status()

@health_router.get("/ready")
async def readiness_check():
    """Readiness check endpoint."""
    health_status = await get_health_status()
    
    if health_status["redis"] == "healthy":
        return {"status": "ready"}
    else:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready"
        )

@health_router.get("/live")
async def liveness_check():
    """Liveness check endpoint."""
    return {"status": "alive"}

# Admin Routes
@admin_router.get("/metrics", response_model=MetricsResponse)
async def get_admin_metrics(
    user: dict = Depends(get_current_user)
):
    """Get system metrics (admin only)."""
    # Check if user is admin
    if user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    return await get_metrics()

@admin_router.get("/users/{user_id}/quota", response_model=UserQuota)
async def get_user_quota(
    user_id: str = Path(..., description="User identifier"),
    admin_user: dict = Depends(get_current_user)
):
    """Get user quota (admin only)."""
    # Check if user is admin
    if admin_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    try:
        redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        
        # Get quota data
        daily_used = int(redis_client.get(f"quota:daily:{user_id}") or 0)
        monthly_used = int(redis_client.get(f"quota:monthly:{user_id}") or 0)
        
        return UserQuota(
            user_id=user_id,
            daily_limit=50,
            daily_used=daily_used,
            monthly_limit=1000,
            monthly_used=monthly_used,
            remaining_daily=50 - daily_used,
            remaining_monthly=1000 - monthly_used,
            reset_daily=datetime.now().replace(hour=0, minute=0, second=0, microsecond=0),
            reset_monthly=datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        )
        
    except Exception as e:
        logger.error(f"Error getting user quota: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting user quota: {str(e)}"
        )

@admin_router.get("/model/config", response_model=ModelConfig)
async def get_model_config(
    admin_user: dict = Depends(get_current_user)
):
    """Get model configuration (admin only)."""
    # Check if user is admin
    if admin_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    return ModelConfig(
        model_name="text-to-video-ms-1.7b",
        model_version="1.0.0",
        supported_formats=["mp4", "avi", "mov", "webm"],
        max_frames=64,
        max_resolution="1024x1024",
        min_resolution="256x256",
        supported_qualities=["low", "medium", "high", "ultra"],
        default_parameters={
            "guidance_scale": 7.5,
            "num_inference_steps": 50,
            "fps": 8
        }
    )

# Export routers
__all__ = [
    "video_router",
    "system_router",
    "api_router",
    "health_router",
    "admin_router"
] 