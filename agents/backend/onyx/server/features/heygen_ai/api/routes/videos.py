from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

from fastapi import APIRouter, Depends, HTTPException, Query, Path, UploadFile, File, Form
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone, timedelta
import structlog
import os
import uuid
import json
from .base import (
from api.models.video import VideoCreate, VideoUpdate, VideoResponse, VideoProcessingOptions
from api.schemas.pagination import PaginationParams
from typing import Any, List, Dict, Optional
import logging
import asyncio
#!/usr/bin/env python3
"""
Video Routes for HeyGen AI FastAPI
Well-structured video processing routes with clear dependencies.
"""


    BaseRoute, RouteCategory, BaseResponse, ErrorResponse, PaginatedResponse,
    route_metrics, require_auth, rate_limit, cache_response,
    get_database_operations, get_current_user, get_request_id
)

logger = structlog.get_logger()

# =============================================================================
# Video Route Class
# =============================================================================

class VideoRoutes(BaseRoute):
    """Video processing routes with clear structure and dependencies."""
    
    def __init__(self, db_operations, api_operations, file_storage) -> Any:
        super().__init__(
            name="Video Processing",
            description="Video processing operations including upload, processing, and management",
            category=RouteCategory.VIDEOS,
            tags=["videos", "processing", "ai"],
            prefix="/videos",
            dependencies={
                "db_ops": db_operations,
                "api_ops": api_operations,
                "file_storage": file_storage
            }
        )
        
        # Register routes
        self._register_routes()
    
    def _register_routes(self) -> Any:
        """Register all video routes with clear organization."""
        
        # =====================================================================
        # Video CRUD Operations
        # =====================================================================
        
        @self.router.get(
            "/",
            response_model=PaginatedResponse,
            summary="Get all videos",
            description="Retrieve a paginated list of all videos with optional filtering"
        )
        @route_metrics
        @require_auth
        @rate_limit(requests_per_minute=100)
        @cache_response(ttl=300)
        async def get_videos(
            pagination: PaginationParams = Depends(),
            user_id: Optional[int] = Query(None, description="Filter by user ID"),
            status: Optional[str] = Query(None, description="Filter by video status"),
            type: Optional[str] = Query(None, description="Filter by video type"),
            search: Optional[str] = Query(None, description="Search videos by title or description"),
            current_user: Dict[str, Any] = Depends(get_current_user),
            request_id: str = Depends(get_request_id)
        ):
            """Get paginated list of videos with filtering."""
            try:
                db_ops = self.get_dependency("db_ops")
                
                # Build query with filters
                query = "SELECT v.*, u.name as user_name FROM videos v JOIN users u ON v.user_id = u.id WHERE 1=1"
                params = {}
                
                # User can only see their own videos unless admin
                if current_user.get("role") != "admin":
                    query += " AND v.user_id = :current_user_id"
                    params["current_user_id"] = int(current_user["user_id"])
                elif user_id:
                    query += " AND v.user_id = :user_id"
                    params["user_id"] = user_id
                
                if status:
                    query += " AND v.status = :status"
                    params["status"] = status
                
                if type:
                    query += " AND v.type = :type"
                    params["type"] = type
                
                if search:
                    query += " AND (v.title ILIKE :search OR v.description ILIKE :search)"
                    params["search"] = f"%{search}%"
                
                # Add pagination
                query += " ORDER BY v.created_at DESC LIMIT :limit OFFSET :offset"
                params["limit"] = pagination.page_size
                params["offset"] = (pagination.page - 1) * pagination.page_size
                
                # Execute query
                videos = await db_ops.execute_query(query, parameters=params)
                
                # Get total count
                count_query = query.replace("SELECT v.*, u.name as user_name", "SELECT COUNT(*) as total")
                count_query = count_query.split("ORDER BY")[0]  # Remove ORDER BY and LIMIT
                count_result = await db_ops.execute_query(count_query, parameters=params)
                total_count = count_result[0]["total"] if count_result else 0
                
                return self.paginated_response(
                    data=videos,
                    total_count=total_count,
                    page=pagination.page,
                    page_size=pagination.page_size,
                    request_id=request_id
                )
                
            except Exception as e:
                logger.error(f"Error getting videos: {e}")
                raise HTTPException(status_code=500, detail="Failed to retrieve videos")
        
        @self.router.get(
            "/{video_id}",
            response_model=VideoResponse,
            summary="Get video by ID",
            description="Retrieve a specific video by its ID"
        )
        @route_metrics
        @require_auth
        @rate_limit(requests_per_minute=200)
        @cache_response(ttl=600)
        async def get_video(
            video_id: int = Path(..., description="Video ID"),
            current_user: Dict[str, Any] = Depends(get_current_user),
            request_id: str = Depends(get_request_id)
        ):
            """Get video by ID."""
            try:
                db_ops = self.get_dependency("db_ops")
                
                # Get video with user information
                videos = await db_ops.execute_query(
                    """
                    SELECT v.*, u.name as user_name 
                    FROM videos v 
                    JOIN users u ON v.user_id = u.id 
                    WHERE v.id = :video_id
                    """,
                    parameters={"video_id": video_id}
                )
                
                if not videos:
                    raise HTTPException(status_code=404, detail="Video not found")
                
                video = videos[0]
                
                # Check permissions (users can only view their own videos unless admin)
                if current_user["user_id"] != str(video["user_id"]) and current_user.get("role") != "admin":
                    raise HTTPException(status_code=403, detail="Insufficient permissions")
                
                return self.success_response(
                    data=video,
                    request_id=request_id
                )
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error getting video {video_id}: {e}")
                raise HTTPException(status_code=500, detail="Failed to retrieve video")
        
        @self.router.post(
            "/",
            response_model=VideoResponse,
            summary="Create new video",
            description="Create a new video processing job"
        )
        @route_metrics
        @require_auth
        @rate_limit(requests_per_minute=10)
        async def create_video(
            title: str = Form(..., description="Video title"),
            description: Optional[str] = Form(None, description="Video description"),
            video_type: str = Form("ai_generated", description="Video type"),
            processing_options: Optional[str] = Form(None, description="Processing options JSON"),
            current_user: Dict[str, Any] = Depends(get_current_user),
            request_id: str = Depends(get_request_id)
        ):
            """Create a new video processing job."""
            try:
                db_ops = self.get_dependency("db_ops")
                api_ops = self.get_dependency("api_ops")
                
                # Parse processing options
                options = {}
                if processing_options:
                    try:
                        options = json.loads(processing_options)
                    except json.JSONDecodeError:
                        raise HTTPException(status_code=400, detail="Invalid processing options format")
                
                # Create video record
                video_data = {
                    "title": title,
                    "description": description,
                    "type": video_type,
                    "user_id": int(current_user["user_id"]),
                    "status": "pending",
                    "processing_options": json.dumps(options),
                    "created_at": datetime.now(timezone.utc)
                }
                
                result = await db_ops.execute_insert(
                    table="videos",
                    data=video_data,
                    returning="id, title, description, type, status, user_id, created_at"
                )
                
                # Start processing job
                try:
                    processing_response = await api_ops.post(
                        endpoint="/process-video",
                        data={
                            "video_id": result["id"],
                            "title": title,
                            "description": description,
                            "type": video_type,
                            "options": options
                        }
                    )
                    
                    # Update video with job ID
                    await db_ops.execute_update(
                        table="videos",
                        data={
                            "external_job_id": processing_response["data"]["job_id"],
                            "status": "processing"
                        },
                        where_conditions={"id": result["id"]}
                    )
                    
                    result["external_job_id"] = processing_response["data"]["job_id"]
                    result["status"] = "processing"
                    
                except Exception as e:
                    logger.error(f"Failed to start video processing: {e}")
                    # Update status to failed
                    await db_ops.execute_update(
                        table="videos",
                        data={"status": "failed", "error_message": str(e)},
                        where_conditions={"id": result["id"]}
                    )
                    result["status"] = "failed"
                    result["error_message"] = str(e)
                
                return self.success_response(
                    data=result,
                    message="Video processing job created successfully",
                    request_id=request_id
                )
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error creating video: {e}")
                raise HTTPException(status_code=500, detail="Failed to create video")
        
        @self.router.put(
            "/{video_id}",
            response_model=VideoResponse,
            summary="Update video",
            description="Update video information"
        )
        @route_metrics
        @require_auth
        @rate_limit(requests_per_minute=50)
        async def update_video(
            video_id: int = Path(..., description="Video ID"),
            video_data: VideoUpdate,
            current_user: Dict[str, Any] = Depends(get_current_user),
            request_id: str = Depends(get_request_id)
        ):
            """Update video information."""
            try:
                db_ops = self.get_dependency("db_ops")
                
                # Check if video exists and get ownership
                videos = await db_ops.execute_query(
                    "SELECT user_id FROM videos WHERE id = :video_id",
                    parameters={"video_id": video_id}
                )
                
                if not videos:
                    raise HTTPException(status_code=404, detail="Video not found")
                
                video = videos[0]
                
                # Check permissions
                if current_user["user_id"] != str(video["user_id"]) and current_user.get("role") != "admin":
                    raise HTTPException(status_code=403, detail="Insufficient permissions")
                
                # Update video
                update_data = video_data.dict(exclude_unset=True)
                update_data["updated_at"] = datetime.now(timezone.utc)
                
                result = await db_ops.execute_update(
                    table="videos",
                    data=update_data,
                    where_conditions={"id": video_id},
                    returning="id, title, description, type, status, updated_at"
                )
                
                return self.success_response(
                    data=result,
                    message="Video updated successfully",
                    request_id=request_id
                )
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error updating video {video_id}: {e}")
                raise HTTPException(status_code=500, detail="Failed to update video")
        
        @self.router.delete(
            "/{video_id}",
            response_model=BaseResponse,
            summary="Delete video",
            description="Delete a video (soft delete)"
        )
        @route_metrics
        @require_auth
        @rate_limit(requests_per_minute=20)
        async def delete_video(
            video_id: int = Path(..., description="Video ID"),
            current_user: Dict[str, Any] = Depends(get_current_user),
            request_id: str = Depends(get_request_id)
        ):
            """Delete video (soft delete)."""
            try:
                db_ops = self.get_dependency("db_ops")
                api_ops = self.get_dependency("api_ops")
                
                # Check if video exists and get ownership
                videos = await db_ops.execute_query(
                    "SELECT user_id, external_job_id FROM videos WHERE id = :video_id",
                    parameters={"video_id": video_id}
                )
                
                if not videos:
                    raise HTTPException(status_code=404, detail="Video not found")
                
                video = videos[0]
                
                # Check permissions
                if current_user["user_id"] != str(video["user_id"]) and current_user.get("role") != "admin":
                    raise HTTPException(status_code=403, detail="Insufficient permissions")
                
                # Cancel external processing job if exists
                if video.get("external_job_id"):
                    try:
                        await api_ops.delete(f"/jobs/{video['external_job_id']}")
                    except Exception as e:
                        logger.warning(f"Failed to cancel external job: {e}")
                
                # Soft delete video
                await db_ops.execute_update(
                    table="videos",
                    data={
                        "status": "deleted",
                        "deleted_at": datetime.now(timezone.utc)
                    },
                    where_conditions={"id": video_id}
                )
                
                return self.success_response(
                    message="Video deleted successfully",
                    request_id=request_id
                )
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error deleting video {video_id}: {e}")
                raise HTTPException(status_code=500, detail="Failed to delete video")
        
        # =====================================================================
        # Video Upload Operations
        # =====================================================================
        
        @self.router.post(
            "/upload",
            response_model=VideoResponse,
            summary="Upload video file",
            description="Upload a video file for processing"
        )
        @route_metrics
        @require_auth
        @rate_limit(requests_per_minute=5)
        async def upload_video(
            file: UploadFile = File(..., description="Video file to upload"),
            title: str = Form(..., description="Video title"),
            description: Optional[str] = Form(None, description="Video description"),
            video_type: str = Form("uploaded", description="Video type"),
            current_user: Dict[str, Any] = Depends(get_current_user),
            request_id: str = Depends(get_request_id)
        ):
            """Upload video file for processing."""
            try:
                db_ops = self.get_dependency("db_ops")
                file_storage = self.get_dependency("file_storage")
                api_ops = self.get_dependency("api_ops")
                
                # Validate file type
                allowed_types = ["video/mp4", "video/avi", "video/mov", "video/wmv"]
                if file.content_type not in allowed_types:
                    raise HTTPException(status_code=400, detail="Invalid file type")
                
                # Validate file size (max 100MB)
                max_size = 100 * 1024 * 1024  # 100MB
                if file.size > max_size:
                    raise HTTPException(status_code=400, detail="File too large (max 100MB)")
                
                # Generate unique filename
                file_extension = os.path.splitext(file.filename)[1]
                unique_filename = f"{uuid.uuid4()}{file_extension}"
                
                # Save file to storage
                file_path = await file_storage.save_file(file, unique_filename)
                
                # Create video record
                video_data = {
                    "title": title,
                    "description": description,
                    "type": video_type,
                    "user_id": int(current_user["user_id"]),
                    "status": "uploaded",
                    "file_path": file_path,
                    "file_size": file.size,
                    "original_filename": file.filename,
                    "created_at": datetime.now(timezone.utc)
                }
                
                result = await db_ops.execute_insert(
                    table="videos",
                    data=video_data,
                    returning="id, title, description, type, status, file_path, created_at"
                )
                
                # Start processing
                try:
                    processing_response = await api_ops.post(
                        endpoint="/process-uploaded-video",
                        data={
                            "video_id": result["id"],
                            "file_path": file_path,
                            "title": title,
                            "description": description
                        }
                    )
                    
                    # Update video with job ID
                    await db_ops.execute_update(
                        table="videos",
                        data={
                            "external_job_id": processing_response["data"]["job_id"],
                            "status": "processing"
                        },
                        where_conditions={"id": result["id"]}
                    )
                    
                    result["external_job_id"] = processing_response["data"]["job_id"]
                    result["status"] = "processing"
                    
                except Exception as e:
                    logger.error(f"Failed to start video processing: {e}")
                    await db_ops.execute_update(
                        table="videos",
                        data={"status": "failed", "error_message": str(e)},
                        where_conditions={"id": result["id"]}
                    )
                    result["status"] = "failed"
                    result["error_message"] = str(e)
                
                return self.success_response(
                    data=result,
                    message="Video uploaded and processing started",
                    request_id=request_id
                )
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error uploading video: {e}")
                raise HTTPException(status_code=500, detail="Failed to upload video")
        
        # =====================================================================
        # Video Processing Operations
        # =====================================================================
        
        @self.router.post(
            "/{video_id}/process",
            response_model=VideoResponse,
            summary="Start video processing",
            description="Start or restart video processing"
        )
        @route_metrics
        @require_auth
        @rate_limit(requests_per_minute=10)
        async def start_video_processing(
            video_id: int = Path(..., description="Video ID"),
            processing_options: Optional[VideoProcessingOptions] = None,
            current_user: Dict[str, Any] = Depends(get_current_user),
            request_id: str = Depends(get_request_id)
        ):
            """Start or restart video processing."""
            try:
                db_ops = self.get_dependency("db_ops")
                api_ops = self.get_dependency("api_ops")
                
                # Check if video exists and get ownership
                videos = await db_ops.execute_query(
                    "SELECT * FROM videos WHERE id = :video_id",
                    parameters={"video_id": video_id}
                )
                
                if not videos:
                    raise HTTPException(status_code=404, detail="Video not found")
                
                video = videos[0]
                
                # Check permissions
                if current_user["user_id"] != str(video["user_id"]) and current_user.get("role") != "admin":
                    raise HTTPException(status_code=403, detail="Insufficient permissions")
                
                # Start processing
                processing_data = {
                    "video_id": video_id,
                    "title": video["title"],
                    "description": video["description"],
                    "type": video["type"]
                }
                
                if processing_options:
                    processing_data["options"] = processing_options.dict()
                elif video.get("processing_options"):
                    processing_data["options"] = json.loads(video["processing_options"])
                
                if video.get("file_path"):
                    processing_data["file_path"] = video["file_path"]
                
                processing_response = await api_ops.post(
                    endpoint="/process-video",
                    data=processing_data
                )
                
                # Update video status
                await db_ops.execute_update(
                    table="videos",
                    data={
                        "external_job_id": processing_response["data"]["job_id"],
                        "status": "processing",
                        "processing_started_at": datetime.now(timezone.utc)
                    },
                    where_conditions={"id": video_id}
                )
                
                result = {
                    **video,
                    "external_job_id": processing_response["data"]["job_id"],
                    "status": "processing",
                    "processing_started_at": datetime.now(timezone.utc).isoformat()
                }
                
                return self.success_response(
                    data=result,
                    message="Video processing started successfully",
                    request_id=request_id
                )
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error starting video processing {video_id}: {e}")
                raise HTTPException(status_code=500, detail="Failed to start video processing")
        
        @self.router.get(
            "/{video_id}/status",
            response_model=Dict[str, Any],
            summary="Get video processing status",
            description="Get current processing status and progress"
        )
        @route_metrics
        @require_auth
        @rate_limit(requests_per_minute=100)
        @cache_response(ttl=30)  # Short cache for status updates
        async def get_video_status(
            video_id: int = Path(..., description="Video ID"),
            current_user: Dict[str, Any] = Depends(get_current_user),
            request_id: str = Depends(get_request_id)
        ):
            """Get video processing status."""
            try:
                db_ops = self.get_dependency("db_ops")
                api_ops = self.get_dependency("api_ops")
                
                # Check if video exists and get ownership
                videos = await db_ops.execute_query(
                    "SELECT * FROM videos WHERE id = :video_id",
                    parameters={"video_id": video_id}
                )
                
                if not videos:
                    raise HTTPException(status_code=404, detail="Video not found")
                
                video = videos[0]
                
                # Check permissions
                if current_user["user_id"] != str(video["user_id"]) and current_user.get("role") != "admin":
                    raise HTTPException(status_code=403, detail="Insufficient permissions")
                
                # Get status from external API if processing
                status_data = {
                    "video_id": video_id,
                    "status": video["status"],
                    "created_at": video["created_at"].isoformat() if video["created_at"] else None,
                    "updated_at": video["updated_at"].isoformat() if video.get("updated_at") else None
                }
                
                if video.get("external_job_id") and video["status"] in ["processing", "pending"]:
                    try:
                        external_status = await api_ops.get(
                            endpoint=f"/jobs/{video['external_job_id']}/status",
                            cache_key=f"job_status:{video['external_job_id']}",
                            cache_ttl=30
                        )
                        
                        status_data.update({
                            "external_status": external_status.get("data", {}),
                            "progress": external_status.get("data", {}).get("progress", 0),
                            "estimated_completion": external_status.get("data", {}).get("estimated_completion")
                        })
                        
                        # Update local status if changed
                        external_status_value = external_status.get("data", {}).get("status")
                        if external_status_value and external_status_value != video["status"]:
                            await db_ops.execute_update(
                                table="videos",
                                data={
                                    "status": external_status_value,
                                    "updated_at": datetime.now(timezone.utc)
                                },
                                where_conditions={"id": video_id}
                            )
                            status_data["status"] = external_status_value
                        
                    except Exception as e:
                        logger.warning(f"Failed to get external status for video {video_id}: {e}")
                        status_data["external_status_error"] = str(e)
                
                return self.success_response(
                    data=status_data,
                    request_id=request_id
                )
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error getting video status {video_id}: {e}")
                raise HTTPException(status_code=500, detail="Failed to get video status")
        
        @self.router.post(
            "/{video_id}/cancel",
            response_model=BaseResponse,
            summary="Cancel video processing",
            description="Cancel ongoing video processing"
        )
        @route_metrics
        @require_auth
        @rate_limit(requests_per_minute=20)
        async def cancel_video_processing(
            video_id: int = Path(..., description="Video ID"),
            current_user: Dict[str, Any] = Depends(get_current_user),
            request_id: str = Depends(get_request_id)
        ):
            """Cancel video processing."""
            try:
                db_ops = self.get_dependency("db_ops")
                api_ops = self.get_dependency("api_ops")
                
                # Check if video exists and get ownership
                videos = await db_ops.execute_query(
                    "SELECT * FROM videos WHERE id = :video_id",
                    parameters={"video_id": video_id}
                )
                
                if not videos:
                    raise HTTPException(status_code=404, detail="Video not found")
                
                video = videos[0]
                
                # Check permissions
                if current_user["user_id"] != str(video["user_id"]) and current_user.get("role") != "admin":
                    raise HTTPException(status_code=403, detail="Insufficient permissions")
                
                # Cancel external job if exists
                if video.get("external_job_id"):
                    try:
                        await api_ops.delete(f"/jobs/{video['external_job_id']}")
                    except Exception as e:
                        logger.warning(f"Failed to cancel external job: {e}")
                
                # Update video status
                await db_ops.execute_update(
                    table="videos",
                    data={
                        "status": "cancelled",
                        "updated_at": datetime.now(timezone.utc)
                    },
                    where_conditions={"id": video_id}
                )
                
                return self.success_response(
                    message="Video processing cancelled successfully",
                    request_id=request_id
                )
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error cancelling video processing {video_id}: {e}")
                raise HTTPException(status_code=500, detail="Failed to cancel video processing")
        
        # =====================================================================
        # Video Download Operations
        # =====================================================================
        
        @self.router.get(
            "/{video_id}/download",
            summary="Download video",
            description="Download the processed video file"
        )
        @route_metrics
        @require_auth
        @rate_limit(requests_per_minute=30)
        async def download_video(
            video_id: int = Path(..., description="Video ID"),
            current_user: Dict[str, Any] = Depends(get_current_user),
            request_id: str = Depends(get_request_id)
        ):
            """Download processed video file."""
            try:
                db_ops = self.get_dependency("db_ops")
                file_storage = self.get_dependency("file_storage")
                
                # Check if video exists and get ownership
                videos = await db_ops.execute_query(
                    "SELECT * FROM videos WHERE id = :video_id",
                    parameters={"video_id": video_id}
                )
                
                if not videos:
                    raise HTTPException(status_code=404, detail="Video not found")
                
                video = videos[0]
                
                # Check permissions
                if current_user["user_id"] != str(video["user_id"]) and current_user.get("role") != "admin":
                    raise HTTPException(status_code=403, detail="Insufficient permissions")
                
                # Check if video is ready for download
                if video["status"] != "completed":
                    raise HTTPException(status_code=400, detail="Video is not ready for download")
                
                # Get download URL
                download_url = await file_storage.get_download_url(video["output_path"])
                
                return self.success_response(
                    data={
                        "download_url": download_url,
                        "filename": video.get("output_filename", f"video_{video_id}.mp4"),
                        "file_size": video.get("output_size")
                    },
                    request_id=request_id
                )
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error downloading video {video_id}: {e}")
                raise HTTPException(status_code=500, detail="Failed to download video")
        
        # =====================================================================
        # Video Analytics Operations
        # =====================================================================
        
        @self.router.get(
            "/{video_id}/analytics",
            response_model=Dict[str, Any],
            summary="Get video analytics",
            description="Get detailed analytics and metrics for a video"
        )
        @route_metrics
        @require_auth
        @rate_limit(requests_per_minute=50)
        @cache_response(ttl=600)
        async def get_video_analytics(
            video_id: int = Path(..., description="Video ID"),
            current_user: Dict[str, Any] = Depends(get_current_user),
            request_id: str = Depends(get_request_id)
        ):
            """Get video analytics and metrics."""
            try:
                db_ops = self.get_dependency("db_ops")
                api_ops = self.get_dependency("api_ops")
                
                # Check if video exists and get ownership
                videos = await db_ops.execute_query(
                    "SELECT * FROM videos WHERE id = :video_id",
                    parameters={"video_id": video_id}
                )
                
                if not videos:
                    raise HTTPException(status_code=404, detail="Video not found")
                
                video = videos[0]
                
                # Check permissions
                if current_user["user_id"] != str(video["user_id"]) and current_user.get("role") != "admin":
                    raise HTTPException(status_code=403, detail="Insufficient permissions")
                
                # Get processing metrics
                processing_metrics = await db_ops.execute_query(
                    """
                    SELECT 
                        AVG(EXTRACT(EPOCH FROM (updated_at - created_at))) as avg_processing_time,
                        COUNT(*) as total_processing_attempts
                    FROM video_processing_logs 
                    WHERE video_id = :video_id
                    """,
                    parameters={"video_id": video_id}
                )
                
                # Get usage analytics
                usage_analytics = await db_ops.execute_query(
                    """
                    SELECT 
                        COUNT(*) as total_views,
                        COUNT(DISTINCT user_id) as unique_viewers,
                        AVG(view_duration) as avg_view_duration
                    FROM video_views 
                    WHERE video_id = :video_id
                    """,
                    parameters={"video_id": video_id}
                )
                
                # Get external analytics if available
                external_analytics = {}
                if video.get("external_job_id"):
                    try:
                        external_response = await api_ops.get(
                            endpoint=f"/jobs/{video['external_job_id']}/analytics",
                            cache_key=f"job_analytics:{video['external_job_id']}",
                            cache_ttl=300
                        )
                        external_analytics = external_response.get("data", {})
                    except Exception as e:
                        logger.warning(f"Failed to get external analytics for video {video_id}: {e}")
                
                analytics = {
                    "video_id": video_id,
                    "processing_metrics": processing_metrics[0] if processing_metrics else {},
                    "usage_analytics": usage_analytics[0] if usage_analytics else {},
                    "external_analytics": external_analytics
                }
                
                return self.success_response(
                    data=analytics,
                    request_id=request_id
                )
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error getting video analytics {video_id}: {e}")
                raise HTTPException(status_code=500, detail="Failed to get video analytics")

# =============================================================================
# Route Factory
# =============================================================================

def create_video_routes(db_operations, api_operations, file_storage) -> VideoRoutes:
    """Factory function to create video routes with dependencies."""
    return VideoRoutes(db_operations, api_operations, file_storage)

# =============================================================================
# Export
# =============================================================================

__all__ = [
    "VideoRoutes",
    "create_video_routes"
] 