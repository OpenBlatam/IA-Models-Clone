"""
🚀 FastAPI Best Practices Implementation

Comprehensive implementation of FastAPI best practices for:
- Data Models (Pydantic v2)
- Path Operations (HTTP methods, status codes, validation)
- Middleware (CORS, authentication, logging, compression)
- Error Handling
- Documentation
- Performance Optimization

Based on official FastAPI documentation and real-world best practices.
"""

import asyncio
import time
import logging
from typing import Any, Dict, List, Optional, Union, Annotated
from datetime import datetime, timedelta
from pathlib import Path
import json
from functools import wraps

from fastapi import (
    FastAPI, APIRouter, Depends, HTTPException, status, Request, Response,
    BackgroundTasks, Query, Path, Body, Header, Cookie, Form, File, UploadFile,
    WebSocket, WebSocketDisconnect
)
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.utils import get_openapi
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from pydantic import BaseModel, Field, ConfigDict, validator, root_validator, computed_field
from pydantic.types import EmailStr, UUID4
import structlog

logger = structlog.get_logger(__name__)

# =============================================================================
# 1. DATA MODELS (Pydantic v2 Best Practices)
# =============================================================================

class BaseResponseModel(BaseModel):
    """Base response model with common configuration."""
    
    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat()},
        from_attributes=True,
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
        use_enum_values=True
    )

class VideoStatus(str, Enum):
    """Video processing status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ProcessingPriority(str, Enum):
    """Processing priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"

# Request Models
class VideoCreateRequest(BaseModel):
    """Request model for creating a video processing job."""
    
    title: str = Field(..., min_length=1, max_length=200, description="Video title")
    description: Optional[str] = Field(None, max_length=1000, description="Video description")
    url: str = Field(..., description="Video URL")
    duration: Optional[float] = Field(None, ge=0, description="Video duration in seconds")
    resolution: Optional[str] = Field(None, pattern=r"^\d+x\d+$", description="Video resolution (e.g., 1920x1080)")
    priority: ProcessingPriority = Field(ProcessingPriority.NORMAL, description="Processing priority")
    tags: List[str] = Field(default_factory=list, max_items=10, description="Video tags")
    
    @validator('url')
    def validate_url(cls, v):
        """Validate video URL format."""
        if not v.startswith(('http://', 'https://')):
            raise ValueError('URL must start with http:// or https://')
        return v
    
    @validator('tags')
    def validate_tags(cls, v):
        """Validate and clean tags."""
        cleaned_tags = [tag.strip().lower() for tag in v if tag.strip()]
        return list(set(cleaned_tags))  # Remove duplicates
    
    @root_validator
    def validate_video_data(cls, values):
        """Root validator for video data."""
        title = values.get('title')
        description = values.get('description')
        
        if description and len(description) < len(title):
            raise ValueError('Description should be longer than title')
        
        return values

class VideoUpdateRequest(BaseModel):
    """Request model for updating a video processing job."""
    
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=1000)
    priority: Optional[ProcessingPriority] = None
    tags: Optional[List[str]] = Field(None, max_items=10)
    
    @validator('tags')
    def validate_tags(cls, v):
        if v is not None:
            cleaned_tags = [tag.strip().lower() for tag in v if tag.strip()]
            return list(set(cleaned_tags))
        return v

class BatchVideoRequest(BaseModel):
    """Request model for batch video processing."""
    
    videos: List[VideoCreateRequest] = Field(..., min_items=1, max_items=100)
    batch_name: Optional[str] = Field(None, max_length=100)
    priority: ProcessingPriority = Field(ProcessingPriority.NORMAL)
    
    @validator('videos')
    def validate_videos(cls, v):
        """Validate batch size and video data."""
        if len(v) > 50:
            raise ValueError('Batch size cannot exceed 50 videos')
        return v

# Response Models
class VideoResponse(BaseResponseModel):
    """Response model for video data."""
    
    id: int = Field(..., description="Video ID")
    title: str = Field(..., description="Video title")
    description: Optional[str] = Field(None, description="Video description")
    url: str = Field(..., description="Video URL")
    duration: Optional[float] = Field(None, description="Video duration in seconds")
    resolution: Optional[str] = Field(None, description="Video resolution")
    status: VideoStatus = Field(..., description="Processing status")
    priority: ProcessingPriority = Field(..., description="Processing priority")
    tags: List[str] = Field(default_factory=list, description="Video tags")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    user_id: Optional[str] = Field(None, description="User ID")
    
    @computed_field
    @property
    def processing_time(self) -> Optional[float]:
        """Calculate processing time if completed."""
        if self.status == VideoStatus.COMPLETED and self.updated_at:
            return (self.updated_at - self.created_at).total_seconds()
        return None

class BatchVideoResponse(BaseResponseModel):
    """Response model for batch video processing."""
    
    batch_id: str = Field(..., description="Batch ID")
    batch_name: Optional[str] = Field(None, description="Batch name")
    total_videos: int = Field(..., description="Total videos in batch")
    processed_videos: int = Field(..., description="Number of processed videos")
    failed_videos: int = Field(..., description="Number of failed videos")
    status: VideoStatus = Field(..., description="Batch status")
    created_at: datetime = Field(..., description="Creation timestamp")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    
    @computed_field
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_videos == 0:
            return 0.0
        return (self.processed_videos / self.total_videos) * 100

class PaginatedResponse(BaseResponseModel):
    """Generic paginated response model."""
    
    items: List[Any] = Field(..., description="List of items")
    total: int = Field(..., description="Total number of items")
    page: int = Field(..., description="Current page number")
    per_page: int = Field(..., description="Items per page")
    has_next: bool = Field(..., description="Whether there are more pages")
    has_prev: bool = Field(..., description="Whether there are previous pages")
    
    @computed_field
    @property
    def total_pages(self) -> int:
        """Calculate total number of pages."""
        return (self.total + self.per_page - 1) // self.per_page

class ErrorResponse(BaseResponseModel):
    """Standard error response model."""
    
    error: str = Field(..., description="Error message")
    status_code: int = Field(..., description="HTTP status code")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
    path: Optional[str] = Field(None, description="Request path")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")

class HealthResponse(BaseResponseModel):
    """Health check response model."""
    
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=datetime.now, description="Check timestamp")
    version: str = Field(..., description="API version")
    uptime: float = Field(..., description="Service uptime in seconds")
    services: Dict[str, Dict[str, Any]] = Field(..., description="Service health status")

# =============================================================================
# 2. PATH OPERATIONS (HTTP Methods, Status Codes, Validation)
# =============================================================================

class VideoRouter:
    """Video processing router with best practices."""
    
    def __init__(self):
        self.router = APIRouter(prefix="/api/v1/videos", tags=["videos"])
        self.setup_routes()
    
    def setup_routes(self):
        """Setup all video routes."""
        
        @self.router.post(
            "/",
            response_model=VideoResponse,
            status_code=status.HTTP_201_CREATED,
            summary="Create video processing job",
            description="Create a new video processing job with AI analysis",
            response_description="Video processing job created successfully",
            openapi_extra={
                "examples": {
                    "normal": {
                        "summary": "Normal video",
                        "value": {
                            "title": "Sample Video",
                            "description": "A sample video for processing",
                            "url": "https://example.com/video.mp4",
                            "duration": 120.5,
                            "resolution": "1920x1080",
                            "priority": "normal",
                            "tags": ["sample", "test"]
                        }
                    },
                    "urgent": {
                        "summary": "Urgent video",
                        "value": {
                            "title": "Urgent Video",
                            "description": "An urgent video that needs quick processing",
                            "url": "https://example.com/urgent.mp4",
                            "priority": "urgent"
                        }
                    }
                }
            }
        )
        async def create_video(
            video_data: VideoCreateRequest = Body(..., description="Video data"),
            current_user: Dict[str, Any] = Depends(self.get_current_user),
            background_tasks: BackgroundTasks = BackgroundTasks()
        ) -> VideoResponse:
            """
            Create a new video processing job.
            
            - **title**: Video title (required)
            - **description**: Video description (optional)
            - **url**: Video URL (required)
            - **duration**: Video duration in seconds (optional)
            - **resolution**: Video resolution (optional)
            - **priority**: Processing priority (default: normal)
            - **tags**: List of video tags (optional)
            
            Returns the created video processing job.
            """
            try:
                # Validate user permissions
                if not current_user:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Authentication required"
                    )
                
                # Create video record
                video_record = {
                    "id": self.generate_video_id(),
                    "title": video_data.title,
                    "description": video_data.description,
                    "url": video_data.url,
                    "duration": video_data.duration,
                    "resolution": video_data.resolution,
                    "status": VideoStatus.PENDING,
                    "priority": video_data.priority,
                    "tags": video_data.tags,
                    "created_at": datetime.now(),
                    "user_id": current_user.get("id")
                }
                
                # Add background task for processing
                background_tasks.add_task(self.process_video_background, video_record["id"])
                
                return VideoResponse(**video_record)
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to create video: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to create video processing job"
                )
        
        @self.router.get(
            "/{video_id}",
            response_model=VideoResponse,
            summary="Get video by ID",
            description="Retrieve a video processing job by its ID",
            responses={
                200: {"description": "Video found successfully"},
                404: {"description": "Video not found", "model": ErrorResponse},
                422: {"description": "Validation error"}
            }
        )
        async def get_video(
            video_id: int = Path(..., gt=0, description="Video ID"),
            current_user: Dict[str, Any] = Depends(self.get_current_user)
        ) -> VideoResponse:
            """
            Get video processing job by ID.
            
            - **video_id**: The ID of the video to retrieve (must be positive)
            
            Returns the video processing job if found.
            """
            try:
                # Check if video exists
                video = await self.get_video_by_id(video_id)
                if not video:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Video with ID {video_id} not found"
                    )
                
                # Check user permissions
                if video["user_id"] != current_user.get("id"):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Access denied to this video"
                    )
                
                return VideoResponse(**video)
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to get video {video_id}: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to retrieve video"
                )
        
        @self.router.get(
            "/",
            response_model=PaginatedResponse,
            summary="List videos",
            description="List video processing jobs with pagination and filtering"
        )
        async def list_videos(
            page: int = Query(1, ge=1, description="Page number"),
            per_page: int = Query(10, ge=1, le=100, description="Items per page"),
            status: Optional[VideoStatus] = Query(None, description="Filter by status"),
            priority: Optional[ProcessingPriority] = Query(None, description="Filter by priority"),
            search: Optional[str] = Query(None, min_length=1, max_length=100, description="Search in title and description"),
            current_user: Dict[str, Any] = Depends(self.get_current_user)
        ) -> PaginatedResponse:
            """
            List video processing jobs with pagination and filtering.
            
            - **page**: Page number (default: 1)
            - **per_page**: Items per page (default: 10, max: 100)
            - **status**: Filter by processing status
            - **priority**: Filter by priority level
            - **search**: Search in title and description
            
            Returns paginated list of videos.
            """
            try:
                # Get videos with filters
                videos, total = await self.get_videos_paginated(
                    page=page,
                    per_page=per_page,
                    status=status,
                    priority=priority,
                    search=search,
                    user_id=current_user.get("id")
                )
                
                # Calculate pagination info
                has_next = (page * per_page) < total
                has_prev = page > 1
                
                return PaginatedResponse(
                    items=videos,
                    total=total,
                    page=page,
                    per_page=per_page,
                    has_next=has_next,
                    has_prev=has_prev
                )
                
            except Exception as e:
                logger.error(f"Failed to list videos: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to retrieve video list"
                )
        
        @self.router.patch(
            "/{video_id}",
            response_model=VideoResponse,
            summary="Update video",
            description="Update video processing job details"
        )
        async def update_video(
            video_id: int = Path(..., gt=0, description="Video ID"),
            video_data: VideoUpdateRequest = Body(..., description="Update data"),
            current_user: Dict[str, Any] = Depends(self.get_current_user)
        ) -> VideoResponse:
            """
            Update video processing job.
            
            - **video_id**: The ID of the video to update
            - **video_data**: Updated video data
            
            Returns the updated video processing job.
            """
            try:
                # Check if video exists and user has access
                video = await self.get_video_by_id(video_id)
                if not video:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Video with ID {video_id} not found"
                    )
                
                if video["user_id"] != current_user.get("id"):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Access denied to this video"
                    )
                
                # Update video data
                update_data = video_data.dict(exclude_unset=True)
                update_data["updated_at"] = datetime.now()
                
                updated_video = {**video, **update_data}
                
                return VideoResponse(**updated_video)
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to update video {video_id}: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to update video"
                )
        
        @self.router.delete(
            "/{video_id}",
            status_code=status.HTTP_204_NO_CONTENT,
            summary="Delete video",
            description="Delete a video processing job"
        )
        async def delete_video(
            video_id: int = Path(..., gt=0, description="Video ID"),
            current_user: Dict[str, Any] = Depends(self.get_current_user)
        ):
            """
            Delete video processing job.
            
            - **video_id**: The ID of the video to delete
            
            Returns 204 No Content on successful deletion.
            """
            try:
                # Check if video exists and user has access
                video = await self.get_video_by_id(video_id)
                if not video:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Video with ID {video_id} not found"
                    )
                
                if video["user_id"] != current_user.get("id"):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Access denied to this video"
                    )
                
                # Delete video (soft delete)
                await self.delete_video_by_id(video_id)
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to delete video {video_id}: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to delete video"
                )
    
    # Helper methods (mock implementations)
    async def get_current_user(self, credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())) -> Dict[str, Any]:
        """Get current authenticated user."""
        # Mock implementation
        return {"id": "user_123", "email": "user@example.com", "role": "user"}
    
    def generate_video_id(self) -> int:
        """Generate a unique video ID."""
        return int(time.time() * 1000)
    
    async def process_video_background(self, video_id: int):
        """Background task for video processing."""
        await asyncio.sleep(5)  # Simulate processing
        logger.info(f"Background processing completed for video {video_id}")
    
    async def get_video_by_id(self, video_id: int) -> Optional[Dict[str, Any]]:
        """Get video by ID (mock implementation)."""
        # Mock implementation
        return {
            "id": video_id,
            "title": "Sample Video",
            "status": VideoStatus.PENDING,
            "created_at": datetime.now(),
            "user_id": "user_123"
        }
    
    async def get_videos_paginated(self, **kwargs) -> tuple[List[Dict[str, Any]], int]:
        """Get paginated videos (mock implementation)."""
        # Mock implementation
        return ([], 0)
    
    async def delete_video_by_id(self, video_id: int):
        """Delete video by ID (mock implementation)."""
        pass

# =============================================================================
# 3. MIDDLEWARE (CORS, Authentication, Logging, Compression)
# =============================================================================

class FastAPIMiddleware:
    """FastAPI middleware configuration."""
    
    @staticmethod
    def setup_middleware(app: FastAPI):
        """Setup all middleware for the application."""
        
        # CORS Middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://localhost:3000", "https://yourdomain.com"],
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
            allow_headers=["*"],
            expose_headers=["X-Process-Time", "X-Total-Count"]
        )
        
        # GZip Compression Middleware
        app.add_middleware(
            GZipMiddleware,
            minimum_size=1000
        )
        
        # Trusted Host Middleware
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["localhost", "127.0.0.1", "yourdomain.com"]
        )
        
        # Request Logging Middleware
        @app.middleware("http")
        async def log_requests(request: Request, call_next):
            start_time = time.time()
            
            # Log request
            logger.info(
                "Request started",
                method=request.method,
                url=str(request.url),
                client_ip=request.client.host if request.client else None,
                user_agent=request.headers.get("user-agent")
            )
            
            # Process request
            response = await call_next(request)
            
            # Log response
            process_time = time.time() - start_time
            logger.info(
                "Request completed",
                method=request.method,
                url=str(request.url),
                status_code=response.status_code,
                process_time=process_time
            )
            
            # Add performance headers
            response.headers["X-Process-Time"] = str(process_time)
            response.headers["X-Request-ID"] = request.headers.get("X-Request-ID", "unknown")
            
            return response
        
        # Rate Limiting Middleware
        @app.middleware("http")
        async def rate_limit_middleware(request: Request, call_next):
            # Simple rate limiting implementation
            client_ip = request.client.host if request.client else "unknown"
            
            # Check rate limit (mock implementation)
            if not await FastAPIMiddleware.check_rate_limit(client_ip):
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "error": "Rate limit exceeded",
                        "retry_after": 60
                    },
                    headers={"Retry-After": "60"}
                )
            
            response = await call_next(request)
            return response
        
        # Authentication Middleware
        @app.middleware("http")
        async def auth_middleware(request: Request, call_next):
            # Skip authentication for public endpoints
            if request.url.path in ["/health", "/docs", "/redoc", "/openapi.json"]:
                return await call_next(request)
            
            # Check authentication
            auth_header = request.headers.get("authorization")
            if not auth_header:
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"error": "Authentication required"}
                )
            
            # Validate token (mock implementation)
            if not await FastAPIMiddleware.validate_token(auth_header):
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"error": "Invalid authentication token"}
                )
            
            return await call_next(request)
    
    @staticmethod
    async def check_rate_limit(client_ip: str) -> bool:
        """Check rate limit for client IP (mock implementation)."""
        # Mock implementation - always allow
        return True
    
    @staticmethod
    async def validate_token(token: str) -> bool:
        """Validate authentication token (mock implementation)."""
        # Mock implementation - always valid
        return True

# =============================================================================
# 4. ERROR HANDLING
# =============================================================================

class FastAPIErrorHandlers:
    """FastAPI error handling configuration."""
    
    @staticmethod
    def setup_error_handlers(app: FastAPI):
        """Setup error handlers for the application."""
        
        @app.exception_handler(HTTPException)
        async def http_exception_handler(request: Request, exc: HTTPException):
            """Handle HTTP exceptions."""
            return JSONResponse(
                status_code=exc.status_code,
                content=ErrorResponse(
                    error=exc.detail,
                    status_code=exc.status_code,
                    path=request.url.path
                ).dict()
            )
        
        @app.exception_handler(ValueError)
        async def value_error_handler(request: Request, exc: ValueError):
            """Handle ValueError exceptions."""
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=ErrorResponse(
                    error=str(exc),
                    status_code=400,
                    path=request.url.path
                ).dict()
            )
        
        @app.exception_handler(Exception)
        async def generic_exception_handler(request: Request, exc: Exception):
            """Handle generic exceptions."""
            logger.error(f"Unhandled exception: {exc}", exc_info=True)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=ErrorResponse(
                    error="Internal server error",
                    status_code=500,
                    path=request.url.path
                ).dict()
            )

# =============================================================================
# 5. APPLICATION FACTORY
# =============================================================================

def create_fastapi_app() -> FastAPI:
    """Create FastAPI application with best practices."""
    
    # Create FastAPI app
    app = FastAPI(
        title="Video-OpusClip API",
        description="AI-powered video processing system with FastAPI best practices",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        debug=False
    )
    
    # Setup middleware
    FastAPIMiddleware.setup_middleware(app)
    
    # Setup error handlers
    FastAPIErrorHandlers.setup_error_handlers(app)
    
    # Setup routes
    video_router = VideoRouter()
    app.include_router(video_router.router)
    
    # Health check endpoint
    @app.get(
        "/health",
        response_model=HealthResponse,
        summary="Health check",
        description="Check system health status"
    )
    async def health_check():
        """Health check endpoint."""
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            uptime=time.time(),
            services={
                "database": {"status": "healthy", "response_time": 0.001},
                "cache": {"status": "healthy", "response_time": 0.0005},
                "models": {"status": "healthy", "loaded_models": ["video_processor"]}
            }
        )
    
    # Custom OpenAPI schema
    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        
        openapi_schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )
        
        # Add custom info
        openapi_schema["info"]["x-logo"] = {
            "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
        }
        
        app.openapi_schema = openapi_schema
        return app.openapi_schema
    
    app.openapi = custom_openapi
    
    return app

# =============================================================================
# 6. USAGE EXAMPLE
# =============================================================================

def main():
    """Main function to run the FastAPI application."""
    app = create_fastapi_app()
    
    # Run the application
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

if __name__ == "__main__":
    main() 