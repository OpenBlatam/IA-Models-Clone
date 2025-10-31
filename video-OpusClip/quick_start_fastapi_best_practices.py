"""
🚀 Quick Start Guide for FastAPI Best Practices

This script provides practical examples of FastAPI best practices for:
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
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum
import structlog

from fastapi import (
    FastAPI, APIRouter, Depends, HTTPException, status, Request, Response,
    BackgroundTasks, Query, Path, Body, Header, Form, File, UploadFile
)
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, ConfigDict, validator, root_validator, computed_field

logger = structlog.get_logger(__name__)

# =============================================================================
# 1. Data Models (Pydantic v2 Best Practices)
# =============================================================================

def example_1_data_models():
    """Example 1: Data Models with Pydantic v2."""
    print("\n=== Example 1: Data Models with Pydantic v2 ===")
    
    # Base model configuration
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
    
    # Enums
    class VideoStatus(str, Enum):
        PENDING = "pending"
        PROCESSING = "processing"
        COMPLETED = "completed"
        FAILED = "failed"
    
    class ProcessingPriority(str, Enum):
        LOW = "low"
        NORMAL = "normal"
        HIGH = "high"
        URGENT = "urgent"
    
    # Request model
    class VideoCreateRequest(BaseModel):
        title: str = Field(..., min_length=1, max_length=200, description="Video title")
        description: Optional[str] = Field(None, max_length=1000, description="Video description")
        url: str = Field(..., description="Video URL")
        duration: Optional[float] = Field(None, ge=0, description="Video duration in seconds")
        priority: ProcessingPriority = Field(ProcessingPriority.NORMAL, description="Processing priority")
        tags: List[str] = Field(default_factory=list, max_items=10, description="Video tags")
        
        @validator('url')
        def validate_url(cls, v):
            if not v.startswith(('http://', 'https://')):
                raise ValueError('URL must start with http:// or https://')
            return v
        
        @validator('tags')
        def validate_tags(cls, v):
            cleaned_tags = [tag.strip().lower() for tag in v if tag.strip()]
            return list(set(cleaned_tags))
    
    # Response model
    class VideoResponse(BaseResponseModel):
        id: int = Field(..., description="Video ID")
        title: str = Field(..., description="Video title")
        description: Optional[str] = Field(None, description="Video description")
        url: str = Field(..., description="Video URL")
        duration: Optional[float] = Field(None, description="Video duration in seconds")
        status: VideoStatus = Field(..., description="Processing status")
        priority: ProcessingPriority = Field(..., description="Processing priority")
        tags: List[str] = Field(default_factory=list, description="Video tags")
        created_at: datetime = Field(..., description="Creation timestamp")
        user_id: Optional[str] = Field(None, description="User ID")
        
        @computed_field
        @property
        def processing_time(self) -> Optional[float]:
            if self.status == VideoStatus.COMPLETED:
                return (datetime.now() - self.created_at).total_seconds()
            return None
    
    # Test the models
    video_request = VideoCreateRequest(
        title="Sample Video",
        description="A sample video for processing",
        url="https://example.com/video.mp4",
        duration=120.5,
        priority=ProcessingPriority.NORMAL,
        tags=["sample", "test"]
    )
    
    video_response = VideoResponse(
        id=1,
        title=video_request.title,
        description=video_request.description,
        url=video_request.url,
        duration=video_request.duration,
        status=VideoStatus.PENDING,
        priority=video_request.priority,
        tags=video_request.tags,
        created_at=datetime.now(),
        user_id="user_123"
    )
    
    print("✅ Created data models:")
    print(f"  - Request model: {video_request}")
    print(f"  - Response model: {video_response}")
    print(f"  - Processing time: {video_response.processing_time}")

# =============================================================================
# 2. Path Operations (HTTP Methods, Status Codes, Validation)
# =============================================================================

def example_2_path_operations():
    """Example 2: Path Operations with best practices."""
    print("\n=== Example 2: Path Operations ===")
    
    router = APIRouter(prefix="/api/v1/videos", tags=["videos"])
    
    # Mock dependencies
    async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())) -> Dict[str, Any]:
        return {"id": "user_123", "email": "user@example.com"}
    
    # POST - Create video
    @router.post(
        "/",
        response_model=VideoResponse,
        status_code=status.HTTP_201_CREATED,
        summary="Create video processing job",
        description="Create a new video processing job with AI analysis"
    )
    async def create_video(
        video_data: VideoCreateRequest = Body(..., description="Video data"),
        current_user: Dict[str, Any] = Depends(get_current_user),
        background_tasks: BackgroundTasks = BackgroundTasks()
    ) -> VideoResponse:
        """Create a new video processing job."""
        # Simulate video creation
        video_response = VideoResponse(
            id=int(time.time()),
            title=video_data.title,
            description=video_data.description,
            url=video_data.url,
            duration=video_data.duration,
            status=VideoStatus.PENDING,
            priority=video_data.priority,
            tags=video_data.tags,
            created_at=datetime.now(),
            user_id=current_user["id"]
        )
        
        # Add background task
        background_tasks.add_task(process_video_background, video_response.id)
        
        return video_response
    
    # GET - Get video by ID
    @router.get(
        "/{video_id}",
        response_model=VideoResponse,
        summary="Get video by ID",
        description="Retrieve a video processing job by its ID",
        responses={
            200: {"description": "Video found successfully"},
            404: {"description": "Video not found"},
            422: {"description": "Validation error"}
        }
    )
    async def get_video(
        video_id: int = Path(..., gt=0, description="Video ID"),
        current_user: Dict[str, Any] = Depends(get_current_user)
    ) -> VideoResponse:
        """Get video processing job by ID."""
        # Simulate video retrieval
        if video_id > 1000:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Video with ID {video_id} not found"
            )
        
        return VideoResponse(
            id=video_id,
            title=f"Video {video_id}",
            description="Sample video description",
            url="https://example.com/video.mp4",
            duration=120.5,
            status=VideoStatus.COMPLETED,
            priority=ProcessingPriority.NORMAL,
            tags=["sample"],
            created_at=datetime.now(),
            user_id=current_user["id"]
        )
    
    # GET - List videos with pagination
    @router.get(
        "/",
        response_model=List[VideoResponse],
        summary="List videos",
        description="List video processing jobs with pagination and filtering"
    )
    async def list_videos(
        page: int = Query(1, ge=1, description="Page number"),
        per_page: int = Query(10, ge=1, le=100, description="Items per page"),
        status: Optional[VideoStatus] = Query(None, description="Filter by status"),
        search: Optional[str] = Query(None, min_length=1, max_length=100, description="Search in title"),
        current_user: Dict[str, Any] = Depends(get_current_user)
    ) -> List[VideoResponse]:
        """List video processing jobs with pagination and filtering."""
        # Simulate video list
        videos = []
        for i in range(per_page):
            video = VideoResponse(
                id=i + 1,
                title=f"Video {i + 1}",
                description=f"Description for video {i + 1}",
                url=f"https://example.com/video{i + 1}.mp4",
                duration=120.5 + i,
                status=VideoStatus.COMPLETED if i % 2 == 0 else VideoStatus.PENDING,
                priority=ProcessingPriority.NORMAL,
                tags=["sample"],
                created_at=datetime.now(),
                user_id=current_user["id"]
            )
            videos.append(video)
        
        return videos
    
    # PATCH - Update video
    @router.patch(
        "/{video_id}",
        response_model=VideoResponse,
        summary="Update video",
        description="Update video processing job details"
    )
    async def update_video(
        video_id: int = Path(..., gt=0, description="Video ID"),
        video_data: VideoCreateRequest = Body(..., description="Update data"),
        current_user: Dict[str, Any] = Depends(get_current_user)
    ) -> VideoResponse:
        """Update video processing job."""
        # Simulate video update
        return VideoResponse(
            id=video_id,
            title=video_data.title,
            description=video_data.description,
            url=video_data.url,
            duration=video_data.duration,
            status=VideoStatus.PROCESSING,
            priority=video_data.priority,
            tags=video_data.tags,
            created_at=datetime.now(),
            user_id=current_user["id"]
        )
    
    # DELETE - Delete video
    @router.delete(
        "/{video_id}",
        status_code=status.HTTP_204_NO_CONTENT,
        summary="Delete video",
        description="Delete a video processing job"
    )
    async def delete_video(
        video_id: int = Path(..., gt=0, description="Video ID"),
        current_user: Dict[str, Any] = Depends(get_current_user)
    ):
        """Delete video processing job."""
        # Simulate video deletion
        if video_id > 1000:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Video with ID {video_id} not found"
            )
        
        return None
    
    print("✅ Created path operations:")
    print("  - POST /api/v1/videos/ (create video)")
    print("  - GET /api/v1/videos/{video_id} (get video)")
    print("  - GET /api/v1/videos/ (list videos with pagination)")
    print("  - PATCH /api/v1/videos/{video_id} (update video)")
    print("  - DELETE /api/v1/videos/{video_id} (delete video)")
    
    return router

# =============================================================================
# 3. Middleware (CORS, Authentication, Logging, Compression)
# =============================================================================

def example_3_middleware():
    """Example 3: Middleware configuration."""
    print("\n=== Example 3: Middleware Configuration ===")
    
    app = FastAPI(title="Middleware Example", version="1.0.0")
    
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
    
    # Request Logging Middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start_time = time.time()
        
        # Log request
        logger.info(
            "Request started",
            method=request.method,
            url=str(request.url),
            client_ip=request.client.host if request.client else None
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
        client_ip = request.client.host if request.client else "unknown"
        
        # Simple rate limiting (mock implementation)
        if not await check_rate_limit(client_ip):
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
        if not await validate_token(auth_header):
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"error": "Invalid authentication token"}
            )
        
        return await call_next(request)
    
    print("✅ Configured middleware:")
    print("  - CORS middleware (cross-origin requests)")
    print("  - GZip compression middleware")
    print("  - Request logging middleware")
    print("  - Rate limiting middleware")
    print("  - Authentication middleware")
    
    return app

# =============================================================================
# 4. Error Handling
# =============================================================================

def example_4_error_handling():
    """Example 4: Error handling patterns."""
    print("\n=== Example 4: Error Handling ===")
    
    app = FastAPI(title="Error Handling Example", version="1.0.0")
    
    # Global exception handlers
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions."""
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail,
                "status_code": exc.status_code,
                "timestamp": datetime.now().isoformat(),
                "path": request.url.path
            }
        )
    
    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        """Handle ValueError exceptions."""
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": str(exc),
                "status_code": 400,
                "timestamp": datetime.now().isoformat(),
                "path": request.url.path
            }
        )
    
    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        """Handle generic exceptions."""
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal server error",
                "status_code": 500,
                "timestamp": datetime.now().isoformat(),
                "path": request.url.path
            }
        )
    
    # Route with error handling
    @app.get("/test-error/{error_type}")
    async def test_error(error_type: str):
        """Test different types of errors."""
        if error_type == "validation":
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Validation error: Invalid input data"
            )
        elif error_type == "not_found":
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Resource not found"
            )
        elif error_type == "server_error":
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error"
            )
        elif error_type == "value_error":
            raise ValueError("This is a value error")
        else:
            return {"message": "No error simulated"}
    
    print("✅ Configured error handling:")
    print("  - HTTP exception handler")
    print("  - ValueError handler")
    print("  - Generic exception handler")
    print("  - Test endpoint: GET /test-error/{error_type}")
    
    return app

# =============================================================================
# 5. Documentation
# =============================================================================

def example_5_documentation():
    """Example 5: OpenAPI documentation."""
    print("\n=== Example 5: OpenAPI Documentation ===")
    
    app = FastAPI(
        title="Video-OpusClip API",
        description="AI-powered video processing system with FastAPI best practices",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json"
    )
    
    # Custom OpenAPI schema
    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        
        from fastapi.openapi.utils import get_openapi
        
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
        
        # Add security schemes
        openapi_schema["components"]["securitySchemes"] = {
            "BearerAuth": {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT"
            }
        }
        
        app.openapi_schema = openapi_schema
        return app.openapi_schema
    
    app.openapi = custom_openapi
    
    # Example route with comprehensive documentation
    @app.post(
        "/videos",
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
        current_user: Dict[str, Any] = Depends(lambda: {"id": "user_123"})
    ) -> VideoResponse:
        """
        Create a new video processing job.
        
        - **title**: Video title (required)
        - **description**: Video description (optional)
        - **url**: Video URL (required)
        - **duration**: Video duration in seconds (optional)
        - **priority**: Processing priority (default: normal)
        - **tags**: List of video tags (optional)
        
        Returns the created video processing job.
        """
        return VideoResponse(
            id=1,
            title=video_data.title,
            description=video_data.description,
            url=video_data.url,
            duration=video_data.duration,
            status=VideoStatus.PENDING,
            priority=video_data.priority,
            tags=video_data.tags,
            created_at=datetime.now(),
            user_id=current_user["id"]
        )
    
    print("✅ Configured OpenAPI documentation:")
    print("  - Custom OpenAPI schema")
    print("  - Security schemes")
    print("  - Example requests")
    print("  - Comprehensive route documentation")
    print("  - Available at: /docs and /redoc")
    
    return app

# =============================================================================
# 6. Performance Optimization
# =============================================================================

def example_6_performance_optimization():
    """Example 6: Performance optimization patterns."""
    print("\n=== Example 6: Performance Optimization ===")
    
    app = FastAPI(title="Performance Example", version="1.0.0")
    
    # Dependency injection
    async def get_db_session():
        """Get database session."""
        # Simulate database session
        return {"connection": "mock_db"}
    
    async def get_cache_client():
        """Get cache client."""
        # Simulate cache client
        return {"connection": "mock_cache"}
    
    async def get_current_user(
        credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())
    ) -> Dict[str, Any]:
        """Get current authenticated user."""
        return {"id": "user_123", "email": "user@example.com"}
    
    # Route with dependency injection
    @app.post("/videos")
    async def create_video(
        video_data: VideoCreateRequest,
        db_session = Depends(get_db_session),
        cache_client = Depends(get_cache_client),
        current_user: Dict[str, Any] = Depends(get_current_user),
        background_tasks: BackgroundTasks = BackgroundTasks()
    ) -> VideoResponse:
        """Create a new video processing job with performance optimization."""
        
        # Add background task
        background_tasks.add_task(process_video_background, video_data.title)
        
        return VideoResponse(
            id=1,
            title=video_data.title,
            description=video_data.description,
            url=video_data.url,
            duration=video_data.duration,
            status=VideoStatus.PENDING,
            priority=video_data.priority,
            tags=video_data.tags,
            created_at=datetime.now(),
            user_id=current_user["id"]
        )
    
    # Caching example
    from functools import lru_cache
    
    @lru_cache()
    def get_config():
        """Get application configuration (cached)."""
        return {"max_videos": 100, "timeout": 30}
    
    @app.get("/config")
    async def get_application_config():
        """Get application configuration."""
        return get_config()
    
    print("✅ Implemented performance optimizations:")
    print("  - Dependency injection for shared resources")
    print("  - Background tasks for long-running operations")
    print("  - LRU caching for expensive operations")
    print("  - Efficient database and cache access")
    
    return app

# =============================================================================
# 7. Security Best Practices
# =============================================================================

def example_7_security():
    """Example 7: Security best practices."""
    print("\n=== Example 7: Security Best Practices ===")
    
    app = FastAPI(title="Security Example", version="1.0.0")
    
    # Authentication
    async def get_current_user(
        credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())
    ) -> Dict[str, Any]:
        """Get current authenticated user."""
        # Mock JWT validation
        if not credentials.credentials:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication token"
            )
        
        return {"id": "user_123", "email": "user@example.com", "role": "user"}
    
    # Authorization
    async def require_permission(permission: str):
        """Dependency to require specific permission."""
        async def permission_checker(
            current_user: Dict[str, Any] = Depends(get_current_user)
        ):
            # Mock permission check
            if permission == "admin" and current_user["role"] != "admin":
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Insufficient permissions"
                )
            return current_user
        return permission_checker
    
    # Secure route
    @app.post("/admin/videos")
    async def create_admin_video(
        video_data: VideoCreateRequest,
        current_user: Dict[str, Any] = Depends(require_permission("admin"))
    ) -> VideoResponse:
        """Create video (admin only)."""
        return VideoResponse(
            id=1,
            title=video_data.title,
            description=video_data.description,
            url=video_data.url,
            duration=video_data.duration,
            status=VideoStatus.PENDING,
            priority=video_data.priority,
            tags=video_data.tags,
            created_at=datetime.now(),
            user_id=current_user["id"]
        )
    
    # Input validation
    class SecureVideoRequest(BaseModel):
        title: str = Field(..., min_length=1, max_length=200)
        url: str = Field(..., regex=r"^https?://.*")
        
        @validator('url')
        def validate_url(cls, v):
            if not v.startswith(('http://', 'https://')):
                raise ValueError('URL must start with http:// or https://')
            return v
    
    @app.post("/secure/videos")
    async def create_secure_video(
        video_data: SecureVideoRequest,
        current_user: Dict[str, Any] = Depends(get_current_user)
    ) -> VideoResponse:
        """Create video with secure input validation."""
        return VideoResponse(
            id=1,
            title=video_data.title,
            description=None,
            url=video_data.url,
            duration=None,
            status=VideoStatus.PENDING,
            priority=ProcessingPriority.NORMAL,
            tags=[],
            created_at=datetime.now(),
            user_id=current_user["id"]
        )
    
    print("✅ Implemented security best practices:")
    print("  - JWT authentication")
    print("  - Role-based authorization")
    print("  - Input validation and sanitization")
    print("  - Secure route protection")
    
    return app

# =============================================================================
# Helper Functions
# =============================================================================

async def process_video_background(video_title: str):
    """Background task for video processing."""
    await asyncio.sleep(2)  # Simulate processing
    logger.info(f"Background processing completed for video: {video_title}")

async def check_rate_limit(client_ip: str) -> bool:
    """Check rate limit for client IP (mock implementation)."""
    return True  # Always allow for demo

async def validate_token(token: str) -> bool:
    """Validate authentication token (mock implementation)."""
    return True  # Always valid for demo

# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Run all FastAPI best practices examples."""
    print("🚀 FastAPI Best Practices Quick Start Guide")
    print("=" * 60)
    
    # Run examples
    example_1_data_models()
    example_2_path_operations()
    example_3_middleware()
    example_4_error_handling()
    example_5_documentation()
    example_6_performance_optimization()
    example_7_security()
    
    print("\n✅ All examples completed successfully!")
    print("\n📚 Next Steps:")
    print("  1. Review the fastapi_best_practices.py file for full implementation")
    print("  2. Check FASTAPI_BEST_PRACTICES_GUIDE.md for detailed documentation")
    print("  3. Start building your own FastAPI application using these patterns")
    print("  4. Customize the patterns for your specific use case")

if __name__ == "__main__":
    main() 