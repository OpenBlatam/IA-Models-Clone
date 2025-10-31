# FastAPI Best Practices Guide

## Overview

This guide provides comprehensive best practices for FastAPI development, covering Data Models (Pydantic v2), Path Operations, and Middleware based on official FastAPI documentation. These practices ensure maintainable, scalable, and performant API development.

## 📋 Table of Contents

1. [Data Models (Pydantic v2)](#data-models-pydantic-v2)
2. [Path Operations](#path-operations)
3. [Middleware](#middleware)
4. [Error Handling](#error-handling)
5. [Documentation](#documentation)
6. [Performance Optimization](#performance-optimization)
7. [Security Best Practices](#security-best-practices)
8. [Testing](#testing)
9. [Deployment](#deployment)

## 🏗️ Data Models (Pydantic v2)

### Base Model Configuration

```python
from pydantic import BaseModel, ConfigDict, Field
from datetime import datetime
from typing import Optional, List, Dict, Any

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
```

### Request Models

#### ✅ Good Practices

```python
class VideoCreateRequest(BaseModel):
    """Request model for creating a video processing job."""
    
    title: str = Field(
        ..., 
        min_length=1, 
        max_length=200, 
        description="Video title"
    )
    description: Optional[str] = Field(
        None, 
        max_length=1000, 
        description="Video description"
    )
    url: str = Field(..., description="Video URL")
    duration: Optional[float] = Field(
        None, 
        ge=0, 
        description="Video duration in seconds"
    )
    resolution: Optional[str] = Field(
        None, 
        pattern=r"^\d+x\d+$", 
        description="Video resolution (e.g., 1920x1080)"
    )
    priority: ProcessingPriority = Field(
        ProcessingPriority.NORMAL, 
        description="Processing priority"
    )
    tags: List[str] = Field(
        default_factory=list, 
        max_items=10, 
        description="Video tags"
    )
    
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
```

#### ❌ Avoid These Practices

```python
# Don't use generic types without constraints
class BadRequest(BaseModel):
    data: Any  # Too generic
    
# Don't skip validation
class BadRequest(BaseModel):
    title: str  # No validation
    
# Don't use mutable defaults
class BadRequest(BaseModel):
    tags: List[str] = []  # Mutable default
```

### Response Models

#### ✅ Good Practices

```python
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
```

### Enum Usage

```python
from enum import Enum

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
```

## 🛣️ Path Operations

### HTTP Methods and Status Codes

#### ✅ Good Practices

```python
@router.post(
    "/",
    response_model=VideoResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create video processing job",
    description="Create a new video processing job with AI analysis",
    response_description="Video processing job created successfully"
)
async def create_video(
    video_data: VideoCreateRequest = Body(..., description="Video data"),
    current_user: Dict[str, Any] = Depends(get_current_user),
    background_tasks: BackgroundTasks = BackgroundTasks()
) -> VideoResponse:
    """Create a new video processing job."""
    pass

@router.get(
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
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> VideoResponse:
    """Get video processing job by ID."""
    pass

@router.patch(
    "/{video_id}",
    response_model=VideoResponse,
    summary="Update video",
    description="Update video processing job details"
)
async def update_video(
    video_id: int = Path(..., gt=0, description="Video ID"),
    video_data: VideoUpdateRequest = Body(..., description="Update data"),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> VideoResponse:
    """Update video processing job."""
    pass

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
    pass
```

### Query Parameters

#### ✅ Good Practices

```python
@router.get("/")
async def list_videos(
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(10, ge=1, le=100, description="Items per page"),
    status: Optional[VideoStatus] = Query(None, description="Filter by status"),
    priority: Optional[ProcessingPriority] = Query(None, description="Filter by priority"),
    search: Optional[str] = Query(
        None, 
        min_length=1, 
        max_length=100, 
        description="Search in title and description"
    ),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> PaginatedResponse:
    """List video processing jobs with pagination and filtering."""
    pass
```

### Path Parameters

#### ✅ Good Practices

```python
@router.get("/{video_id}")
async def get_video(
    video_id: int = Path(..., gt=0, description="Video ID"),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> VideoResponse:
    """Get video by ID."""
    pass

@router.get("/{video_id}/clips/{clip_id}")
async def get_video_clip(
    video_id: int = Path(..., gt=0, description="Video ID"),
    clip_id: int = Path(..., gt=0, description="Clip ID"),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> ClipResponse:
    """Get specific clip from video."""
    pass
```

### Request Body

#### ✅ Good Practices

```python
@router.post("/")
async def create_video(
    video_data: VideoCreateRequest = Body(
        ..., 
        description="Video data",
        examples={
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
    ),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> VideoResponse:
    """Create a new video processing job."""
    pass
```

### File Uploads

#### ✅ Good Practices

```python
@router.post("/upload")
async def upload_video(
    file: UploadFile = File(..., description="Video file to upload"),
    title: str = Form(..., description="Video title"),
    description: Optional[str] = Form(None, description="Video description"),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> VideoResponse:
    """Upload a video file."""
    
    # Validate file type
    if not file.content_type.startswith('video/'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be a video"
        )
    
    # Validate file size (e.g., 100MB limit)
    if file.size > 100 * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File size too large"
        )
    
    # Process file
    contents = await file.read()
    # ... process video file
    
    return VideoResponse(...)
```

## 🔧 Middleware

### CORS Middleware

#### ✅ Good Practices

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://yourdomain.com",
        "https://api.yourdomain.com"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Process-Time", "X-Total-Count"],
    max_age=3600
)
```

### Compression Middleware

#### ✅ Good Practices

```python
from fastapi.middleware.gzip import GZipMiddleware

app.add_middleware(
    GZipMiddleware,
    minimum_size=1000  # Only compress responses larger than 1KB
)
```

### Trusted Host Middleware

#### ✅ Good Practices

```python
from fastapi.middleware.trustedhost import TrustedHostMiddleware

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=[
        "localhost",
        "127.0.0.1",
        "yourdomain.com",
        "*.yourdomain.com"
    ]
)
```

### Custom Middleware

#### ✅ Good Practices

```python
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

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host if request.client else "unknown"
    
    # Check rate limit
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
    
    # Validate token
    if not await validate_token(auth_header):
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"error": "Invalid authentication token"}
        )
    
    return await call_next(request)
```

## 🚨 Error Handling

### Global Exception Handlers

#### ✅ Good Practices

```python
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
```

### Route-Level Error Handling

#### ✅ Good Practices

```python
@router.post("/")
async def create_video(
    video_data: VideoCreateRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> VideoResponse:
    """Create a new video processing job."""
    try:
        # Validate user permissions
        if not current_user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required"
            )
        
        # Check if user has permission to create videos
        if not await check_user_permissions(current_user["id"], "create_video"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        
        # Create video
        video = await create_video_record(video_data, current_user["id"])
        return VideoResponse(**video)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create video: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create video processing job"
        )
```

## 📚 Documentation

### OpenAPI Configuration

#### ✅ Good Practices

```python
app = FastAPI(
    title="Video-OpusClip API",
    description="AI-powered video processing system with FastAPI best practices",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    debug=False
)

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
```

### Route Documentation

#### ✅ Good Practices

```python
@router.post(
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
            }
        }
    }
)
async def create_video(
    video_data: VideoCreateRequest = Body(..., description="Video data"),
    current_user: Dict[str, Any] = Depends(get_current_user)
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
    pass
```

## ⚡ Performance Optimization

### Dependency Injection

#### ✅ Good Practices

```python
# Use dependency injection for shared resources
async def get_db_session():
    """Get database session."""
    async with session_factory() as session:
        try:
            yield session
        finally:
            await session.close()

async def get_cache_client():
    """Get cache client."""
    return redis_client

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())
) -> Dict[str, Any]:
    """Get current authenticated user."""
    # Validate token and return user
    pass

@router.post("/")
async def create_video(
    video_data: VideoCreateRequest,
    db_session = Depends(get_db_session),
    cache_client = Depends(get_cache_client),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> VideoResponse:
    """Create a new video processing job."""
    pass
```

### Background Tasks

#### ✅ Good Practices

```python
@router.post("/")
async def create_video(
    video_data: VideoCreateRequest,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> VideoResponse:
    """Create a new video processing job."""
    
    # Create video record
    video = await create_video_record(video_data, current_user["id"])
    
    # Add background task for processing
    background_tasks.add_task(process_video_background, video["id"])
    
    return VideoResponse(**video)

async def process_video_background(video_id: int):
    """Background task for video processing."""
    try:
        # Process video
        await process_video(video_id)
        logger.info(f"Background processing completed for video {video_id}")
    except Exception as e:
        logger.error(f"Background processing failed for video {video_id}: {e}")
```

### Caching

#### ✅ Good Practices

```python
from functools import lru_cache
import redis.asyncio as redis

@lru_cache()
def get_config():
    """Get application configuration (cached)."""
    return load_config()

async def get_cached_video(video_id: int, cache_client: redis.Redis):
    """Get video with caching."""
    cache_key = f"video:{video_id}"
    
    # Try to get from cache
    cached_data = await cache_client.get(cache_key)
    if cached_data:
        return json.loads(cached_data)
    
    # Get from database
    video = await get_video_from_db(video_id)
    
    # Cache for 5 minutes
    await cache_client.setex(cache_key, 300, json.dumps(video))
    
    return video
```

## 🔒 Security Best Practices

### Authentication

#### ✅ Good Practices

```python
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

security = HTTPBearer()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict[str, Any]:
    """Get current authenticated user."""
    try:
        payload = jwt.decode(
            credentials.credentials, 
            SECRET_KEY, 
            algorithms=["HS256"]
        )
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication token"
            )
        return {"id": user_id, "email": payload.get("email")}
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )
```

### Authorization

#### ✅ Good Practices

```python
async def check_user_permissions(user_id: str, permission: str) -> bool:
    """Check if user has specific permission."""
    # Check user permissions
    pass

async def require_permission(permission: str):
    """Dependency to require specific permission."""
    async def permission_checker(
        current_user: Dict[str, Any] = Depends(get_current_user)
    ):
        if not await check_user_permissions(current_user["id"], permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        return current_user
    return permission_checker

@router.post("/")
async def create_video(
    video_data: VideoCreateRequest,
    current_user: Dict[str, Any] = Depends(require_permission("create_video"))
) -> VideoResponse:
    """Create a new video processing job."""
    pass
```

### Input Validation

#### ✅ Good Practices

```python
class VideoCreateRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    url: str = Field(..., regex=r"^https?://.*")
    
    @validator('url')
    def validate_url(cls, v):
        """Validate video URL format."""
        if not v.startswith(('http://', 'https://')):
            raise ValueError('URL must start with http:// or https://')
        return v
```

## 🧪 Testing

### Test Configuration

#### ✅ Good Practices

```python
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

@pytest.fixture
def client():
    """Create test client."""
    app = create_fastapi_app()
    return TestClient(app)

@pytest.fixture
def mock_auth():
    """Mock authentication."""
    with patch("app.auth.get_current_user") as mock:
        mock.return_value = {"id": "test_user", "email": "test@example.com"}
        yield mock

def test_create_video(client, mock_auth):
    """Test video creation."""
    response = client.post(
        "/api/v1/videos/",
        json={
            "title": "Test Video",
            "url": "https://example.com/video.mp4"
        }
    )
    assert response.status_code == 201
    assert response.json()["title"] == "Test Video"
```

### Integration Tests

#### ✅ Good Practices

```python
@pytest.mark.asyncio
async def test_video_workflow():
    """Test complete video workflow."""
    # Create video
    video_data = VideoCreateRequest(
        title="Test Video",
        url="https://example.com/video.mp4"
    )
    
    # Test creation
    video = await create_video(video_data)
    assert video.status == VideoStatus.PENDING
    
    # Test processing
    await process_video(video.id)
    updated_video = await get_video(video.id)
    assert updated_video.status == VideoStatus.COMPLETED
```

## 🚀 Deployment

### Production Configuration

#### ✅ Good Practices

```python
import os
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await startup_event()
    yield
    # Shutdown
    await shutdown_event()

def create_production_app() -> FastAPI:
    """Create production FastAPI application."""
    app = FastAPI(
        title="Video-OpusClip API",
        version="1.0.0",
        lifespan=lifespan,
        docs_url=None,  # Disable docs in production
        redoc_url=None
    )
    
    # Production middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.getenv("ALLOWED_ORIGINS", "").split(","),
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
        allow_headers=["*"]
    )
    
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Setup routes
    setup_routes(app)
    
    return app
```

### Docker Configuration

#### ✅ Good Practices

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 📊 Monitoring and Logging

### Structured Logging

#### ✅ Good Practices

```python
import structlog

logger = structlog.get_logger(__name__)

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
    
    return response
```

### Health Checks

#### ✅ Good Practices

```python
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "uptime": time.time(),
        "services": {
            "database": {"status": "healthy", "response_time": 0.001},
            "cache": {"status": "healthy", "response_time": 0.0005},
            "models": {"status": "healthy", "loaded_models": ["video_processor"]}
        }
    }
```

## 🎯 Summary

This guide covers the essential FastAPI best practices for:

1. **Data Models**: Proper Pydantic v2 configuration, validation, and response models
2. **Path Operations**: HTTP methods, status codes, query parameters, and request/response handling
3. **Middleware**: CORS, compression, authentication, logging, and custom middleware
4. **Error Handling**: Global and route-level exception handling
5. **Documentation**: OpenAPI configuration and route documentation
6. **Performance**: Dependency injection, background tasks, and caching
7. **Security**: Authentication, authorization, and input validation
8. **Testing**: Unit and integration test patterns
9. **Deployment**: Production configuration and Docker setup
10. **Monitoring**: Logging and health checks

Following these best practices ensures your FastAPI application is maintainable, scalable, secure, and performant. 