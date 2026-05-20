# 🚀 FASTAPI BEST PRACTICES GUIDE - AI VIDEO SYSTEM

## 📋 Table of Contents

1. [Overview](#overview)
2. [Data Models (Pydantic)](#data-models-pydantic)
3. [Path Operations](#path-operations)
4. [Middleware](#middleware)
5. [Dependency Injection](#dependency-injection)
6. [Error Handling](#error-handling)
7. [Performance Optimization](#performance-optimization)
8. [Security](#security)
9. [Testing](#testing)
10. [Deployment](#deployment)

## 🎯 Overview

This guide demonstrates FastAPI best practices for building high-performance, maintainable APIs following official FastAPI documentation standards. The implementation focuses on:

- **Data Models**: Proper Pydantic validation and serialization
- **Path Operations**: RESTful API design with correct HTTP methods
- **Middleware**: Performance monitoring, error handling, and security
- **Dependency Injection**: Clean separation of concerns
- **Error Handling**: Comprehensive error responses and logging

## 📊 Data Models (Pydantic)

### Core Principles

1. **Use Pydantic for all data validation**
2. **Define clear field constraints and descriptions**
3. **Use enums for constrained values**
4. **Implement computed fields for derived data**
5. **Add custom validators for business logic**

### Example: VideoData Model

```python
class VideoData(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
        use_enum_values=True,
        json_encoders={
            datetime: lambda v: v.isoformat(),
            FilePath: str
        }
    )
    
    video_id: str = Field(..., min_length=1, max_length=50, description="Video identifier")
    title: str = Field(..., max_length=200, description="Video title")
    duration: float = Field(..., ge=0, le=3600, description="Duration in seconds")
    quality: VideoQuality = Field(default=VideoQuality.MEDIUM, description="Video quality")
    
    @computed_field
    @property
    def duration_minutes(self) -> float:
        return self.duration / 60
    
    @validator('title')
    @classmethod
    def validate_title(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError('Title cannot be empty')
        return v.strip()
```

### Best Practices

- **Field Descriptions**: Always provide clear descriptions for API documentation
- **Validation**: Use appropriate validators (min_length, max_length, ge, le)
- **Defaults**: Provide sensible defaults where appropriate
- **Computed Fields**: Use `@computed_field` for derived data
- **Custom Validators**: Implement business logic validation

## 🔗 Path Operations

### RESTful Design Principles

1. **Use appropriate HTTP methods**
2. **Return correct status codes**
3. **Provide comprehensive documentation**
4. **Implement proper error handling**
5. **Use path parameters for resource identification**

### Example: Video Processing Endpoint

```python
@video_router.post(
    "/process",
    response_model=VideoResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Process a video with AI enhancement",
    description="Process a single video using AI algorithms for enhancement and optimization.",
    response_description="Video processing result with status and metadata",
    responses={
        201: {"description": "Video processing started successfully"},
        400: {"description": "Bad request", "model": ErrorResponse},
        422: {"description": "Validation error", "model": ErrorResponse},
        500: {"description": "Internal server error", "model": ErrorResponse}
    }
)
async def process_video(
    video_data: VideoData,
    background_tasks: BackgroundTasks,
    video_service: VideoService = Depends(get_video_service),
    current_user: dict = Depends(get_current_user)
) -> VideoResponse:
    """
    Process a video with AI enhancement.
    
    - **video_data**: Video information and processing parameters
    - **background_tasks**: FastAPI background tasks for async processing
    
    Returns:
    - **VideoResponse**: Processing result with status and metadata
    """
    try:
        # Validate input
        if not video_data.title or not video_data.title.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Video title is required"
            )
        
        # Process video
        result = await video_service.process_video(video_data)
        
        # Add background task for cleanup
        background_tasks.add_task(cleanup_temp_files, video_data.video_id)
        
        return result
        
    except Exception as e:
        logger.error(f"Video processing error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during video processing"
        )
```

### HTTP Methods Usage

| Method | Usage | Status Code |
|--------|-------|-------------|
| GET | Retrieve resources | 200 OK |
| POST | Create new resources | 201 Created |
| PUT | Update entire resource | 200 OK |
| PATCH | Partial update | 200 OK |
| DELETE | Remove resource | 204 No Content |

### Query Parameters

```python
@video_router.get("/")
async def list_videos(
    skip: int = Query(0, ge=0, description="Number of videos to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of videos to return"),
    quality: Optional[VideoQuality] = Query(None, description="Filter by video quality"),
    video_service: VideoService = Depends(get_video_service)
) -> VideoListResponse:
    return await video_service.list_videos(skip=skip, limit=limit, quality=quality)
```

### Path Parameters

```python
@video_router.get("/{video_id}")
async def get_video(
    video_id: str = Path(..., description="Unique video identifier", min_length=1),
    video_service: VideoService = Depends(get_video_service)
) -> VideoResponse:
    video = await video_service.get_video(video_id)
    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video with ID {video_id} not found"
        )
    return video
```

## 🛡️ Middleware

### Performance Monitoring Middleware

```python
@app.middleware("http")
async def performance_middleware(request: Request, call_next):
    start_time = time.time()
    
    # Log request
    logger.info(
        f"Request: {request.method} {request.url.path} "
        f"from {request.client.host if request.client else 'unknown'}"
    )
    
    # Process request
    response = await call_next(request)
    
    # Calculate processing time
    process_time = time.time() - start_time
    
    # Add performance headers
    response.headers["X-Process-Time"] = str(process_time)
    response.headers["X-Request-ID"] = request.headers.get("X-Request-ID", "unknown")
    
    # Log response
    logger.info(
        f"Response: {response.status_code} "
        f"took {process_time:.4f}s "
        f"for {request.method} {request.url.path}"
    )
    
    # Log slow requests
    if process_time > 1.0:
        logger.warning(
            f"Slow request: {request.method} {request.url.path} "
            f"took {process_time:.4f}s"
        )
    
    return response
```

### Error Handling Middleware

```python
@app.middleware("http")
async def error_handling_middleware(request: Request, call_next):
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        # Log error
        logger.error(
            f"Unhandled error in {request.method} {request.url.path}: {str(e)}",
            exc_info=True
        )
        
        # Return error response
        error_response = {
            "error_code": "INTERNAL_ERROR",
            "error_type": "server_error",
            "message": "An unexpected error occurred",
            "timestamp": datetime.now().isoformat(),
            "request_id": request.headers.get("X-Request-ID", "unknown")
        }
        
        return Response(
            content=json.dumps(error_response),
            status_code=500,
            media_type="application/json"
        )
```

### Security Middleware

```python
# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "yourdomain.com", "*.yourdomain.com"]
)
```

## 🔧 Dependency Injection

### Service Dependencies

```python
class VideoService:
    def __init__(self):
        self.processing_queue: Dict[str, VideoData] = {}
        self.results_cache: Dict[str, VideoResponse] = {}
    
    async def process_video(self, video_data: VideoData) -> VideoResponse:
        # Implementation
        pass

# Global service instance
video_service = VideoService()

# Dependency
async def get_video_service() -> VideoService:
    return video_service
```

### Authentication Dependencies

```python
async def get_current_user(request: Request):
    # Simulate authentication
    auth_header = request.headers.get("authorization")
    if not auth_header:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header required"
        )
    return {"user_id": "user_123", "username": "test_user"}
```

### Using Dependencies in Endpoints

```python
@video_router.post("/process")
async def process_video(
    video_data: VideoData,
    background_tasks: BackgroundTasks,
    video_service: VideoService = Depends(get_video_service),
    current_user: dict = Depends(get_current_user)
) -> VideoResponse:
    # Implementation
    pass
```

## ❌ Error Handling

### HTTP Exceptions

```python
# 400 Bad Request
raise HTTPException(
    status_code=status.HTTP_400_BAD_REQUEST,
    detail="Video title is required"
)

# 404 Not Found
raise HTTPException(
    status_code=status.HTTP_404_NOT_FOUND,
    detail=f"Video with ID {video_id} not found"
)

# 500 Internal Server Error
raise HTTPException(
    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    detail="Internal server error during video processing"
)
```

### Error Response Model

```python
class ErrorResponse(BaseModel):
    error_code: str = Field(..., description="Error code")
    error_type: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request identifier for tracking")
```

### Try-Catch Pattern

```python
try:
    result = await video_service.process_video(video_data)
    return result
except ValidationError as e:
    logger.error(f"Validation error: {e}")
    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail="Invalid input data"
    )
except Exception as e:
    logger.error(f"Video processing error: {e}")
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Internal server error during video processing"
    )
```

## ⚡ Performance Optimization

### Background Tasks

```python
async def cleanup_temp_files(video_id: str):
    """Background task to cleanup temporary files."""
    try:
        await asyncio.sleep(1)  # Simulate cleanup
        logger.info(f"Cleaned up temporary files for video {video_id}")
    except Exception as e:
        logger.error(f"Error cleaning up files for video {video_id}: {e}")

@video_router.post("/process")
async def process_video(
    video_data: VideoData,
    background_tasks: BackgroundTasks,
    video_service: VideoService = Depends(get_video_service)
) -> VideoResponse:
    result = await video_service.process_video(video_data)
    
    # Add background task for cleanup
    background_tasks.add_task(cleanup_temp_files, video_data.video_id)
    
    return result
```

### Async Operations

```python
async def process_batch(self, batch_request: BatchVideoRequest) -> BatchVideoResponse:
    # Process videos concurrently
    tasks = [self.process_video(video) for video in batch_request.videos]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    completed = sum(1 for r in results if isinstance(r, VideoResponse) and r.is_completed)
    failed = sum(1 for r in results if isinstance(r, Exception))
    
    return BatchVideoResponse(
        # ... response data
    )
```

### Response Headers

```python
# Add performance headers
response.headers["X-Process-Time"] = str(process_time)
response.headers["X-Request-ID"] = request.headers.get("X-Request-ID", "unknown")
response.headers["Cache-Control"] = "public, max-age=300"  # 5 minutes cache
```

## 🔒 Security

### CORS Configuration

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)
```

### Trusted Hosts

```python
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "yourdomain.com", "*.yourdomain.com"]
)
```

### Request Validation

```python
@app.middleware("http")
async def validation_middleware(request: Request, call_next):
    # Validate request size
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > 10 * 1024 * 1024:  # 10MB limit
        return Response(
            content=json.dumps({
                "error_code": "PAYLOAD_TOO_LARGE",
                "error_type": "validation_error",
                "message": "Request payload too large",
                "timestamp": datetime.now().isoformat()
            }),
            status_code=413,
            media_type="application/json"
        )
    
    # Validate content type for POST/PUT requests
    if request.method in ["POST", "PUT", "PATCH"]:
        content_type = request.headers.get("content-type", "")
        if not content_type.startswith("application/json"):
            return Response(
                content=json.dumps({
                    "error_code": "INVALID_CONTENT_TYPE",
                    "error_type": "validation_error",
                    "message": "Content-Type must be application/json",
                    "timestamp": datetime.now().isoformat()
                }),
                status_code=415,
                media_type="application/json"
            )
    
    return await call_next(request)
```

## 🧪 Testing

### Unit Testing Example

```python
import pytest
from fastapi.testclient import TestClient
from fastapi_best_practices import app

client = TestClient(app)

def test_process_video():
    video_data = {
        "video_id": "test_123",
        "title": "Test Video",
        "duration": 120.0,
        "quality": "medium"
    }
    
    response = client.post("/api/v1/videos/process", json=video_data)
    assert response.status_code == 201
    assert response.json()["video_id"] == "test_123"
    assert response.json()["status"] == "completed"

def test_get_video_not_found():
    response = client.get("/api/v1/videos/nonexistent")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]
```

### Integration Testing

```python
@pytest.mark.asyncio
async def test_video_service():
    service = VideoService()
    
    video_data = VideoData(
        video_id="test_123",
        title="Test Video",
        duration=120.0
    )
    
    result = await service.process_video(video_data)
    assert result.video_id == "test_123"
    assert result.status == VideoStatus.COMPLETED
```

## 🚀 Deployment

### Application Factory

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 AI Video System starting up...")
    # Initialize resources (database, cache, etc.)
    yield
    # Shutdown
    logger.info("🛑 AI Video System shutting down...")
    # Cleanup resources

# Create FastAPI app
app = FastAPI(
    title="AI Video System",
    description="High-performance video processing API with AI enhancement",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)
```

### Production Configuration

```python
# main.py
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        workers=4,
        log_level="info"
    )
```

### Docker Configuration

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "fastapi_best_practices:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 📚 Additional Resources

### FastAPI Documentation
- [FastAPI Official Documentation](https://fastapi.tiangolo.com/)
- [Pydantic Documentation](https://pydantic-docs.helpmanual.io/)
- [FastAPI Best Practices](https://fastapi.tiangolo.com/tutorial/best-practices/)

### Related Files
- `fastapi_best_practices.py` - Complete implementation
- `pydantic_validation.py` - Pydantic validation examples
- `error_handling.py` - Error handling patterns
- `performance_optimization.py` - Performance optimization techniques

### Key Takeaways

1. **Always use Pydantic for data validation**
2. **Follow RESTful conventions for API design**
3. **Implement comprehensive error handling**
4. **Use middleware for cross-cutting concerns**
5. **Leverage dependency injection for clean architecture**
6. **Monitor performance with appropriate metrics**
7. **Implement security best practices**
8. **Write comprehensive tests**
9. **Document your API thoroughly**
10. **Plan for production deployment**

This implementation provides a solid foundation for building scalable, maintainable FastAPI applications following industry best practices and official FastAPI documentation standards. 