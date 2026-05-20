# FastAPI Implementation Documentation

## Overview

The FastAPI implementation for the AI Video System follows modern best practices including Pydantic v2, SQLAlchemy 2.0, async database libraries, and lifespan context managers for startup/shutdown management.

## Key Features

### 🚀 **Modern FastAPI Best Practices**
- Lifespan context managers instead of `app.on_event()`
- Pydantic v2 models for validation and serialization
- SQLAlchemy 2.0 async ORM
- Async database libraries (asyncpg, aiomysql)
- Functional components and declarative routes
- Clear return type annotations

### 🎯 **Comprehensive Error Handling**
- Custom exception handlers
- Pydantic validation error handling
- Request/response middleware
- Error context tracking
- Structured error responses

### 💾 **Async Database Integration**
- SQLAlchemy 2.0 async session management
- Connection pooling and health checks
- Transaction management
- Background task processing

### 📊 **Performance Optimization**
- Request tracking middleware
- Active request counting
- Performance metrics collection
- GPU utilization monitoring

## Architecture

### Core Components

1. **`dependencies.py`** - Dependency injection and Pydantic models
2. **`routes.py`** - Declarative route definitions
3. **`main.py`** - FastAPI application with lifespan
4. **Database Models** - SQLAlchemy 2.0 async models
5. **Error Handling** - Comprehensive error management

### File Structure

```
api/
├── dependencies.py      # Dependencies and Pydantic models
├── routes.py           # Route definitions
├── main.py            # FastAPI application
└── FASTAPI_DOCS.md    # This documentation
```

## Pydantic v2 Models

### VideoRequest Model

```python
class VideoRequest(BaseModel):
    """Video generation request model."""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid"
    )
    
    prompt: str = Field(..., min_length=1, max_length=1000, description="Video generation prompt")
    num_steps: int = Field(default=20, ge=1, le=100, description="Number of inference steps")
    quality: str = Field(default="medium", pattern="^(low|medium|high)$", description="Video quality")
    width: int = Field(default=512, ge=256, le=1024, description="Video width")
    height: int = Field(default=512, ge=256, le=1024, description="Video height")
    seed: Optional[int] = Field(default=None, ge=0, le=2**32-1, description="Random seed")
    
    def model_post_init(self, __context: Any) -> None:
        """Post-initialization validation."""
        if self.width % 8 != 0 or self.height % 8 != 0:
            raise ValueError("Width and height must be divisible by 8")
```

### VideoResponse Model

```python
class VideoResponse(BaseModel):
    """Video generation response model."""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True
    )
    
    video_id: str = Field(..., description="Unique video identifier")
    status: str = Field(..., pattern="^(pending|processing|completed|failed)$", description="Processing status")
    video_url: Optional[str] = Field(default=None, description="Video download URL")
    thumbnail_url: Optional[str] = Field(default=None, description="Thumbnail URL")
    processing_time: Optional[float] = Field(default=None, ge=0, description="Processing time in seconds")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
```

## SQLAlchemy 2.0 Models

### Base and VideoRecord

```python
class Base(DeclarativeBase):
    """SQLAlchemy 2.0 declarative base."""
    pass


class VideoRecord(Base):
    """Video record model."""
    
    __tablename__ = "videos"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    created_at: Mapped[DateTime] = mapped_column(DateTime, nullable=False)
    completed_at: Mapped[Optional[DateTime]] = mapped_column(DateTime, nullable=True)
    processing_time: Mapped[Optional[float]] = mapped_column(Integer, nullable=True)
    video_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    thumbnail_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    metadata: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON string
```

## Dependency Injection

### Database Session Dependency

```python
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Get database session dependency."""
    if app_state.db_session_maker is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not initialized"
        )
    
    async with app_state.db_session_maker() as session:
        try:
            yield session
        except Exception as e:
            await session.rollback()
            raise
        finally:
            await session.close()
```

### Performance Optimizer Dependency

```python
async def get_performance_optimizer() -> PerformanceOptimizer:
    """Get performance optimizer dependency."""
    if app_state.performance_optimizer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Performance optimizer not initialized"
        )
    return app_state.performance_optimizer
```

## Declarative Route Definitions

### Video Generation Route

```python
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
```

## Lifespan Context Manager

### Application Lifespan

```python
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan context manager."""
    # Startup
    logger.info("Starting AI Video System...")
    app_state.startup_time = asyncio.get_event_loop().time()
    
    try:
        # Initialize database
        await initialize_database()
        
        # Initialize performance optimizer
        await initialize_performance_optimizer()
        
        logger.info("AI Video System started successfully")
        yield
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    finally:
        # Shutdown
        logger.info("Shutting down AI Video System...")
        await cleanup_resources()
        logger.info("AI Video System shutdown completed")
```

### Database Initialization

```python
async def initialize_database() -> None:
    """Initialize database connection."""
    try:
        # Parse database URL
        db_url = "postgresql+asyncpg://user:password@localhost/ai_video_db"
        
        # Create async engine
        engine = create_async_engine(
            db_url,
            pool_size=10,
            max_overflow=20,
            echo=False,
            pool_pre_ping=True
        )
        
        # Create session maker
        app_state.db_session_maker = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        # Test connection
        async with app_state.db_session_maker() as session:
            await session.execute(text("SELECT 1"))
        
        logger.info("Database initialized successfully")
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise
```

## Functional Components

### Validation Functions

```python
def validate_video_request(request: VideoRequest) -> VideoRequest:
    """Validate video request with business logic."""
    # Early return for invalid quality/step combinations
    if request.quality == "high" and request.num_steps < 50:
        raise ValueError("High quality requires at least 50 steps")
    
    if request.quality == "low" and request.num_steps > 30:
        raise ValueError("Low quality should use 30 steps or fewer")
    
    # Validate aspect ratio
    aspect_ratio = request.width / request.height
    if aspect_ratio < 0.5 or aspect_ratio > 2.0:
        raise ValueError("Aspect ratio must be between 0.5 and 2.0")
    
    return request


def generate_video_id() -> str:
    """Generate unique video ID."""
    import uuid
    return str(uuid.uuid4())


def format_processing_time(seconds: float) -> str:
    """Format processing time for display."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"
```

### Database Operations

```python
async def create_video_record(
    session: AsyncSession,
    video_id: str,
    prompt: str,
    metadata: Dict[str, Any]
) -> VideoRecord:
    """Create video record in database."""
    from datetime import datetime
    
    video_record = VideoRecord(
        id=video_id,
        prompt=prompt,
        status="pending",
        created_at=datetime.utcnow(),
        metadata=str(metadata)
    )
    
    session.add(video_record)
    await session.commit()
    await session.refresh(video_record)
    
    return video_record


async def update_video_status(
    session: AsyncSession,
    video_id: str,
    status: str,
    video_path: Optional[str] = None,
    thumbnail_path: Optional[str] = None,
    error_message: Optional[str] = None,
    processing_time: Optional[float] = None
) -> VideoRecord:
    """Update video status in database."""
    from datetime import datetime
    
    result = await session.execute(
        text("""
            UPDATE videos 
            SET status = :status, 
                completed_at = :completed_at,
                video_path = :video_path,
                thumbnail_path = :thumbnail_path,
                error_message = :error_message,
                processing_time = :processing_time
            WHERE id = :video_id
        """),
        {
            "status": status,
            "completed_at": datetime.utcnow() if status in ["completed", "failed"] else None,
            "video_path": video_path,
            "thumbnail_path": thumbnail_path,
            "error_message": error_message,
            "processing_time": processing_time,
            "video_id": video_id
        }
    )
    
    await session.commit()
    
    # Fetch updated record
    result = await session.execute(
        text("SELECT * FROM videos WHERE id = :video_id"),
        {"video_id": video_id}
    )
    return result.fetchone()
```

## Error Handling

### Exception Handlers

```python
@app.exception_handler(AIVideoError)
async def ai_video_error_handler(request: Request, exc: AIVideoError):
    """Handle AI Video system errors."""
    error_context = ErrorContext(
        operation=request.url.path,
        user_id=request.headers.get("X-User-ID"),
        request_id=request.headers.get("X-Request-ID")
    )
    
    error = error_handler.handle_error(exc, error_context)
    
    return JSONResponse(
        status_code=400,
        content={
            "error": error.message,
            "error_code": error.error_code or "AI_VIDEO_ERROR",
            "details": error.details,
            "timestamp": asyncio.get_event_loop().time()
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    """Handle Pydantic validation errors."""
    error_context = ErrorContext(
        operation=request.url.path,
        user_id=request.headers.get("X-User-ID"),
        request_id=request.headers.get("X-Request-ID")
    )
    
    logger.error(f"Validation Error: {exc.errors()}", extra={"context": error_context.to_dict()})
    
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation error",
            "error_code": "VALIDATION_ERROR",
            "details": {"errors": exc.errors()},
            "timestamp": asyncio.get_event_loop().time()
        }
    )
```

### Request Middleware

```python
@app.middleware("http")
async def request_middleware(request: Request, call_next):
    """Middleware for request tracking and error handling."""
    # Increment active requests
    app_state.increment_requests()
    
    # Add request ID to headers if not present
    if "X-Request-ID" not in request.headers:
        import uuid
        request.headers.__dict__["_list"].append(
            (b"x-request-id", str(uuid.uuid4()).encode())
        )
    
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        # Handle unexpected errors
        error_context = ErrorContext(
            operation=request.url.path,
            user_id=request.headers.get("X-User-ID"),
            request_id=request.headers.get("X-Request-ID")
        )
        
        error = error_handler.handle_error(e, error_context)
        logger.error(f"Request failed: {error.message}")
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "error_code": "INTERNAL_ERROR",
                "details": {"type": type(e).__name__},
                "timestamp": asyncio.get_event_loop().time()
            }
        )
    finally:
        # Decrement active requests
        app_state.decrement_requests()
```

## Background Tasks

### Video Generation Background Task

```python
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
```

## API Endpoints

### Video Endpoints

- `POST /api/v1/videos/generate` - Generate video from prompt
- `GET /api/v1/videos/{video_id}` - Get video status
- `GET /api/v1/videos/{video_id}/download` - Download video
- `GET /api/v1/videos/{video_id}/thumbnail` - Get video thumbnail
- `GET /api/v1/videos/` - List videos with pagination

### System Endpoints

- `GET /api/v1/system/status` - Get system status
- `GET /api/v1/system/health` - Health check
- `GET /api/v1/system/metrics` - Get system metrics

## Usage Examples

### Generate Video

```bash
curl -X POST "http://localhost:8000/api/v1/videos/generate" \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "A beautiful sunset over mountains",
       "num_steps": 30,
       "quality": "high",
       "width": 512,
       "height": 512
     }'
```

### Get Video Status

```bash
curl "http://localhost:8000/api/v1/videos/{video_id}"
```

### Get System Status

```bash
curl "http://localhost:8000/api/v1/system/status"
```

## Best Practices Followed

### 1. **Lifespan Context Managers**
- ✅ Use `lifespan` context manager instead of `app.on_event()`
- ✅ Proper startup and shutdown handling
- ✅ Resource cleanup in finally block

### 2. **Functional Components**
- ✅ Plain functions for validation and utilities
- ✅ Pydantic models for input validation and response schemas
- ✅ Clear separation of concerns

### 3. **Declarative Routes**
- ✅ Clear return type annotations
- ✅ Comprehensive docstrings
- ✅ Proper error handling with early returns

### 4. **Async/Await Patterns**
- ✅ `async def` for asynchronous operations
- ✅ `def` for synchronous operations
- ✅ Proper async database operations

### 5. **Error Handling**
- ✅ Custom exception handlers
- ✅ Structured error responses
- ✅ Error context tracking
- ✅ Request/response middleware

### 6. **Database Integration**
- ✅ SQLAlchemy 2.0 async ORM
- ✅ Connection pooling
- ✅ Transaction management
- ✅ Health checks

## Testing

### Unit Tests

```python
import pytest
from fastapi.testclient import TestClient
from .main import app

client = TestClient(app)

def test_generate_video():
    """Test video generation endpoint."""
    response = client.post(
        "/api/v1/videos/generate",
        json={
            "prompt": "Test video",
            "num_steps": 20,
            "quality": "medium"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "video_id" in data
    assert data["status"] == "pending"

def test_validation_error():
    """Test validation error handling."""
    response = client.post(
        "/api/v1/videos/generate",
        json={
            "prompt": "",  # Invalid empty prompt
            "num_steps": 200  # Invalid number of steps
        }
    )
    assert response.status_code == 422
```

### Integration Tests

```python
async def test_database_operations():
    """Test database operations."""
    async with get_db_session() as session:
        video_id = generate_video_id()
        video_record = await create_video_record(
            session, video_id, "Test prompt", {}
        )
        assert video_record.id == video_id
        assert video_record.status == "pending"
```

## Performance Considerations

### 1. **Connection Pooling**
- Configure appropriate pool sizes
- Enable connection health checks
- Monitor connection usage

### 2. **Background Tasks**
- Use FastAPI background tasks for long-running operations
- Implement proper error handling in background tasks
- Monitor task completion rates

### 3. **Caching**
- Implement response caching where appropriate
- Use Redis or similar for distributed caching
- Cache expensive database queries

### 4. **Monitoring**
- Track request/response times
- Monitor database connection usage
- Log performance metrics

## Deployment

### Docker Configuration

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Variables

```bash
# Database
DATABASE_URL=postgresql+asyncpg://user:password@localhost/ai_video_db
DATABASE_POOL_SIZE=10
DATABASE_MAX_OVERFLOW=20

# Performance
ENABLE_PROFILING=true
CACHE_ENABLED=true
MAX_CONCURRENT_TASKS=4

# Logging
LOG_LEVEL=INFO
```

## Future Enhancements

- **GraphQL Support**: Add GraphQL endpoints
- **WebSocket Support**: Real-time video generation updates
- **Rate Limiting**: Implement request rate limiting
- **Authentication**: Add JWT authentication
- **File Upload**: Support for video file uploads
- **Batch Processing**: Batch video generation endpoints

## Contributing

When contributing to the FastAPI implementation:

1. Follow FastAPI best practices
2. Use Pydantic v2 models
3. Implement proper error handling
4. Add comprehensive tests
5. Update documentation
6. Use type hints throughout
7. Follow async/await patterns

## License

This implementation is part of the AI Video System and follows the same licensing terms. 