# Declarative Route Definitions Guide

A comprehensive guide for implementing declarative route definitions with clear return type annotations in FastAPI applications.

## 🚀 Overview

This guide covers:
- **Declarative Route Patterns**: Clear, self-documenting API endpoints
- **Return Type Annotations**: Comprehensive type safety for responses
- **Response Models**: Structured and consistent API responses
- **Parameter Validation**: Type-safe request parameter handling
- **Error Handling**: Consistent error response patterns
- **Documentation**: Auto-generated API documentation
- **Best Practices**: Modern FastAPI patterns and conventions

## 📋 Table of Contents

1. [Declarative Route Principles](#declarative-route-principles)
2. [Return Type Annotations](#return-type-annotations)
3. [Response Models](#response-models)
4. [Parameter Types](#parameter-types)
5. [Error Handling](#error-handling)
6. [Route Documentation](#route-documentation)
7. [Advanced Patterns](#advanced-patterns)
8. [Testing Declarative Routes](#testing-declarative-routes)
9. [Best Practices](#best-practices)

## 🎯 Declarative Route Principles

### Clear Intent and Purpose

```python
@router.post(
    "/users",
    response_model=SuccessResponse[UserResponse],
    status_code=status.HTTP_201_CREATED,
    summary="Create a new user",
    description="Create a new user account with validation",
    response_description="User created successfully"
)
async def create_user(
    user_data: UserCreate = Body(..., description="User creation data"),
    db: Annotated[Any, Depends(get_db)] = None
) -> SuccessResponse[UserResponse]:
    """
    Create a new user account.
    
    Args:
        user_data: User creation data with validation
        db: Database session dependency
        
    Returns:
        SuccessResponse with created user data
        
    Raises:
        HTTPException: If validation fails or user creation fails
    """
    # Implementation here
    pass
```

### Consistent Response Structure

```python
# Good: Consistent response structure
@router.get("/users/{user_id}")
async def get_user(user_id: int) -> SuccessResponse[UserResponse]:
    """Get user by ID with consistent response structure."""
    return SuccessResponse(
        message="User retrieved successfully",
        data=user_data
    )

# Bad: Inconsistent response structure
@router.get("/users/{user_id}")
async def get_user(user_id: int):
    """Get user by ID with inconsistent response."""
    return {"user": user_data, "status": "success"}  # Inconsistent
```

## 📝 Return Type Annotations

### Basic Return Types

```python
from typing import List, Optional, Dict, Any, Union

# Simple return types
@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    return {"status": "healthy"}

# List return types
@router.get("/users")
async def get_users() -> List[UserResponse]:
    """Get all users."""
    return [UserResponse(**user.dict()) for user in users]

# Optional return types
@router.get("/users/{user_id}")
async def get_user(user_id: int) -> Optional[UserResponse]:
    """Get user by ID, may not exist."""
    return UserResponse(**user.dict()) if user else None
```

### Generic Response Types

```python
from typing import TypeVar, Generic

T = TypeVar('T')

class SuccessResponse(BaseModel, Generic[T]):
    """Generic success response."""
    success: Literal[True] = True
    message: str
    data: T
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ErrorResponse(BaseModel):
    """Error response model."""
    success: Literal[False] = False
    message: str
    error_code: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    details: Optional[Dict[str, Any]] = None

# Usage with generic types
@router.post("/users")
async def create_user(user_data: UserCreate) -> SuccessResponse[UserResponse]:
    """Create user with generic response type."""
    return SuccessResponse(
        message="User created successfully",
        data=UserResponse(**created_user.dict())
    )

@router.get("/users")
async def get_users() -> SuccessResponse[List[UserResponse]]:
    """Get users with generic list response."""
    return SuccessResponse(
        message="Users retrieved successfully",
        data=[UserResponse(**user.dict()) for user in users]
    )
```

### Union Types for Multiple Response Types

```python
from typing import Union

@router.get("/users/{user_id}")
async def get_user(user_id: int) -> Union[SuccessResponse[UserResponse], ErrorResponse]:
    """Get user with multiple possible response types."""
    try:
        user = await get_user_by_id(user_id)
        if user:
            return SuccessResponse(
                message="User retrieved successfully",
                data=UserResponse(**user.dict())
            )
        else:
            return ErrorResponse(
                message="User not found",
                error_code="NOT_FOUND"
            )
    except Exception as e:
        return ErrorResponse(
            message="Failed to retrieve user",
            error_code="INTERNAL_ERROR"
        )
```

## 📊 Response Models

### Standard Response Models

```python
from pydantic import BaseModel, Field
from datetime import datetime, timezone
from typing import Literal, Any

class SuccessResponse(BaseModel):
    """Standard success response model."""
    success: Literal[True] = True
    message: str
    data: Any
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ErrorResponse(BaseModel):
    """Standard error response model."""
    success: Literal[False] = False
    message: str
    error_code: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    details: Optional[Dict[str, Any]] = None

class PaginatedResponse(BaseModel, Generic[T]):
    """Paginated response model."""
    success: Literal[True] = True
    message: str
    data: List[T]
    pagination: Dict[str, Any] = Field(
        description="Pagination metadata",
        example={
            "page": 1,
            "per_page": 10,
            "total": 100,
            "total_pages": 10,
            "has_next": True,
            "has_prev": False
        }
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
```

### Specialized Response Models

```python
class HealthCheckResponse(BaseModel):
    """Health check response model."""
    status: Literal["healthy", "unhealthy"] = Field(description="Service status")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    version: str = Field(description="API version")
    uptime: float = Field(description="Service uptime in seconds")
    services: Dict[str, str] = Field(description="Service health status")

class VideoProcessingResponse(BaseModel):
    """Video processing response model."""
    video_id: str = Field(description="Video identifier")
    status: VideoStatus = Field(description="Processing status")
    progress: Optional[float] = Field(None, ge=0, le=100, description="Processing progress")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    message: str = Field(description="Status message")

class AnalyticsResponse(BaseModel):
    """Analytics response model."""
    period: Dict[str, datetime] = Field(description="Analysis period")
    metrics: Dict[str, Any] = Field(description="Calculated metrics")
    total_videos: int = Field(description="Total videos in period")
    total_users: int = Field(description="Total active users")
    average_processing_time: float = Field(description="Average processing time")
    success_rate: float = Field(ge=0, le=100, description="Success rate percentage")
```

## 🔧 Parameter Types

### Query Parameter Models

```python
class PaginationParams(BaseModel):
    """Pagination parameters."""
    page: int = Query(1, ge=1, description="Page number")
    per_page: int = Query(10, ge=1, le=100, description="Items per page")
    sort_by: Optional[str] = Query(None, description="Sort field")
    sort_order: Literal["asc", "desc"] = Query("desc", description="Sort order")

class VideoFilterParams(BaseModel):
    """Video filter parameters."""
    status: Optional[VideoStatus] = Query(None, description="Filter by status")
    quality: Optional[VideoQuality] = Query(None, description="Filter by quality")
    created_after: Optional[datetime] = Query(None, description="Created after date")
    created_before: Optional[datetime] = Query(None, description="Created before date")
    search: Optional[str] = Query(None, description="Search in script content")

class AnalyticsParams(BaseModel):
    """Analytics parameters."""
    start_date: datetime = Query(..., description="Start date for analytics")
    end_date: datetime = Query(..., description="End date for analytics")
    group_by: Optional[str] = Query(None, description="Grouping field")
    metrics: List[str] = Query(default_factory=list, description="Metrics to calculate")
```

### Path Parameter Types

```python
@router.get("/users/{user_id}")
async def get_user(
    user_id: int = Path(..., gt=0, description="User ID")
) -> SuccessResponse[UserResponse]:
    """Get user by ID with validated path parameter."""
    pass

@router.put("/videos/{video_id}/status")
async def update_video_status(
    video_id: int = Path(..., gt=0, description="Video ID"),
    status_update: Dict[str, Any] = Body(..., description="Status update data")
) -> SuccessResponse[VideoResponse]:
    """Update video status with validated path and body parameters."""
    pass
```

### Dependency Injection Types

```python
from fastapi import Depends
from typing import Annotated

@router.post("/users")
async def create_user(
    user_data: UserCreate = Body(..., description="User creation data"),
    db: Annotated[Any, Depends(get_db)] = None,
    current_user: Annotated[Any, Depends(get_current_active_user)] = None
) -> SuccessResponse[UserResponse]:
    """Create user with dependency injection."""
    pass

@router.get("/videos")
async def get_videos(
    pagination: Annotated[PaginationParams, Depends()] = None,
    filters: Annotated[VideoFilterParams, Depends()] = None,
    current_user: Annotated[Any, Depends(get_current_active_user)] = None
) -> PaginatedResponse[VideoSummary]:
    """Get videos with parameter dependencies."""
    pass
```

## 🛡️ Error Handling

### Consistent Error Responses

```python
from fastapi import HTTPException, status

def handle_validation_error(error: str) -> HTTPException:
    """Create consistent validation error response."""
    return HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail={
            "message": error,
            "error_code": "VALIDATION_ERROR"
        }
    )

def handle_not_found_error(resource: str) -> HTTPException:
    """Create consistent not found error response."""
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={
            "message": f"{resource} not found",
            "error_code": "NOT_FOUND"
        }
    )

def handle_unauthorized_error() -> HTTPException:
    """Create consistent unauthorized error response."""
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={
            "message": "Authentication required",
            "error_code": "UNAUTHORIZED"
        }
    )

def handle_forbidden_error() -> HTTPException:
    """Create consistent forbidden error response."""
    return HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail={
            "message": "Access denied",
            "error_code": "FORBIDDEN"
        }
    )
```

### Error Handling in Routes

```python
@router.post("/users")
async def create_user(
    user_data: UserCreate = Body(..., description="User creation data"),
    db: Annotated[Any, Depends(get_db)] = None
) -> SuccessResponse[UserResponse]:
    """
    Create a new user account.
    
    Raises:
        HTTPException: If validation fails or user creation fails
    """
    try:
        # Process user registration using functional pipeline
        user_dict, error = process_user_registration(user_data.dict())
        if error:
            raise handle_validation_error(error)
        
        # Create user in database
        user_repo = get_user_repository(db)
        user = await user_repo.create(**user_dict)
        
        # Transform to response
        response_data = transform_user_to_response(user.to_dict())
        
        return SuccessResponse(
            message="User created successfully",
            data=response_data.dict()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error creating user", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": "Failed to create user",
                "error_code": "INTERNAL_ERROR"
            }
        )
```

## 📚 Route Documentation

### Comprehensive Route Documentation

```python
@router.post(
    "/users",
    response_model=SuccessResponse[UserResponse],
    status_code=status.HTTP_201_CREATED,
    summary="Create a new user",
    description="Create a new user account with comprehensive validation including username, email, and password strength requirements",
    response_description="User created successfully with generated API key",
    tags=["Users"],
    responses={
        201: {
            "description": "User created successfully",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "message": "User created successfully",
                        "data": {
                            "id": 1,
                            "username": "john_doe",
                            "email": "john@example.com",
                            "created_at": "2024-01-01T00:00:00Z"
                        },
                        "timestamp": "2024-01-01T00:00:00Z"
                    }
                }
            }
        },
        422: {
            "description": "Validation error",
            "content": {
                "application/json": {
                    "example": {
                        "success": False,
                        "message": "Username must be at least 3 characters long",
                        "error_code": "VALIDATION_ERROR",
                        "timestamp": "2024-01-01T00:00:00Z"
                    }
                }
            }
        }
    }
)
async def create_user(
    user_data: UserCreate = Body(
        ...,
        description="User creation data with validation",
        example={
            "username": "john_doe",
            "email": "john@example.com",
            "password": "SecurePassword123",
            "confirm_password": "SecurePassword123"
        }
    ),
    db: Annotated[Any, Depends(get_db)] = None
) -> SuccessResponse[UserResponse]:
    """
    Create a new user account with comprehensive validation.
    
    This endpoint creates a new user account with the following validations:
    - Username must be 3-50 characters and contain only letters, numbers, and underscores
    - Email must be a valid email format
    - Password must be at least 8 characters with uppercase, lowercase, and digit
    - Password confirmation must match the password
    
    Args:
        user_data: User creation data with validation
        db: Database session dependency
        
    Returns:
        SuccessResponse with created user data including generated API key
        
    Raises:
        HTTPException: 
            - 422: If validation fails (invalid username, email, or password)
            - 500: If user creation fails due to internal error
            
    Example:
        ```python
        response = await create_user({
            "username": "john_doe",
            "email": "john@example.com",
            "password": "SecurePassword123",
            "confirm_password": "SecurePassword123"
        })
        ```
    """
    # Implementation here
    pass
```

### Router-Level Documentation

```python
router = APIRouter(
    prefix="/api/v1",
    tags=["HeyGen AI API"],
    responses={
        200: {"description": "Success"},
        201: {"description": "Created"},
        400: {"description": "Bad Request"},
        401: {"description": "Unauthorized"},
        403: {"description": "Forbidden"},
        404: {"description": "Not Found"},
        422: {"description": "Validation Error"},
        500: {"description": "Internal Server Error"}
    }
)
```

## 🔄 Advanced Patterns

### Background Tasks

```python
from fastapi import BackgroundTasks

@router.post(
    "/videos",
    response_model=SuccessResponse[VideoResponse],
    status_code=status.HTTP_201_CREATED,
    summary="Create a new video",
    description="Create a new video generation request with background processing"
)
async def create_video(
    video_data: VideoCreate = Body(..., description="Video creation data"),
    background_tasks: BackgroundTasks = None,
    current_user: Annotated[Any, Depends(get_current_active_user)] = None,
    db: Annotated[Any, Depends(get_db)] = None
) -> SuccessResponse[VideoResponse]:
    """
    Create a new video generation request with background processing.
    
    Args:
        video_data: Video creation data with validation
        background_tasks: Background tasks for video processing
        current_user: Current authenticated user
        db: Database session dependency
        
    Returns:
        SuccessResponse with created video data
        
    Raises:
        HTTPException: If validation fails or video creation fails
    """
    try:
        # Process video creation using functional pipeline
        video_dict, error = process_video_creation(video_data.dict(), current_user.id)
        if error:
            raise handle_validation_error(error)
        
        # Create video in database
        video_repo = get_video_repository(db)
        video = await video_repo.create(**video_dict)
        
        # Add background task for video processing
        if background_tasks:
            background_tasks.add_task(process_video_background, video.id, db)
        
        # Transform to response
        response_data = transform_video_to_response(video.to_dict())
        
        return SuccessResponse(
            message="Video creation request submitted successfully",
            data=response_data.dict()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error creating video", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": "Failed to create video",
                "error_code": "INTERNAL_ERROR"
            }
        )

async def process_video_background(video_id: int, db: Any) -> None:
    """
    Background task for video processing.
    
    Args:
        video_id: Video identifier
        db: Database session
    """
    try:
        logger.info("Starting background video processing", video_id=video_id)
        
        # Simulate video processing
        await asyncio.sleep(5)  # Simulate processing time
        
        # Update video status to completed
        video_repo = get_video_repository(db)
        await video_repo.update(video_id, status=VideoStatus.COMPLETED)
        
        logger.info("Background video processing completed", video_id=video_id)
        
    except Exception as e:
        logger.error("Background video processing failed", video_id=video_id, error=str(e))
        
        # Update video status to failed
        try:
            video_repo = get_video_repository(db)
            await video_repo.update(
                video_id, 
                status=VideoStatus.FAILED,
                error_message=str(e)
            )
        except Exception as update_error:
            logger.error("Failed to update video status to failed", video_id=video_id, error=str(update_error))
```

### Streaming Responses

```python
from fastapi.responses import StreamingResponse
import json

@router.get(
    "/videos/{video_id}/stream",
    summary="Stream video processing status",
    description="Stream real-time video processing status updates"
)
async def stream_video_status(
    video_id: int = Path(..., gt=0, description="Video ID"),
    current_user: Annotated[Any, Depends(get_current_active_user)] = None
) -> StreamingResponse:
    """
    Stream video processing status updates.
    
    Args:
        video_id: Video identifier
        current_user: Current authenticated user
        
    Returns:
        StreamingResponse with real-time status updates
    """
    async def generate_status_updates():
        """Generate status update stream."""
        while True:
            # Get current video status
            video = await get_video_by_id(video_id)
            if not video or video.user_id != current_user.id:
                yield f"data: {json.dumps({'error': 'Video not found'})}\n\n"
                break
            
            # Send status update
            status_data = {
                "video_id": video_id,
                "status": video.status,
                "progress": video.progress,
                "message": f"Video {video.status}"
            }
            
            yield f"data: {json.dumps(status_data)}\n\n"
            
            # Check if processing is complete
            if video.status in [VideoStatus.COMPLETED, VideoStatus.FAILED]:
                break
            
            # Wait before next update
            await asyncio.sleep(2)
    
    return StreamingResponse(
        generate_status_updates(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream"
        }
    )
```

### File Upload and Download

```python
from fastapi import UploadFile, File
from fastapi.responses import FileResponse

@router.post(
    "/videos/{video_id}/upload",
    summary="Upload video file",
    description="Upload video file for processing"
)
async def upload_video_file(
    video_id: int = Path(..., gt=0, description="Video ID"),
    file: UploadFile = File(..., description="Video file to upload"),
    current_user: Annotated[Any, Depends(get_current_active_user)] = None
) -> SuccessResponse[Dict[str, str]]:
    """
    Upload video file for processing.
    
    Args:
        video_id: Video identifier
        file: Video file to upload
        current_user: Current authenticated user
        
    Returns:
        SuccessResponse with upload confirmation
    """
    # Validate file type
    if not file.content_type.startswith('video/'):
        raise handle_validation_error("File must be a video")
    
    # Save file
    file_path = f"uploads/video_{video_id}_{file.filename}"
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    
    return SuccessResponse(
        message="Video file uploaded successfully",
        data={"file_path": file_path, "file_size": len(content)}
    )

@router.get(
    "/videos/{video_id}/download",
    summary="Download video file",
    description="Download completed video file"
)
async def download_video_file(
    video_id: int = Path(..., gt=0, description="Video ID"),
    current_user: Annotated[Any, Depends(get_current_active_user)] = None
) -> FileResponse:
    """
    Download completed video file.
    
    Args:
        video_id: Video identifier
        current_user: Current authenticated user
        
    Returns:
        FileResponse with video file
    """
    # Get video and check ownership
    video = await get_video_by_id(video_id)
    if not video or video.user_id != current_user.id:
        raise handle_not_found_error("Video")
    
    if video.status != VideoStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "message": "Video processing not completed",
                "error_code": "PROCESSING_INCOMPLETE"
            }
        )
    
    return FileResponse(
        path=video.file_path,
        filename=f"video_{video_id}.mp4",
        media_type="video/mp4"
    )
```

## 🧪 Testing Declarative Routes

### Unit Testing Routes

```python
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

class TestUserRoutes:
    """Test user route endpoints."""
    
    @pytest.mark.asyncio
    async def test_create_user_success(self, client: TestClient):
        """Test successful user creation."""
        user_data = {
            "username": "john_doe",
            "email": "john@example.com",
            "password": "SecurePassword123",
            "confirm_password": "SecurePassword123"
        }
        
        with patch('app.services.functional_services.process_user_registration') as mock_process:
            mock_process.return_value = ({"username": "john_doe"}, None)
            
            with patch('app.core.database.get_user_repository') as mock_repo:
                mock_user = AsyncMock()
                mock_user.id = 1
                mock_user.username = "john_doe"
                mock_user.to_dict.return_value = {"id": 1, "username": "john_doe"}
                mock_repo.return_value.create.return_value = mock_user
                
                response = client.post("/api/v1/users", json=user_data)
                
                assert response.status_code == 201
                data = response.json()
                assert data["success"] is True
                assert data["message"] == "User created successfully"
                assert data["data"]["username"] == "john_doe"
    
    @pytest.mark.asyncio
    async def test_create_user_validation_error(self, client: TestClient):
        """Test user creation with validation error."""
        user_data = {
            "username": "jo",  # Too short
            "email": "invalid-email",
            "password": "weak",
            "confirm_password": "weak"
        }
        
        with patch('app.services.functional_services.process_user_registration') as mock_process:
            mock_process.return_value = (None, "Username must be at least 3 characters long")
            
            response = client.post("/api/v1/users", json=user_data)
            
            assert response.status_code == 422
            data = response.json()
            assert data["detail"]["message"] == "Username must be at least 3 characters long"
            assert data["detail"]["error_code"] == "VALIDATION_ERROR"
    
    @pytest.mark.asyncio
    async def test_get_user_success(self, client: TestClient):
        """Test successful user retrieval."""
        with patch('app.core.database.get_user_repository') as mock_repo:
            mock_user = AsyncMock()
            mock_user.id = 1
            mock_user.username = "john_doe"
            mock_user.to_dict.return_value = {"id": 1, "username": "john_doe"}
            mock_repo.return_value.get_by_id.return_value = mock_user
            
            response = client.get("/api/v1/users/1")
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["data"]["username"] == "john_doe"
    
    @pytest.mark.asyncio
    async def test_get_user_not_found(self, client: TestClient):
        """Test user retrieval when user not found."""
        with patch('app.core.database.get_user_repository') as mock_repo:
            mock_repo.return_value.get_by_id.return_value = None
            
            response = client.get("/api/v1/users/999")
            
            assert response.status_code == 404
            data = response.json()
            assert data["detail"]["message"] == "User not found"
            assert data["detail"]["error_code"] == "NOT_FOUND"
```

### Integration Testing

```python
class TestVideoRoutesIntegration:
    """Integration tests for video routes."""
    
    @pytest.mark.asyncio
    async def test_video_creation_workflow(self, client: TestClient, db_session):
        """Test complete video creation workflow."""
        # Create user first
        user_data = {
            "username": "test_user",
            "email": "test@example.com",
            "password": "SecurePassword123",
            "confirm_password": "SecurePassword123"
        }
        
        user_response = client.post("/api/v1/users", json=user_data)
        assert user_response.status_code == 201
        
        # Get authentication token (simplified)
        auth_headers = {"Authorization": "Bearer test_token"}
        
        # Create video
        video_data = {
            "script": "Hello world, this is a test video.",
            "voice_id": "voice_001",
            "quality": "medium"
        }
        
        video_response = client.post(
            "/api/v1/videos",
            json=video_data,
            headers=auth_headers
        )
        
        assert video_response.status_code == 201
        video_data = video_response.json()
        video_id = video_data["data"]["id"]
        
        # Get video status
        status_response = client.get(
            f"/api/v1/videos/{video_id}",
            headers=auth_headers
        )
        
        assert status_response.status_code == 200
        status_data = status_response.json()
        assert status_data["data"]["status"] in ["pending", "processing", "completed"]
```

## 🛠️ Best Practices

### Route Organization

```python
# Good: Organized by resource
router = APIRouter(prefix="/api/v1", tags=["Users"])

@router.post("/users")
async def create_user() -> SuccessResponse[UserResponse]:
    pass

@router.get("/users")
async def get_users() -> PaginatedResponse[UserSummary]:
    pass

@router.get("/users/{user_id}")
async def get_user() -> SuccessResponse[UserResponse]:
    pass

@router.put("/users/{user_id}")
async def update_user() -> SuccessResponse[UserResponse]:
    pass

# Bad: Mixed resources
@router.post("/users")
async def create_user():
    pass

@router.post("/videos")  # Different resource in same router
async def create_video():
    pass
```

### Consistent Naming

```python
# Good: Consistent naming conventions
@router.post("/users")  # Create user
@router.get("/users")   # Get users list
@router.get("/users/{user_id}")  # Get specific user
@router.put("/users/{user_id}")  # Update user
@router.delete("/users/{user_id}")  # Delete user

# Bad: Inconsistent naming
@router.post("/users")
@router.get("/get-users")
@router.get("/users/{user_id}")
@router.patch("/users/{user_id}/update")
@router.delete("/remove-user/{user_id}")
```

### Error Handling

```python
# Good: Consistent error handling
@router.post("/users")
async def create_user(user_data: UserCreate) -> SuccessResponse[UserResponse]:
    try:
        # Business logic
        result = await process_user_creation(user_data)
        return SuccessResponse(message="User created", data=result)
    except ValidationError as e:
        raise handle_validation_error(str(e))
    except DatabaseError as e:
        logger.error("Database error", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

# Bad: Inconsistent error handling
@router.post("/users")
async def create_user(user_data: UserCreate):
    if not user_data.username:
        return {"error": "Username required"}  # Inconsistent format
    try:
        result = await process_user_creation(user_data)
        return {"success": True, "data": result}  # Inconsistent format
    except Exception as e:
        return {"error": str(e)}  # No proper error codes
```

### Documentation

```python
# Good: Comprehensive documentation
@router.post(
    "/users",
    response_model=SuccessResponse[UserResponse],
    summary="Create a new user",
    description="Create a new user account with validation",
    response_description="User created successfully"
)
async def create_user(
    user_data: UserCreate = Body(..., description="User creation data")
) -> SuccessResponse[UserResponse]:
    """
    Create a new user account.
    
    This endpoint creates a new user with comprehensive validation.
    
    Args:
        user_data: User creation data
        
    Returns:
        SuccessResponse with created user data
        
    Raises:
        HTTPException: If validation fails
    """
    pass

# Bad: Minimal documentation
@router.post("/users")
async def create_user(user_data: UserCreate):
    """Create user."""
    pass
```

## 📚 Additional Resources

- [FastAPI Route Documentation](https://fastapi.tiangolo.com/tutorial/routing/)
- [Pydantic Response Models](https://fastapi.tiangolo.com/tutorial/response-model/)
- [FastAPI Dependencies](https://fastapi.tiangolo.com/tutorial/dependencies/)
- [FastAPI Background Tasks](https://fastapi.tiangolo.com/tutorial/background-tasks/)
- [FastAPI Testing](https://fastapi.tiangolo.com/tutorial/testing/)

## 🚀 Next Steps

1. **Implement declarative routes** in your existing FastAPI application
2. **Add comprehensive return type annotations** for all endpoints
3. **Create consistent response models** for success and error cases
4. **Implement proper error handling** with structured error responses
5. **Add comprehensive documentation** for all routes
6. **Write integration tests** for route workflows
7. **Follow best practices** for route organization and naming

This declarative route definitions guide provides a comprehensive framework for building well-documented, type-safe, and maintainable FastAPI applications with clear return type annotations and consistent response patterns. 