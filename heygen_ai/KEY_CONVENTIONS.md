# Key Conventions - HeyGen AI FastAPI Project

A comprehensive guide to coding standards, naming conventions, and best practices for the HeyGen AI FastAPI application.

## 🎯 Overview

This guide establishes consistent conventions across:
- **Code Style**: Python, FastAPI, and Pydantic conventions
- **Naming Conventions**: Files, classes, functions, and variables
- **Project Structure**: Directory organization and file placement
- **API Design**: Endpoint naming, HTTP methods, and response formats
- **Database**: Models, migrations, and query conventions
- **Testing**: Test organization and naming patterns
- **Documentation**: Code comments and API documentation

## 📋 Table of Contents

1. [Code Style Conventions](#code-style-conventions)
2. [Naming Conventions](#naming-conventions)
3. [Project Structure](#project-structure)
4. [API Design Conventions](#api-design-conventions)
5. [Database Conventions](#database-conventions)
6. [Testing Conventions](#testing-conventions)
7. [Documentation Conventions](#documentation-conventions)
8. [Security Conventions](#security-conventions)
9. [Performance Conventions](#performance-conventions)
10. [Error Handling Conventions](#error-handling-conventions)

## 🐍 Code Style Conventions

### Python Style Guide

#### **PEP 8 Compliance**
```python
# ✅ Good: Follow PEP 8
def calculate_user_analytics(user_id: UUID, start_date: datetime) -> Dict[str, Any]:
    """Calculate analytics for a specific user."""
    analytics_data = {
        'total_videos': 0,
        'total_views': 0,
        'engagement_rate': 0.0
    }
    return analytics_data

# ❌ Bad: Not following PEP 8
def calculateUserAnalytics(userId:UUID,startDate:datetime)->Dict[str,Any]:
    analyticsData={'total_videos':0,'total_views':0,'engagement_rate':0.0}
    return analyticsData
```

#### **Import Organization**
```python
# Standard library imports
import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Union
from uuid import UUID

# Third-party imports
import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, validator
from sqlalchemy.ext.asyncio import AsyncSession

# Local application imports
from api.serialization.pydantic_optimizer import PydanticSerializationOptimizer
from api.lazy_loading.lazy_loader import LazyLoadingManager
from core.database import get_db_session
from models.user import User
from schemas.video import VideoCreate, VideoResponse
from services.video_service import VideoService
```

#### **Type Hints**
```python
# ✅ Good: Comprehensive type hints
async def process_video_data(
    video_id: UUID,
    processing_options: Dict[str, Any],
    user_id: Optional[UUID] = None
) -> VideoProcessingResult:
    """Process video data with specified options."""
    pass

# ❌ Bad: Missing or incomplete type hints
async def process_video_data(video_id, processing_options, user_id=None):
    pass
```

### FastAPI Conventions

#### **Router Organization**
```python
# ✅ Good: Organized router structure
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional

router = APIRouter(prefix="/api/v1/videos", tags=["videos"])

@router.get("/", response_model=List[VideoResponse])
async def get_videos(
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(20, ge=1, le=100, description="Items per page"),
    status: Optional[VideoStatus] = Query(None, description="Video status filter"),
    db: AsyncSession = Depends(get_db_session)
) -> List[VideoResponse]:
    """Get paginated list of videos."""
    pass

@router.post("/", response_model=VideoResponse, status_code=201)
async def create_video(
    video_data: VideoCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session)
) -> VideoResponse:
    """Create a new video."""
    pass

@router.get("/{video_id}", response_model=VideoResponse)
async def get_video(
    video_id: UUID,
    db: AsyncSession = Depends(get_db_session)
) -> VideoResponse:
    """Get a specific video by ID."""
    pass
```

#### **Dependency Injection**
```python
# ✅ Good: Clean dependency injection
def get_video_service(db: AsyncSession = Depends(get_db_session)) -> VideoService:
    """Get video service instance."""
    return VideoService(db)

@router.get("/{video_id}/analytics")
async def get_video_analytics(
    video_id: UUID,
    video_service: VideoService = Depends(get_video_service),
    current_user: User = Depends(get_current_user)
) -> VideoAnalyticsResponse:
    """Get analytics for a specific video."""
    pass
```

### Pydantic Conventions

#### **Model Definition**
```python
# ✅ Good: Well-structured Pydantic models
from pydantic import BaseModel, Field, validator, ConfigDict
from datetime import datetime
from typing import Optional, List
from uuid import UUID

class VideoBase(BaseModel):
    """Base video model with common fields."""
    title: str = Field(..., min_length=1, max_length=200, description="Video title")
    description: Optional[str] = Field(None, max_length=1000, description="Video description")
    duration: Optional[int] = Field(None, ge=0, description="Duration in seconds")
    
    model_config = ConfigDict(
        validate_assignment=True,
        populate_by_name=True,
        use_enum_values=True,
        extra='ignore'
    )

class VideoCreate(VideoBase):
    """Model for creating a new video."""
    user_id: UUID = Field(..., description="User ID who created the video")
    template_id: Optional[UUID] = Field(None, description="Template ID if using template")

class VideoResponse(VideoBase):
    """Model for video responses."""
    id: UUID = Field(..., description="Video ID")
    user_id: UUID = Field(..., description="User ID who created the video")
    status: VideoStatus = Field(..., description="Current video status")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    
    @validator('title')
    def validate_title(cls, v):
        """Validate video title."""
        if not v.strip():
            raise ValueError('Title cannot be empty')
        return v.strip()
```

## 📝 Naming Conventions

### File and Directory Naming

#### **Python Files**
```python
# ✅ Good: Snake case for Python files
video_service.py
user_analytics.py
pydantic_optimizer.py
lazy_loader.py
specialized_loaders.py

# ❌ Bad: Camel case or other formats
videoService.py
UserAnalytics.py
pydanticOptimizer.py
```

#### **Directories**
```python
# ✅ Good: Snake case for directories
api/
├── serialization/
├── lazy_loading/
├── caching/
└── middleware/

models/
├── user/
├── video/
└── analytics/

services/
├── video_service/
├── user_service/
└── analytics_service/
```

### Class Naming

#### **Pascal Case for Classes**
```python
# ✅ Good: Pascal case for classes
class VideoService:
    """Service for video operations."""
    pass

class UserAnalytics:
    """Analytics for user data."""
    pass

class PydanticSerializationOptimizer:
    """Optimizer for Pydantic serialization."""
    pass

class LazyLoadingManager:
    """Manager for lazy loading operations."""
    pass

# ❌ Bad: Snake case or camel case
class video_service:
    pass

class userAnalytics:
    pass
```

### Function and Method Naming

#### **Snake Case for Functions**
```python
# ✅ Good: Snake case for functions and methods
async def get_user_videos(user_id: UUID) -> List[Video]:
    """Get videos for a specific user."""
    pass

async def calculate_engagement_rate(video_id: UUID) -> float:
    """Calculate engagement rate for a video."""
    pass

async def process_video_upload(file_data: bytes) -> Video:
    """Process uploaded video file."""
    pass

# ❌ Bad: Camel case or other formats
async def getUserVideos(userId: UUID) -> List[Video]:
    pass

async def calculateEngagementRate(videoId: UUID) -> float:
    pass
```

### Variable Naming

#### **Snake Case for Variables**
```python
# ✅ Good: Snake case for variables
user_videos = await get_user_videos(user_id)
engagement_rate = calculate_engagement_rate(video_id)
total_view_count = sum(video.views for video in user_videos)
processing_status = "completed"

# ❌ Bad: Camel case or other formats
userVideos = await get_user_videos(user_id)
engagementRate = calculate_engagement_rate(video_id)
totalViewCount = sum(video.views for video in user_videos)
```

### Constant Naming

#### **Uppercase with Underscores**
```python
# ✅ Good: Uppercase for constants
MAX_VIDEO_SIZE = 100 * 1024 * 1024  # 100MB
DEFAULT_PAGE_SIZE = 20
MAX_RETRY_ATTEMPTS = 3
SUPPORTED_VIDEO_FORMATS = ['.mp4', '.avi', '.mov']
API_VERSION = "v1"

# ❌ Bad: Other formats
maxVideoSize = 100 * 1024 * 1024
defaultPageSize = 20
maxRetryAttempts = 3
```

## 🏗️ Project Structure

### Directory Organization

```
heygen_ai/
├── api/                          # API layer
│   ├── serialization/            # Serialization optimization
│   ├── lazy_loading/             # Lazy loading system
│   ├── caching/                  # Caching system
│   └── middleware/               # Custom middleware
├── core/                         # Core functionality
│   ├── database/                 # Database configuration
│   ├── config/                   # Application configuration
│   ├── security/                 # Security utilities
│   └── utils/                    # Utility functions
├── models/                       # Database models
│   ├── user/                     # User-related models
│   ├── video/                    # Video-related models
│   └── analytics/                # Analytics models
├── schemas/                      # Pydantic schemas
│   ├── user/                     # User schemas
│   ├── video/                    # Video schemas
│   └── analytics/                # Analytics schemas
├── services/                     # Business logic
│   ├── video_service/            # Video operations
│   ├── user_service/             # User operations
│   └── analytics_service/        # Analytics operations
├── repositories/                 # Data access layer
│   ├── user_repository/          # User data access
│   ├── video_repository/         # Video data access
│   └── analytics_repository/     # Analytics data access
├── tests/                        # Test files
│   ├── unit/                     # Unit tests
│   ├── integration/              # Integration tests
│   └── e2e/                      # End-to-end tests
└── docs/                         # Documentation
    ├── api/                      # API documentation
    ├── guides/                   # Implementation guides
    └── examples/                 # Code examples
```

### File Organization

#### **Module Structure**
```python
# ✅ Good: Well-organized module structure
"""
Video service module for handling video operations.

This module provides comprehensive video management functionality including
upload, processing, analytics, and optimization features.
"""

# Standard library imports
import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional, Union
from uuid import UUID

# Third-party imports
import structlog
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

# Local imports
from models.video import Video
from schemas.video import VideoCreate, VideoResponse
from repositories.video_repository import VideoRepository

logger = structlog.get_logger()

class VideoService:
    """Service for video operations."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.repository = VideoRepository(db)
    
    async def create_video(self, video_data: VideoCreate, user_id: UUID) -> VideoResponse:
        """Create a new video."""
        pass
    
    async def get_video(self, video_id: UUID) -> VideoResponse:
        """Get a video by ID."""
        pass
    
    async def update_video(self, video_id: UUID, video_data: VideoCreate) -> VideoResponse:
        """Update a video."""
        pass
    
    async def delete_video(self, video_id: UUID) -> bool:
        """Delete a video."""
        pass
```

## 🌐 API Design Conventions

### Endpoint Naming

#### **RESTful Endpoints**
```python
# ✅ Good: RESTful endpoint naming
@router.get("/videos")                    # GET /api/v1/videos
@router.post("/videos")                   # POST /api/v1/videos
@router.get("/videos/{video_id}")         # GET /api/v1/videos/{video_id}
@router.put("/videos/{video_id}")         # PUT /api/v1/videos/{video_id}
@router.delete("/videos/{video_id}")      # DELETE /api/v1/videos/{video_id}
@router.get("/videos/{video_id}/analytics")  # GET /api/v1/videos/{video_id}/analytics
@router.post("/videos/{video_id}/process")   # POST /api/v1/videos/{video_id}/process

# ❌ Bad: Non-RESTful naming
@router.get("/getVideos")
@router.post("/createVideo")
@router.get("/video/{video_id}/getAnalytics")
```

#### **Query Parameters**
```python
# ✅ Good: Consistent query parameter naming
@router.get("/videos")
async def get_videos(
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(20, ge=1, le=100, description="Items per page"),
    status: Optional[VideoStatus] = Query(None, description="Video status filter"),
    sort_by: str = Query("created_at", description="Sort field"),
    sort_order: str = Query("desc", regex="^(asc|desc)$", description="Sort order"),
    search: Optional[str] = Query(None, description="Search term")
):
    pass

# ❌ Bad: Inconsistent parameter naming
@router.get("/videos")
async def get_videos(
    pageNum: int = Query(1),
    itemsPerPage: int = Query(20),
    videoStatus: Optional[VideoStatus] = Query(None),
    sortField: str = Query("created_at")
):
    pass
```

### HTTP Status Codes

#### **Consistent Status Code Usage**
```python
# ✅ Good: Proper HTTP status codes
@router.get("/videos", response_model=List[VideoResponse])
async def get_videos() -> List[VideoResponse]:
    """Get list of videos - returns 200 OK."""
    pass

@router.post("/videos", response_model=VideoResponse, status_code=201)
async def create_video(video_data: VideoCreate) -> VideoResponse:
    """Create video - returns 201 Created."""
    pass

@router.put("/videos/{video_id}", response_model=VideoResponse)
async def update_video(video_id: UUID, video_data: VideoCreate) -> VideoResponse:
    """Update video - returns 200 OK."""
    pass

@router.delete("/videos/{video_id}", status_code=204)
async def delete_video(video_id: UUID):
    """Delete video - returns 204 No Content."""
    pass

# Error responses
@router.get("/videos/{video_id}")
async def get_video(video_id: UUID) -> VideoResponse:
    """Get video - returns 404 Not Found if not found."""
    video = await video_service.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    return video
```

### Response Format

#### **Consistent Response Structure**
```python
# ✅ Good: Consistent response format
from pydantic import BaseModel
from typing import Optional, Any

class APIResponse(BaseModel):
    """Standard API response format."""
    success: bool = Field(..., description="Request success status")
    message: str = Field(..., description="Response message")
    data: Optional[Any] = Field(None, description="Response data")
    errors: List[str] = Field(default_factory=list, description="Error messages")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class PaginatedResponse(BaseModel):
    """Paginated response format."""
    data: List[Any] = Field(..., description="Response data")
    total_count: int = Field(..., description="Total number of items")
    page: int = Field(..., description="Current page number")
    per_page: int = Field(..., description="Items per page")
    total_pages: int = Field(..., description="Total number of pages")
    has_next: bool = Field(..., description="Has next page")
    has_previous: bool = Field(..., description="Has previous page")

# Usage in endpoints
@router.get("/videos", response_model=PaginatedResponse[VideoResponse])
async def get_videos(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100)
) -> PaginatedResponse[VideoResponse]:
    """Get paginated list of videos."""
    videos, total_count = await video_service.get_videos_paginated(page, per_page)
    
    return PaginatedResponse(
        data=videos,
        total_count=total_count,
        page=page,
        per_page=per_page,
        total_pages=(total_count + per_page - 1) // per_page,
        has_next=page * per_page < total_count,
        has_previous=page > 1
    )
```

## 🗄️ Database Conventions

### Model Naming

#### **SQLAlchemy Models**
```python
# ✅ Good: Clear model naming and structure
from sqlalchemy import Column, String, Integer, DateTime, Boolean, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from core.database import Base

class Video(Base):
    """Video model for storing video information."""
    __tablename__ = "videos"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    title = Column(String(200), nullable=False, index=True)
    description = Column(String(1000), nullable=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    status = Column(String(50), nullable=False, default="pending", index=True)
    duration = Column(Integer, nullable=True)
    file_size = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc), nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="videos")
    analytics = relationship("VideoAnalytics", back_populates="video", cascade="all, delete-orphan")

# ❌ Bad: Poor naming and structure
class video(Base):
    __tablename__ = "video"
    
    id = Column(UUID(as_uuid=True), primary_key=True)
    name = Column(String(200))
    desc = Column(String(1000))
    user = Column(UUID(as_uuid=True), ForeignKey("user.id"))
```

### Migration Naming

#### **Alembic Migration Files**
```python
# ✅ Good: Descriptive migration names
# 2024_01_15_10_30_00_create_videos_table.py
# 2024_01_15_11_00_00_add_video_analytics_table.py
# 2024_01_15_11_30_00_add_video_status_index.py

# ❌ Bad: Generic names
# migration_001.py
# migration_002.py
# migration_003.py
```

### Query Conventions

#### **Repository Pattern**
```python
# ✅ Good: Repository pattern with clear method names
class VideoRepository:
    """Repository for video data access."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def get_by_id(self, video_id: UUID) -> Optional[Video]:
        """Get video by ID."""
        result = await self.db.execute(
            select(Video).where(Video.id == video_id)
        )
        return result.scalar_one_or_none()
    
    async def get_by_user_id(self, user_id: UUID, limit: int = 100) -> List[Video]:
        """Get videos by user ID."""
        result = await self.db.execute(
            select(Video)
            .where(Video.user_id == user_id)
            .order_by(Video.created_at.desc())
            .limit(limit)
        )
        return result.scalars().all()
    
    async def create(self, video_data: VideoCreate, user_id: UUID) -> Video:
        """Create a new video."""
        video = Video(
            title=video_data.title,
            description=video_data.description,
            user_id=user_id,
            status="pending"
        )
        self.db.add(video)
        await self.db.commit()
        await self.db.refresh(video)
        return video

# ❌ Bad: Direct database access in services
class VideoService:
    async def get_video(self, video_id: UUID):
        result = await self.db.execute(
            select(Video).where(Video.id == video_id)
        )
        return result.scalar_one_or_none()
```

## 🧪 Testing Conventions

### Test File Naming

#### **Test File Structure**
```python
# ✅ Good: Clear test file naming
test_video_service.py
test_user_analytics.py
test_pydantic_optimizer.py
test_lazy_loader.py

# ❌ Bad: Generic names
test_service.py
test_analytics.py
test_optimizer.py
```

### Test Class and Method Naming

#### **Descriptive Test Names**
```python
# ✅ Good: Clear test class and method names
import pytest
from unittest.mock import AsyncMock, patch

class TestVideoService:
    """Test cases for VideoService."""
    
    @pytest.fixture
    async def video_service(self, db_session):
        """Create video service instance for testing."""
        return VideoService(db_session)
    
    @pytest.fixture
    def sample_video_data(self):
        """Sample video data for testing."""
        return VideoCreate(
            title="Test Video",
            description="Test description",
            user_id=uuid4()
        )
    
    async def test_create_video_success(self, video_service, sample_video_data):
        """Test successful video creation."""
        # Arrange
        user_id = uuid4()
        
        # Act
        result = await video_service.create_video(sample_video_data, user_id)
        
        # Assert
        assert result.title == sample_video_data.title
        assert result.user_id == user_id
        assert result.status == "pending"
    
    async def test_get_video_not_found(self, video_service):
        """Test getting non-existent video."""
        # Arrange
        video_id = uuid4()
        
        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await video_service.get_video(video_id)
        
        assert exc_info.value.status_code == 404
        assert "Video not found" in exc_info.value.detail

# ❌ Bad: Unclear test names
class TestService:
    def test_1(self):
        pass
    
    def test_2(self):
        pass
```

### Test Organization

#### **Test Directory Structure**
```
tests/
├── unit/                          # Unit tests
│   ├── services/                  # Service tests
│   │   ├── test_video_service.py
│   │   ├── test_user_service.py
│   │   └── test_analytics_service.py
│   ├── models/                    # Model tests
│   │   ├── test_video_model.py
│   │   └── test_user_model.py
│   └── utils/                     # Utility tests
│       ├── test_serialization.py
│       └── test_lazy_loading.py
├── integration/                   # Integration tests
│   ├── test_api_endpoints.py
│   ├── test_database_operations.py
│   └── test_external_services.py
└── e2e/                          # End-to-end tests
    ├── test_video_workflow.py
    ├── test_user_workflow.py
    └── test_analytics_workflow.py
```

## 📚 Documentation Conventions

### Code Comments

#### **Docstring Standards**
```python
# ✅ Good: Comprehensive docstrings
class VideoService:
    """Service for video operations and management.
    
    This service provides comprehensive video management functionality including
    creation, retrieval, updating, and deletion of videos. It also handles
    video processing, analytics, and optimization features.
    
    Attributes:
        db: Database session for data access
        repository: Video repository for data operations
    """
    
    def __init__(self, db: AsyncSession):
        """Initialize VideoService with database session.
        
        Args:
            db: Async database session for data access
        """
        self.db = db
        self.repository = VideoRepository(db)
    
    async def create_video(self, video_data: VideoCreate, user_id: UUID) -> VideoResponse:
        """Create a new video for the specified user.
        
        Args:
            video_data: Video creation data containing title, description, etc.
            user_id: ID of the user creating the video
            
        Returns:
            VideoResponse: Created video information
            
        Raises:
            HTTPException: If video creation fails or validation errors occur
            
        Example:
            >>> video_data = VideoCreate(title="My Video", description="Test video")
            >>> video = await video_service.create_video(video_data, user_id)
            >>> print(video.title)
            'My Video'
        """
        pass

# ❌ Bad: Missing or poor docstrings
class VideoService:
    def __init__(self, db):
        self.db = db
    
    async def create_video(self, video_data, user_id):
        # Create video
        pass
```

### API Documentation

#### **FastAPI Documentation**
```python
# ✅ Good: Comprehensive API documentation
@router.post(
    "/videos",
    response_model=VideoResponse,
    status_code=201,
    summary="Create a new video",
    description="Create a new video for the authenticated user. The video will be "
                "initially set to 'pending' status and will be processed asynchronously.",
    response_description="Successfully created video",
    responses={
        201: {
            "description": "Video created successfully",
            "content": {
                "application/json": {
                    "example": {
                        "id": "123e4567-e89b-12d3-a456-426614174000",
                        "title": "My First Video",
                        "description": "A test video",
                        "status": "pending",
                        "created_at": "2024-01-15T10:30:00Z"
                    }
                }
            }
        },
        400: {
            "description": "Invalid video data",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Title cannot be empty"
                    }
                }
            }
        },
        401: {
            "description": "Authentication required"
        }
    }
)
async def create_video(
    video_data: VideoCreate = Body(
        ...,
        example={
            "title": "My First Video",
            "description": "A test video description",
            "template_id": "123e4567-e89b-12d3-a456-426614174000"
        }
    ),
    current_user: User = Depends(get_current_user),
    video_service: VideoService = Depends(get_video_service)
) -> VideoResponse:
    """Create a new video for the authenticated user.
    
    This endpoint allows users to create new videos. The video will be
    initially set to 'pending' status and will be processed asynchronously.
    
    - **title**: Video title (required, 1-200 characters)
    - **description**: Video description (optional, max 1000 characters)
    - **template_id**: Optional template ID to use for video creation
    
    Returns the created video information including the generated video ID.
    """
    return await video_service.create_video(video_data, current_user.id)
```

## 🔒 Security Conventions

### Authentication and Authorization

#### **Security Decorators**
```python
# ✅ Good: Consistent security patterns
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> User:
    """Get current authenticated user from JWT token."""
    try:
        payload = jwt.decode(
            credentials.credentials,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM]
        )
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials"
            )
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
    
    user = await user_service.get_user_by_id(UUID(user_id))
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    
    return user

async def require_admin_role(current_user: User = Depends(get_current_user)) -> User:
    """Require admin role for endpoint access."""
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user

# Usage in endpoints
@router.delete("/videos/{video_id}")
async def delete_video(
    video_id: UUID,
    current_user: User = Depends(get_current_user),
    video_service: VideoService = Depends(get_video_service)
):
    """Delete a video (only owner or admin can delete)."""
    video = await video_service.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    if video.user_id != current_user.id and current_user.role != UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="Not authorized to delete this video")
    
    await video_service.delete_video(video_id)
    return {"message": "Video deleted successfully"}
```

### Input Validation

#### **Comprehensive Validation**
```python
# ✅ Good: Comprehensive input validation
from pydantic import BaseModel, Field, validator, root_validator
import re

class VideoCreate(BaseModel):
    """Video creation model with comprehensive validation."""
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
    template_id: Optional[UUID] = Field(
        None,
        description="Template ID for video creation"
    )
    
    @validator('title')
    def validate_title(cls, v):
        """Validate video title."""
        if not v.strip():
            raise ValueError('Title cannot be empty or whitespace only')
        
        # Check for prohibited characters
        if re.search(r'[<>:"/\\|?*]', v):
            raise ValueError('Title contains invalid characters')
        
        return v.strip()
    
    @validator('description')
    def validate_description(cls, v):
        """Validate video description."""
        if v is not None:
            if not v.strip():
                raise ValueError('Description cannot be whitespace only')
            return v.strip()
        return v
    
    @root_validator
    def validate_video_data(cls, values):
        """Validate overall video data."""
        title = values.get('title')
        description = values.get('description')
        
        # Ensure title and description are different
        if title and description and title.lower() == description.lower():
            raise ValueError('Title and description cannot be identical')
        
        return values
```

## ⚡ Performance Conventions

### Database Optimization

#### **Query Optimization**
```python
# ✅ Good: Optimized database queries
class VideoRepository:
    async def get_videos_with_analytics(self, user_id: UUID) -> List[Video]:
        """Get videos with analytics using optimized query."""
        result = await self.db.execute(
            select(Video, VideoAnalytics)
            .outerjoin(VideoAnalytics, Video.id == VideoAnalytics.video_id)
            .where(Video.user_id == user_id)
            .options(
                selectinload(Video.analytics),
                selectinload(Video.user)
            )
        )
        return result.unique().all()
    
    async def get_videos_paginated(
        self,
        page: int,
        per_page: int,
        user_id: Optional[UUID] = None
    ) -> tuple[List[Video], int]:
        """Get paginated videos with count."""
        # Get total count
        count_query = select(func.count(Video.id))
        if user_id:
            count_query = count_query.where(Video.user_id == user_id)
        
        count_result = await self.db.execute(count_query)
        total_count = count_result.scalar()
        
        # Get paginated data
        offset = (page - 1) * per_page
        videos_query = select(Video).options(selectinload(Video.user))
        if user_id:
            videos_query = videos_query.where(Video.user_id == user_id)
        
        videos_query = videos_query.offset(offset).limit(per_page)
        videos_result = await self.db.execute(videos_query)
        videos = videos_result.scalars().all()
        
        return videos, total_count

# ❌ Bad: Inefficient queries
class VideoRepository:
    async def get_videos_with_analytics(self, user_id: UUID) -> List[Video]:
        # N+1 query problem
        videos = await self.db.execute(
            select(Video).where(Video.user_id == user_id)
        )
        videos = videos.scalars().all()
        
        for video in videos:
            analytics = await self.db.execute(
                select(VideoAnalytics).where(VideoAnalytics.video_id == video.id)
            )
            video.analytics = analytics.scalars().all()
        
        return videos
```

### Caching Conventions

#### **Consistent Caching Patterns**
```python
# ✅ Good: Consistent caching implementation
from functools import lru_cache
import redis

class VideoService:
    def __init__(self, db: AsyncSession, redis_client: redis.Redis):
        self.db = db
        self.redis = redis_client
        self.cache_ttl = 3600  # 1 hour
    
    async def get_video(self, video_id: UUID) -> Optional[VideoResponse]:
        """Get video with caching."""
        # Try cache first
        cache_key = f"video:{video_id}"
        cached_video = await self.redis.get(cache_key)
        
        if cached_video:
            return VideoResponse.parse_raw(cached_video)
        
        # Get from database
        video = await self.repository.get_by_id(video_id)
        if not video:
            return None
        
        # Cache the result
        video_response = VideoResponse.from_orm(video)
        await self.redis.setex(
            cache_key,
            self.cache_ttl,
            video_response.json()
        )
        
        return video_response
    
    async def invalidate_video_cache(self, video_id: UUID):
        """Invalidate video cache when updated."""
        cache_key = f"video:{video_id}"
        await self.redis.delete(cache_key)
    
    @lru_cache(maxsize=100)
    def get_supported_formats(self) -> List[str]:
        """Get supported video formats (cached)."""
        return ['.mp4', '.avi', '.mov', '.mkv']

# ❌ Bad: Inconsistent caching
class VideoService:
    async def get_video(self, video_id: UUID):
        # No caching
        return await self.repository.get_by_id(video_id)
    
    def get_supported_formats(self):
        # No caching for expensive operation
        return self._load_formats_from_database()
```

## 🚨 Error Handling Conventions

### Exception Handling

#### **Consistent Error Handling**
```python
# ✅ Good: Comprehensive error handling
import structlog
from fastapi import HTTPException, status
from typing import Union

logger = structlog.get_logger()

class VideoServiceError(Exception):
    """Base exception for video service errors."""
    pass

class VideoNotFoundError(VideoServiceError):
    """Raised when video is not found."""
    pass

class VideoProcessingError(VideoServiceError):
    """Raised when video processing fails."""
    pass

class VideoService:
    async def get_video(self, video_id: UUID) -> VideoResponse:
        """Get video with proper error handling."""
        try:
            video = await self.repository.get_by_id(video_id)
            if not video:
                raise VideoNotFoundError(f"Video {video_id} not found")
            
            return VideoResponse.from_orm(video)
            
        except VideoNotFoundError:
            logger.warning(f"Video not found: {video_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Video not found"
            )
        except Exception as e:
            logger.error(f"Error getting video {video_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error"
            )
    
    async def process_video(self, video_id: UUID) -> VideoResponse:
        """Process video with error handling."""
        try:
            # Get video
            video = await self.get_video(video_id)
            
            # Process video
            processed_video = await self._process_video_file(video)
            
            # Update status
            updated_video = await self.repository.update_status(
                video_id, "completed"
            )
            
            return VideoResponse.from_orm(updated_video)
            
        except VideoProcessingError as e:
            logger.error(f"Video processing failed for {video_id}: {e}")
            await self.repository.update_status(video_id, "failed")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Video processing failed: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Unexpected error processing video {video_id}: {e}")
            await self.repository.update_status(video_id, "failed")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error"
            )

# ❌ Bad: Poor error handling
class VideoService:
    async def get_video(self, video_id: UUID):
        video = await self.repository.get_by_id(video_id)
        return video  # No error handling
    
    async def process_video(self, video_id: UUID):
        # No error handling
        video = await self.get_video(video_id)
        return await self._process_video_file(video)
```

### Logging Conventions

#### **Structured Logging**
```python
# ✅ Good: Structured logging with context
import structlog
from typing import Dict, Any

logger = structlog.get_logger()

class VideoService:
    async def create_video(self, video_data: VideoCreate, user_id: UUID) -> VideoResponse:
        """Create video with structured logging."""
        logger.info(
            "Creating video",
            user_id=str(user_id),
            title=video_data.title,
            has_description=bool(video_data.description)
        )
        
        try:
            video = await self.repository.create(video_data, user_id)
            
            logger.info(
                "Video created successfully",
                video_id=str(video.id),
                user_id=str(user_id),
                status=video.status
            )
            
            return VideoResponse.from_orm(video)
            
        except Exception as e:
            logger.error(
                "Failed to create video",
                user_id=str(user_id),
                title=video_data.title,
                error=str(e),
                exc_info=True
            )
            raise
    
    async def process_video(self, video_id: UUID) -> VideoResponse:
        """Process video with detailed logging."""
        logger.info("Starting video processing", video_id=str(video_id))
        
        try:
            # Get video
            video = await self.get_video(video_id)
            logger.debug("Retrieved video for processing", video_id=str(video_id))
            
            # Process video
            processed_video = await self._process_video_file(video)
            logger.info(
                "Video processing completed",
                video_id=str(video_id),
                duration=processed_video.duration,
                file_size=processed_video.file_size
            )
            
            return processed_video
            
        except Exception as e:
            logger.error(
                "Video processing failed",
                video_id=str(video_id),
                error=str(e),
                exc_info=True
            )
            raise

# ❌ Bad: Poor logging
class VideoService:
    async def create_video(self, video_data, user_id):
        print(f"Creating video for user {user_id}")  # No structured logging
        video = await self.repository.create(video_data, user_id)
        print("Video created")  # No context
        return video
```

## 📋 Summary

This comprehensive key conventions guide ensures:

### **1. Code Consistency**
- **PEP 8 compliance** for Python code
- **Consistent naming** across all components
- **Standardized structure** for files and directories

### **2. API Design Standards**
- **RESTful endpoints** with proper HTTP methods
- **Consistent response formats** with proper status codes
- **Comprehensive documentation** with examples

### **3. Database Best Practices**
- **Repository pattern** for data access
- **Optimized queries** with proper indexing
- **Consistent model structure** with relationships

### **4. Testing Standards**
- **Clear test organization** with descriptive names
- **Comprehensive test coverage** for all components
- **Proper test fixtures** and mocking

### **5. Security Guidelines**
- **Authentication patterns** with JWT tokens
- **Authorization checks** with role-based access
- **Input validation** with comprehensive rules

### **6. Performance Optimization**
- **Efficient database queries** with proper joins
- **Caching strategies** for frequently accessed data
- **Memory management** with monitoring

### **7. Error Handling**
- **Comprehensive exception handling** with proper logging
- **Structured error responses** with meaningful messages
- **Graceful degradation** for system failures

### **8. Documentation Standards**
- **Comprehensive docstrings** for all functions and classes
- **API documentation** with examples and responses
- **Code comments** for complex logic

Following these conventions ensures:
- **Maintainable code** that's easy to understand and modify
- **Consistent user experience** across all API endpoints
- **Reliable performance** with proper optimization
- **Secure application** with proper validation and authorization
- **Comprehensive testing** with good coverage
- **Clear documentation** for developers and users

These conventions should be followed consistently across all components of the HeyGen AI FastAPI application to maintain code quality and ensure a professional, reliable, and scalable system. 