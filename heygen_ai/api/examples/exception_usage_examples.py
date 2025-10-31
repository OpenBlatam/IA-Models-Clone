from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, EmailStr
import asyncio
from ..exceptions.http_exceptions import (
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
HTTP Exception Usage Examples for HeyGen AI API
Practical examples of using HTTP exceptions in API endpoints.
"""


    ValidationError, InvalidInputError, MissingRequiredFieldError,
    AuthenticationError, InvalidCredentialsError, ExpiredTokenError,
    AuthorizationError, InsufficientPermissionsError, SubscriptionRequiredError,
    NotFoundError, UserNotFoundError, VideoNotFoundError, TemplateNotFoundError,
    RateLimitError, VideoCreationRateLimitError, APIRateLimitError,
    ExternalServiceError, HeyGenAPIError, VideoProcessingError,
    ResourceConflictError, DuplicateResourceError, VideoAlreadyProcessingError,
    PayloadTooLargeError, VideoFileTooLargeError, ScriptTooLongError,
    UnsupportedMediaError, UnsupportedVideoFormatError,
    BusinessLogicError, VideoDurationLimitError, InvalidVideoTemplateError,
    InternalServerError, DatabaseError, CacheError,
    ExceptionFactory, log_exception
)

# =============================================================================
# Data Models
# =============================================================================

class UserCreateRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8)
    name: str = Field(..., min_length=1, max_length=100)

class VideoCreateRequest(BaseModel):
    script: str = Field(..., min_length=1, max_length=5000)
    template_id: str
    format: str = Field(default="mp4")
    duration: Optional[int] = Field(None, ge=1, le=600)  # 1-600 seconds

class VideoUpdateRequest(BaseModel):
    script: Optional[str] = Field(None, min_length=1, max_length=5000)
    template_id: Optional[str] = None

class User(BaseModel):
    id: int
    email: str
    name: str
    subscription_type: str = "free"
    
    def can_create_videos(self) -> bool:
        return self.subscription_type in ["premium", "enterprise"]
    
    def has_premium_subscription(self) -> bool:
        return self.subscription_type == "premium"

class Video(BaseModel):
    id: str
    user_id: int
    script: str
    template_id: str
    status: str
    created_at: str

# =============================================================================
# Mock Services
# =============================================================================

class MockUserService:
    """Mock user service for demonstration."""
    
    def __init__(self) -> Any:
        self.users = {
            1: User(id=1, email="user1@example.com", name="John Doe", subscription_type="free"),
            2: User(id=2, email="user2@example.com", name="Jane Smith", subscription_type="premium"),
        }
    
    async def get_user(self, user_id: int) -> Optional[User]:
        """Get user by ID."""
        await asyncio.sleep(0.1)  # Simulate database call
        return self.users.get(user_id)
    
    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        await asyncio.sleep(0.1)  # Simulate database call
        for user in self.users.values():
            if user.email == email:
                return user
        return None
    
    async def create_user(self, user_data: UserCreateRequest) -> User:
        """Create a new user."""
        await asyncio.sleep(0.1)  # Simulate database call
        
        # Check if user already exists
        existing_user = await self.get_user_by_email(user_data.email)
        if existing_user:
            raise DuplicateResourceError(
                message="User with this email already exists"
            )
        
        # Create new user
        new_user = User(
            id=len(self.users) + 1,
            email=user_data.email,
            name=user_data.name,
            subscription_type="free"
        )
        self.users[new_user.id] = new_user
        return new_user

class MockVideoService:
    """Mock video service for demonstration."""
    
    def __init__(self) -> Any:
        self.videos = {}
        self.processing_videos = set()
    
    async def get_video(self, video_id: str) -> Optional[Video]:
        """Get video by ID."""
        await asyncio.sleep(0.1)  # Simulate database call
        return self.videos.get(video_id)
    
    async def create_video(self, video_data: VideoCreateRequest, user: User) -> Video:
        """Create a new video."""
        await asyncio.sleep(0.1)  # Simulate database call
        
        # Validate template
        if not await self.is_valid_template(video_data.template_id):
            raise TemplateNotFoundError(
                message=f"Template '{video_data.template_id}' not found"
            )
        
        # Check if user can create videos
        if not user.can_create_videos():
            raise InsufficientPermissionsError(
                message="You don't have permission to create videos"
            )
        
        # Check rate limits
        if await self.is_rate_limited(user.id):
            raise VideoCreationRateLimitError(
                message="Video creation rate limit exceeded",
                retry_after=3600  # 1 hour
            )
        
        # Create video
        video_id = f"vid_{len(self.videos) + 1}"
        video = Video(
            id=video_id,
            user_id=user.id,
            script=video_data.script,
            template_id=video_data.template_id,
            status="processing",
            created_at="2024-01-01T12:00:00Z"
        )
        
        self.videos[video_id] = video
        self.processing_videos.add(video_id)
        
        return video
    
    async def update_video(self, video_id: str, video_data: VideoUpdateRequest, user: User) -> Video:
        """Update video."""
        video = await self.get_video(video_id)
        if not video:
            raise VideoNotFoundError(
                message=f"Video with ID {video_id} not found"
            )
        
        # Check ownership
        if video.user_id != user.id:
            raise AuthorizationError(
                message="You don't have permission to update this video"
            )
        
        # Check if video is processing
        if video_id in self.processing_videos:
            raise VideoAlreadyProcessingError(
                message="Cannot update video while it's being processed"
            )
        
        # Update video
        if video_data.script:
            video.script = video_data.script
        if video_data.template_id:
            video.template_id = video_data.template_id
        
        return video
    
    async def is_valid_template(self, template_id: str) -> bool:
        """Check if template is valid."""
        valid_templates = ["template_1", "template_2", "template_3"]
        return template_id in valid_templates
    
    async def is_rate_limited(self, user_id: int) -> bool:
        """Check if user is rate limited."""
        # Simulate rate limiting
        return user_id == 1  # User 1 is rate limited for demo

class MockRateLimiter:
    """Mock rate limiter for demonstration."""
    
    def __init__(self) -> Any:
        self.limits = {}
    
    async def is_limited(self, user_id: int, action: str) -> bool:
        """Check if user is rate limited for action."""
        key = f"{user_id}:{action}"
        return self.limits.get(key, False)
    
    async def set_limit(self, user_id: int, action: str, limited: bool):
        """Set rate limit for user and action."""
        key = f"{user_id}:{action}"
        self.limits[key] = limited

# =============================================================================
# Dependencies
# =============================================================================

# Mock services
user_service = MockUserService()
video_service = MockVideoService()
rate_limiter = MockRateLimiter()

async def get_current_user(user_id: int = 1) -> User:
    """Get current user (mock authentication)."""
    user = await user_service.get_user(user_id)
    if not user:
        raise AuthenticationError("User not found")
    return user

async def require_premium_subscription(user: User = Depends(get_current_user)) -> User:
    """Require premium subscription."""
    if not user.has_premium_subscription():
        raise SubscriptionRequiredError(
            message="Premium subscription required for this feature"
        )
    return user

# =============================================================================
# API Routes with Exception Examples
# =============================================================================

router = APIRouter()

@router.post("/users", response_model=User)
async def create_user(user_data: UserCreateRequest):
    """
    Create a new user with validation error handling.
    
    Raises:
        - ValidationError: If input validation fails
        - DuplicateResourceError: If user already exists
        - DatabaseError: If database operation fails
    """
    try:
        # Validate password strength
        if len(user_data.password) < 8:
            raise ValidationError(
                message="Password validation failed",
                details=[
                    {
                        "field": "password",
                        "message": "Password must be at least 8 characters long",
                        "value": "***",  # Don't log actual password
                        "suggestion": "Use a password with at least 8 characters"
                    }
                ]
            )
        
        # Create user
        user = await user_service.create_user(user_data)
        return user
        
    except DuplicateResourceError:
        # Re-raise as is
        raise
    except Exception as e:
        # Log and raise internal server error
        logger.error("Failed to create user", error=str(e))
        raise DatabaseError("Failed to create user")

@router.get("/users/{user_id}", response_model=User)
async def get_user(user_id: int):
    """
    Get user by ID with not found error handling.
    
    Raises:
        - UserNotFoundError: If user doesn't exist
        - DatabaseError: If database operation fails
    """
    try:
        user = await user_service.get_user(user_id)
        if not user:
            raise UserNotFoundError(
                message=f"User with ID {user_id} not found"
            )
        return user
        
    except UserNotFoundError:
        # Re-raise as is
        raise
    except Exception as e:
        logger.error("Failed to get user", error=str(e))
        raise DatabaseError("Failed to retrieve user")

@router.post("/videos", response_model=Video)
async def create_video(
    video_data: VideoCreateRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Create a new video with comprehensive error handling.
    
    Raises:
        - ValidationError: If input validation fails
        - InsufficientPermissionsError: If user lacks permissions
        - TemplateNotFoundError: If template doesn't exist
        - VideoCreationRateLimitError: If rate limit exceeded
        - VideoProcessingError: If external service fails
    """
    try:
        # Validate script content
        if not video_data.script or len(video_data.script.strip()) == 0:
            raise ValidationError(
                message="Script is required",
                details=[
                    {
                        "field": "script",
                        "message": "Script cannot be empty",
                        "suggestion": "Please provide a script for the video"
                    }
                ]
            )
        
        # Validate script length
        if len(video_data.script) > 5000:
            raise ScriptTooLongError(
                message="Script is too long",
                details=[
                    {
                        "field": "script",
                        "message": "Script exceeds 5000 character limit",
                        "value": f"{len(video_data.script)} characters",
                        "suggestion": "Please shorten the script to 5000 characters or less"
                    }
                ]
            )
        
        # Validate video format
        supported_formats = ["mp4", "mov", "avi"]
        if video_data.format not in supported_formats:
            raise UnsupportedVideoFormatError(
                message=f"Format '{video_data.format}' is not supported",
                details=[
                    {
                        "field": "format",
                        "message": f"Format '{video_data.format}' is not supported",
                        "value": video_data.format,
                        "suggestion": f"Please use one of: {', '.join(supported_formats)}"
                    }
                ]
            )
        
        # Validate video duration
        if video_data.duration and video_data.duration > 600:
            raise VideoDurationLimitError(
                message="Video duration exceeds 10-minute limit",
                details=[
                    {
                        "field": "duration",
                        "message": "Duration cannot exceed 600 seconds",
                        "value": video_data.duration,
                        "suggestion": "Please reduce the video duration to 10 minutes or less"
                    }
                ]
            )
        
        # Create video
        video = await video_service.create_video(video_data, current_user)
        return video
        
    except (ValidationError, InsufficientPermissionsError, 
            TemplateNotFoundError, VideoCreationRateLimitError):
        # Re-raise validation and business logic errors
        raise
    except Exception as e:
        logger.error("Failed to create video", error=str(e))
        raise VideoProcessingError("Failed to create video")

@router.get("/videos/{video_id}", response_model=Video)
async def get_video(
    video_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get video by ID with access control.
    
    Raises:
        - VideoNotFoundError: If video doesn't exist
        - ResourceAccessDeniedError: If user lacks access
        - DatabaseError: If database operation fails
    """
    try:
        video = await video_service.get_video(video_id)
        if not video:
            raise VideoNotFoundError(
                message=f"Video with ID {video_id} not found"
            )
        
        # Check access permissions
        if video.user_id != current_user.id:
            raise AuthorizationError(
                message="You don't have access to this video"
            )
        
        return video
        
    except (VideoNotFoundError, AuthorizationError):
        # Re-raise as is
        raise
    except Exception as e:
        logger.error("Failed to get video", error=str(e))
        raise DatabaseError("Failed to retrieve video")

@router.put("/videos/{video_id}", response_model=Video)
async def update_video(
    video_id: str,
    video_data: VideoUpdateRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Update video with conflict handling.
    
    Raises:
        - VideoNotFoundError: If video doesn't exist
        - AuthorizationError: If user lacks permission
        - VideoAlreadyProcessingError: If video is being processed
        - ValidationError: If update data is invalid
    """
    try:
        # Validate update data
        if video_data.script and len(video_data.script.strip()) == 0:
            raise ValidationError(
                message="Script cannot be empty",
                details=[
                    {
                        "field": "script",
                        "message": "Script cannot be empty",
                        "suggestion": "Please provide a non-empty script"
                    }
                ]
            )
        
        video = await video_service.update_video(video_id, video_data, current_user)
        return video
        
    except (VideoNotFoundError, AuthorizationError, VideoAlreadyProcessingError, ValidationError):
        # Re-raise as is
        raise
    except Exception as e:
        logger.error("Failed to update video", error=str(e))
        raise DatabaseError("Failed to update video")

@router.get("/videos/{video_id}/status")
async def get_video_status(
    video_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get video processing status.
    
    Raises:
        - VideoNotFoundError: If video doesn't exist
        - HeyGenAPIError: If external API fails
    """
    try:
        # Check if video exists
        video = await video_service.get_video(video_id)
        if not video:
            raise VideoNotFoundError(
                message=f"Video with ID {video_id} not found"
            )
        
        # Check access permissions
        if video.user_id != current_user.id:
            raise AuthorizationError(
                message="You don't have access to this video"
            )
        
        # Simulate external API call
        if video_id == "vid_1":
            # Simulate external API error
            raise HeyGenAPIError(
                message="HeyGen AI service is temporarily unavailable"
            )
        
        return {"video_id": video_id, "status": video.status}
        
    except (VideoNotFoundError, AuthorizationError, HeyGenAPIError):
        # Re-raise as is
        raise
    except Exception as e:
        logger.error("Failed to get video status", error=str(e))
        raise ExternalServiceError("Failed to get video status")

@router.post("/videos/batch")
async def create_videos_batch(
    videos_data: List[VideoCreateRequest],
    current_user: User = Depends(require_premium_subscription)
):
    """
    Create multiple videos (premium feature).
    
    Raises:
        - SubscriptionRequiredError: If user lacks premium subscription
        - ValidationError: If any video data is invalid
        - VideoCreationRateLimitError: If rate limit exceeded
    """
    try:
        # Validate batch size
        if len(videos_data) > 10:
            raise ValidationError(
                message="Batch size too large",
                details=[
                    {
                        "field": "videos_data",
                        "message": "Maximum 10 videos per batch",
                        "value": len(videos_data),
                        "suggestion": "Please reduce batch size to 10 or fewer videos"
                    }
                ]
            )
        
        # Check rate limits
        if await rate_limiter.is_limited(current_user.id, "batch_video_creation"):
            raise VideoCreationRateLimitError(
                message="Batch video creation rate limit exceeded",
                retry_after=7200  # 2 hours
            )
        
        # Create videos
        created_videos = []
        for video_data in videos_data:
            try:
                video = await video_service.create_video(video_data, current_user)
                created_videos.append(video)
            except Exception as e:
                logger.error(f"Failed to create video in batch", error=str(e))
                # Continue with other videos
        
        return {"created_videos": created_videos, "total": len(created_videos)}
        
    except (SubscriptionRequiredError, ValidationError, VideoCreationRateLimitError):
        # Re-raise as is
        raise
    except Exception as e:
        logger.error("Failed to create videos batch", error=str(e))
        raise VideoProcessingError("Failed to create videos batch")

# =============================================================================
# Exception Factory Examples
# =============================================================================

@router.post("/examples/factory")
async def exception_factory_examples():
    """Examples of using ExceptionFactory."""
    
    # Create validation error with factory
    validation_error = ExceptionFactory.create_validation_error(
        field="email",
        message="Invalid email format",
        value="invalid-email",
        suggestion="Please provide a valid email address"
    )
    
    # Create rate limit error with factory
    rate_limit_error = ExceptionFactory.create_rate_limit_error(
        retry_after=60,
        message="Too many requests"
    )
    
    # Create external service error with factory
    external_error = ExceptionFactory.create_external_service_error(
        service_name="HeyGen API",
        error_message="Service temporarily unavailable",
        status_code=503
    )
    
    return {
        "validation_error": validation_error.to_error_response().dict(),
        "rate_limit_error": rate_limit_error.to_error_response().dict(),
        "external_error": external_error.to_error_response().dict()
    }

# =============================================================================
# Error Testing Endpoints
# =============================================================================

@router.get("/test/errors/{error_type}")
async def test_error(error_type: str):
    """Test different error types."""
    
    error_map = {
        "validation": ValidationError("Test validation error"),
        "authentication": AuthenticationError("Test authentication error"),
        "authorization": AuthorizationError("Test authorization error"),
        "not_found": NotFoundError("Test not found error"),
        "rate_limit": RateLimitError("Test rate limit error", retry_after=60),
        "external_service": ExternalServiceError("Test external service error"),
        "conflict": ResourceConflictError("Test conflict error"),
        "payload_too_large": PayloadTooLargeError("Test payload too large error"),
        "unsupported_media": UnsupportedMediaError("Test unsupported media error"),
        "business_logic": BusinessLogicError("Test business logic error"),
        "internal_server": InternalServerError("Test internal server error"),
    }
    
    if error_type not in error_map:
        raise NotFoundError(f"Error type '{error_type}' not found")
    
    raise error_map[error_type]

# =============================================================================
# Export
# =============================================================================

__all__ = [
    "router",
    "user_service",
    "video_service",
    "rate_limiter",
    "get_current_user",
    "require_premium_subscription",
] 