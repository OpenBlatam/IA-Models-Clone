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

from typing import Dict, Any, List, Optional, Union, Literal
from pydantic import BaseModel, Field, validator, root_validator
from datetime import datetime
from enum import Enum
        import re
        import re
        import re
        import re
        import re
        import re
        import re
        import re
from typing import Any, List, Dict, Optional
import logging
import asyncio
#!/usr/bin/env python3
"""
Pydantic models for HeyGen AI API
Provides comprehensive input validation with type hints.
"""



class QualityLevel(str, Enum):
    """Video quality levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class LanguageCode(str, Enum):
    """Supported language codes"""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    RUSSIAN = "ru"
    JAPANESE = "ja"
    KOREAN = "ko"
    CHINESE = "zh"


class VoiceID(str, Enum):
    """Available voice IDs"""
    VOICE_1 = "Voice 1"
    VOICE_2 = "Voice 2"
    VOICE_3 = "Voice 3"
    VOICE_4 = "Voice 4"
    VOICE_5 = "Voice 5"


class VideoStatus(str, Enum):
    """Video processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ProcessingSettings(BaseModel):
    """Video processing settings"""
    num_inference_steps: int = Field(default=50, ge=10, le=200, description="Number of inference steps")
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0, description="Guidance scale for generation")
    fps: int = Field(default=12, ge=1, le=60, description="Frames per second")
    resolution: tuple[int, int] = Field(default=(768, 768), description="Video resolution (width, height)")
    
    @validator('resolution')
    def validate_resolution(cls, v: tuple[int, int]) -> tuple[int, int]:
        """Validate resolution dimensions"""
        width, height = v
        if width < 256 or height < 256:
            raise ValueError("Resolution must be at least 256x256")
        if width > 2048 or height > 2048:
            raise ValueError("Resolution cannot exceed 2048x2048")
        return v


class VideoGenerationInput(BaseModel):
    """Input model for video generation"""
    script: str = Field(..., min_length=10, max_length=1000, description="Script for video generation")
    voice_id: VoiceID = Field(default=VoiceID.VOICE_1, description="Voice selection")
    language: LanguageCode = Field(default=LanguageCode.ENGLISH, description="Language code")
    quality: QualityLevel = Field(default=QualityLevel.MEDIUM, description="Video quality level")
    duration: Optional[int] = Field(None, ge=5, le=300, description="Video duration in seconds")
    custom_settings: Optional[ProcessingSettings] = Field(None, description="Custom processing settings")
    
    @validator('script')
    def validate_script_content(cls, v: str) -> str:
        """Validate script content"""
        if not v.strip():
            raise ValueError("Script cannot be empty")
        return v.strip()
    
    @root_validator
    def validate_duration_consistency(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate duration consistency with quality settings"""
        quality = values.get('quality')
        duration = values.get('duration')
        
        if quality == QualityLevel.LOW and duration and duration > 120:
            raise ValueError("Low quality videos cannot exceed 120 seconds")
        
        return values


class VideoStatusInput(BaseModel):
    """Input model for video status check"""
    video_id: str = Field(..., description="Video ID to check")
    
    @validator('video_id')
    def validate_video_id_format(cls, v: str) -> str:
        """Validate video ID format"""
        pattern = re.compile(r'^video_\d+_[a-zA-Z0-9_-]+$')
        if not pattern.match(v):
            raise ValueError("Invalid video ID format")
        return v


class UserVideosInput(BaseModel):
    """Input model for user videos list"""
    limit: int = Field(default=50, ge=1, le=100, description="Number of videos to return")
    offset: int = Field(default=0, ge=0, description="Number of videos to skip")
    status_filter: Optional[VideoStatus] = Field(None, description="Filter by status")
    date_from: Optional[datetime] = Field(None, description="Filter from date")
    date_to: Optional[datetime] = Field(None, description="Filter to date")
    
    @root_validator
    def validate_date_range(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate date range consistency"""
        date_from = values.get('date_from')
        date_to = values.get('date_to')
        
        if date_from and date_to and date_from > date_to:
            raise ValueError("date_from cannot be after date_to")
        
        return values


class HealthCheckInput(BaseModel):
    """Input model for health check"""
    include_details: bool = Field(default=False, description="Include detailed component status")
    check_external_services: bool = Field(default=False, description="Check external services")


class UserCreateInput(BaseModel):
    """Input model for user creation"""
    username: str = Field(..., min_length=3, max_length=50, description="Username")
    email: str = Field(..., description="Email address")
    password: str = Field(..., min_length=8, description="Password")
    full_name: Optional[str] = Field(None, max_length=100, description="Full name")
    
    @validator('username')
    def validate_username_format(cls, v: str) -> str:
        """Validate username format"""
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError("Username can only contain letters, numbers, underscores, and hyphens")
        return v
    
    @validator('email')
    def validate_email_format(cls, v: str) -> str:
        """Validate email format"""
        pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        if not pattern.match(v):
            raise ValueError("Invalid email format")
        return v.lower()
    
    @validator('password')
    def validate_password_strength(cls, v: str) -> str:
        """Validate password strength"""
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        
        if not re.search(r'[A-Z]', v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not re.search(r'[a-z]', v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not re.search(r'\d', v):
            raise ValueError("Password must contain at least one number")
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', v):
            raise ValueError("Password must contain at least one special character")
        
        return v


class UserUpdateInput(BaseModel):
    """Input model for user updates"""
    username: Optional[str] = Field(None, min_length=3, max_length=50, description="Username")
    email: Optional[str] = Field(None, description="Email address")
    full_name: Optional[str] = Field(None, max_length=100, description="Full name")
    is_active: Optional[bool] = Field(None, description="User active status")
    
    @validator('username')
    def validate_username_format(cls, v: Optional[str]) -> Optional[str]:
        """Validate username format"""
        if v is None:
            return v
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError("Username can only contain letters, numbers, underscores, and hyphens")
        return v
    
    @validator('email')
    def validate_email_format(cls, v: Optional[str]) -> Optional[str]:
        """Validate email format"""
        if v is None:
            return v
        pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        if not pattern.match(v):
            raise ValueError("Invalid email format")
        return v.lower()


class APIKeyInput(BaseModel):
    """Input model for API key operations"""
    api_key: str = Field(..., min_length=32, description="API key")
    
    @validator('api_key')
    async def validate_api_key_format(cls, v: str) -> str:
        """Validate API key format"""
        if not re.match(r'^[a-zA-Z0-9]{32,}$', v):
            raise ValueError("Invalid API key format")
        return v


class PaginationInput(BaseModel):
    """Input model for pagination"""
    page: int = Field(default=1, ge=1, description="Page number")
    page_size: int = Field(default=50, ge=1, le=100, description="Items per page")
    
    @property
    def offset(self) -> int:
        """Calculate offset from page and page_size"""
        return (self.page - 1) * self.page_size
    
    @property
    def limit(self) -> int:
        """Get limit from page_size"""
        return self.page_size


class FileUploadInput(BaseModel):
    """Input model for file uploads"""
    filename: str = Field(..., description="Original filename")
    content_type: str = Field(..., description="File content type")
    file_size: int = Field(..., ge=1, le=100*1024*1024, description="File size in bytes")  # 100MB max
    
    @validator('filename')
    def validate_filename(cls, v: str) -> str:
        """Validate filename"""
        unsafe_chars = r'[<>:"/\\|?*]'
        if re.search(unsafe_chars, v):
            raise ValueError("Filename contains unsafe characters")
        if len(v) > 255:
            raise ValueError("Filename too long")
        return v
    
    @validator('content_type')
    def validate_content_type(cls, v: str) -> str:
        """Validate content type"""
        allowed_types = [
            'video/mp4', 'video/avi', 'video/mov', 'video/wmv',
            'image/jpeg', 'image/png', 'image/gif', 'image/webp',
            'audio/mpeg', 'audio/wav', 'audio/ogg'
        ]
        if v not in allowed_types:
            raise ValueError(f"Unsupported content type: {v}")
        return v


class SearchInput(BaseModel):
    """Input model for search operations"""
    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    filters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Search filters")
    sort_by: Optional[str] = Field(default="created_at", description="Sort field")
    sort_order: Literal["asc", "desc"] = Field(default="desc", description="Sort order")
    
    @validator('query')
    def validate_query(cls, v: str) -> str:
        """Validate search query"""
        if not v.strip():
            raise ValueError("Search query cannot be empty")
        return v.strip()


class NotificationInput(BaseModel):
    """Input model for notifications"""
    title: str = Field(..., min_length=1, max_length=100, description="Notification title")
    message: str = Field(..., min_length=1, max_length=1000, description="Notification message")
    notification_type: Literal["info", "success", "warning", "error"] = Field(default="info", description="Notification type")
    priority: Literal["low", "medium", "high"] = Field(default="medium", description="Notification priority")
    expires_at: Optional[datetime] = Field(None, description="Expiration time")


class RateLimitInput(BaseModel):
    """Input model for rate limiting"""
    requests_per_minute: int = Field(default=60, ge=1, le=1000, description="Requests per minute")
    burst_limit: int = Field(default=10, ge=1, le=100, description="Burst limit")
    
    @root_validator
    def validate_rate_limit_consistency(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate rate limit consistency"""
        requests_per_minute = values.get('requests_per_minute')
        burst_limit = values.get('burst_limit')
        
        if burst_limit > requests_per_minute:
            raise ValueError("Burst limit cannot exceed requests per minute")
        
        return values


# Response models
class VideoGenerationOutput(BaseModel):
    """Output model for video generation"""
    video_id: str = Field(..., description="Generated video ID")
    status: VideoStatus = Field(..., description="Video processing status")
    processing_time: float = Field(default=0.0, ge=0.0, description="Processing time in seconds")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Video metadata")


class VideoStatusOutput(BaseModel):
    """Output model for video status"""
    video_id: str = Field(..., description="Video ID")
    status: VideoStatus = Field(..., description="Current status")
    progress: float = Field(..., ge=0.0, le=100.0, description="Processing progress (0-100)")
    file_size: Optional[int] = Field(None, ge=0, description="File size in bytes")
    duration: Optional[float] = Field(None, ge=0.0, description="Video duration in seconds")
    thumbnail_url: Optional[str] = Field(None, description="Thumbnail URL")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")


class UserVideosOutput(BaseModel):
    """Output model for user videos list"""
    videos: List[VideoStatusOutput] = Field(default_factory=list, description="List of video objects")
    total_count: int = Field(..., ge=0, description="Total number of videos")
    has_more: bool = Field(..., description="Whether there are more videos")
    pagination: Dict[str, Any] = Field(default_factory=dict, description="Pagination info")


class HealthCheckOutput(BaseModel):
    """Output model for health check"""
    status: Literal["healthy", "degraded", "unhealthy"] = Field(..., description="Overall health status")
    version: str = Field(..., description="API version")
    uptime: Dict[str, Any] = Field(default_factory=dict, description="System uptime info")
    components: Dict[str, bool] = Field(default_factory=dict, description="Component status")
    external_services: Optional[Dict[str, Any]] = Field(None, description="External service status")


class ErrorOutput(BaseModel):
    """Output model for errors"""
    error_code: str = Field(..., description="Error code")
    error_type: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")


# Named exports
__all__ = [
    # Enums
    "QualityLevel",
    "LanguageCode", 
    "VoiceID",
    "VideoStatus",
    
    # Input models
    "ProcessingSettings",
    "VideoGenerationInput",
    "VideoStatusInput",
    "UserVideosInput",
    "HealthCheckInput",
    "UserCreateInput",
    "UserUpdateInput",
    "APIKeyInput",
    "PaginationInput",
    "FileUploadInput",
    "SearchInput",
    "NotificationInput",
    "RateLimitInput",
    
    # Output models
    "VideoGenerationOutput",
    "VideoStatusOutput",
    "UserVideosOutput",
    "HealthCheckOutput",
    "ErrorOutput"
] 