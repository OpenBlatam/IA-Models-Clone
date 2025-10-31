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

from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timezone
from enum import Enum
from pydantic import (
import re
from .base_schemas import (
from typing import Any, List, Dict, Optional
import logging
import asyncio
#!/usr/bin/env python3
"""
Video Schemas for HeyGen AI API
Video creation, management, and processing operations.
"""

    BaseModel, Field, validator, root_validator, 
    ConfigDict, computed_field, model_validator
)

    BaseRequest, BaseResponse, DataResponse, PaginatedDataResponse,
    IDField, TimestampFields, StatusFields, MetadataFields
)

# =============================================================================
# Video Enums
# =============================================================================

class VideoStatus(str, Enum):
    """Video status enumeration."""
    DRAFT = "draft"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PUBLISHED = "published"
    ARCHIVED = "archived"

class VideoQuality(str, Enum):
    """Video quality enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA_HD = "ultra_hd"
    CUSTOM = "custom"

class VideoFormat(str, Enum):
    """Video format enumeration."""
    MP4 = "mp4"
    MOV = "mov"
    AVI = "avi"
    WEBM = "webm"
    MKV = "mkv"

class VideoAspectRatio(str, Enum):
    """Video aspect ratio enumeration."""
    SQUARE = "1:1"
    PORTRAIT = "9:16"
    LANDSCAPE = "16:9"
    WIDESCREEN = "21:9"
    CUSTOM = "custom"

class VideoProcessingStage(str, Enum):
    """Video processing stage enumeration."""
    INITIALIZED = "initialized"
    SCRIPT_PROCESSING = "script_processing"
    VOICE_GENERATION = "voice_generation"
    VIDEO_GENERATION = "video_generation"
    RENDERING = "rendering"
    UPLOADING = "uploading"
    COMPLETED = "completed"

# =============================================================================
# Video Base Models
# =============================================================================

class VideoBase(BaseModel):
    """Base video model."""
    title: str = Field(
        min_length=1,
        max_length=200,
        description="Video title"
    )
    description: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Video description"
    )
    quality: VideoQuality = Field(
        default=VideoQuality.HIGH,
        description="Video quality"
    )
    format: VideoFormat = Field(
        default=VideoFormat.MP4,
        description="Video format"
    )
    aspect_ratio: VideoAspectRatio = Field(
        default=VideoAspectRatio.LANDSCAPE,
        description="Video aspect ratio"
    )
    duration: Optional[int] = Field(
        default=None,
        ge=1,
        le=3600,
        description="Video duration in seconds"
    )
    
    @validator('title')
    def validate_title(cls, v) -> bool:
        """Validate video title."""
        if not v or not v.strip():
            raise ValueError('Video title cannot be empty')
        return v.strip()
    
    @validator('description')
    def validate_description(cls, v) -> bool:
        """Validate video description."""
        if v is not None:
            return v.strip()
        return v

class VideoScript(BaseModel):
    """Video script model."""
    content: str = Field(
        min_length=1,
        max_length=10000,
        description="Script content"
    )
    language: str = Field(
        default="en",
        min_length=2,
        max_length=5,
        description="Script language code"
    )
    voice_id: Optional[str] = Field(
        default=None,
        description="Voice ID for narration"
    )
    speed: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="Speech speed multiplier"
    )
    tone: Optional[str] = Field(
        default=None,
        description="Voice tone/style"
    )
    
    @validator('content')
    def validate_content(cls, v) -> bool:
        """Validate script content."""
        if not v or not v.strip():
            raise ValueError('Script content cannot be empty')
        return v.strip()
    
    @validator('speed')
    def validate_speed(cls, v) -> bool:
        """Validate speech speed."""
        if v < 0.5 or v > 2.0:
            raise ValueError('Speed must be between 0.5 and 2.0')
        return round(v, 2)

class VideoTemplate(BaseModel):
    """Video template model."""
    id: str = Field(
        description="Template ID"
    )
    name: str = Field(
        description="Template name"
    )
    category: str = Field(
        description="Template category"
    )
    thumbnail_url: Optional[str] = Field(
        default=None,
        description="Template thumbnail URL"
    )
    preview_url: Optional[str] = Field(
        default=None,
        description="Template preview URL"
    )
    settings: Dict[str, Any] = Field(
        default={},
        description="Template settings"
    )

class VideoSettings(BaseModel):
    """Video settings model."""
    resolution: Optional[str] = Field(
        default=None,
        description="Video resolution (e.g., '1920x1080')"
    )
    frame_rate: Optional[int] = Field(
        default=None,
        ge=1,
        le=120,
        description="Frame rate (fps)"
    )
    bitrate: Optional[int] = Field(
        default=None,
        ge=1000,
        description="Video bitrate (kbps)"
    )
    audio_enabled: bool = Field(
        default=True,
        description="Enable audio"
    )
    background_music: Optional[str] = Field(
        default=None,
        description="Background music URL or ID"
    )
    watermark: Optional[str] = Field(
        default=None,
        description="Watermark URL or ID"
    )
    subtitles: bool = Field(
        default=False,
        description="Enable subtitles"
    )
    subtitles_language: Optional[str] = Field(
        default=None,
        description="Subtitles language"
    )
    
    @validator('resolution')
    def validate_resolution(cls, v) -> bool:
        """Validate resolution format."""
        if v and not re.match(r'^\d+x\d+$', v):
            raise ValueError('Resolution must be in format WIDTHxHEIGHT (e.g., 1920x1080)')
        return v

# =============================================================================
# Video Request Models
# =============================================================================

class VideoCreateRequest(BaseRequest):
    """Video creation request model."""
    title: str = Field(
        min_length=1,
        max_length=200,
        description="Video title"
    )
    description: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Video description"
    )
    script: VideoScript = Field(
        description="Video script"
    )
    template: VideoTemplate = Field(
        description="Video template"
    )
    settings: Optional[VideoSettings] = Field(
        default=None,
        description="Video settings"
    )
    quality: VideoQuality = Field(
        default=VideoQuality.HIGH,
        description="Video quality"
    )
    format: VideoFormat = Field(
        default=VideoFormat.MP4,
        description="Video format"
    )
    aspect_ratio: VideoAspectRatio = Field(
        default=VideoAspectRatio.LANDSCAPE,
        description="Video aspect ratio"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata"
    )
    tags: Optional[List[str]] = Field(
        default=None,
        description="Video tags"
    )

class VideoUpdateRequest(BaseRequest):
    """Video update request model."""
    title: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=200,
        description="Video title"
    )
    description: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Video description"
    )
    script: Optional[VideoScript] = Field(
        default=None,
        description="Video script"
    )
    settings: Optional[VideoSettings] = Field(
        default=None,
        description="Video settings"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata"
    )
    tags: Optional[List[str]] = Field(
        default=None,
        description="Video tags"
    )

class VideoSearchRequest(BaseRequest):
    """Video search request model."""
    query: Optional[str] = Field(
        default=None,
        description="Search query"
    )
    status: Optional[VideoStatus] = Field(
        default=None,
        description="Filter by video status"
    )
    quality: Optional[VideoQuality] = Field(
        default=None,
        description="Filter by video quality"
    )
    format: Optional[VideoFormat] = Field(
        default=None,
        description="Filter by video format"
    )
    template_category: Optional[str] = Field(
        default=None,
        description="Filter by template category"
    )
    created_after: Optional[datetime] = Field(
        default=None,
        description="Filter videos created after this date"
    )
    created_before: Optional[datetime] = Field(
        default=None,
        description="Filter videos created before this date"
    )
    tags: Optional[List[str]] = Field(
        default=None,
        description="Filter by tags"
    )
    page: int = Field(
        default=1,
        ge=1,
        description="Page number"
    )
    per_page: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Items per page"
    )
    sort_by: Optional[str] = Field(
        default="created_at",
        description="Sort field"
    )
    sort_order: Optional[str] = Field(
        default="desc",
        regex="^(asc|desc)$",
        description="Sort order"
    )

class VideoProcessingRequest(BaseRequest):
    """Video processing request model."""
    video_id: str = Field(
        description="Video ID to process"
    )
    priority: Optional[str] = Field(
        default="normal",
        regex="^(low|normal|high|urgent)$",
        description="Processing priority"
    )
    callback_url: Optional[str] = Field(
        default=None,
        description="Callback URL for processing updates"
    )

class VideoPublishRequest(BaseRequest):
    """Video publish request model."""
    video_id: str = Field(
        description="Video ID to publish"
    )
    platforms: Optional[List[str]] = Field(
        default=None,
        description="Target platforms for publishing"
    )
    privacy: Optional[str] = Field(
        default="public",
        regex="^(public|private|unlisted)$",
        description="Privacy setting"
    )
    scheduled_at: Optional[datetime] = Field(
        default=None,
        description="Scheduled publish time"
    )

# =============================================================================
# Video Response Models
# =============================================================================

class VideoFileInfo(BaseModel):
    """Video file information model."""
    url: str = Field(
        description="Video file URL"
    )
    size: int = Field(
        description="File size in bytes"
    )
    duration: int = Field(
        description="Video duration in seconds"
    )
    resolution: str = Field(
        description="Video resolution"
    )
    format: VideoFormat = Field(
        description="Video format"
    )
    bitrate: Optional[int] = Field(
        default=None,
        description="Video bitrate (kbps)"
    )
    frame_rate: Optional[int] = Field(
        default=None,
        description="Frame rate (fps)"
    )
    thumbnail_url: Optional[str] = Field(
        default=None,
        description="Thumbnail URL"
    )
    preview_url: Optional[str] = Field(
        default=None,
        description="Preview URL"
    )
    
    @computed_field
    @property
    def size_mb(self) -> float:
        """Get file size in MB."""
        return round(self.size / (1024 * 1024), 2)
    
    @computed_field
    @property
    def duration_formatted(self) -> str:
        """Get formatted duration."""
        minutes = self.duration // 60
        seconds = self.duration % 60
        return f"{minutes:02d}:{seconds:02d}"
    
    model_config = ConfigDict(
        use_enum_values=True
    )

class VideoProcessingInfo(BaseModel):
    """Video processing information model."""
    stage: VideoProcessingStage = Field(
        description="Current processing stage"
    )
    progress: float = Field(
        ge=0.0,
        le=100.0,
        description="Processing progress percentage"
    )
    estimated_completion: Optional[datetime] = Field(
        default=None,
        description="Estimated completion time"
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if processing failed"
    )
    retry_count: int = Field(
        default=0,
        description="Number of retry attempts"
    )
    started_at: Optional[datetime] = Field(
        default=None,
        description="Processing start time"
    )
    completed_at: Optional[datetime] = Field(
        default=None,
        description="Processing completion time"
    )
    
    @computed_field
    @property
    def is_completed(self) -> bool:
        """Check if processing is completed."""
        return self.stage == VideoProcessingStage.COMPLETED
    
    @computed_field
    @property
    def is_failed(self) -> bool:
        """Check if processing failed."""
        return self.error_message is not None
    
    @computed_field
    @property
    def processing_time(self) -> Optional[int]:
        """Get processing time in seconds."""
        if self.started_at and self.completed_at:
            return int((self.completed_at - self.started_at).total_seconds())
        return None
    
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        },
        use_enum_values=True
    )

class VideoResponse(BaseModel):
    """Video response model."""
    id: str = Field(
        description="Video ID"
    )
    title: str = Field(
        description="Video title"
    )
    description: Optional[str] = Field(
        default=None,
        description="Video description"
    )
    status: VideoStatus = Field(
        description="Video status"
    )
    quality: VideoQuality = Field(
        description="Video quality"
    )
    format: VideoFormat = Field(
        description="Video format"
    )
    aspect_ratio: VideoAspectRatio = Field(
        description="Video aspect ratio"
    )
    script: VideoScript = Field(
        description="Video script"
    )
    template: VideoTemplate = Field(
        description="Video template"
    )
    settings: Optional[VideoSettings] = Field(
        default=None,
        description="Video settings"
    )
    file_info: Optional[VideoFileInfo] = Field(
        default=None,
        description="Video file information"
    )
    processing_info: Optional[VideoProcessingInfo] = Field(
        default=None,
        description="Processing information"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata"
    )
    tags: Optional[List[str]] = Field(
        default=None,
        description="Video tags"
    )
    user_id: str = Field(
        description="User ID who created the video"
    )
    created_at: datetime = Field(
        description="Creation timestamp"
    )
    updated_at: datetime = Field(
        description="Last update timestamp"
    )
    published_at: Optional[datetime] = Field(
        default=None,
        description="Publication timestamp"
    )
    
    @computed_field
    @property
    def is_ready(self) -> bool:
        """Check if video is ready for viewing."""
        return (
            self.status == VideoStatus.COMPLETED and
            self.file_info is not None
        )
    
    @computed_field
    @property
    def is_processing(self) -> bool:
        """Check if video is being processed."""
        return self.status == VideoStatus.PROCESSING
    
    @computed_field
    @property
    def is_published(self) -> bool:
        """Check if video is published."""
        return self.status == VideoStatus.PUBLISHED
    
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        },
        use_enum_values=True
    )

class VideoListResponse(PaginatedDataResponse[VideoResponse]):
    """Video list response model."""
    data: List[VideoResponse] = Field(
        description="List of videos"
    )

class VideoDetailResponse(DataResponse[VideoResponse]):
    """Video detail response model."""
    data: VideoResponse = Field(
        description="Video details"
    )

# =============================================================================
# Video Analytics Models
# =============================================================================

class VideoAnalytics(BaseModel):
    """Video analytics model."""
    video_id: str = Field(
        description="Video ID"
    )
    views: int = Field(
        default=0,
        description="Number of views"
    )
    likes: int = Field(
        default=0,
        description="Number of likes"
    )
    shares: int = Field(
        default=0,
        description="Number of shares"
    )
    comments: int = Field(
        default=0,
        description="Number of comments"
    )
    watch_time: int = Field(
        default=0,
        description="Total watch time in seconds"
    )
    engagement_rate: float = Field(
        default=0.0,
        description="Engagement rate percentage"
    )
    retention_rate: float = Field(
        default=0.0,
        description="Retention rate percentage"
    )
    last_updated: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last analytics update"
    )
    
    @computed_field
    @property
    def watch_time_formatted(self) -> str:
        """Get formatted watch time."""
        hours = self.watch_time // 3600
        minutes = (self.watch_time % 3600) // 60
        return f"{hours}h {minutes}m"
    
    @computed_field
    @property
    def engagement_score(self) -> float:
        """Calculate engagement score."""
        if self.views == 0:
            return 0.0
        return round((self.likes + self.shares + self.comments) / self.views * 100, 2)
    
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )

class VideoAnalyticsResponse(DataResponse[VideoAnalytics]):
    """Video analytics response model."""
    data: VideoAnalytics = Field(
        description="Video analytics"
    )

# =============================================================================
# Video Export Models
# =============================================================================

class VideoExportRequest(BaseRequest):
    """Video export request model."""
    video_id: str = Field(
        description="Video ID to export"
    )
    format: VideoFormat = Field(
        description="Export format"
    )
    quality: VideoQuality = Field(
        description="Export quality"
    )
    resolution: Optional[str] = Field(
        default=None,
        description="Export resolution"
    )
    include_watermark: bool = Field(
        default=False,
        description="Include watermark in export"
    )
    include_subtitles: bool = Field(
        default=False,
        description="Include subtitles in export"
    )

class VideoExportResponse(DataResponse[Dict[str, Any]]):
    """Video export response model."""
    data: Dict[str, Any] = Field(
        description="Export information"
    )
    download_url: Optional[str] = Field(
        default=None,
        description="Download URL"
    )
    expires_at: Optional[datetime] = Field(
        default=None,
        description="Download URL expiration time"
    )

# =============================================================================
# Export
# =============================================================================

__all__ = [
    # Enums
    "VideoStatus",
    "VideoQuality",
    "VideoFormat",
    "VideoAspectRatio",
    "VideoProcessingStage",
    
    # Base Models
    "VideoBase",
    "VideoScript",
    "VideoTemplate",
    "VideoSettings",
    
    # Request Models
    "VideoCreateRequest",
    "VideoUpdateRequest",
    "VideoSearchRequest",
    "VideoProcessingRequest",
    "VideoPublishRequest",
    
    # Response Models
    "VideoFileInfo",
    "VideoProcessingInfo",
    "VideoResponse",
    "VideoListResponse",
    "VideoDetailResponse",
    
    # Analytics Models
    "VideoAnalytics",
    "VideoAnalyticsResponse",
    
    # Export Models
    "VideoExportRequest",
    "VideoExportResponse",
] 