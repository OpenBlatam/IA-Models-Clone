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

from pydantic import BaseModel, Field, validator, HttpUrl, conint, confloat
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timezone
from enum import Enum
import re
from . import (
from typing import Any, List, Dict, Optional
import logging
import asyncio
#!/usr/bin/env python3
"""
Video Data Models for HeyGen AI FastAPI
FastAPI best practices for video data models with comprehensive validation and documentation.
"""


    TimestampedModel, IdentifiedModel, StatusEnum,
    validate_file_size, validate_video_duration
)

# =============================================================================
# Video Enums
# =============================================================================

class VideoStatusEnum(str, Enum):
    """Video status enumeration following FastAPI best practices."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UPLOADED = "uploaded"
    DELETED = "deleted"

class VideoTypeEnum(str, Enum):
    """Video type enumeration following FastAPI best practices."""
    AI_GENERATED = "ai_generated"
    UPLOADED = "uploaded"
    EDITED = "edited"
    COMPOSITE = "composite"
    TEMPLATE = "template"

class VideoQualityEnum(str, Enum):
    """Video quality enumeration following FastAPI best practices."""
    LOW = "low"          # 480p
    MEDIUM = "medium"    # 720p
    HIGH = "high"        # 1080p
    ULTRA = "ultra"      # 4K

class VideoFormatEnum(str, Enum):
    """Video format enumeration following FastAPI best practices."""
    MP4 = "mp4"
    AVI = "avi"
    MOV = "mov"
    WEBM = "webm"
    MKV = "mkv"

class VideoAspectRatioEnum(str, Enum):
    """Video aspect ratio enumeration following FastAPI best practices."""
    SQUARE = "1:1"       # 1:1
    PORTRAIT = "9:16"    # 9:16
    LANDSCAPE = "16:9"   # 16:9
    CUSTOM = "custom"    # Custom ratio

# =============================================================================
# Video Base Models
# =============================================================================

class VideoBase(BaseModel):
    """Base video model following FastAPI best practices."""
    
    title: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Video title",
        example="My Amazing AI Video"
    )
    
    description: Optional[str] = Field(
        None,
        max_length=1000,
        description="Video description",
        example="A beautiful AI-generated video showcasing amazing visuals"
    )
    
    video_type: VideoTypeEnum = Field(
        ...,
        description="Type of video",
        example=VideoTypeEnum.AI_GENERATED
    )
    
    quality: VideoQualityEnum = Field(
        default=VideoQualityEnum.HIGH,
        description="Video quality",
        example=VideoQualityEnum.HIGH
    )
    
    format: VideoFormatEnum = Field(
        default=VideoFormatEnum.MP4,
        description="Video format",
        example=VideoFormatEnum.MP4
    )
    
    aspect_ratio: VideoAspectRatioEnum = Field(
        default=VideoAspectRatioEnum.LANDSCAPE,
        description="Video aspect ratio",
        example=VideoAspectRatioEnum.LANDSCAPE
    )
    
    tags: List[str] = Field(
        default_factory=list,
        max_items=20,
        description="Video tags for categorization",
        example=["ai", "creative", "art"]
    )
    
    is_public: bool = Field(
        default=False,
        description="Whether video is publicly accessible",
        example=False
    )
    
    thumbnail_url: Optional[HttpUrl] = Field(
        None,
        description="URL to video thumbnail",
        example="https://example.com/thumbnails/video_123.jpg"
    )
    
    class Config:
        """Pydantic configuration following FastAPI best practices."""
        schema_extra = {
            "example": {
                "title": "My Amazing AI Video",
                "description": "A beautiful AI-generated video showcasing amazing visuals",
                "video_type": "ai_generated",
                "quality": "high",
                "format": "mp4",
                "aspect_ratio": "16:9",
                "tags": ["ai", "creative", "art"],
                "is_public": False,
                "thumbnail_url": "https://example.com/thumbnails/video_123.jpg"
            }
        }

# =============================================================================
# Video Request Models
# =============================================================================

class VideoCreate(VideoBase):
    """Video creation model following FastAPI best practices."""
    
    processing_options: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Video processing options and parameters",
        example={
            "style": "cinematic",
            "duration": 30,
            "background_music": True,
            "voice_over": False
        }
    )
    
    template_id: Optional[str] = Field(
        None,
        description="Template ID if using a video template",
        example="template_123"
    )
    
    @validator('tags')
    def validate_tags(cls, v) -> bool:
        """Tags validation following FastAPI best practices."""
        if len(v) > 20:
            raise ValueError('Maximum 20 tags allowed')
        
        # Validate tag format (alphanumeric, underscore, hyphen)
        for tag in v:
            if not re.match(r'^[a-zA-Z0-9_-]+$', tag):
                raise ValueError(f'Invalid tag format: {tag}')
        
        return v
    
    class Config:
        """Pydantic configuration following FastAPI best practices."""
        schema_extra = {
            "example": {
                "title": "My Amazing AI Video",
                "description": "A beautiful AI-generated video showcasing amazing visuals",
                "video_type": "ai_generated",
                "quality": "high",
                "format": "mp4",
                "aspect_ratio": "16:9",
                "tags": ["ai", "creative", "art"],
                "is_public": False,
                "processing_options": {
                    "style": "cinematic",
                    "duration": 30,
                    "background_music": True,
                    "voice_over": False
                },
                "template_id": "template_123"
            }
        }

class VideoUpdate(BaseModel):
    """Video update model following FastAPI best practices."""
    
    title: Optional[str] = Field(
        None,
        min_length=1,
        max_length=200,
        description="Video title",
        example="Updated Video Title"
    )
    
    description: Optional[str] = Field(
        None,
        max_length=1000,
        description="Video description",
        example="Updated video description"
    )
    
    quality: Optional[VideoQualityEnum] = Field(
        None,
        description="Video quality",
        example=VideoQualityEnum.ULTRA
    )
    
    format: Optional[VideoFormatEnum] = Field(
        None,
        description="Video format",
        example=VideoFormatEnum.MP4
    )
    
    aspect_ratio: Optional[VideoAspectRatioEnum] = Field(
        None,
        description="Video aspect ratio",
        example=VideoAspectRatioEnum.LANDSCAPE
    )
    
    tags: Optional[List[str]] = Field(
        None,
        max_items=20,
        description="Video tags for categorization",
        example=["ai", "creative", "art", "updated"]
    )
    
    is_public: Optional[bool] = Field(
        None,
        description="Whether video is publicly accessible",
        example=True
    )
    
    thumbnail_url: Optional[HttpUrl] = Field(
        None,
        description="URL to video thumbnail",
        example="https://example.com/thumbnails/video_123_updated.jpg"
    )
    
    @validator('tags')
    def validate_tags(cls, v) -> bool:
        """Tags validation following FastAPI best practices."""
        if v is not None:
            if len(v) > 20:
                raise ValueError('Maximum 20 tags allowed')
            
            # Validate tag format
            for tag in v:
                if not re.match(r'^[a-zA-Z0-9_-]+$', tag):
                    raise ValueError(f'Invalid tag format: {tag}')
        
        return v
    
    class Config:
        """Pydantic configuration following FastAPI best practices."""
        schema_extra = {
            "example": {
                "title": "Updated Video Title",
                "description": "Updated video description",
                "quality": "ultra",
                "tags": ["ai", "creative", "art", "updated"],
                "is_public": True
            }
        }

class VideoUpload(BaseModel):
    """Video upload model following FastAPI best practices."""
    
    title: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Video title",
        example="My Uploaded Video"
    )
    
    description: Optional[str] = Field(
        None,
        max_length=1000,
        description="Video description",
        example="A video I uploaded for processing"
    )
    
    file_size: conint(gt=0, le=100*1024*1024) = Field(
        ...,
        description="Video file size in bytes (max 100MB)",
        example=52428800  # 50MB
    )
    
    file_format: VideoFormatEnum = Field(
        ...,
        description="Video file format",
        example=VideoFormatEnum.MP4
    )
    
    duration: Optional[confloat(gt=0, le=3600)] = Field(
        None,
        description="Video duration in seconds (max 1 hour)",
        example=120.5
    )
    
    width: Optional[conint(gt=0)] = Field(
        None,
        description="Video width in pixels",
        example=1920
    )
    
    height: Optional[conint(gt=0)] = Field(
        None,
        description="Video height in pixels",
        example=1080
    )
    
    tags: List[str] = Field(
        default_factory=list,
        max_items=20,
        description="Video tags for categorization",
        example=["uploaded", "personal"]
    )
    
    is_public: bool = Field(
        default=False,
        description="Whether video is publicly accessible",
        example=False
    )
    
    @validator('file_size')
    def validate_file_size(cls, v) -> bool:
        """File size validation following FastAPI best practices."""
        return validate_file_size(v, max_size_mb=100)
    
    @validator('duration')
    def validate_duration(cls, v) -> bool:
        """Duration validation following FastAPI best practices."""
        if v is not None:
            return validate_video_duration(v, max_duration=3600)
        return v
    
    @validator('tags')
    def validate_tags(cls, v) -> bool:
        """Tags validation following FastAPI best practices."""
        if len(v) > 20:
            raise ValueError('Maximum 20 tags allowed')
        
        for tag in v:
            if not re.match(r'^[a-zA-Z0-9_-]+$', tag):
                raise ValueError(f'Invalid tag format: {tag}')
        
        return v
    
    class Config:
        """Pydantic configuration following FastAPI best practices."""
        schema_extra = {
            "example": {
                "title": "My Uploaded Video",
                "description": "A video I uploaded for processing",
                "file_size": 52428800,
                "file_format": "mp4",
                "duration": 120.5,
                "width": 1920,
                "height": 1080,
                "tags": ["uploaded", "personal"],
                "is_public": False
            }
        }

class VideoProcessingOptions(BaseModel):
    """Video processing options model following FastAPI best practices."""
    
    style: Optional[str] = Field(
        None,
        description="Video style/theme",
        example="cinematic"
    )
    
    duration: Optional[conint(gt=0, le=300)] = Field(
        None,
        description="Target video duration in seconds (max 5 minutes)",
        example=30
    )
    
    background_music: bool = Field(
        default=True,
        description="Whether to add background music",
        example=True
    )
    
    voice_over: bool = Field(
        default=False,
        description="Whether to add voice over",
        example=False
    )
    
    voice_language: Optional[str] = Field(
        None,
        description="Voice over language",
        example="en-US"
    )
    
    voice_gender: Optional[str] = Field(
        None,
        description="Voice gender",
        example="female"
    )
    
    text_overlay: bool = Field(
        default=False,
        description="Whether to add text overlays",
        example=False
    )
    
    custom_prompts: List[str] = Field(
        default_factory=list,
        description="Custom AI prompts for video generation",
        example=["A beautiful sunset over mountains", "People walking in a city"]
    )
    
    color_scheme: Optional[str] = Field(
        None,
        description="Color scheme for the video",
        example="warm"
    )
    
    transition_effects: List[str] = Field(
        default_factory=list,
        description="Transition effects to apply",
        example=["fade", "slide"]
    )
    
    @validator('duration')
    def validate_duration(cls, v) -> bool:
        """Duration validation following FastAPI best practices."""
        if v is not None:
            if v > 300:
                raise ValueError('Duration must be 5 minutes or less')
        return v
    
    @validator('custom_prompts')
    def validate_prompts(cls, v) -> bool:
        """Prompts validation following FastAPI best practices."""
        if len(v) > 10:
            raise ValueError('Maximum 10 custom prompts allowed')
        
        for prompt in v:
            if len(prompt) > 500:
                raise ValueError('Prompt must be 500 characters or less')
        
        return v
    
    class Config:
        """Pydantic configuration following FastAPI best practices."""
        schema_extra = {
            "example": {
                "style": "cinematic",
                "duration": 30,
                "background_music": True,
                "voice_over": False,
                "voice_language": "en-US",
                "voice_gender": "female",
                "text_overlay": False,
                "custom_prompts": [
                    "A beautiful sunset over mountains",
                    "People walking in a city"
                ],
                "color_scheme": "warm",
                "transition_effects": ["fade", "slide"]
            }
        }

# =============================================================================
# Video Response Models
# =============================================================================

class VideoResponse(IdentifiedModel, VideoBase):
    """Video response model following FastAPI best practices."""
    
    id: int = Field(
        ...,
        description="Video unique identifier",
        example=1
    )
    
    user_id: int = Field(
        ...,
        description="User ID who created the video",
        example=1
    )
    
    status: VideoStatusEnum = Field(
        ...,
        description="Video processing status",
        example=VideoStatusEnum.COMPLETED
    )
    
    file_path: Optional[str] = Field(
        None,
        description="Path to video file",
        example="/uploads/videos/video_123.mp4"
    )
    
    file_size: Optional[int] = Field(
        None,
        description="Video file size in bytes",
        example=52428800
    )
    
    original_filename: Optional[str] = Field(
        None,
        description="Original uploaded filename",
        example="my_video.mp4"
    )
    
    output_path: Optional[str] = Field(
        None,
        description="Path to processed video file",
        example="/processed/videos/video_123_final.mp4"
    )
    
    output_filename: Optional[str] = Field(
        None,
        description="Processed video filename",
        example="video_123_final.mp4"
    )
    
    output_size: Optional[int] = Field(
        None,
        description="Processed video file size in bytes",
        example=31457280
    )
    
    duration: Optional[float] = Field(
        None,
        description="Video duration in seconds",
        example=30.5
    )
    
    width: Optional[int] = Field(
        None,
        description="Video width in pixels",
        example=1920
    )
    
    height: Optional[int] = Field(
        None,
        description="Video height in pixels",
        example=1080
    )
    
    fps: Optional[float] = Field(
        None,
        description="Video frames per second",
        example=30.0
    )
    
    bitrate: Optional[int] = Field(
        None,
        description="Video bitrate in kbps",
        example=5000
    )
    
    external_job_id: Optional[str] = Field(
        None,
        description="External processing job ID",
        example="job_abc123"
    )
    
    processing_started_at: Optional[datetime] = Field(
        None,
        description="When video processing started",
        example="2024-01-15T10:30:00Z"
    )
    
    processing_completed_at: Optional[datetime] = Field(
        None,
        description="When video processing completed",
        example="2024-01-15T10:35:00Z"
    )
    
    error_message: Optional[str] = Field(
        None,
        description="Error message if processing failed",
        example="Processing failed due to invalid input"
    )
    
    processing_options: Optional[Dict[str, Any]] = Field(
        None,
        description="Processing options used",
        example={
            "style": "cinematic",
            "duration": 30,
            "background_music": True
        }
    )
    
    download_url: Optional[HttpUrl] = Field(
        None,
        description="URL to download the video",
        example="https://example.com/downloads/video_123.mp4"
    )
    
    preview_url: Optional[HttpUrl] = Field(
        None,
        description="URL to video preview",
        example="https://example.com/previews/video_123.mp4"
    )
    
    created_at: datetime = Field(
        ...,
        description="Video creation timestamp",
        example="2024-01-15T10:00:00Z"
    )
    
    updated_at: Optional[datetime] = Field(
        None,
        description="Video last update timestamp",
        example="2024-01-15T10:35:00Z"
    )
    
    class Config:
        """Pydantic configuration following FastAPI best practices."""
        orm_mode = True
        schema_extra = {
            "example": {
                "id": 1,
                "title": "My Amazing AI Video",
                "description": "A beautiful AI-generated video showcasing amazing visuals",
                "video_type": "ai_generated",
                "quality": "high",
                "format": "mp4",
                "aspect_ratio": "16:9",
                "tags": ["ai", "creative", "art"],
                "is_public": False,
                "thumbnail_url": "https://example.com/thumbnails/video_123.jpg",
                "user_id": 1,
                "status": "completed",
                "file_path": "/uploads/videos/video_123.mp4",
                "file_size": 52428800,
                "output_path": "/processed/videos/video_123_final.mp4",
                "output_filename": "video_123_final.mp4",
                "output_size": 31457280,
                "duration": 30.5,
                "width": 1920,
                "height": 1080,
                "fps": 30.0,
                "bitrate": 5000,
                "external_job_id": "job_abc123",
                "processing_started_at": "2024-01-15T10:30:00Z",
                "processing_completed_at": "2024-01-15T10:35:00Z",
                "processing_options": {
                    "style": "cinematic",
                    "duration": 30,
                    "background_music": True
                },
                "download_url": "https://example.com/downloads/video_123.mp4",
                "preview_url": "https://example.com/previews/video_123.mp4",
                "created_at": "2024-01-15T10:00:00Z",
                "updated_at": "2024-01-15T10:35:00Z"
            }
        }

class VideoListResponse(BaseModel):
    """Video list response model following FastAPI best practices."""
    
    videos: List[VideoResponse] = Field(
        ...,
        description="List of videos"
    )
    
    total_count: int = Field(
        ...,
        description="Total number of videos",
        example=100
    )
    
    page: int = Field(
        ...,
        description="Current page number",
        example=1
    )
    
    page_size: int = Field(
        ...,
        description="Number of videos per page",
        example=20
    )
    
    total_pages: int = Field(
        ...,
        description="Total number of pages",
        example=5
    )
    
    has_next: bool = Field(
        ...,
        description="Whether there are more pages",
        example=True
    )
    
    has_prev: bool = Field(
        ...,
        description="Whether there are previous pages",
        example=False
    )
    
    class Config:
        """Pydantic configuration following FastAPI best practices."""
        schema_extra = {
            "example": {
                "videos": [
                    {
                        "id": 1,
                        "title": "My Amazing AI Video",
                        "video_type": "ai_generated",
                        "status": "completed",
                        "user_id": 1,
                        "created_at": "2024-01-15T10:00:00Z"
                    }
                ],
                "total_count": 100,
                "page": 1,
                "page_size": 20,
                "total_pages": 5,
                "has_next": True,
                "has_prev": False
            }
        }

class VideoStatusResponse(BaseModel):
    """Video status response model following FastAPI best practices."""
    
    video_id: int = Field(
        ...,
        description="Video ID",
        example=1
    )
    
    status: VideoStatusEnum = Field(
        ...,
        description="Current video status",
        example=VideoStatusEnum.PROCESSING
    )
    
    progress: Optional[float] = Field(
        None,
        ge=0.0,
        le=100.0,
        description="Processing progress percentage",
        example=75.5
    )
    
    estimated_completion: Optional[datetime] = Field(
        None,
        description="Estimated completion time",
        example="2024-01-15T10:35:00Z"
    )
    
    current_step: Optional[str] = Field(
        None,
        description="Current processing step",
        example="Rendering video"
    )
    
    total_steps: Optional[int] = Field(
        None,
        description="Total number of processing steps",
        example=5
    )
    
    current_step_number: Optional[int] = Field(
        None,
        description="Current step number",
        example=3
    )
    
    error_message: Optional[str] = Field(
        None,
        description="Error message if processing failed",
        example="Processing failed due to invalid input"
    )
    
    external_status: Optional[Dict[str, Any]] = Field(
        None,
        description="External processing service status",
        example={
            "job_id": "job_abc123",
            "status": "processing",
            "progress": 75.5
        }
    )
    
    class Config:
        """Pydantic configuration following FastAPI best practices."""
        schema_extra = {
            "example": {
                "video_id": 1,
                "status": "processing",
                "progress": 75.5,
                "estimated_completion": "2024-01-15T10:35:00Z",
                "current_step": "Rendering video",
                "total_steps": 5,
                "current_step_number": 3,
                "external_status": {
                    "job_id": "job_abc123",
                    "status": "processing",
                    "progress": 75.5
                }
            }
        }

# =============================================================================
# Video Analytics Models
# =============================================================================

class VideoAnalytics(BaseModel):
    """Video analytics model following FastAPI best practices."""
    
    video_id: int = Field(
        ...,
        description="Video ID",
        example=1
    )
    
    views: int = Field(
        default=0,
        description="Number of video views",
        example=150
    )
    
    unique_viewers: int = Field(
        default=0,
        description="Number of unique viewers",
        example=120
    )
    
    avg_view_duration: float = Field(
        default=0.0,
        description="Average view duration in seconds",
        example=25.5
    )
    
    completion_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Video completion rate percentage",
        example=85.2
    )
    
    likes: int = Field(
        default=0,
        description="Number of likes",
        example=25
    )
    
    dislikes: int = Field(
        default=0,
        description="Number of dislikes",
        example=2
    )
    
    shares: int = Field(
        default=0,
        description="Number of shares",
        example=10
    )
    
    downloads: int = Field(
        default=0,
        description="Number of downloads",
        example=5
    )
    
    processing_time: Optional[float] = Field(
        None,
        description="Total processing time in seconds",
        example=300.5
    )
    
    file_size_reduction: Optional[float] = Field(
        None,
        ge=0.0,
        le=100.0,
        description="File size reduction percentage after processing",
        example=40.2
    )
    
    quality_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=100.0,
        description="AI-generated quality score",
        example=92.5
    )
    
    engagement_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=100.0,
        description="Overall engagement score",
        example=78.3
    )
    
    class Config:
        """Pydantic configuration following FastAPI best practices."""
        schema_extra = {
            "example": {
                "video_id": 1,
                "views": 150,
                "unique_viewers": 120,
                "avg_view_duration": 25.5,
                "completion_rate": 85.2,
                "likes": 25,
                "dislikes": 2,
                "shares": 10,
                "downloads": 5,
                "processing_time": 300.5,
                "file_size_reduction": 40.2,
                "quality_score": 92.5,
                "engagement_score": 78.3
            }
        }

# =============================================================================
# Export
# =============================================================================

__all__ = [
    "VideoStatusEnum",
    "VideoTypeEnum",
    "VideoQualityEnum",
    "VideoFormatEnum",
    "VideoAspectRatioEnum",
    "VideoBase",
    "VideoCreate",
    "VideoUpdate",
    "VideoUpload",
    "VideoProcessingOptions",
    "VideoResponse",
    "VideoListResponse",
    "VideoStatusResponse",
    "VideoAnalytics"
] 