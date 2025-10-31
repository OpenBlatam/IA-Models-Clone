from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

from pydantic import BaseModel, Field, ConfigDict, computed_field, validator
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from enum import Enum
from pathlib import Path as FilePath
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
🚀 FASTAPI MODELS - AI VIDEO SYSTEM
===================================

Pydantic data models for the AI Video system following FastAPI best practices.
"""


# ============================================================================
# ENUMS
# ============================================================================

class VideoStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class VideoQuality(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"

class ProcessingPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"

# ============================================================================
# CORE MODELS
# ============================================================================

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
    priority: ProcessingPriority = Field(default=ProcessingPriority.NORMAL, description="Processing priority")
    
    description: Optional[str] = Field(default=None, max_length=1000, description="Video description")
    tags: List[str] = Field(default_factory=list, description="Video tags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @computed_field
    @property
    def duration_minutes(self) -> float:
        return self.duration / 60
    
    @computed_field
    @property
    def file_size_mb(self) -> float:
        quality_multiplier = {"low": 0.5, "medium": 1.0, "high": 2.0, "ultra": 4.0}
        return 10 * quality_multiplier[self.quality] * (self.duration / 60)
    
    @validator('title')
    @classmethod
    def validate_title(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError('Title cannot be empty')
        return v.strip()
    
    @validator('tags')
    @classmethod
    def validate_tags(cls, v: List[str]) -> List[str]:
        return [tag.strip().lower() for tag in v if tag.strip()]

class VideoResponse(BaseModel):
    video_id: str = Field(..., description="Video identifier")
    status: VideoStatus = Field(..., description="Processing status")
    message: str = Field(..., description="Status message")
    
    video_url: Optional[str] = Field(None, description="Video download URL")
    thumbnail_url: Optional[str] = Field(None, description="Thumbnail URL")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    
    @computed_field
    @property
    def is_completed(self) -> bool:
        return self.status == VideoStatus.COMPLETED
    
    @computed_field
    @property
    def is_failed(self) -> bool:
        return self.status == VideoStatus.FAILED

# ============================================================================
# BATCH MODELS
# ============================================================================

class BatchVideoRequest(BaseModel):
    videos: List[VideoData] = Field(..., min_length=1, max_length=100, description="List of videos to process")
    batch_name: Optional[str] = Field(default=None, max_length=100, description="Batch name")
    priority: ProcessingPriority = Field(default=ProcessingPriority.NORMAL, description="Batch priority")
    
    @validator('videos')
    @classmethod
    def validate_batch_size(cls, v: List[VideoData]) -> List[VideoData]:
        if len(v) > 100:
            raise ValueError("Batch size cannot exceed 100 videos")
        return v
    
    @computed_field
    @property
    def total_duration(self) -> float:
        return sum(video.duration for video in self.videos)
    
    @computed_field
    @property
    def estimated_processing_time(self) -> float:
        return self.total_duration / 60 * 30  # 30 seconds per minute

class BatchVideoResponse(BaseModel):
    batch_id: str = Field(..., description="Batch identifier")
    batch_name: Optional[str] = Field(None, description="Batch name")
    
    total_videos: int = Field(..., description="Total number of videos")
    completed_videos: int = Field(..., ge=0, description="Number of completed videos")
    failed_videos: int = Field(..., ge=0, description="Number of failed videos")
    processing_videos: int = Field(..., ge=0, description="Number of processing videos")
    
    overall_progress: float = Field(..., ge=0.0, le=100.0, description="Overall progress percentage")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    
    status: VideoStatus = Field(..., description="Overall batch status")
    message: str = Field(..., description="Status message")
    
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    
    @computed_field
    @property
    def success_rate(self) -> float:
        if self.total_videos == 0:
            return 0.0
        return self.completed_videos / self.total_videos

# ============================================================================
# RESPONSE MODELS
# ============================================================================

class ErrorResponse(BaseModel):
    error_code: str = Field(..., description="Error code")
    error_type: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request identifier for tracking")

class PaginationParams(BaseModel):
    skip: int = Field(0, ge=0, description="Number of items to skip")
    limit: int = Field(100, ge=1, le=1000, description="Maximum number of items to return")

class VideoListResponse(BaseModel):
    items: List[VideoResponse] = Field(..., description="List of videos")
    total: int = Field(..., description="Total number of videos")
    skip: int = Field(..., description="Number of items skipped")
    limit: int = Field(..., description="Number of items returned")
    has_next: bool = Field(..., description="Whether there are more items")
    has_previous: bool = Field(..., description="Whether there are previous items") 