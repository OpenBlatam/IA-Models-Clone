from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
from uuid import UUID
from pydantic import Field, field_validator
from .base import AggregateRoot
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Video Entity
===========

Video entity representing video generation requests, processing status, and results.
"""





class VideoStatus(str, Enum):
    """Video processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class VideoQuality(str, Enum):
    """Video quality options."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"


class VideoFormat(str, Enum):
    """Video format options."""
    MP4 = "mp4"
    MOV = "mov"
    AVI = "avi"
    WEBM = "webm"


class ProcessingStage(str, Enum):
    """Video processing stages."""
    SCRIPT_GENERATION = "script_generation"
    AVATAR_CREATION = "avatar_creation"
    IMAGE_SYNC = "image_sync"
    VIDEO_COMPOSITION = "video_composition"
    FINAL_RENDER = "final_render"


class Video(AggregateRoot):
    """
    Video entity for video generation.
    
    Videos represent the complete video generation process including
    template selection, avatar creation, script generation, and final rendering.
    """
    
    # Basic information
    title: str = Field(..., min_length=1, max_length=100, description="Video title")
    description: Optional[str] = Field(None, max_length=500, description="Video description")
    
    # User and ownership
    user_id: UUID = Field(..., description="Video owner ID")
    creator_id: UUID = Field(..., description="Video creator ID")
    
    # Template and configuration
    template_id: UUID = Field(..., description="Template ID")
    avatar_id: UUID = Field(..., description="Avatar ID")
    
    # Video settings
    quality: VideoQuality = Field(default=VideoQuality.HIGH, description="Video quality")
    format: VideoFormat = Field(default=VideoFormat.MP4, description="Video format")
    aspect_ratio: str = Field(default="16:9", description="Video aspect ratio")
    duration: Optional[float] = Field(None, ge=0, description="Video duration in seconds")
    
    # Status and processing
    status: VideoStatus = Field(default=VideoStatus.PENDING, description="Video status")
    processing_stages: Dict[str, str] = Field(
        default={
            ProcessingStage.SCRIPT_GENERATION: VideoStatus.PENDING.value,
            ProcessingStage.AVATAR_CREATION: VideoStatus.PENDING.value,
            ProcessingStage.IMAGE_SYNC: VideoStatus.PENDING.value,
            ProcessingStage.VIDEO_COMPOSITION: VideoStatus.PENDING.value,
            ProcessingStage.FINAL_RENDER: VideoStatus.PENDING.value,
        },
        description="Processing stage statuses"
    )
    
    # Generated content
    script_content: Optional[str] = Field(None, description="Generated script content")
    avatar_video_url: Optional[str] = Field(None, description="Avatar video URL")
    final_video_url: Optional[str] = Field(None, description="Final video URL")
    thumbnail_url: Optional[str] = Field(None, description="Video thumbnail URL")
    
    # Image synchronization
    image_sync_config: Dict = Field(
        default={},
        description="Image synchronization configuration"
    )
    sync_timeline: Optional[List[Dict]] = Field(
        None,
        description="Image synchronization timeline"
    )
    
    # Processing metadata
    estimated_completion: Optional[datetime] = Field(
        None,
        description="Estimated completion time"
    )
    processing_time: Optional[float] = Field(
        None,
        ge=0,
        description="Actual processing time in seconds"
    )
    file_size: Optional[int] = Field(
        None,
        ge=0,
        description="Final video file size in bytes"
    )
    
    # Error handling
    error_message: Optional[str] = Field(None, description="Error message if failed")
    error_code: Optional[str] = Field(None, description="Error code if failed")
    retry_count: int = Field(default=0, description="Number of retry attempts")
    
    # Additional settings
    background_music: Optional[str] = Field(None, description="Background music URL")
    watermark: Optional[str] = Field(None, description="Watermark text or URL")
    custom_branding: Optional[Dict] = Field(None, description="Custom branding settings")
    
    # Metadata
    tags: List[str] = Field(default=[], description="Video tags")
    is_public: bool = Field(default=False, description="Public video flag")
    view_count: int = Field(default=0, description="Number of views")
    
    @field_validator("title")
    @classmethod
    def validate_title(cls, v: str) -> str:
        """Validate video title."""
        if not v.strip():
            raise ValueError("Video title cannot be empty")
        return v.strip()
    
    @field_validator("aspect_ratio")
    @classmethod
    def validate_aspect_ratio(cls, v: str) -> str:
        """Validate aspect ratio."""
        valid_ratios = ["16:9", "9:16", "1:1", "4:3", "3:4", "21:9"]
        if v not in valid_ratios:
            raise ValueError(f"Invalid aspect ratio: {v}")
        return v
    
    def _validate_entity(self) -> None:
        """Validate video business rules."""
        if self.status == VideoStatus.COMPLETED and not self.final_video_url:
            raise ValueError("Completed videos must have a final video URL")
        
        if self.retry_count > 3:
            raise ValueError("Maximum retry count exceeded")
    
    def start_processing(self) -> None:
        """Start video processing."""
        if self.status != VideoStatus.PENDING:
            raise ValueError("Can only start processing for pending videos")
        
        self.status = VideoStatus.PROCESSING
        self.mark_as_dirty()
    
    def complete_processing(self, video_url: str, duration: float, file_size: int) -> None:
        """Complete video processing."""
        self.status = VideoStatus.COMPLETED
        self.final_video_url = video_url
        self.duration = duration
        self.file_size = file_size
        self.processing_time = (datetime.utcnow() - self.created_at).total_seconds()
        self.mark_as_dirty()
    
    def fail_processing(self, error_message: str, error_code: str = None) -> None:
        """Mark video processing as failed."""
        self.status = VideoStatus.FAILED
        self.error_message = error_message
        self.error_code = error_code
        self.mark_as_dirty()
    
    def cancel_processing(self) -> None:
        """Cancel video processing."""
        if self.status not in [VideoStatus.PENDING, VideoStatus.PROCESSING]:
            raise ValueError("Can only cancel pending or processing videos")
        
        self.status = VideoStatus.CANCELLED
        self.mark_as_dirty()
    
    def retry_processing(self) -> None:
        """Retry video processing."""
        if self.status != VideoStatus.FAILED:
            raise ValueError("Can only retry failed videos")
        
        if self.retry_count >= 3:
            raise ValueError("Maximum retry count reached")
        
        self.status = VideoStatus.PENDING
        self.retry_count += 1
        self.error_message = None
        self.error_code = None
        self.mark_as_dirty()
    
    def update_stage_status(self, stage: ProcessingStage, status: VideoStatus) -> None:
        """Update processing stage status."""
        self.processing_stages[stage.value] = status.value
        self.mark_as_dirty()
    
    def set_script_content(self, content: str) -> None:
        """Set generated script content."""
        self.script_content = content
        self.update_stage_status(ProcessingStage.SCRIPT_GENERATION, VideoStatus.COMPLETED)
    
    def set_avatar_video(self, video_url: str) -> None:
        """Set avatar video URL."""
        self.avatar_video_url = video_url
        self.update_stage_status(ProcessingStage.AVATAR_CREATION, VideoStatus.COMPLETED)
    
    def set_sync_timeline(self, timeline: List[Dict]) -> None:
        """Set image synchronization timeline."""
        self.sync_timeline = timeline
        self.update_stage_status(ProcessingStage.IMAGE_SYNC, VideoStatus.COMPLETED)
    
    def increment_view_count(self) -> None:
        """Increment view count."""
        self.view_count += 1
        self.mark_as_dirty()
    
    def is_processing_complete(self) -> bool:
        """Check if all processing stages are complete."""
        return all(
            status == VideoStatus.COMPLETED.value
            for status in self.processing_stages.values()
        )
    
    def get_progress_percentage(self) -> float:
        """Get processing progress percentage."""
        completed_stages = sum(
            1 for status in self.processing_stages.values()
            if status == VideoStatus.COMPLETED.value
        )
        total_stages = len(self.processing_stages)
        return (completed_stages / total_stages) * 100
    
    def to_summary_dict(self) -> Dict:
        """Convert to summary dictionary for listings."""
        return {
            "id": str(self.id),
            "title": self.title,
            "description": self.description,
            "status": self.status.value,
            "thumbnail_url": self.thumbnail_url,
            "duration": self.duration,
            "created_at": self.created_at.isoformat(),
            "progress": self.get_progress_percentage(),
            "view_count": self.view_count,
        }
    
    def get_processing_log(self) -> List[Dict]:
        """Get processing log with timestamps."""
        return [
            {
                "stage": stage,
                "status": status,
                "timestamp": self.updated_at.isoformat(),
            }
            for stage, status in self.processing_stages.items()
        ] 