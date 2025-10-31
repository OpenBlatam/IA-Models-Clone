from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Pydantic v2 Schemas for Video API
================================

Modern Pydantic v2 models with:
- Field validation
- Type safety
- Performance optimization
- Clear documentation
"""




class VideoQuality(str, Enum):
    """Video quality options."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"


class VideoFormat(str, Enum):
    """Video format options."""
    MP4 = "mp4"
    WEBM = "webm"
    AVI = "avi"


class ProcessingStatus(str, Enum):
    """Processing status options."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class VideoRequest(BaseModel):
    """Video generation request schema."""
    
    input_text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Text prompt for video generation",
        examples=["Create a video about artificial intelligence"]
    )
    
    user_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="User identifier",
        examples=["user_123"]
    )
    
    quality: VideoQuality = Field(
        default=VideoQuality.MEDIUM,
        description="Video quality setting"
    )
    
    format: VideoFormat = Field(
        default=VideoFormat.MP4,
        description="Output video format"
    )
    
    duration: int = Field(
        default=60,
        ge=5,
        le=600,
        description="Video duration in seconds"
    )
    
    plugins: Optional[List[str]] = Field(
        default=None,
        description="Optional list of plugin names to use",
        examples=[["background_music", "text_overlay"]]
    )
    
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata for processing"
    )
    
    @field_validator("input_text")
    @classmethod
    def validate_input_text(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Input text cannot be empty")
        return v.strip()
    
    @field_validator("plugins")
    @classmethod
    def validate_plugins(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is not None:
            if len(v) > 10:
                raise ValueError("Maximum 10 plugins allowed")
            if len(set(v)) != len(v):
                raise ValueError("Duplicate plugins not allowed")
        return v


class VideoResponse(BaseModel):
    """Video generation response schema."""
    
    request_id: str = Field(
        ...,
        description="Unique request identifier"
    )
    
    status: ProcessingStatus = Field(
        ...,
        description="Current processing status"
    )
    
    output_url: Optional[str] = Field(
        default=None,
        description="URL to generated video (when completed)"
    )
    
    thumbnail_url: Optional[str] = Field(
        default=None,
        description="URL to video thumbnail"
    )
    
    duration: Optional[float] = Field(
        default=None,
        description="Actual video duration in seconds"
    )
    
    quality: Optional[VideoQuality] = Field(
        default=None,
        description="Video quality used"
    )
    
    format: Optional[VideoFormat] = Field(
        default=None,
        description="Video format used"
    )
    
    file_size: Optional[int] = Field(
        default=None,
        description="File size in bytes"
    )
    
    processing_time: Optional[float] = Field(
        default=None,
        description="Processing time in seconds"
    )
    
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if failed"
    )
    
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Creation timestamp"
    )
    
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Last update timestamp"
    )
    
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional response metadata"
    )


class BatchVideoRequest(BaseModel):
    """Batch video status request schema."""
    
    request_ids: List[str] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="List of request IDs to query"
    )
    
    @field_validator("request_ids")
    @classmethod
    async def validate_request_ids(cls, v: List[str]) -> List[str]:
        if len(set(v)) != len(v):
            raise ValueError("Duplicate request IDs not allowed")
        return v


class BatchVideoResponse(BaseModel):
    """Batch video status response schema."""
    
    results: Dict[str, VideoResponse] = Field(
        ...,
        description="Map of request_id to video response"
    )
    
    success_count: int = Field(
        ...,
        description="Number of successful queries"
    )
    
    error_count: int = Field(
        ...,
        description="Number of failed queries"
    )


class VideoLogEntry(BaseModel):
    """Video processing log entry schema."""
    
    timestamp: datetime = Field(
        ...,
        description="Log entry timestamp"
    )
    
    level: str = Field(
        ...,
        description="Log level (info, warning, error)"
    )
    
    message: str = Field(
        ...,
        description="Log message"
    )
    
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional log details"
    )


class VideoLogsResponse(BaseModel):
    """Video logs response schema."""
    
    request_id: str = Field(
        ...,
        description="Request identifier"
    )
    
    logs: List[VideoLogEntry] = Field(
        ...,
        description="List of log entries"
    )
    
    total_count: int = Field(
        ...,
        description="Total number of log entries"
    )
    
    has_more: bool = Field(
        ...,
        description="Whether there are more logs available"
    )


class APIResponse(BaseModel):
    """Standard API response wrapper."""
    
    success: bool = Field(
        ...,
        description="Whether the operation was successful"
    )
    
    data: Optional[Any] = Field(
        default=None,
        description="Response data"
    )
    
    error: Optional[str] = Field(
        default=None,
        description="Error message if failed"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Response timestamp"
    )
    
    request_id: Optional[str] = Field(
        default=None,
        description="Request tracking ID"
    )
    
    processing_time_ms: Optional[float] = Field(
        default=None,
        description="Processing time in milliseconds"
    )


class HealthResponse(BaseModel):
    """Health check response schema."""
    
    status: str = Field(
        ...,
        description="Service status"
    )
    
    version: str = Field(
        ...,
        description="API version"
    )
    
    uptime: float = Field(
        ...,
        description="Service uptime in seconds"
    )
    
    components: Dict[str, bool] = Field(
        ...,
        description="Component health status"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Health check timestamp"
    )


class MetricsResponse(BaseModel):
    """Metrics response schema."""
    
    total_requests: int = Field(
        ...,
        description="Total requests processed"
    )
    
    successful_requests: int = Field(
        ...,
        description="Successful requests"
    )
    
    failed_requests: int = Field(
        ...,
        description="Failed requests"
    )
    
    average_processing_time: float = Field(
        ...,
        description="Average processing time in seconds"
    )
    
    active_requests: int = Field(
        ...,
        description="Currently active requests"
    )
    
    cache_hit_rate: float = Field(
        ...,
        description="Cache hit rate percentage"
    )
    
    memory_usage: Optional[float] = Field(
        default=None,
        description="Memory usage percentage"
    )
    
    cpu_usage: Optional[float] = Field(
        default=None,
        description="CPU usage percentage"
    ) 