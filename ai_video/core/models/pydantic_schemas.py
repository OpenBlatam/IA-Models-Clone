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

from pydantic import (
from typing import Optional, List, Dict, Any, Union, Literal
from datetime import datetime, timedelta
from enum import Enum
import re
import uuid
from pathlib import Path
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Pydantic Schemas for AI Video System
===================================

Comprehensive Pydantic BaseModel schemas for consistent input/output validation
and response schemas in the AI Video system.

Features:
- Advanced validation with custom validators
- Nested model relationships
- Response schemas with proper serialization
- Input sanitization and transformation
- Error handling with detailed validation messages
- Type safety and documentation
"""

    BaseModel, Field, ConfigDict, validator, root_validator,
    ValidationError, computed_field, field_validator, model_validator
)

# =============================================================================
# ENUMERATIONS
# =============================================================================

class VideoStatus(str, Enum):
    """Video processing status enumeration."""
    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"

class VideoFormat(str, Enum):
    """Supported video formats."""
    MP4 = "mp4"
    AVI = "avi"
    MOV = "mov"
    WEBM = "webm"
    GIF = "gif"

class QualityLevel(str, Enum):
    """Video quality levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"

class ProcessingPriority(str, Enum):
    """Processing priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"

class ModelType(str, Enum):
    """AI model types."""
    STABLE_DIFFUSION = "stable_diffusion"
    MIDJOURNEY = "midjourney"
    DALLE = "dalle"
    CUSTOM = "custom"

# =============================================================================
# BASE CONFIGURATION
# =============================================================================

class BaseConfig:
    """Base configuration for all models."""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
        use_enum_values=True,
        json_encoders={
            datetime: lambda v: v.isoformat(),
            Path: str
        },
        populate_by_name=True,
        validate_default=True
    )

# =============================================================================
# INPUT VALIDATION MODELS
# =============================================================================

class VideoGenerationInput(BaseModel, BaseConfig):
    """Input model for video generation requests."""
    
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Text prompt describing the video to generate",
        examples=["A beautiful sunset over mountains", "A cat playing with a ball"]
    )
    
    negative_prompt: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Negative prompt to avoid certain elements",
        examples=["blurry, low quality, distorted"]
    )
    
    num_frames: int = Field(
        default=16,
        ge=8,
        le=128,
        description="Number of video frames to generate"
    )
    
    height: int = Field(
        default=512,
        ge=256,
        le=1024,
        description="Video height in pixels"
    )
    
    width: int = Field(
        default=512,
        ge=256,
        le=1024,
        description="Video width in pixels"
    )
    
    fps: int = Field(
        default=8,
        ge=1,
        le=60,
        description="Frames per second"
    )
    
    guidance_scale: float = Field(
        default=7.5,
        ge=1.0,
        le=20.0,
        description="Guidance scale for generation"
    )
    
    num_inference_steps: int = Field(
        default=50,
        ge=10,
        le=200,
        description="Number of inference steps"
    )
    
    quality: QualityLevel = Field(
        default=QualityLevel.MEDIUM,
        description="Quality level for generation"
    )
    
    format: VideoFormat = Field(
        default=VideoFormat.MP4,
        description="Output video format"
    )
    
    seed: Optional[int] = Field(
        default=None,
        ge=0,
        le=2**32-1,
        description="Random seed for reproducible generation"
    )
    
    model_type: ModelType = Field(
        default=ModelType.STABLE_DIFFUSION,
        description="AI model type to use"
    )
    
    priority: ProcessingPriority = Field(
        default=ProcessingPriority.NORMAL,
        description="Processing priority"
    )
    
    # Custom validators
    @field_validator('prompt')
    @classmethod
    def validate_prompt_content(cls, v: str) -> str:
        """Validate and sanitize prompt content."""
        if not v or not v.strip():
            raise ValueError('Prompt cannot be empty')
        
        # Remove excessive whitespace
        v = ' '.join(v.split())
        
        # Check for inappropriate content (basic filter)
        inappropriate_words = ['inappropriate', 'explicit', 'nsfw']
        if any(word in v.lower() for word in inappropriate_words):
            raise ValueError('Prompt contains inappropriate content')
        
        return v
    
    @field_validator('height', 'width')
    @classmethod
    def validate_dimensions(cls, v: int) -> int:
        """Validate video dimensions."""
        if v % 8 != 0:
            raise ValueError('Dimensions must be divisible by 8')
        return v
    
    @model_validator(mode='after')
    def validate_aspect_ratio(self) -> 'VideoGenerationInput':
        """Validate aspect ratio constraints."""
        aspect_ratio = self.width / self.height
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            raise ValueError('Aspect ratio must be between 0.5 and 2.0')
        return self
    
    @computed_field
    @property
    def total_pixels(self) -> int:
        """Calculate total pixels for the video."""
        return self.width * self.height
    
    @computed_field
    @property
    def estimated_size_mb(self) -> float:
        """Estimate video file size in MB."""
        # Rough estimation: pixels * frames * 3 bytes per pixel / 1M
        return (self.total_pixels * self.num_frames * 3) / (1024 * 1024)

class BatchGenerationInput(BaseModel, BaseConfig):
    """Input model for batch video generation."""
    
    requests: List[VideoGenerationInput] = Field(
        ...,
        min_length=1,
        max_length=20,
        description="List of video generation requests"
    )
    
    batch_name: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Optional name for the batch"
    )
    
    priority: ProcessingPriority = Field(
        default=ProcessingPriority.NORMAL,
        description="Overall batch priority"
    )
    
    @field_validator('requests')
    @classmethod
    def validate_batch_size(cls, v: List[VideoGenerationInput]) -> List[VideoGenerationInput]:
        """Validate batch size and content."""
        if len(v) > 20:
            raise ValueError('Maximum 20 requests allowed per batch')
        
        # Check for duplicate prompts
        prompts = [req.prompt for req in v]
        if len(prompts) != len(set(prompts)):
            raise ValueError('Duplicate prompts not allowed in batch')
        
        return v
    
    @computed_field
    @property
    def total_estimated_size_mb(self) -> float:
        """Calculate total estimated size for all videos."""
        return sum(req.estimated_size_mb for req in self.requests)

class VideoEditInput(BaseModel, BaseConfig):
    """Input model for video editing operations."""
    
    video_id: str = Field(..., description="ID of the video to edit")
    
    operation: Literal["trim", "resize", "filter", "speed", "loop"] = Field(
        ...,
        description="Type of editing operation"
    )
    
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Operation-specific parameters"
    )
    
    @field_validator('video_id')
    @classmethod
    def validate_video_id(cls, v: str) -> str:
        """Validate video ID format."""
        if not re.match(r'^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$', v):
            raise ValueError('Invalid video ID format')
        return v
    
    @field_validator('parameters')
    @classmethod
    def validate_operation_parameters(cls, v: Dict[str, Any], info) -> Dict[str, Any]:
        """Validate operation-specific parameters."""
        operation = info.data.get('operation')
        
        if operation == "trim":
            required_keys = ["start_time", "end_time"]
            if not all(key in v for key in required_keys):
                raise ValueError('Trim operation requires start_time and end_time')
        
        elif operation == "resize":
            required_keys = ["width", "height"]
            if not all(key in v for key in required_keys):
                raise ValueError('Resize operation requires width and height')
        
        elif operation == "speed":
            if "speed_factor" not in v:
                raise ValueError('Speed operation requires speed_factor')
            if not 0.1 <= v["speed_factor"] <= 10.0:
                raise ValueError('Speed factor must be between 0.1 and 10.0')
        
        return v

# =============================================================================
# RESPONSE MODELS
# =============================================================================

class VideoMetadata(BaseModel, BaseConfig):
    """Video metadata model."""
    
    video_id: str = Field(..., description="Unique video identifier")
    prompt: str = Field(..., description="Original generation prompt")
    negative_prompt: Optional[str] = Field(None, description="Negative prompt used")
    
    # Technical specifications
    width: int = Field(..., description="Video width in pixels")
    height: int = Field(..., description="Video height in pixels")
    fps: int = Field(..., description="Frames per second")
    num_frames: int = Field(..., description="Number of frames")
    duration: float = Field(..., description="Video duration in seconds")
    file_size: int = Field(..., description="File size in bytes")
    format: VideoFormat = Field(..., description="Video format")
    
    # Generation parameters
    guidance_scale: float = Field(..., description="Guidance scale used")
    num_inference_steps: int = Field(..., description="Inference steps used")
    seed: Optional[int] = Field(None, description="Random seed used")
    model_type: ModelType = Field(..., description="Model type used")
    quality: QualityLevel = Field(..., description="Quality level used")
    
    # Timestamps
    created_at: datetime = Field(..., description="Creation timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
    
    @computed_field
    @property
    def processing_time(self) -> Optional[float]:
        """Calculate processing time in seconds."""
        if self.completed_at:
            return (self.completed_at - self.created_at).total_seconds()
        return None
    
    @computed_field
    @property
    def resolution(self) -> str:
        """Get video resolution string."""
        return f"{self.width}x{self.height}"
    
    @computed_field
    @property
    def file_size_mb(self) -> float:
        """Get file size in MB."""
        return self.file_size / (1024 * 1024)

class VideoGenerationResponse(BaseModel, BaseConfig):
    """Response model for video generation."""
    
    video_id: str = Field(..., description="Unique video identifier")
    status: VideoStatus = Field(..., description="Current processing status")
    message: str = Field(..., description="Status message")
    
    # URLs and paths
    video_url: Optional[str] = Field(None, description="Video download URL")
    thumbnail_url: Optional[str] = Field(None, description="Thumbnail URL")
    preview_url: Optional[str] = Field(None, description="Preview URL")
    
    # Progress information
    progress: float = Field(..., ge=0.0, le=100.0, description="Progress percentage")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    
    # Error information
    error_code: Optional[str] = Field(None, description="Error code if failed")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    error_details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    
    # Metadata
    metadata: Optional[VideoMetadata] = Field(None, description="Video metadata")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    
    @computed_field
    @property
    def is_completed(self) -> bool:
        """Check if video generation is completed."""
        return self.status == VideoStatus.COMPLETED
    
    @computed_field
    @property
    def is_failed(self) -> bool:
        """Check if video generation failed."""
        return self.status == VideoStatus.FAILED
    
    @computed_field
    @property
    async def can_download(self) -> bool:
        """Check if video can be downloaded."""
        return self.is_completed and self.video_url is not None

class BatchGenerationResponse(BaseModel, BaseConfig):
    """Response model for batch generation."""
    
    batch_id: str = Field(..., description="Unique batch identifier")
    batch_name: Optional[str] = Field(None, description="Batch name")
    
    # Job information
    video_ids: List[str] = Field(..., description="List of video identifiers")
    total_videos: int = Field(..., description="Total number of videos")
    completed_videos: int = Field(..., ge=0, description="Number of completed videos")
    failed_videos: int = Field(..., ge=0, description="Number of failed videos")
    processing_videos: int = Field(..., ge=0, description="Number of processing videos")
    
    # Progress information
    overall_progress: float = Field(..., ge=0.0, le=100.0, description="Overall progress percentage")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    
    # Status
    status: VideoStatus = Field(..., description="Overall batch status")
    message: str = Field(..., description="Status message")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    
    @computed_field
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_videos == 0:
            return 0.0
        return self.completed_videos / self.total_videos
    
    @computed_field
    @property
    def is_completed(self) -> bool:
        """Check if batch is completed."""
        return self.status == VideoStatus.COMPLETED
    
    @computed_field
    @property
    def is_failed(self) -> bool:
        """Check if batch failed."""
        return self.status == VideoStatus.FAILED

class VideoEditResponse(BaseModel, BaseConfig):
    """Response model for video editing operations."""
    
    edit_id: str = Field(..., description="Unique edit identifier")
    original_video_id: str = Field(..., description="Original video ID")
    
    # Edit information
    operation: str = Field(..., description="Applied operation")
    parameters: Dict[str, Any] = Field(..., description="Applied parameters")
    
    # Result information
    status: VideoStatus = Field(..., description="Edit status")
    edited_video_url: Optional[str] = Field(None, description="Edited video URL")
    thumbnail_url: Optional[str] = Field(None, description="Thumbnail URL")
    
    # Processing information
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")

# =============================================================================
# SYSTEM MODELS
# =============================================================================

class SystemHealth(BaseModel, BaseConfig):
    """System health status model."""
    
    status: Literal["healthy", "degraded", "unhealthy"] = Field(..., description="Overall system status")
    version: str = Field(..., description="System version")
    uptime: float = Field(..., ge=0, description="System uptime in seconds")
    
    # Resource usage
    cpu_usage: float = Field(..., ge=0, le=100, description="CPU usage percentage")
    memory_usage: float = Field(..., ge=0, le=100, description="Memory usage percentage")
    gpu_usage: Optional[float] = Field(None, ge=0, le=100, description="GPU usage percentage")
    disk_usage: float = Field(..., ge=0, le=100, description="Disk usage percentage")
    
    # Performance metrics
    active_requests: int = Field(..., ge=0, description="Number of active requests")
    queue_size: int = Field(..., ge=0, description="Queue size")
    average_response_time: float = Field(..., ge=0, description="Average response time in seconds")
    
    # Service status
    database_status: Literal["healthy", "degraded", "unhealthy"] = Field(..., description="Database status")
    cache_status: Literal["healthy", "degraded", "unhealthy"] = Field(..., description="Cache status")
    storage_status: Literal["healthy", "degraded", "unhealthy"] = Field(..., description="Storage status")
    
    # Timestamp
    timestamp: datetime = Field(default_factory=datetime.now, description="Health check timestamp")
    
    @computed_field
    @property
    def is_healthy(self) -> bool:
        """Check if system is healthy."""
        return self.status == "healthy"
    
    @computed_field
    @property
    def resource_usage_critical(self) -> bool:
        """Check if resource usage is critical."""
        return (
            self.cpu_usage > 90 or
            self.memory_usage > 90 or
            self.disk_usage > 90 or
            (self.gpu_usage and self.gpu_usage > 95)
        )

class UserQuota(BaseModel, BaseConfig):
    """User quota and usage model."""
    
    user_id: str = Field(..., description="User identifier")
    
    # Daily limits
    daily_limit: int = Field(..., ge=0, description="Daily video generation limit")
    daily_used: int = Field(..., ge=0, description="Videos generated today")
    daily_reset: datetime = Field(..., description="Daily reset timestamp")
    
    # Monthly limits
    monthly_limit: int = Field(..., ge=0, description="Monthly video generation limit")
    monthly_used: int = Field(..., ge=0, description="Videos generated this month")
    monthly_reset: datetime = Field(..., description="Monthly reset timestamp")
    
    # Storage limits
    storage_limit_mb: int = Field(..., ge=0, description="Storage limit in MB")
    storage_used_mb: float = Field(..., ge=0, description="Storage used in MB")
    
    # Priority settings
    max_priority: ProcessingPriority = Field(..., description="Maximum allowed priority")
    
    @computed_field
    @property
    def daily_remaining(self) -> int:
        """Calculate remaining daily quota."""
        return max(0, self.daily_limit - self.daily_used)
    
    @computed_field
    @property
    def monthly_remaining(self) -> int:
        """Calculate remaining monthly quota."""
        return max(0, self.monthly_limit - self.monthly_used)
    
    @computed_field
    @property
    def storage_remaining_mb(self) -> float:
        """Calculate remaining storage in MB."""
        return max(0, self.storage_limit_mb - self.storage_used_mb)
    
    @computed_field
    @property
    def daily_usage_percentage(self) -> float:
        """Calculate daily usage percentage."""
        if self.daily_limit == 0:
            return 0.0
        return (self.daily_used / self.daily_limit) * 100
    
    @computed_field
    @property
    def monthly_usage_percentage(self) -> float:
        """Calculate monthly usage percentage."""
        if self.monthly_limit == 0:
            return 0.0
        return (self.monthly_used / self.monthly_limit) * 100
    
    @computed_field
    @property
    def storage_usage_percentage(self) -> float:
        """Calculate storage usage percentage."""
        if self.storage_limit_mb == 0:
            return 0.0
        return (self.storage_used_mb / self.storage_limit_mb) * 100
    
    @computed_field
    @property
    def can_generate_video(self) -> bool:
        """Check if user can generate a video."""
        return (
            self.daily_remaining > 0 and
            self.monthly_remaining > 0 and
            self.storage_remaining_mb > 10  # Minimum 10MB required
        )

class APIError(BaseModel, BaseConfig):
    """Standardized API error response."""
    
    error_code: str = Field(..., description="Error code")
    error_type: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    
    # Request information
    request_id: Optional[str] = Field(None, description="Request identifier for tracking")
    endpoint: Optional[str] = Field(None, description="API endpoint that caused the error")
    
    # Timestamp
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
    
    # Help information
    help_url: Optional[str] = Field(None, description="URL to help documentation")
    retry_after: Optional[int] = Field(None, ge=0, description="Retry after seconds")
    
    @computed_field
    @property
    def is_retryable(self) -> bool:
        """Check if the error is retryable."""
        retryable_codes = [
            "RATE_LIMIT_EXCEEDED",
            "SERVICE_UNAVAILABLE",
            "TIMEOUT",
            "TEMPORARY_ERROR"
        ]
        return self.error_code in retryable_codes

# =============================================================================
# VALIDATION UTILITIES
# =============================================================================

class ValidationUtils:
    """Utility class for validation helpers."""
    
    @staticmethod
    def validate_uuid(value: str) -> str:
        """Validate UUID format."""
        try:
            uuid.UUID(value)
            return value
        except ValueError:
            raise ValueError("Invalid UUID format")
    
    @staticmethod
    def validate_url(value: str) -> str:
        """Validate URL format."""
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        
        if not url_pattern.match(value):
            raise ValueError("Invalid URL format")
        return value
    
    @staticmethod
    def sanitize_filename(value: str) -> str:
        """Sanitize filename for safe storage."""
        # Remove or replace unsafe characters
        unsafe_chars = '<>:"/\\|?*'
        for char in unsafe_chars:
            value = value.replace(char, '_')
        
        # Limit length
        if len(value) > 255:
            value = value[:255]
        
        return value.strip()

# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_video_id() -> str:
    """Generate a unique video ID."""
    return str(uuid.uuid4())

def create_batch_id() -> str:
    """Generate a unique batch ID."""
    return f"batch_{uuid.uuid4().hex[:8]}"

def create_error_response(
    error_code: str,
    error_type: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None,
    endpoint: Optional[str] = None
) -> APIError:
    """Create a standardized error response."""
    return APIError(
        error_code=error_code,
        error_type=error_type,
        message=message,
        details=details,
        request_id=request_id,
        endpoint=endpoint
    )

def create_success_response(
    video_id: str,
    status: VideoStatus,
    message: str,
    **kwargs
) -> VideoGenerationResponse:
    """Create a standardized success response."""
    return VideoGenerationResponse(
        video_id=video_id,
        status=status,
        message=message,
        **kwargs
    )

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enumerations
    'VideoStatus',
    'VideoFormat', 
    'QualityLevel',
    'ProcessingPriority',
    'ModelType',
    
    # Input Models
    'VideoGenerationInput',
    'BatchGenerationInput',
    'VideoEditInput',
    
    # Response Models
    'VideoMetadata',
    'VideoGenerationResponse',
    'BatchGenerationResponse',
    'VideoEditResponse',
    
    # System Models
    'SystemHealth',
    'UserQuota',
    'APIError',
    
    # Utilities
    'ValidationUtils',
    'create_video_id',
    'create_batch_id',
    'create_error_response',
    'create_success_response'
] 