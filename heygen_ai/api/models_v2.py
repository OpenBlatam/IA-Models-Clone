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

from typing import Dict, List, Optional, Union, Any, Annotated, Literal
from datetime import datetime, timezone
from decimal import Decimal
import re
import uuid
from pathlib import Path
from pydantic import (
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema
from enum import Enum
import structlog
        import re
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Enhanced Pydantic v2 Models for HeyGen AI API
Advanced models with computed fields, validators, serialization, and performance optimizations.
"""


    BaseModel, Field, HttpUrl, EmailStr, validator, root_validator,
    computed_field, model_validator, field_validator, ConfigDict,
    PlainSerializer, PlainValidator, BeforeValidator, AfterValidator,
    WithJsonSchema, GetJsonSchemaHandler, GetCoreSchemaHandler
)

logger = structlog.get_logger()

# =============================================================================
# Enhanced Enums with Metadata
# =============================================================================

class VideoStatus(str, Enum):
    """Video generation status with metadata."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    QUEUED = "queued"
    
    @classmethod
    def get_description(cls, value: str) -> str:
        """Get human-readable description for status."""
        descriptions = {
            "pending": "Video is waiting to be processed",
            "processing": "Video is currently being generated",
            "completed": "Video generation finished successfully",
            "failed": "Video generation failed",
            "cancelled": "Video generation was cancelled",
            "queued": "Video is in processing queue"
        }
        return descriptions.get(value, "Unknown status")


class LanguageCode(str, Enum):
    """Supported language codes with metadata."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    CHINESE = "zh"
    JAPANESE = "ja"
    KOREAN = "ko"
    RUSSIAN = "ru"
    ARABIC = "ar"
    HINDI = "hi"
    
    @classmethod
    def get_language_name(cls, code: str) -> str:
        """Get language name from code."""
        names = {
            "en": "English",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "it": "Italian",
            "pt": "Portuguese",
            "zh": "Chinese",
            "ja": "Japanese",
            "ko": "Korean",
            "ru": "Russian",
            "ar": "Arabic",
            "hi": "Hindi"
        }
        return names.get(code, "Unknown")


class VideoStyle(str, Enum):
    """Video style options with metadata."""
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    EDUCATIONAL = "educational"
    MARKETING = "marketing"
    ENTERTAINMENT = "entertainment"
    NEWS = "news"
    DOCUMENTARY = "documentary"
    TUTORIAL = "tutorial"
    
    @classmethod
    def get_style_description(cls, style: str) -> str:
        """Get style description."""
        descriptions = {
            "professional": "Formal, business-appropriate style",
            "casual": "Relaxed, conversational style",
            "educational": "Academic, instructional style",
            "marketing": "Persuasive, promotional style",
            "entertainment": "Fun, engaging style",
            "news": "Journalistic, factual style",
            "documentary": "Informative, narrative style",
            "tutorial": "Step-by-step instructional style"
        }
        return descriptions.get(style, "Unknown style")


class Resolution(str, Enum):
    """Video resolution options with metadata."""
    HD_720P = "720p"
    FULL_HD_1080P = "1080p"
    UHD_4K = "4k"
    UHD_8K = "8k"
    
    @classmethod
    def get_dimensions(cls, resolution: str) -> tuple[int, int]:
        """Get width and height for resolution."""
        dimensions = {
            "720p": (1280, 720),
            "1080p": (1920, 1080),
            "4k": (3840, 2160),
            "8k": (7680, 4320)
        }
        return dimensions.get(resolution, (1920, 1080))


class OutputFormat(str, Enum):
    """Video output format options with metadata."""
    MP4 = "mp4"
    MOV = "mov"
    AVI = "avi"
    WEBM = "webm"
    MKV = "mkv"
    
    @classmethod
    def get_mime_type(cls, format: str) -> str:
        """Get MIME type for format."""
        mime_types = {
            "mp4": "video/mp4",
            "mov": "video/quicktime",
            "avi": "video/x-msvideo",
            "webm": "video/webm",
            "mkv": "video/x-matroska"
        }
        return mime_types.get(format, "video/mp4")


class QualityLevel(str, Enum):
    """Video quality levels with metadata."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"
    
    @classmethod
    def get_bitrate(cls, quality: str, resolution: str) -> int:
        """Get recommended bitrate for quality and resolution."""
        bitrates = {
            ("low", "720p"): 1000000,
            ("low", "1080p"): 2000000,
            ("medium", "720p"): 2000000,
            ("medium", "1080p"): 4000000,
            ("high", "720p"): 4000000,
            ("high", "1080p"): 8000000,
            ("ultra", "720p"): 8000000,
            ("ultra", "1080p"): 16000000,
        }
        return bitrates.get((quality, resolution), 4000000)


# =============================================================================
# Custom Validators and Serializers
# =============================================================================

def validate_script_content(v: str) -> str:
    """Validate script content."""
    if not v or not v.strip():
        raise ValueError("Script cannot be empty")
    
    if len(v.strip()) < 10:
        raise ValueError("Script must be at least 10 characters long")
    
    if len(v) > 10000:
        raise ValueError("Script cannot exceed 10,000 characters")
    
    # Check for inappropriate content
    inappropriate_words = ["spam", "scam", "inappropriate"]
    if any(word in v.lower() for word in inappropriate_words):
        raise ValueError("Script contains inappropriate content")
    
    return v.strip()


def validate_voice_id(v: str) -> str:
    """Validate voice ID format."""
    if not re.match(r'^[a-zA-Z0-9_-]{3,50}$', v):
        raise ValueError("Voice ID must be 3-50 characters long and contain only letters, numbers, hyphens, and underscores")
    return v


def validate_avatar_id(v: str) -> str:
    """Validate avatar ID format."""
    if not re.match(r'^[a-zA-Z0-9_-]{3,50}$', v):
        raise ValueError("Avatar ID must be 3-50 characters long and contain only letters, numbers, hyphens, and underscores")
    return v


def validate_duration(v: int) -> int:
    """Validate video duration."""
    if v < 5:
        raise ValueError("Video duration must be at least 5 seconds")
    if v > 3600:
        raise ValueError("Video duration cannot exceed 1 hour (3600 seconds)")
    return v


def validate_file_size(v: int) -> int:
    """Validate file size in bytes."""
    if v < 0:
        raise ValueError("File size cannot be negative")
    if v > 100 * 1024 * 1024:  # 100MB limit
        raise ValueError("File size cannot exceed 100MB")
    return v


def serialize_datetime(v: datetime) -> str:
    """Serialize datetime to ISO format."""
    if v.tzinfo is None:
        v = v.replace(tzinfo=timezone.utc)
    return v.isoformat()


def serialize_decimal(v: Decimal) -> float:
    """Serialize Decimal to float."""
    return float(v)


# =============================================================================
# Base Models with Enhanced Configuration
# =============================================================================

class BaseHeyGenModel(BaseModel):
    """Base model with enhanced Pydantic v2 configuration."""
    
    model_config = ConfigDict(
        # Performance optimizations
        validate_assignment=True,
        validate_default=True,
        extra='forbid',  # Reject extra fields
        frozen=False,  # Allow mutation for now
        use_enum_values=True,
        
        # JSON configuration
        json_encoders={
            datetime: serialize_datetime,
            Decimal: serialize_decimal,
        },
        
        # Schema generation
        json_schema_extra={
            "examples": [],
            "additionalProperties": False
        },
        
        # Validation
        str_strip_whitespace=True,
        str_min_length=1,
        
        # Error handling
        error_msg_templates={
            'value_error.missing': 'This field is required',
            'value_error.any_str.min_length': 'Minimum length is {limit_value}',
            'value_error.any_str.max_length': 'Maximum length is {limit_value}',
        }
    )


class TimestampedModel(BaseHeyGenModel):
    """Base model with automatic timestamps."""
    
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last update timestamp"
    )
    
    @model_validator(mode='after')
    def update_timestamp(self) -> 'TimestampedModel':
        """Update timestamp on model changes."""
        self.updated_at = datetime.now(timezone.utc)
        return self


class IdentifiedModel(BaseHeyGenModel):
    """Base model with automatic ID generation."""
    
    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier"
    )
    
    @computed_field
    @property
    def short_id(self) -> str:
        """Get short version of ID."""
        return self.id[:8]


# =============================================================================
# Enhanced Request Models
# =============================================================================

class CreateVideoRequest(BaseHeyGenModel):
    """Enhanced request model for creating a video."""
    
    script: Annotated[str, PlainValidator(validate_script_content)] = Field(
        ..., 
        min_length=10, 
        max_length=10000,
        description="Script text for the video"
    )
    avatar_id: Annotated[str, PlainValidator(validate_avatar_id)] = Field(
        ..., 
        description="ID of the avatar to use"
    )
    voice_id: Annotated[str, PlainValidator(validate_voice_id)] = Field(
        ..., 
        description="ID of the voice to use"
    )
    language: LanguageCode = Field(
        default=LanguageCode.ENGLISH,
        description="Language code"
    )
    style: VideoStyle = Field(
        default=VideoStyle.PROFESSIONAL,
        description="Video style"
    )
    resolution: Resolution = Field(
        default=Resolution.FULL_HD_1080P,
        description="Video resolution"
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.MP4,
        description="Output video format"
    )
    quality: QualityLevel = Field(
        default=QualityLevel.HIGH,
        description="Video quality level"
    )
    duration: Annotated[Optional[int], PlainValidator(validate_duration)] = Field(
        None, 
        ge=5, 
        le=3600,
        description="Video duration in seconds (5-3600)"
    )
    background: Optional[HttpUrl] = Field(
        None, 
        description="Background image/video URL"
    )
    custom_settings: Dict[str, Any] = Field(
        default_factory=dict,
        description="Custom video settings"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )
    
    @computed_field
    @property
    def estimated_file_size(self) -> int:
        """Estimate file size based on duration and quality."""
        if not self.duration:
            return 0
        
        bitrate = QualityLevel.get_bitrate(self.quality, self.resolution)
        return int((bitrate * self.duration) / 8)  # Convert bits to bytes
    
    @computed_field
    @property
    def estimated_processing_time(self) -> int:
        """Estimate processing time in seconds."""
        base_time = 30  # Base processing time
        duration_factor = (self.duration or 60) / 60  # Factor based on duration
        quality_factor = {"low": 0.5, "medium": 1.0, "high": 1.5, "ultra": 2.0}[self.quality]
        
        return int(base_time * duration_factor * quality_factor)
    
    @computed_field
    @property
    def word_count(self) -> int:
        """Calculate word count from script."""
        return len(self.script.split())
    
    @computed_field
    @property
    def reading_time(self) -> float:
        """Estimate reading time in minutes."""
        words_per_minute = 150
        return self.word_count / words_per_minute
    
    @model_validator(mode='after')
    def validate_script_duration_match(self) -> 'CreateVideoRequest':
        """Validate that script length matches duration if specified."""
        if self.duration and self.reading_time > self.duration / 60:
            raise ValueError(
                f"Script is too long for {self.duration} seconds. "
                f"Estimated reading time: {self.reading_time:.1f} minutes"
            )
        return self
    
    @field_validator('custom_settings')
    @classmethod
    def validate_custom_settings(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate custom settings."""
        allowed_keys = {
            'fps', 'bitrate', 'audio_quality', 'background_music',
            'transitions', 'effects', 'subtitles', 'watermark'
        }
        
        invalid_keys = set(v.keys()) - allowed_keys
        if invalid_keys:
            raise ValueError(f"Invalid custom settings: {invalid_keys}")
        
        return v


class BatchCreateVideoRequest(BaseHeyGenModel):
    """Enhanced request model for batch video creation."""
    
    videos: List[CreateVideoRequest] = Field(
        ..., 
        min_length=1, 
        max_length=20,
        description="List of video requests"
    )
    batch_name: Optional[str] = Field(
        None,
        max_length=100,
        description="Name for the batch operation"
    )
    priority: Literal["low", "normal", "high"] = Field(
        default="normal",
        description="Processing priority"
    )
    notify_on_completion: bool = Field(
        default=True,
        description="Send notification when batch completes"
    )
    
    @computed_field
    @property
    def total_estimated_size(self) -> int:
        """Calculate total estimated file size."""
        return sum(video.estimated_file_size for video in self.videos)
    
    @computed_field
    @property
    def total_estimated_time(self) -> int:
        """Calculate total estimated processing time."""
        return sum(video.estimated_processing_time for video in self.videos)
    
    @computed_field
    @property
    def languages_used(self) -> List[str]:
        """Get list of languages used in the batch."""
        return list(set(video.language for video in self.videos))
    
    @model_validator(mode='after')
    def validate_batch_limits(self) -> 'BatchCreateVideoRequest':
        """Validate batch limits."""
        if self.total_estimated_size > 1024 * 1024 * 1024:  # 1GB
            raise ValueError("Total estimated file size exceeds 1GB limit")
        
        if self.total_estimated_time > 3600:  # 1 hour
            raise ValueError("Total estimated processing time exceeds 1 hour")
        
        return self


class GenerateScriptRequest(BaseHeyGenModel):
    """Enhanced request model for script generation."""
    
    topic: str = Field(
        ..., 
        min_length=3, 
        max_length=200,
        description="Topic for the script"
    )
    language: LanguageCode = Field(
        default=LanguageCode.ENGLISH,
        description="Language code"
    )
    style: VideoStyle = Field(
        default=VideoStyle.PROFESSIONAL,
        description="Script style"
    )
    target_duration: str = Field(
        default="2 minutes",
        pattern=r'^\d+\s*(seconds?|minutes?|hours?)$',
        description="Target duration (e.g., '2 minutes', '30 seconds')"
    )
    additional_context: Optional[str] = Field(
        None,
        max_length=1000,
        description="Additional context for script generation"
    )
    tone: Literal["formal", "casual", "friendly", "professional", "enthusiastic"] = Field(
        default="professional",
        description="Desired tone for the script"
    )
    include_hooks: bool = Field(
        default=True,
        description="Include attention-grabbing hooks"
    )
    include_call_to_action: bool = Field(
        default=True,
        description="Include call-to-action at the end"
    )
    
    @field_validator('target_duration')
    @classmethod
    def validate_target_duration(cls, v: str) -> str:
        """Validate and normalize target duration."""
        # Parse duration string
        match = re.match(r'^(\d+)\s*(seconds?|minutes?|hours?)$', v.lower())
        if not match:
            raise ValueError("Invalid duration format. Use 'X seconds/minutes/hours'")
        
        value, unit = match.groups()
        value = int(value)
        
        # Convert to seconds and validate
        if unit.startswith('hour'):
            total_seconds = value * 3600
        elif unit.startswith('minute'):
            total_seconds = value * 60
        else:
            total_seconds = value
        
        if total_seconds < 5:
            raise ValueError("Target duration must be at least 5 seconds")
        if total_seconds > 3600:
            raise ValueError("Target duration cannot exceed 1 hour")
        
        return v


class OptimizeScriptRequest(BaseHeyGenModel):
    """Enhanced request model for script optimization."""
    
    script: Annotated[str, PlainValidator(validate_script_content)] = Field(
        ...,
        description="Script text to optimize"
    )
    target_duration: str = Field(
        default="2 minutes",
        pattern=r'^\d+\s*(seconds?|minutes?|hours?)$',
        description="Target duration"
    )
    style: VideoStyle = Field(
        default=VideoStyle.PROFESSIONAL,
        description="Script style"
    )
    language: LanguageCode = Field(
        default=LanguageCode.ENGLISH,
        description="Language code"
    )
    optimization_goals: List[Literal["clarity", "engagement", "brevity", "impact"]] = Field(
        default=["clarity"],
        description="Optimization goals"
    )
    preserve_key_points: bool = Field(
        default=True,
        description="Preserve key points during optimization"
    )
    
    @computed_field
    @property
    def current_word_count(self) -> int:
        """Get current word count."""
        return len(self.script.split())
    
    @computed_field
    @property
    def current_reading_time(self) -> float:
        """Get current reading time in minutes."""
        words_per_minute = 150
        return self.current_word_count / words_per_minute


# =============================================================================
# Enhanced Response Models
# =============================================================================

class VideoResponse(TimestampedModel, IdentifiedModel):
    """Enhanced response model for video generation."""
    
    video_id: str = Field(..., description="Unique video ID")
    status: VideoStatus = Field(..., description="Video generation status")
    output_url: Optional[HttpUrl] = Field(None, description="URL to generated video")
    duration: Optional[float] = Field(None, ge=0, description="Video duration in seconds")
    file_size: Annotated[Optional[int], PlainValidator(validate_file_size)] = Field(
        None, 
        description="Video file size in bytes"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Video metadata"
    )
    error_message: Optional[str] = Field(None, description="Error message if failed")
    progress: Optional[float] = Field(None, ge=0, le=100, description="Generation progress percentage")
    processing_time: Optional[float] = Field(None, ge=0, description="Total processing time in seconds")
    
    # Original request data
    script: str = Field(..., description="Original script")
    avatar_id: str = Field(..., description="Avatar ID used")
    voice_id: str = Field(..., description="Voice ID used")
    language: LanguageCode = Field(..., description="Language used")
    style: VideoStyle = Field(..., description="Style used")
    resolution: Resolution = Field(..., description="Resolution used")
    output_format: OutputFormat = Field(..., description="Output format used")
    quality: QualityLevel = Field(..., description="Quality level used")
    
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
    def file_size_mb(self) -> Optional[float]:
        """Get file size in MB."""
        if self.file_size is None:
            return None
        return round(self.file_size / (1024 * 1024), 2)
    
    @computed_field
    @property
    def duration_formatted(self) -> Optional[str]:
        """Get formatted duration string."""
        if self.duration is None:
            return None
        
        minutes = int(self.duration // 60)
        seconds = int(self.duration % 60)
        return f"{minutes:02d}:{seconds:02d}"
    
    @computed_field
    @property
    def status_description(self) -> str:
        """Get human-readable status description."""
        return VideoStatus.get_description(self.status)
    
    @computed_field
    @property
    def language_name(self) -> str:
        """Get language name."""
        return LanguageCode.get_language_name(self.language)
    
    @computed_field
    @property
    def style_description(self) -> str:
        """Get style description."""
        return VideoStyle.get_style_description(self.style)
    
    @computed_field
    @property
    def resolution_dimensions(self) -> tuple[int, int]:
        """Get resolution dimensions."""
        return Resolution.get_dimensions(self.resolution)
    
    @computed_field
    @property
    def mime_type(self) -> str:
        """Get MIME type for output format."""
        return OutputFormat.get_mime_type(self.output_format)


class BatchVideoResponse(TimestampedModel, IdentifiedModel):
    """Enhanced response model for batch video creation."""
    
    batch_id: str = Field(..., description="Batch operation ID")
    videos: List[VideoResponse] = Field(..., description="List of video responses")
    total_count: int = Field(..., description="Total number of videos")
    completed_count: int = Field(..., description="Number of completed videos")
    failed_count: int = Field(..., description="Number of failed videos")
    processing_count: int = Field(..., description="Number of videos being processed")
    queued_count: int = Field(..., description="Number of videos in queue")
    batch_name: Optional[str] = Field(None, description="Batch name")
    priority: str = Field(..., description="Processing priority")
    
    @computed_field
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_count == 0:
            return 0.0
        return round((self.completed_count / self.total_count) * 100, 2)
    
    @computed_field
    @property
    def total_file_size(self) -> int:
        """Calculate total file size of completed videos."""
        return sum(video.file_size or 0 for video in self.videos if video.is_completed)
    
    @computed_field
    @property
    def total_duration(self) -> float:
        """Calculate total duration of completed videos."""
        return sum(video.duration or 0 for video in self.videos if video.is_completed)
    
    @computed_field
    @property
    def average_processing_time(self) -> float:
        """Calculate average processing time."""
        completed_videos = [v for v in self.videos if v.processing_time is not None]
        if not completed_videos:
            return 0.0
        return sum(v.processing_time for v in completed_videos) / len(completed_videos)
    
    @computed_field
    @property
    def languages_used(self) -> List[str]:
        """Get list of languages used."""
        return list(set(video.language for video in self.videos))
    
    @computed_field
    @property
    def styles_used(self) -> List[str]:
        """Get list of styles used."""
        return list(set(video.style for video in self.videos))
    
    @computed_field
    @property
    def is_completed(self) -> bool:
        """Check if batch is completed."""
        return self.completed_count + self.failed_count == self.total_count


class ScriptResponse(TimestampedModel, IdentifiedModel):
    """Enhanced response model for script generation."""
    
    script_id: str = Field(..., description="Unique script ID")
    script: str = Field(..., description="Generated script text")
    word_count: int = Field(..., description="Number of words in script")
    estimated_duration: float = Field(..., description="Estimated speaking duration")
    language: LanguageCode = Field(..., description="Script language")
    style: VideoStyle = Field(..., description="Script style")
    tone: str = Field(..., description="Script tone")
    readability_score: Optional[float] = Field(None, ge=0, le=100, description="Readability score")
    engagement_score: Optional[float] = Field(None, ge=0, le=100, description="Engagement score")
    keywords: List[str] = Field(default_factory=list, description="Extracted keywords")
    sentiment: Optional[Dict[str, float]] = Field(None, description="Sentiment analysis")
    
    @computed_field
    @property
    def reading_time_minutes(self) -> float:
        """Get reading time in minutes."""
        words_per_minute = 150
        return round(self.word_count / words_per_minute, 2)
    
    @computed_field
    @property
    def speaking_time_minutes(self) -> float:
        """Get speaking time in minutes."""
        return round(self.estimated_duration / 60, 2)
    
    @computed_field
    @property
    def language_name(self) -> str:
        """Get language name."""
        return LanguageCode.get_language_name(self.language)
    
    @computed_field
    @property
    def style_description(self) -> str:
        """Get style description."""
        return VideoStyle.get_style_description(self.style)
    
    @computed_field
    @property
    def is_optimized(self) -> bool:
        """Check if script is optimized."""
        return self.readability_score is not None and self.readability_score > 70


# =============================================================================
# Utility Models
# =============================================================================

class PaginationParams(BaseHeyGenModel):
    """Pagination parameters."""
    
    page: int = Field(default=1, ge=1, description="Page number")
    size: int = Field(default=20, ge=1, le=100, description="Page size")
    
    @computed_field
    @property
    def offset(self) -> int:
        """Calculate offset for database queries."""
        return (self.page - 1) * self.size


class PaginatedResponse(BaseHeyGenModel):
    """Generic paginated response."""
    
    items: List[Any] = Field(..., description="List of items")
    total: int = Field(..., description="Total number of items")
    page: int = Field(..., description="Current page")
    size: int = Field(..., description="Page size")
    
    @computed_field
    @property
    def total_pages(self) -> int:
        """Calculate total number of pages."""
        return (self.total + self.size - 1) // self.size
    
    @computed_field
    @property
    def has_next(self) -> bool:
        """Check if there's a next page."""
        return self.page < self.total_pages
    
    @computed_field
    @property
    def has_previous(self) -> bool:
        """Check if there's a previous page."""
        return self.page > 1


class ErrorResponse(BaseHeyGenModel):
    """Enhanced error response model."""
    
    error: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Error code")
    error_type: str = Field(..., description="Error type")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Error timestamp"
    )
    request_id: Optional[str] = Field(None, description="Request ID for tracking")
    user_id: Optional[str] = Field(None, description="User ID if available")
    
    @computed_field
    @property
    def is_user_error(self) -> bool:
        """Check if this is a user-correctable error."""
        user_errors = {"VALIDATION_ERROR", "AUTHENTICATION_ERROR", "RATE_LIMIT_ERROR"}
        return self.error_code in user_errors
    
    @computed_field
    @property
    def is_system_error(self) -> bool:
        """Check if this is a system error."""
        system_errors = {"DATABASE_ERROR", "EXTERNAL_SERVICE_ERROR", "SYSTEM_ERROR"}
        return self.error_code in system_errors


class HealthResponse(BaseHeyGenModel):
    """Enhanced health check response."""
    
    status: str = Field(..., description="Overall system status")
    components: Dict[str, Dict[str, Any]] = Field(..., description="Component health status")
    version: str = Field(..., description="System version")
    uptime: float = Field(..., description="System uptime in seconds")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Health check timestamp"
    )
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    
    @computed_field
    @property
    def is_healthy(self) -> bool:
        """Check if system is healthy."""
        return self.status == "healthy"
    
    @computed_field
    @property
    def uptime_formatted(self) -> str:
        """Get formatted uptime string."""
        days = int(self.uptime // 86400)
        hours = int((self.uptime % 86400) // 3600)
        minutes = int((self.uptime % 3600) // 60)
        
        if days > 0:
            return f"{days}d {hours}h {minutes}m"
        elif hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m"


# =============================================================================
# WebSocket Message Models
# =============================================================================

class WebSocketMessage(BaseHeyGenModel):
    """Base WebSocket message model."""
    
    type: str = Field(..., description="Message type")
    data: Dict[str, Any] = Field(..., description="Message data")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Message timestamp"
    )
    message_id: Optional[str] = Field(None, description="Message ID")


class VideoProgressMessage(WebSocketMessage):
    """Video progress update message."""
    
    type: Literal["video_progress"] = "video_progress"
    data: Dict[str, Any] = Field(..., description="Progress data")
    
    @field_validator('data')
    @classmethod
    def validate_progress_data(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate progress data."""
        required_fields = ["video_id", "progress", "status"]
        for field in required_fields:
            if field not in v:
                raise ValueError(f"Missing required field: {field}")
        
        if not isinstance(v.get("progress"), (int, float)) or not 0 <= v["progress"] <= 100:
            raise ValueError("Progress must be a number between 0 and 100")
        
        return v


class VideoCompleteMessage(WebSocketMessage):
    """Video completion message."""
    
    type: Literal["video_complete"] = "video_complete"
    data: Dict[str, Any] = Field(..., description="Completion data")
    
    @field_validator('data')
    @classmethod
    def validate_completion_data(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate completion data."""
        required_fields = ["video_id", "output_url"]
        for field in required_fields:
            if field not in v:
                raise ValueError(f"Missing required field: {field}")
        
        return v


class VideoErrorMessage(WebSocketMessage):
    """Video error message."""
    
    type: Literal["video_error"] = "video_error"
    data: Dict[str, Any] = Field(..., description="Error data")
    
    @field_validator('data')
    @classmethod
    def validate_error_data(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate error data."""
        required_fields = ["video_id", "error_message"]
        for field in required_fields:
            if field not in v:
                raise ValueError(f"Missing required field: {field}")
        
        return v


# =============================================================================
# Model Registry and Utilities
# =============================================================================

class ModelRegistry:
    """Registry for managing Pydantic models."""
    
    _models: Dict[str, type[BaseModel]] = {}
    
    @classmethod
    def register(cls, name: str, model: type[BaseModel]) -> None:
        """Register a model."""
        cls._models[name] = model
    
    @classmethod
    def get(cls, name: str) -> Optional[type[BaseModel]]:
        """Get a model by name."""
        return cls._models.get(name)
    
    @classmethod
    def list_models(cls) -> List[str]:
        """List all registered models."""
        return list(cls._models.keys())
    
    @classmethod
    def generate_schema(cls, name: str) -> Optional[Dict[str, Any]]:
        """Generate JSON schema for a model."""
        model = cls.get(name)
        if model:
            return model.model_json_schema()
        return None


# Register all models
ModelRegistry.register("CreateVideoRequest", CreateVideoRequest)
ModelRegistry.register("BatchCreateVideoRequest", BatchCreateVideoRequest)
ModelRegistry.register("GenerateScriptRequest", GenerateScriptRequest)
ModelRegistry.register("OptimizeScriptRequest", OptimizeScriptRequest)
ModelRegistry.register("VideoResponse", VideoResponse)
ModelRegistry.register("BatchVideoResponse", BatchVideoResponse)
ModelRegistry.register("ScriptResponse", ScriptResponse)
ModelRegistry.register("ErrorResponse", ErrorResponse)
ModelRegistry.register("HealthResponse", HealthResponse)


# =============================================================================
# Performance Optimizations
# =============================================================================

def create_model_with_cache(model_class: type[BaseModel], **kwargs) -> BaseModel:
    """Create a model instance with caching for performance."""
    # This is a placeholder for model caching implementation
    return model_class(**kwargs)


def validate_model_batch(models: List[BaseModel]) -> List[BaseModel]:
    """Validate a batch of models efficiently."""
    # This is a placeholder for batch validation implementation
    return models


# =============================================================================
# Export all models
# =============================================================================

__all__ = [
    # Enums
    "VideoStatus", "LanguageCode", "VideoStyle", "Resolution", "OutputFormat", "QualityLevel",
    
    # Base Models
    "BaseHeyGenModel", "TimestampedModel", "IdentifiedModel",
    
    # Request Models
    "CreateVideoRequest", "BatchCreateVideoRequest", "GenerateScriptRequest", "OptimizeScriptRequest",
    
    # Response Models
    "VideoResponse", "BatchVideoResponse", "ScriptResponse",
    
    # Utility Models
    "PaginationParams", "PaginatedResponse", "ErrorResponse", "HealthResponse",
    
    # WebSocket Models
    "WebSocketMessage", "VideoProgressMessage", "VideoCompleteMessage", "VideoErrorMessage",
    
    # Registry and Utilities
    "ModelRegistry", "create_model_with_cache", "validate_model_batch",
] 