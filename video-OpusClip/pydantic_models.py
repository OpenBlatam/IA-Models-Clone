"""
Pydantic Models for Video-OpusClip System

Comprehensive Pydantic BaseModel implementation for consistent input/output validation
and response schemas with enhanced validation, performance optimizations, and integration.
"""

from __future__ import annotations
from typing import List, Dict, Optional, Union, Any, Literal
from datetime import datetime, timedelta
from enum import Enum
import re
import uuid
from pathlib import Path

from pydantic import (
    BaseModel, 
    Field, 
    field_validator, 
    model_validator,
    ConfigDict,
    ValidationError,
    computed_field,
    field_serializer
)
from pydantic.types import HttpUrl, AnyUrl
import structlog

logger = structlog.get_logger()

# =============================================================================
# CONFIGURATION
# =============================================================================

class VideoOpusClipConfig:
    """Configuration constants for Video-OpusClip Pydantic models."""
    
    # URL validation patterns
    YOUTUBE_URL_PATTERNS = [
        r'https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+',
        r'https?://youtu\.be/[\w-]+',
        r'https?://(?:www\.)?youtube\.com/embed/[\w-]+',
    ]
    
    # File size limits (in bytes)
    MAX_VIDEO_SIZE = 100 * 1024 * 1024  # 100MB
    MAX_IMAGE_SIZE = 50 * 1024 * 1024   # 50MB
    MAX_AUDIO_SIZE = 25 * 1024 * 1024   # 25MB
    
    # Duration limits (in seconds)
    MIN_CLIP_DURATION = 3
    MAX_CLIP_DURATION = 600  # 10 minutes
    
    # Text length limits
    MAX_TITLE_LENGTH = 200
    MAX_DESCRIPTION_LENGTH = 5000
    MAX_CAPTION_LENGTH = 1000
    MAX_TAGS_COUNT = 50
    
    # Supported formats
    SUPPORTED_VIDEO_FORMATS = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv']
    SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.webp', '.gif']
    SUPPORTED_AUDIO_FORMATS = ['.mp3', '.wav', '.aac', '.ogg', '.flac']
    
    # Language codes
    SUPPORTED_LANGUAGES = [
        'en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh',
        'en-US', 'en-GB', 'es-MX', 'es-ES', 'fr-FR', 'de-DE', 'it-IT'
    ]

# =============================================================================
# ENUMS
# =============================================================================

class VideoStatus(str, Enum):
    """Video processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    QUEUED = "queued"
    VALIDATING = "validating"

class VideoQuality(str, Enum):
    """Video quality levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"
    CUSTOM = "custom"

class VideoFormat(str, Enum):
    """Video formats."""
    MP4 = "mp4"
    AVI = "avi"
    MOV = "mov"
    MKV = "mkv"
    WEBM = "webm"
    FLV = "flv"

class ProcessingPriority(str, Enum):
    """Processing priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"

class ContentType(str, Enum):
    """Content types for viral analysis."""
    EDUCATIONAL = "educational"
    ENTERTAINMENT = "entertainment"
    NEWS = "news"
    TUTORIAL = "tutorial"
    REVIEW = "review"
    REACTION = "reaction"
    COMEDY = "comedy"
    MUSIC = "music"
    GAMING = "gaming"
    LIFESTYLE = "lifestyle"
    TECH = "tech"
    SPORTS = "sports"

class EngagementType(str, Enum):
    """Engagement patterns."""
    HIGH_RETENTION = "high_retention"
    VIRAL_POTENTIAL = "viral_potential"
    SHAREABLE = "shareable"
    COMMENT_GENERATOR = "comment_generator"
    LIKE_MAGNET = "like_magnet"
    SUBSCRIBER_GROWTH = "subscriber_growth"

# =============================================================================
# BASE MODELS
# =============================================================================

class VideoOpusClipBaseModel(BaseModel):
    """Base model for all Video-OpusClip Pydantic models with optimized configuration."""
    
    model_config = ConfigDict(
        # Performance optimizations
        validate_assignment=True,
        arbitrary_types_allowed=True,
        extra="forbid",
        validate_default=True,
        copy_on_model_validation=False,
        
        # Serialization settings
        use_enum_values=True,
        json_encoders={
            datetime: lambda v: v.isoformat(),
            Path: lambda v: str(v),
        },
        
        # Validation settings
        str_strip_whitespace=True,
        str_min_length=1,
        
        # Example generation
        json_schema_extra={
            "examples": []
        }
    )

class TimestampedModel(VideoOpusClipBaseModel):
    """Base model with timestamp fields."""
    
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=datetime.now,
        description="Last update timestamp"
    )
    
    @field_validator('updated_at', mode='before')
    @classmethod
    def update_timestamp(cls, v):
        """Update timestamp when model is modified."""
        return datetime.now()

class IdentifiedModel(VideoOpusClipBaseModel):
    """Base model with unique identifier."""
    
    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier"
    )
    
    @field_validator('id', mode='before')
    @classmethod
    def validate_id(cls, v):
        """Validate and generate ID if needed."""
        if not v:
            return str(uuid.uuid4())
        return v

# =============================================================================
# INPUT VALIDATION MODELS
# =============================================================================

class YouTubeUrlValidator:
    """YouTube URL validation utilities."""
    
    @staticmethod
    def validate_youtube_url(url: str) -> str:
        """Validate and sanitize YouTube URL."""
        if not url or not isinstance(url, str):
            raise ValueError("YouTube URL is required and must be a string")
        
        url = url.strip()
        if len(url) > 2048:
            raise ValueError("YouTube URL too long (max 2048 characters)")
        
        # Check for malicious patterns
        malicious_patterns = [
            "javascript:", "data:", "vbscript:", "file://", "ftp://",
            "eval(", "exec(", "system(", "shell_exec("
        ]
        url_lower = url.lower()
        for pattern in malicious_patterns:
            if pattern in url_lower:
                raise ValueError(f"Malicious URL pattern detected: {pattern}")
        
        # Validate YouTube URL format
        for pattern in VideoOpusClipConfig.YOUTUBE_URL_PATTERNS:
            if re.match(pattern, url):
                return url
        
        raise ValueError("Invalid YouTube URL format")
    
    @staticmethod
    def extract_video_id(url: str) -> str:
        """Extract video ID from YouTube URL."""
        patterns = [
            r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([\w-]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        raise ValueError("Could not extract video ID from URL")

class VideoClipRequest(VideoOpusClipBaseModel):
    """Request model for video clip processing with comprehensive validation."""
    
    youtube_url: str = Field(
        ..., 
        description="YouTube URL to process",
        min_length=1,
        max_length=2048
    )
    language: str = Field(
        default="en",
        description="Language code for processing",
        min_length=2,
        max_length=10
    )
    max_clip_length: int = Field(
        default=60,
        description="Maximum clip length in seconds",
        ge=VideoOpusClipConfig.MIN_CLIP_DURATION,
        le=VideoOpusClipConfig.MAX_CLIP_DURATION
    )
    min_clip_length: int = Field(
        default=15,
        description="Minimum clip length in seconds",
        ge=VideoOpusClipConfig.MIN_CLIP_DURATION,
        le=VideoOpusClipConfig.MAX_CLIP_DURATION
    )
    quality: VideoQuality = Field(
        default=VideoQuality.HIGH,
        description="Target video quality"
    )
    format: VideoFormat = Field(
        default=VideoFormat.MP4,
        description="Target video format"
    )
    priority: ProcessingPriority = Field(
        default=ProcessingPriority.NORMAL,
        description="Processing priority"
    )
    custom_params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Custom processing parameters"
    )
    audience_profile: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Target audience profile for optimization"
    )
    
    @field_validator('youtube_url')
    @classmethod
    def validate_youtube_url(cls, v: str) -> str:
        """Validate YouTube URL."""
        return YouTubeUrlValidator.validate_youtube_url(v)
    
    @field_validator('language')
    @classmethod
    def validate_language(cls, v: str) -> str:
        """Validate language code."""
        if v not in VideoOpusClipConfig.SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language: {v}")
        return v
    
    @field_validator('min_clip_length', 'max_clip_length')
    @classmethod
    def validate_clip_length(cls, v: int) -> int:
        """Validate clip length."""
        if v < VideoOpusClipConfig.MIN_CLIP_DURATION:
            raise ValueError(f"Clip length must be at least {VideoOpusClipConfig.MIN_CLIP_DURATION} seconds")
        if v > VideoOpusClipConfig.MAX_CLIP_DURATION:
            raise ValueError(f"Clip length cannot exceed {VideoOpusClipConfig.MAX_CLIP_DURATION} seconds")
        return v
    
    @model_validator(mode='after')
    def validate_clip_length_logic(self) -> 'VideoClipRequest':
        """Validate logical constraints between min and max clip length."""
        if self.min_clip_length >= self.max_clip_length:
            raise ValueError("min_clip_length must be less than max_clip_length")
        return self
    
    @computed_field
    @property
    def video_id(self) -> str:
        """Extract video ID from YouTube URL."""
        return YouTubeUrlValidator.extract_video_id(self.youtube_url)
    
    @computed_field
    @property
    def request_hash(self) -> str:
        """Generate unique hash for this request."""
        import hashlib
        content = f"{self.youtube_url}:{self.language}:{self.max_clip_length}:{self.quality}"
        return hashlib.md5(content.encode()).hexdigest()
    
    class Config:
        json_schema_extra = {
            "example": {
                "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                "language": "en",
                "max_clip_length": 60,
                "min_clip_length": 15,
                "quality": "high",
                "format": "mp4",
                "priority": "normal"
            }
        }

class ViralVideoRequest(VideoClipRequest):
    """Request model for viral video generation with enhanced validation."""
    
    n_variants: int = Field(
        default=5,
        description="Number of viral variants to generate",
        ge=1,
        le=20
    )
    use_langchain: bool = Field(
        default=True,
        description="Use LangChain for content analysis"
    )
    viral_optimization: bool = Field(
        default=True,
        description="Enable viral optimization features"
    )
    engagement_focus: Optional[EngagementType] = Field(
        default=None,
        description="Focus on specific engagement type"
    )
    content_type: Optional[ContentType] = Field(
        default=None,
        description="Content type for optimization"
    )
    
    @field_validator('n_variants')
    @classmethod
    def validate_variants_count(cls, v: int) -> int:
        """Validate number of variants."""
        if v < 1:
            raise ValueError("Must generate at least 1 variant")
        if v > 20:
            raise ValueError("Cannot generate more than 20 variants")
        return v

class BatchVideoRequest(VideoOpusClipBaseModel):
    """Request model for batch video processing."""
    
    requests: List[VideoClipRequest] = Field(
        ...,
        description="List of video processing requests",
        min_length=1,
        max_length=100
    )
    batch_id: Optional[str] = Field(
        default=None,
        description="Batch identifier"
    )
    priority: ProcessingPriority = Field(
        default=ProcessingPriority.NORMAL,
        description="Batch processing priority"
    )
    max_workers: int = Field(
        default=8,
        description="Maximum parallel workers",
        ge=1,
        le=32
    )
    timeout: float = Field(
        default=300.0,
        description="Batch timeout in seconds",
        ge=60.0,
        le=3600.0
    )
    
    @field_validator('requests')
    @classmethod
    def validate_requests(cls, v: List[VideoClipRequest]) -> List[VideoClipRequest]:
        """Validate batch requests."""
        if not v:
            raise ValueError("Batch cannot be empty")
        if len(v) > 100:
            raise ValueError("Batch size cannot exceed 100 requests")
        return v
    
    @computed_field
    @property
    def total_requests(self) -> int:
        """Get total number of requests."""
        return len(self.requests)
    
    @computed_field
    @property
    def estimated_processing_time(self) -> float:
        """Estimate processing time based on requests."""
        return sum(req.max_clip_length for req in self.requests) * 0.1  # Rough estimate

# =============================================================================
# RESPONSE MODELS
# =============================================================================

class VideoClipResponse(VideoOpusClipBaseModel):
    """Response model for video clip processing."""
    
    success: bool = Field(..., description="Processing success status")
    clip_id: Optional[str] = Field(default=None, description="Generated clip ID")
    request_id: Optional[str] = Field(default=None, description="Original request ID")
    
    # Video metadata
    title: Optional[str] = Field(default=None, description="Video title")
    description: Optional[str] = Field(default=None, description="Video description")
    duration: Optional[float] = Field(default=None, description="Clip duration in seconds")
    language: Optional[str] = Field(default=None, description="Processing language")
    
    # File information
    file_path: Optional[str] = Field(default=None, description="Output file path")
    file_size: Optional[int] = Field(default=None, description="File size in bytes")
    resolution: Optional[str] = Field(default=None, description="Video resolution")
    fps: Optional[float] = Field(default=None, description="Frames per second")
    bitrate: Optional[int] = Field(default=None, description="Video bitrate")
    
    # Processing metrics
    processing_time: float = Field(default=0.0, description="Processing time in seconds")
    quality: Optional[VideoQuality] = Field(default=None, description="Achieved quality")
    format: Optional[VideoFormat] = Field(default=None, description="Output format")
    
    # Status and errors
    status: VideoStatus = Field(default=VideoStatus.PENDING, description="Processing status")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    warnings: List[str] = Field(default_factory=list, description="Processing warnings")
    
    # Additional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(default_factory=datetime.now, description="Response creation time")
    
    @field_validator('processing_time')
    @classmethod
    def validate_processing_time(cls, v: float) -> float:
        """Validate processing time."""
        if v < 0:
            raise ValueError("Processing time cannot be negative")
        return v
    
    @field_validator('duration')
    @classmethod
    def validate_duration(cls, v: Optional[float]) -> Optional[float]:
        """Validate video duration."""
        if v is not None and v <= 0:
            raise ValueError("Duration must be positive")
        return v
    
    @computed_field
    @property
    def is_successful(self) -> bool:
        """Check if processing was successful."""
        return self.success and self.status == VideoStatus.COMPLETED
    
    @computed_field
    @property
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return len(self.warnings) > 0
    
    @computed_field
    @property
    def file_size_mb(self) -> Optional[float]:
        """Get file size in megabytes."""
        if self.file_size is not None:
            return round(self.file_size / (1024 * 1024), 2)
        return None

class ViralVideoVariant(VideoOpusClipBaseModel):
    """Model for viral video variant with engagement metrics."""
    
    variant_id: str = Field(..., description="Unique variant identifier")
    title: str = Field(..., description="Variant title")
    description: str = Field(..., description="Variant description")
    
    # Engagement metrics
    viral_score: float = Field(..., description="Viral potential score", ge=0.0, le=1.0)
    engagement_prediction: float = Field(..., description="Predicted engagement rate", ge=0.0, le=1.0)
    retention_score: float = Field(default=0.0, description="Retention prediction", ge=0.0, le=1.0)
    
    # Video data
    clip_data: Optional[VideoClipResponse] = Field(default=None, description="Associated clip data")
    duration: float = Field(..., description="Variant duration", ge=0.0)
    
    # Content analysis
    content_type: Optional[ContentType] = Field(default=None, description="Content type")
    engagement_type: Optional[EngagementType] = Field(default=None, description="Engagement pattern")
    target_audience: List[str] = Field(default_factory=list, description="Target audience")
    
    # Viral elements
    viral_hooks: List[str] = Field(default_factory=list, description="Viral hook elements")
    trending_elements: List[str] = Field(default_factory=list, description="Trending elements")
    hashtags: List[str] = Field(default_factory=list, description="Optimized hashtags")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    generation_time: float = Field(default=0.0, description="Generation time in seconds")
    
    @field_validator('viral_score', 'engagement_prediction', 'retention_score')
    @classmethod
    def validate_scores(cls, v: float) -> float:
        """Validate score values."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Score must be between 0.0 and 1.0")
        return v
    
    @computed_field
    @property
    def overall_score(self) -> float:
        """Calculate overall variant score."""
        return (self.viral_score + self.engagement_prediction + self.retention_score) / 3
    
    @computed_field
    @property
    def is_high_performing(self) -> bool:
        """Check if variant is high performing."""
        return self.overall_score >= 0.8

class ViralVideoBatchResponse(VideoOpusClipBaseModel):
    """Response model for viral video batch processing."""
    
    success: bool = Field(..., description="Batch processing success status")
    original_clip_id: str = Field(..., description="Original clip identifier")
    batch_id: Optional[str] = Field(default=None, description="Batch identifier")
    
    # Variants
    variants: List[ViralVideoVariant] = Field(default_factory=list, description="Generated variants")
    total_variants_generated: int = Field(default=0, description="Total variants generated")
    successful_variants: int = Field(default=0, description="Successfully generated variants")
    
    # Processing metrics
    processing_time: float = Field(default=0.0, description="Total processing time")
    langchain_analysis_time: float = Field(default=0.0, description="LangChain analysis time")
    content_optimization_time: float = Field(default=0.0, description="Content optimization time")
    
    # Quality metrics
    average_viral_score: float = Field(default=0.0, description="Average viral score")
    best_viral_score: float = Field(default=0.0, description="Best viral score")
    caption_quality_score: float = Field(default=0.0, description="Caption quality score")
    editing_quality_score: float = Field(default=0.0, description="Editing quality score")
    
    # Error handling
    errors: List[str] = Field(default_factory=list, description="Processing errors")
    warnings: List[str] = Field(default_factory=list, description="Processing warnings")
    
    # Performance metrics
    cache_hit_rate: float = Field(default=0.0, description="Cache hit rate")
    processing_efficiency: float = Field(default=0.0, description="Processing efficiency")
    memory_usage: float = Field(default=0.0, description="Memory usage in MB")
    cpu_usage: float = Field(default=0.0, description="CPU usage percentage")
    
    @field_validator('variants')
    @classmethod
    def validate_variants(cls, v: List[ViralVideoVariant]) -> List[ViralVideoVariant]:
        """Validate variants list."""
        if not v:
            raise ValueError("Variants list cannot be empty")
        return v
    
    @computed_field
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_variants_generated == 0:
            return 0.0
        return self.successful_variants / self.total_variants_generated
    
    @computed_field
    @property
    def best_variant(self) -> Optional[ViralVideoVariant]:
        """Get the best performing variant."""
        if not self.variants:
            return None
        return max(self.variants, key=lambda v: v.overall_score)
    
    @computed_field
    @property
    def high_performing_variants(self) -> List[ViralVideoVariant]:
        """Get high performing variants."""
        return [v for v in self.variants if v.is_high_performing]

class BatchVideoResponse(VideoOpusClipBaseModel):
    """Response model for batch video processing."""
    
    success: bool = Field(..., description="Batch processing success status")
    batch_id: str = Field(..., description="Batch identifier")
    
    # Results
    results: List[VideoClipResponse] = Field(default_factory=list, description="Processing results")
    total_requests: int = Field(default=0, description="Total requests processed")
    successful_requests: int = Field(default=0, description="Successful requests")
    failed_requests: int = Field(default=0, description="Failed requests")
    
    # Processing metrics
    processing_time: float = Field(default=0.0, description="Total processing time")
    average_processing_time: float = Field(default=0.0, description="Average processing time per request")
    
    # Error handling
    errors: List[str] = Field(default_factory=list, description="Batch processing errors")
    warnings: List[str] = Field(default_factory=list, description="Batch processing warnings")
    
    # Performance metrics
    memory_usage: float = Field(default=0.0, description="Peak memory usage in MB")
    cpu_usage: float = Field(default=0.0, description="Average CPU usage")
    gpu_usage: Optional[float] = Field(default=None, description="GPU usage if applicable")
    
    @computed_field
    @property
    def success_rate(self) -> float:
        """Calculate batch success rate."""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests
    
    @computed_field
    @property
    def failed_results(self) -> List[VideoClipResponse]:
        """Get failed processing results."""
        return [r for r in self.results if not r.success]
    
    @computed_field
    @property
    def successful_results(self) -> List[VideoClipResponse]:
        """Get successful processing results."""
        return [r for r in self.results if r.success]

# =============================================================================
# VALIDATION MODELS
# =============================================================================

class ValidationResult(VideoOpusClipBaseModel):
    """Base validation result model."""
    
    is_valid: bool = Field(..., description="Validation status")
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    suggestions: List[str] = Field(default_factory=list, description="Improvement suggestions")
    
    @computed_field
    @property
    def has_errors(self) -> bool:
        """Check if there are validation errors."""
        return len(self.errors) > 0
    
    @computed_field
    @property
    def has_warnings(self) -> bool:
        """Check if there are validation warnings."""
        return len(self.warnings) > 0

class VideoValidationResult(ValidationResult):
    """Validation result for video processing requests."""
    
    quality_score: float = Field(default=0.0, description="Quality validation score", ge=0.0, le=1.0)
    duration_score: float = Field(default=0.0, description="Duration validation score", ge=0.0, le=1.0)
    format_score: float = Field(default=0.0, description="Format validation score", ge=0.0, le=1.0)
    overall_score: float = Field(default=0.0, description="Overall validation score", ge=0.0, le=1.0)
    
    # Validation details
    url_valid: bool = Field(default=False, description="URL validation status")
    language_supported: bool = Field(default=False, description="Language support status")
    duration_appropriate: bool = Field(default=False, description="Duration appropriateness")
    format_supported: bool = Field(default=False, description="Format support status")
    
    @computed_field
    @property
    def is_high_quality(self) -> bool:
        """Check if validation indicates high quality."""
        return self.overall_score >= 0.8

class BatchValidationResult(ValidationResult):
    """Validation result for batch processing requests."""
    
    valid_videos: int = Field(default=0, description="Number of valid videos")
    invalid_videos: int = Field(default=0, description="Number of invalid videos")
    validation_results: List[VideoValidationResult] = Field(default_factory=list, description="Individual validation results")
    overall_score: float = Field(default=0.0, description="Overall batch validation score", ge=0.0, le=1.0)
    
    @computed_field
    @property
    def total_videos(self) -> int:
        """Get total number of videos in batch."""
        return self.valid_videos + self.invalid_videos
    
    @computed_field
    @property
    def validation_rate(self) -> float:
        """Calculate validation success rate."""
        if self.total_videos == 0:
            return 0.0
        return self.valid_videos / self.total_videos

# =============================================================================
# CONFIGURATION MODELS
# =============================================================================

class VideoProcessingConfig(VideoOpusClipBaseModel):
    """Configuration model for video processing."""
    
    # Quality settings
    target_quality: VideoQuality = Field(default=VideoQuality.HIGH, description="Target video quality")
    target_format: VideoFormat = Field(default=VideoFormat.MP4, description="Target video format")
    target_resolution: str = Field(default="1920x1080", description="Target resolution")
    target_fps: float = Field(default=30.0, description="Target frames per second", ge=1.0, le=120.0)
    target_bitrate: Optional[int] = Field(default=None, description="Target bitrate in kbps")
    
    # Processing settings
    enable_audio: bool = Field(default=True, description="Enable audio processing")
    enable_subtitles: bool = Field(default=True, description="Enable subtitle generation")
    enable_thumbnails: bool = Field(default=True, description="Enable thumbnail generation")
    enable_metadata: bool = Field(default=True, description="Enable metadata extraction")
    
    # Performance settings
    max_workers: int = Field(default=8, description="Maximum parallel workers", ge=1, le=32)
    chunk_size: int = Field(default=1000, description="Processing chunk size", ge=100, le=10000)
    timeout: float = Field(default=300.0, description="Processing timeout in seconds", ge=60.0, le=3600.0)
    retry_attempts: int = Field(default=3, description="Retry attempts on failure", ge=0, le=10)
    retry_delay: float = Field(default=1.0, description="Delay between retries in seconds", ge=0.1, le=60.0)
    
    # Advanced settings
    use_gpu: bool = Field(default=False, description="Enable GPU acceleration")
    use_hardware_acceleration: bool = Field(default=True, description="Enable hardware acceleration")
    optimize_for_web: bool = Field(default=True, description="Optimize for web delivery")
    preserve_original: bool = Field(default=False, description="Preserve original file")
    
    @field_validator('target_resolution')
    @classmethod
    def validate_resolution(cls, v: str) -> str:
        """Validate resolution format."""
        if not re.match(r'^\d+x\d+$', v):
            raise ValueError("Resolution must be in format 'WIDTHxHEIGHT'")
        return v
    
    @field_validator('target_bitrate')
    @classmethod
    def validate_bitrate(cls, v: Optional[int]) -> Optional[int]:
        """Validate bitrate value."""
        if v is not None and v <= 0:
            raise ValueError("Bitrate must be positive")
        return v

class ViralProcessingConfig(VideoProcessingConfig):
    """Configuration model for viral video processing."""
    
    # Viral optimization settings
    viral_optimization_enabled: bool = Field(default=True, description="Enable viral optimization")
    engagement_analysis: bool = Field(default=True, description="Enable engagement analysis")
    content_analysis: bool = Field(default=True, description="Enable content analysis")
    audience_analysis: bool = Field(default=True, description="Enable audience analysis")
    
    # LangChain settings
    use_langchain: bool = Field(default=True, description="Use LangChain for analysis")
    langchain_model: str = Field(default="gpt-3.5-turbo", description="LangChain model to use")
    langchain_temperature: float = Field(default=0.7, description="LangChain temperature", ge=0.0, le=2.0)
    
    # Viral generation settings
    min_viral_score: float = Field(default=0.6, description="Minimum viral score threshold", ge=0.0, le=1.0)
    max_variants: int = Field(default=10, description="Maximum variants to generate", ge=1, le=50)
    variant_diversity: float = Field(default=0.8, description="Variant diversity factor", ge=0.0, le=1.0)
    
    # Performance settings
    cache_enabled: bool = Field(default=True, description="Enable result caching")
    parallel_generation: bool = Field(default=True, description="Enable parallel variant generation")
    optimization_level: str = Field(default="ultra", description="Optimization level")

# =============================================================================
# UTILITY MODELS
# =============================================================================

class ProcessingMetrics(VideoOpusClipBaseModel):
    """Model for processing performance metrics."""
    
    # Timing metrics
    start_time: datetime = Field(default_factory=datetime.now, description="Processing start time")
    end_time: Optional[datetime] = Field(default=None, description="Processing end time")
    total_time: float = Field(default=0.0, description="Total processing time in seconds")
    
    # Resource usage
    memory_usage_mb: float = Field(default=0.0, description="Memory usage in MB")
    cpu_usage_percent: float = Field(default=0.0, description="CPU usage percentage")
    gpu_usage_percent: Optional[float] = Field(default=None, description="GPU usage percentage")
    
    # Performance metrics
    throughput: float = Field(default=0.0, description="Processing throughput")
    efficiency: float = Field(default=0.0, description="Processing efficiency")
    cache_hit_rate: float = Field(default=0.0, description="Cache hit rate")
    
    @computed_field
    @property
    def is_completed(self) -> bool:
        """Check if processing is completed."""
        return self.end_time is not None
    
    @computed_field
    @property
    def processing_duration(self) -> Optional[timedelta]:
        """Get processing duration."""
        if self.end_time and self.start_time:
            return self.end_time - self.start_time
        return None

class ErrorInfo(VideoOpusClipBaseModel):
    """Model for detailed error information."""
    
    error_code: str = Field(..., description="Error code")
    error_message: str = Field(..., description="Error message")
    error_type: str = Field(..., description="Error type")
    field_name: Optional[str] = Field(default=None, description="Field causing error")
    field_value: Optional[Any] = Field(default=None, description="Field value that caused error")
    
    # Context information
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
    request_id: Optional[str] = Field(default=None, description="Request identifier")
    user_id: Optional[str] = Field(default=None, description="User identifier")
    
    # Technical details
    stack_trace: Optional[str] = Field(default=None, description="Stack trace")
    additional_context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")

# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_video_clip_request(
    youtube_url: str,
    language: str = "en",
    max_clip_length: int = 60,
    quality: VideoQuality = VideoQuality.HIGH
) -> VideoClipRequest:
    """Create a video clip request with default values."""
    return VideoClipRequest(
        youtube_url=youtube_url,
        language=language,
        max_clip_length=max_clip_length,
        quality=quality
    )

def create_viral_video_request(
    youtube_url: str,
    n_variants: int = 5,
    use_langchain: bool = True
) -> ViralVideoRequest:
    """Create a viral video request with default values."""
    return ViralVideoRequest(
        youtube_url=youtube_url,
        n_variants=n_variants,
        use_langchain=use_langchain
    )

def create_batch_request(
    requests: List[VideoClipRequest],
    priority: ProcessingPriority = ProcessingPriority.NORMAL
) -> BatchVideoRequest:
    """Create a batch video request."""
    return BatchVideoRequest(
        requests=requests,
        priority=priority
    )

def create_processing_config(
    quality: VideoQuality = VideoQuality.HIGH,
    format: VideoFormat = VideoFormat.MP4,
    max_workers: int = 8
) -> VideoProcessingConfig:
    """Create a video processing configuration."""
    return VideoProcessingConfig(
        target_quality=quality,
        target_format=format,
        max_workers=max_workers
    )

# =============================================================================
# VALIDATION UTILITIES
# =============================================================================

def validate_video_request(request: VideoClipRequest) -> VideoValidationResult:
    """Validate a video clip request."""
    errors = []
    warnings = []
    suggestions = []
    
    # URL validation
    url_valid = True
    try:
        YouTubeUrlValidator.validate_youtube_url(request.youtube_url)
    except ValueError as e:
        errors.append(f"URL validation failed: {e}")
        url_valid = False
    
    # Language validation
    language_supported = request.language in VideoOpusClipConfig.SUPPORTED_LANGUAGES
    if not language_supported:
        errors.append(f"Unsupported language: {request.language}")
    
    # Duration validation
    duration_appropriate = (
        VideoOpusClipConfig.MIN_CLIP_DURATION <= request.max_clip_length <= VideoOpusClipConfig.MAX_CLIP_DURATION
    )
    if not duration_appropriate:
        warnings.append(f"Clip duration {request.max_clip_length}s may not be optimal")
    
    # Format validation
    format_supported = True  # All formats in enum are supported
    
    # Calculate scores
    quality_score = 1.0 if url_valid and language_supported else 0.0
    duration_score = 1.0 if duration_appropriate else 0.5
    format_score = 1.0 if format_supported else 0.0
    overall_score = (quality_score + duration_score + format_score) / 3
    
    # Generate suggestions
    if not language_supported:
        suggestions.append(f"Consider using one of: {', '.join(VideoOpusClipConfig.SUPPORTED_LANGUAGES[:5])}")
    if not duration_appropriate:
        suggestions.append(f"Optimal duration range: {VideoOpusClipConfig.MIN_CLIP_DURATION}-{VideoOpusClipConfig.MAX_CLIP_DURATION}s")
    
    return VideoValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        suggestions=suggestions,
        quality_score=quality_score,
        duration_score=duration_score,
        format_score=format_score,
        overall_score=overall_score,
        url_valid=url_valid,
        language_supported=language_supported,
        duration_appropriate=duration_appropriate,
        format_supported=format_supported
    )

def validate_batch_request(batch: BatchVideoRequest) -> BatchValidationResult:
    """Validate a batch video request."""
    validation_results = []
    valid_videos = 0
    invalid_videos = 0
    
    for request in batch.requests:
        result = validate_video_request(request)
        validation_results.append(result)
        
        if result.is_valid:
            valid_videos += 1
        else:
            invalid_videos += 1
    
    # Aggregate errors and warnings
    all_errors = []
    all_warnings = []
    all_suggestions = []
    
    for result in validation_results:
        all_errors.extend(result.errors)
        all_warnings.extend(result.warnings)
        all_suggestions.extend(result.suggestions)
    
    # Calculate overall score
    if validation_results:
        overall_score = sum(r.overall_score for r in validation_results) / len(validation_results)
    else:
        overall_score = 0.0
    
    return BatchValidationResult(
        is_valid=invalid_videos == 0,
        errors=all_errors,
        warnings=all_warnings,
        suggestions=all_suggestions,
        valid_videos=valid_videos,
        invalid_videos=invalid_videos,
        validation_results=validation_results,
        overall_score=overall_score
    ) 