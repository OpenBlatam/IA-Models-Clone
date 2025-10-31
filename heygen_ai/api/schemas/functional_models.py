from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

from typing import Optional, List, Dict, Any, Union, Annotated, Literal
from datetime import datetime, timezone, timedelta
from decimal import Decimal
import re
import uuid
from pathlib import Path
from enum import Enum
from pydantic import (
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema
import structlog
from typing import Any, List, Dict, Optional
import logging
import asyncio
#!/usr/bin/env python3
"""
Functional Pydantic Models for HeyGen AI API
Modern Pydantic v2 models with functional components, computed fields, and advanced validation.
"""


    BaseModel, Field, HttpUrl, EmailStr, validator, root_validator,
    computed_field, model_validator, field_validator, ConfigDict,
    PlainSerializer, PlainValidator, BeforeValidator, AfterValidator,
    WithJsonSchema, GetJsonSchemaHandler, GetCoreSchemaHandler,
    ValidationError, ValidationInfo
)

logger = structlog.get_logger()

# =============================================================================
# Functional Validators and Utilities
# =============================================================================

def validate_username(value: str) -> str:
    """Functional username validator."""
    if not value or len(value) < 3:
        raise ValueError("Username must be at least 3 characters long")
    if not re.match(r'^[a-zA-Z0-9_]+$', value):
        raise ValueError("Username can only contain letters, numbers, and underscores")
    return value.lower()

def validate_password_strength(value: str) -> str:
    """Functional password strength validator."""
    if len(value) < 8:
        raise ValueError("Password must be at least 8 characters long")
    if not re.search(r'[A-Z]', value):
        raise ValueError("Password must contain at least one uppercase letter")
    if not re.search(r'[a-z]', value):
        raise ValueError("Password must contain at least one lowercase letter")
    if not re.search(r'\d', value):
        raise ValueError("Password must contain at least one digit")
    return value

def validate_script_content(value: str) -> str:
    """Functional script content validator."""
    if not value or not value.strip():
        raise ValueError("Script cannot be empty")
    if len(value) > 1000:
        raise ValueError("Script cannot exceed 1000 characters")
    return value.strip()

def validate_file_path(value: str) -> str:
    """Functional file path validator."""
    path = Path(value)
    if not path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
        raise ValueError("Invalid video file format")
    return str(path)

def calculate_processing_efficiency(processing_time: float, file_size: int) -> float:
    """Functional efficiency calculator."""
    if processing_time <= 0 or file_size <= 0:
        return 0.0
    # Efficiency = file_size / processing_time (bytes per second)
    return min(file_size / processing_time, 100.0)

def generate_video_id() -> str:
    """Functional video ID generator."""
    return f"video_{uuid.uuid4().hex[:12]}"

async def validate_api_key_format(value: str) -> str:
    """Functional API key format validator."""
    if not value or len(value) < 32:
        raise ValueError("API key must be at least 32 characters long")
    if not re.match(r'^[a-zA-Z0-9_-]+$', value):
        raise ValueError("API key contains invalid characters")
    return value

# =============================================================================
# Enhanced Enums with Functional Methods
# =============================================================================

class VideoStatus(str, Enum):
    """Video processing status with functional methods."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    QUEUED = "queued"
    
    @classmethod
    def get_description(cls, value: str) -> str:
        """Functional method to get status description."""
        descriptions = {
            "pending": "Video is waiting to be processed",
            "processing": "Video is currently being generated",
            "completed": "Video generation finished successfully",
            "failed": "Video generation failed",
            "cancelled": "Video generation was cancelled",
            "queued": "Video is in processing queue"
        }
        return descriptions.get(value, "Unknown status")
    
    @classmethod
    def is_final(cls, value: str) -> bool:
        """Functional method to check if status is final."""
        return value in [cls.COMPLETED, cls.FAILED, cls.CANCELLED]
    
    @classmethod
    def can_retry(cls, value: str) -> bool:
        """Functional method to check if video can be retried."""
        return value == cls.FAILED

class VideoQuality(str, Enum):
    """Video quality with functional methods."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"
    
    @classmethod
    def get_bitrate(cls, value: str) -> int:
        """Functional method to get bitrate for quality."""
        bitrates = {
            "low": 1000000,      # 1 Mbps
            "medium": 2500000,   # 2.5 Mbps
            "high": 5000000,     # 5 Mbps
            "ultra": 10000000    # 10 Mbps
        }
        return bitrates.get(value, 2500000)
    
    @classmethod
    def get_processing_time_multiplier(cls, value: str) -> float:
        """Functional method to get processing time multiplier."""
        multipliers = {
            "low": 0.5,
            "medium": 1.0,
            "high": 1.5,
            "ultra": 2.0
        }
        return multipliers.get(value, 1.0)

class ModelType(str, Enum):
    """Model type with functional methods."""
    TRANSFORMER = "transformer"
    DIFFUSION = "diffusion"
    TTS = "tts"
    FACE_DETECTION = "face_detection"
    EMOTION_ANALYSIS = "emotion_analysis"
    AUDIO_PROCESSING = "audio_processing"
    
    @classmethod
    def get_resource_requirements(cls, value: str) -> Dict[str, float]:
        """Functional method to get resource requirements."""
        requirements = {
            "transformer": {"memory": 4.0, "gpu": 2.0},
            "diffusion": {"memory": 8.0, "gpu": 4.0},
            "tts": {"memory": 2.0, "gpu": 1.0},
            "face_detection": {"memory": 1.0, "gpu": 0.5},
            "emotion_analysis": {"memory": 1.5, "gpu": 0.5},
            "audio_processing": {"memory": 1.0, "gpu": 0.5}
        }
        return requirements.get(value, {"memory": 1.0, "gpu": 0.5})

# =============================================================================
# Base Models with Functional Configuration
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
            datetime: lambda v: v.isoformat(),
            Decimal: lambda v: float(v),
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
        """Functional timestamp updater."""
        self.updated_at = datetime.now(timezone.utc)
        return self

# =============================================================================
# User Models with Functional Components
# =============================================================================

class UserBase(BaseHeyGenModel):
    """Base user model with functional validation."""
    
    username: str = Field(
        ..., 
        min_length=3, 
        max_length=50,
        description="Unique username",
        examples=["john_doe", "alice_smith"]
    )
    email: EmailStr = Field(
        ...,
        description="User email address",
        examples=["user@example.com"]
    )
    full_name: Optional[str] = Field(
        None,
        max_length=100,
        description="User's full name"
    )
    is_active: bool = Field(
        default=True,
        description="Account active status"
    )
    
    @field_validator('username')
    @classmethod
    def validate_username(cls, v: str) -> str:
        """Functional username validation."""
        return validate_username(v)

class UserCreate(UserBase):
    """User creation model with functional password validation."""
    
    password: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="User password"
    )
    confirm_password: str = Field(
        ...,
        description="Password confirmation"
    )
    
    @field_validator('password')
    @classmethod
    def validate_password(cls, v: str) -> str:
        """Functional password validation."""
        return validate_password_strength(v)
    
    @model_validator(mode='after')
    def validate_passwords_match(self) -> 'UserCreate':
        """Functional password confirmation validation."""
        if self.password != self.confirm_password:
            raise ValueError("Passwords do not match")
        return self

class UserUpdate(BaseHeyGenModel):
    """User update model with functional validation."""
    
    username: Optional[str] = Field(None, min_length=3, max_length=50)
    email: Optional[EmailStr] = None
    full_name: Optional[str] = Field(None, max_length=100)
    is_active: Optional[bool] = None
    password: Optional[str] = Field(None, min_length=8, max_length=128)
    
    @field_validator('username')
    @classmethod
    def validate_username(cls, v: Optional[str]) -> Optional[str]:
        """Functional username validation."""
        if v is not None:
            return validate_username(v)
        return v
    
    @field_validator('password')
    @classmethod
    def validate_password(cls, v: Optional[str]) -> Optional[str]:
        """Functional password validation."""
        if v is not None:
            return validate_password_strength(v)
        return v

class UserResponse(UserBase, TimestampedModel):
    """User response model with computed fields."""
    
    id: int = Field(..., description="User ID")
    api_key: Optional[str] = Field(None, description="API key")
    last_login_at: Optional[datetime] = Field(None, description="Last login timestamp")
    login_count: int = Field(default=0, description="Total login count")
    
    @computed_field
    @property
    def display_name(self) -> str:
        """Functional computed field for display name."""
        return self.full_name or self.username
    
    @computed_field
    @property
    def is_online(self) -> bool:
        """Functional computed field for online status."""
        if not self.last_login_at:
            return False
        return (datetime.now(timezone.utc) - self.last_login_at) < timedelta(minutes=5)
    
    @computed_field
    @property
    def account_age_days(self) -> int:
        """Functional computed field for account age."""
        return (datetime.now(timezone.utc) - self.created_at).days

class UserSummary(BaseHeyGenModel):
    """User summary model for list responses."""
    
    id: int
    username: str
    email: str
    full_name: Optional[str] = None
    is_active: bool
    created_at: datetime
    
    @computed_field
    @property
    def display_name(self) -> str:
        """Functional computed field for display name."""
        return self.full_name or self.username

# =============================================================================
# Video Models with Functional Components
# =============================================================================

class VideoBase(BaseHeyGenModel):
    """Base video model with functional validation."""
    
    script: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Video script content"
    )
    voice_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Voice model identifier"
    )
    language: str = Field(
        default="en",
        min_length=2,
        max_length=5,
        description="Video language code"
    )
    quality: VideoQuality = Field(
        default=VideoQuality.MEDIUM,
        description="Video quality setting"
    )
    
    @field_validator('script')
    @classmethod
    def validate_script(cls, v: str) -> str:
        """Functional script validation."""
        return validate_script_content(v)
    
    @field_validator('language')
    @classmethod
    def validate_language(cls, v: str) -> str:
        """Functional language validation."""
        return v.lower()

class VideoCreate(VideoBase):
    """Video creation model with functional validation."""
    
    @model_validator(mode='after')
    def validate_script_length_for_quality(self) -> 'VideoCreate':
        """Functional quality-based script validation."""
        max_lengths = {
            VideoQuality.LOW: 500,
            VideoQuality.MEDIUM: 1000,
            VideoQuality.HIGH: 1500,
            VideoQuality.ULTRA: 2000
        }
        max_length = max_lengths.get(self.quality, 1000)
        if len(self.script) > max_length:
            raise ValueError(f"Script too long for {self.quality} quality (max {max_length} characters)")
        return self

class VideoUpdate(BaseHeyGenModel):
    """Video update model with functional validation."""
    
    script: Optional[str] = Field(None, min_length=1, max_length=1000)
    voice_id: Optional[str] = Field(None, min_length=1, max_length=50)
    language: Optional[str] = Field(None, min_length=2, max_length=5)
    quality: Optional[VideoQuality] = None
    
    @field_validator('script')
    @classmethod
    def validate_script(cls, v: Optional[str]) -> Optional[str]:
        """Functional script validation."""
        if v is not None:
            return validate_script_content(v)
        return v

class VideoResponse(VideoBase, TimestampedModel):
    """Video response model with computed fields."""
    
    id: int = Field(..., description="Video ID")
    video_id: str = Field(..., description="External video identifier")
    user_id: int = Field(..., description="User who created the video")
    status: VideoStatus = Field(..., description="Video processing status")
    file_path: Optional[str] = Field(None, description="Video file path")
    file_size: Optional[int] = Field(None, description="Video file size in bytes")
    duration: Optional[float] = Field(None, description="Video duration in seconds")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    progress: Optional[float] = Field(None, ge=0, le=100, description="Processing progress")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    
    @computed_field
    @property
    def is_completed(self) -> bool:
        """Functional computed field for completion status."""
        return self.status == VideoStatus.COMPLETED
    
    @computed_field
    @property
    def is_failed(self) -> bool:
        """Functional computed field for failure status."""
        return self.status == VideoStatus.FAILED
    
    @computed_field
    @property
    def can_retry(self) -> bool:
        """Functional computed field for retry capability."""
        return VideoStatus.can_retry(self.status)
    
    @computed_field
    @property
    def file_size_mb(self) -> Optional[float]:
        """Functional computed field for file size in MB."""
        if self.file_size:
            return round(self.file_size / (1024 * 1024), 2)
        return None
    
    @computed_field
    @property
    def processing_efficiency(self) -> Optional[float]:
        """Functional computed field for processing efficiency."""
        if self.processing_time and self.file_size:
            return calculate_processing_efficiency(self.processing_time, self.file_size)
        return None
    
    @computed_field
    @property
    def estimated_completion_time(self) -> Optional[datetime]:
        """Functional computed field for estimated completion."""
        if self.status == VideoStatus.PROCESSING and self.progress:
            if self.progress > 0:
                elapsed = datetime.now(timezone.utc) - self.created_at
                total_estimated = elapsed * (100 / self.progress)
                return self.created_at + total_estimated
        return None

class VideoSummary(BaseHeyGenModel):
    """Video summary model for list responses."""
    
    id: int
    video_id: str
    script: str
    status: VideoStatus
    quality: VideoQuality
    created_at: datetime
    processing_time: Optional[float] = None
    
    @computed_field
    @property
    def script_preview(self) -> str:
        """Functional computed field for script preview."""
        return self.script[:100] + "..." if len(self.script) > 100 else self.script
    
    @computed_field
    @property
    def is_completed(self) -> bool:
        """Functional computed field for completion status."""
        return self.status == VideoStatus.COMPLETED

# =============================================================================
# Model Usage Models with Functional Components
# =============================================================================

class ModelUsageBase(BaseHeyGenModel):
    """Base model usage model with functional validation."""
    
    model_type: ModelType = Field(..., description="Type of model used")
    model_name: str = Field(..., min_length=1, max_length=100, description="Model name/version")
    processing_time: float = Field(..., gt=0, description="Processing time in seconds")
    memory_usage: Optional[float] = Field(None, ge=0, description="Memory usage in MB")
    gpu_usage: Optional[float] = Field(None, ge=0, le=100, description="GPU usage percentage")
    
    @field_validator('processing_time')
    @classmethod
    def validate_processing_time(cls, v: float) -> float:
        """Functional processing time validation."""
        if v <= 0:
            raise ValueError("Processing time must be positive")
        return v
    
    @field_validator('gpu_usage')
    @classmethod
    def validate_gpu_usage(cls, v: Optional[float]) -> Optional[float]:
        """Functional GPU usage validation."""
        if v is not None and (v < 0 or v > 100):
            raise ValueError("GPU usage must be between 0 and 100")
        return v

class ModelUsageCreate(ModelUsageBase):
    """Model usage creation model."""
    
    user_id: int = Field(..., description="User ID")
    video_id: int = Field(..., description="Video ID")

class ModelUsageResponse(ModelUsageBase, TimestampedModel):
    """Model usage response model with computed fields."""
    
    id: int = Field(..., description="Usage record ID")
    user_id: int = Field(..., description="User ID")
    video_id: int = Field(..., description="Video ID")
    success: bool = Field(..., description="Whether usage was successful")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    
    @computed_field
    @property
    def efficiency_score(self) -> Optional[float]:
        """Functional computed field for efficiency score."""
        if not self.success or self.processing_time <= 0:
            return None
        
        # Base efficiency calculation
        efficiency = 100.0 / self.processing_time
        
        # Adjust for resource usage
        if self.memory_usage:
            efficiency *= (1000 / max(self.memory_usage, 1))
        
        if self.gpu_usage:
            efficiency *= (self.gpu_usage / 100)
        
        return min(efficiency, 100.0)
    
    @computed_field
    @property
    def resource_requirements(self) -> Dict[str, float]:
        """Functional computed field for resource requirements."""
        return ModelType.get_resource_requirements(self.model_type)

# =============================================================================
# API Key Models with Functional Components
# =============================================================================

class APIKeyCreate(BaseHeyGenModel):
    """API key creation model with functional validation."""
    
    name: str = Field(..., min_length=1, max_length=100, description="API key name")
    expires_at: Optional[datetime] = Field(None, description="Expiration date")
    permissions: List[str] = Field(default_factory=list, description="API key permissions")
    
    @field_validator('permissions')
    @classmethod
    def validate_permissions(cls, v: List[str]) -> List[str]:
        """Functional permissions validation."""
        valid_permissions = ['read', 'write', 'admin', 'video:create', 'video:read']
        for permission in v:
            if permission not in valid_permissions:
                raise ValueError(f"Invalid permission: {permission}")
        return list(set(v))  # Remove duplicates

class APIKeyResponse(BaseHeyGenModel, TimestampedModel):
    """API key response model with computed fields."""
    
    id: int = Field(..., description="API key ID")
    name: str = Field(..., description="API key name")
    key_prefix: str = Field(..., description="API key prefix")
    user_id: int = Field(..., description="User ID")
    expires_at: Optional[datetime] = Field(None, description="Expiration date")
    permissions: List[str] = Field(..., description="API key permissions")
    last_used_at: Optional[datetime] = Field(None, description="Last usage timestamp")
    is_active: bool = Field(..., description="API key active status")
    
    @computed_field
    @property
    def is_expired(self) -> bool:
        """Functional computed field for expiration status."""
        if not self.expires_at:
            return False
        return datetime.now(timezone.utc) > self.expires_at
    
    @computed_field
    @property
    def days_until_expiry(self) -> Optional[int]:
        """Functional computed field for days until expiry."""
        if not self.expires_at:
            return None
        delta = self.expires_at - datetime.now(timezone.utc)
        return max(0, delta.days)
    
    @computed_field
    @property
    def has_admin_permission(self) -> bool:
        """Functional computed field for admin permission."""
        return 'admin' in self.permissions

# =============================================================================
# Analytics Models with Functional Components
# =============================================================================

class AnalyticsRequest(BaseHeyGenModel):
    """Analytics request model with functional validation."""
    
    start_date: datetime = Field(..., description="Start date for analytics")
    end_date: datetime = Field(..., description="End date for analytics")
    metrics: List[str] = Field(default_factory=list, description="Metrics to calculate")
    group_by: Optional[str] = Field(None, description="Grouping field")
    
    @field_validator('end_date')
    @classmethod
    def validate_date_range(cls, v: datetime, info: ValidationInfo) -> datetime:
        """Functional date range validation."""
        start_date = info.data.get('start_date')
        if start_date and v <= start_date:
            raise ValueError("End date must be after start date")
        return v
    
    @field_validator('metrics')
    @classmethod
    def validate_metrics(cls, v: List[str]) -> List[str]:
        """Functional metrics validation."""
        valid_metrics = ['videos_created', 'processing_time', 'success_rate', 'user_activity']
        for metric in v:
            if metric not in valid_metrics:
                raise ValueError(f"Invalid metric: {metric}")
        return list(set(v))

class AnalyticsResponse(BaseHeyGenModel):
    """Analytics response model with computed fields."""
    
    period: Dict[str, datetime] = Field(..., description="Analysis period")
    metrics: Dict[str, Any] = Field(..., description="Calculated metrics")
    total_videos: int = Field(..., description="Total videos in period")
    total_users: int = Field(..., description="Total active users")
    average_processing_time: float = Field(..., description="Average processing time")
    success_rate: float = Field(..., ge=0, le=100, description="Success rate percentage")
    
    @computed_field
    @property
    def period_days(self) -> int:
        """Functional computed field for period duration."""
        return (self.period['end'] - self.period['start']).days
    
    @computed_field
    @property
    def videos_per_day(self) -> float:
        """Functional computed field for videos per day."""
        if self.period_days > 0:
            return round(self.total_videos / self.period_days, 2)
        return 0.0
    
    @computed_field
    @property
    def performance_score(self) -> float:
        """Functional computed field for performance score."""
        # Calculate performance score based on multiple factors
        score = 0.0
        
        # Success rate weight: 40%
        score += (self.success_rate / 100) * 40
        
        # Processing time weight: 30% (lower is better)
        if self.average_processing_time > 0:
            time_score = max(0, 100 - (self.average_processing_time * 10))
            score += (time_score / 100) * 30
        
        # User activity weight: 30%
        activity_score = min(100, self.total_users * 10)
        score += (activity_score / 100) * 30
        
        return round(score, 2)

# =============================================================================
# Functional Model Factories
# =============================================================================

def create_user_response(user_data: Dict[str, Any]) -> UserResponse:
    """Functional factory for user response."""
    return UserResponse(**user_data)

def create_video_response(video_data: Dict[str, Any]) -> VideoResponse:
    """Functional factory for video response."""
    return VideoResponse(**video_data)

def create_analytics_response(
    start_date: datetime,
    end_date: datetime,
    metrics_data: Dict[str, Any]
) -> AnalyticsResponse:
    """Functional factory for analytics response."""
    return AnalyticsResponse(
        period={"start": start_date, "end": end_date},
        **metrics_data
    )

def validate_and_create_video(video_data: Dict[str, Any]) -> VideoCreate:
    """Functional validation and creation for video."""
    return VideoCreate(**video_data)

# =============================================================================
# Export all models and functions
# =============================================================================

__all__ = [
    # Base models
    'BaseHeyGenModel',
    'TimestampedModel',
    
    # User models
    'UserBase',
    'UserCreate',
    'UserUpdate',
    'UserResponse',
    'UserSummary',
    
    # Video models
    'VideoBase',
    'VideoCreate',
    'VideoUpdate',
    'VideoResponse',
    'VideoSummary',
    
    # Model usage models
    'ModelUsageBase',
    'ModelUsageCreate',
    'ModelUsageResponse',
    
    # API key models
    'APIKeyCreate',
    'APIKeyResponse',
    
    # Analytics models
    'AnalyticsRequest',
    'AnalyticsResponse',
    
    # Enums
    'VideoStatus',
    'VideoQuality',
    'ModelType',
    
    # Functional validators
    'validate_username',
    'validate_password_strength',
    'validate_script_content',
    'validate_file_path',
    'validate_api_key_format',
    
    # Functional utilities
    'calculate_processing_efficiency',
    'generate_video_id',
    
    # Functional factories
    'create_user_response',
    'create_video_response',
    'create_analytics_response',
    'validate_and_create_video',
] 