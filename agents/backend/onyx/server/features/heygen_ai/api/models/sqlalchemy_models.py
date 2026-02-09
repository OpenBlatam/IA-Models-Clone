from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

from typing import Optional, List, Dict, Any, Union, TYPE_CHECKING
from datetime import datetime, timezone
from decimal import Decimal
import json
import uuid
from sqlalchemy import (
from sqlalchemy.orm import (
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.sql import func, text
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY as PG_ARRAY
from sqlalchemy.dialects.mysql import JSON as MYSQL_JSON
    from sqlalchemy.orm import Session
    from sqlalchemy.ext.asyncio import AsyncSession
from typing import Any, List, Dict, Optional
import logging
import asyncio
#!/usr/bin/env python3
"""
Enhanced SQLAlchemy 2.0 Models for HeyGen AI API
Modern SQLAlchemy 2.0 implementation with type annotations, relationships, and advanced features.
"""


# SQLAlchemy 2.0 imports
    Column, Integer, String, DateTime, Float, Boolean, Text, ForeignKey,
    Index, UniqueConstraint, CheckConstraint, Enum as SQLEnum, JSON,
    BigInteger, SmallInteger, Numeric, Date, Time, LargeBinary, ARRAY
)
    DeclarativeBase, Mapped, mapped_column, relationship, 
    declarative_mixin, declared_attr, validates, validates
)

# Type checking imports
if TYPE_CHECKING:


# =============================================================================
# Enhanced Declarative Base with SQLAlchemy 2.0 Features
# =============================================================================

class Base(AsyncAttrs, DeclarativeBase):
    """
    Enhanced declarative base with SQLAlchemy 2.0 features.
    Includes async support, type annotations, and common functionality.
    """
    
    # Common metadata for all models
    __abstract__ = True
    
    # Automatic timestamp columns
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        index=True
    )
    
    def __repr__(self) -> str:
        """Enhanced string representation."""
        attrs = []
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                if isinstance(value, str) and len(value) > 50:
                    value = value[:50] + "..."
                attrs.append(f"{key}={value}")
        return f"{self.__class__.__name__}({', '.join(attrs)})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        result = {}
        for column in self.__table__.columns:
            value = getattr(self, column.name)
            if isinstance(value, datetime):
                value = value.isoformat()
            elif isinstance(value, Decimal):
                value = float(value)
            result[column.name] = value
        return result
    
    def to_json(self) -> str:
        """Convert model to JSON string."""
        return json.dumps(self.to_dict(), default=str)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Base':
        """Create model instance from dictionary."""
        # Filter out non-column attributes
        columns = {c.name for c in cls.__table__.columns}
        filtered_data = {k: v for k, v in data.items() if k in columns}
        return cls(**filtered_data)


# =============================================================================
# Mixins for Common Functionality
# =============================================================================

@declarative_mixin
class TimestampMixin:
    """Mixin for automatic timestamp management."""
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
        comment="Record creation timestamp"
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        index=True,
        comment="Record last update timestamp"
    )


@declarative_mixin
class SoftDeleteMixin:
    """Mixin for soft delete functionality."""
    
    deleted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        comment="Soft delete timestamp"
    )
    is_deleted: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
        comment="Soft delete flag"
    )


@declarative_mixin
class UUIDMixin:
    """Mixin for UUID primary keys."""
    
    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        nullable=False,
        index=True,
        comment="Unique identifier"
    )


@declarative_mixin
class AuditMixin:
    """Mixin for audit trail functionality."""
    
    created_by: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        index=True,
        comment="User who created the record"
    )
    updated_by: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        index=True,
        comment="User who last updated the record"
    )
    version: Mapped[int] = mapped_column(
        Integer,
        default=1,
        nullable=False,
        comment="Record version for optimistic locking"
    )


# =============================================================================
# Enhanced User Model
# =============================================================================

class User(Base, TimestampMixin, SoftDeleteMixin, AuditMixin):
    """Enhanced user model with modern SQLAlchemy 2.0 features."""
    
    __tablename__ = "users"
    __table_args__ = (
        # Indexes for performance
        Index('idx_users_username_active', 'username', 'is_active'),
        Index('idx_users_email_active', 'email', 'is_active'),
        Index('idx_users_api_key_active', 'api_key', 'is_active'),
        Index('idx_users_created_at', 'created_at'),
        
        # Unique constraints
        UniqueConstraint('username', name='uq_users_username'),
        UniqueConstraint('email', name='uq_users_email'),
        UniqueConstraint('api_key', name='uq_users_api_key'),
        
        # Check constraints
        CheckConstraint("username ~ '^[a-zA-Z0-9_]{3,50}$'", name='ck_users_username_format'),
        CheckConstraint("email ~ '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}$'", name='ck_users_email_format'),
        
        # Table comments
        {'comment': 'User accounts for authentication and authorization'}
    )
    
    # Primary key
    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
        comment="User unique identifier"
    )
    
    # User information
    username: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        unique=True,
        index=True,
        comment="Unique username for login"
    )
    email: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        unique=True,
        index=True,
        comment="User email address"
    )
    api_key: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        unique=True,
        index=True,
        comment="API key for authentication"
    )
    
    # Status and preferences
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        index=True,
        comment="Account active status"
    )
    is_verified: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
        comment="Email verification status"
    )
    is_admin: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
        comment="Administrator privileges"
    )
    
    # Profile information
    first_name: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        comment="User first name"
    )
    last_name: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        comment="User last name"
    )
    avatar_url: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
        comment="User avatar image URL"
    )
    
    # Settings and preferences
    preferences: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        nullable=True,
        comment="User preferences and settings"
    )
    timezone: Mapped[Optional[str]] = mapped_column(
        String(50),
        default="UTC",
        nullable=True,
        comment="User timezone"
    )
    language: Mapped[Optional[str]] = mapped_column(
        String(10),
        default="en",
        nullable=True,
        comment="User preferred language"
    )
    
    # Usage tracking
    last_login_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        comment="Last login timestamp"
    )
    login_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        comment="Total login count"
    )
    
    # Relationships
    videos: Mapped[List["Video"]] = relationship(
        "Video",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="selectin",
        order_by="Video.created_at.desc()"
    )
    model_usage: Mapped[List["ModelUsage"]] = relationship(
        "ModelUsage",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="selectin"
    )
    
    # Validators
    @validates('username')
    def validate_username(self, key: str, value: str) -> str:
        """Validate username format."""
        if not value or len(value) < 3:
            raise ValueError("Username must be at least 3 characters long")
        if not value.replace('_', '').isalnum():
            raise ValueError("Username can only contain letters, numbers, and underscores")
        return value.lower()
    
    @validates('email')
    def validate_email(self, key: str, value: str) -> str:
        """Validate email format."""
        if not value or '@' not in value:
            raise ValueError("Invalid email format")
        return value.lower()
    
    # Methods
    def get_full_name(self) -> str:
        """Get user's full name."""
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        return self.username
    
    def update_last_login(self) -> None:
        """Update last login timestamp and count."""
        self.last_login_at = datetime.now(timezone.utc)
        self.login_count += 1
    
    def is_authenticated(self) -> bool:
        """Check if user is authenticated and active."""
        return self.is_active and not self.is_deleted


# =============================================================================
# Enhanced Video Model
# =============================================================================

class VideoStatus(str, SQLEnum):
    """Video processing status enumeration."""
    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class VideoQuality(str, SQLEnum):
    """Video quality enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"


class Video(Base, TimestampMixin, SoftDeleteMixin, AuditMixin):
    """Enhanced video model with comprehensive features."""
    
    __tablename__ = "videos"
    __table_args__ = (
        # Indexes for performance
        Index('idx_videos_user_id_status', 'user_id', 'status'),
        Index('idx_videos_video_id_status', 'video_id', 'status'),
        Index('idx_videos_created_at_status', 'created_at', 'status'),
        Index('idx_videos_quality_status', 'quality', 'status'),
        Index('idx_videos_processing_time', 'processing_time'),
        
        # Unique constraints
        UniqueConstraint('video_id', name='uq_videos_video_id'),
        
        # Check constraints
        CheckConstraint("processing_time >= 0", name='ck_videos_processing_time_positive'),
        CheckConstraint("file_size >= 0", name='ck_videos_file_size_positive'),
        CheckConstraint("duration >= 0", name='ck_videos_duration_positive'),
        
        # Table comments
        {'comment': 'Video generation requests and results'}
    )
    
    # Primary key
    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
        comment="Video unique identifier"
    )
    
    # Video identification
    video_id: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        unique=True,
        index=True,
        comment="External video identifier"
    )
    user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="User who created the video"
    )
    
    # Video content
    script: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Video script content"
    )
    voice_id: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
        comment="Voice model identifier"
    )
    language: Mapped[str] = mapped_column(
        String(10),
        default="en",
        nullable=False,
        index=True,
        comment="Video language"
    )
    quality: Mapped[VideoQuality] = mapped_column(
        SQLEnum(VideoQuality),
        default=VideoQuality.MEDIUM,
        nullable=False,
        index=True,
        comment="Video quality setting"
    )
    
    # Processing status
    status: Mapped[VideoStatus] = mapped_column(
        SQLEnum(VideoStatus),
        default=VideoStatus.PENDING,
        nullable=False,
        index=True,
        comment="Video processing status"
    )
    progress: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Processing progress (0-100)"
    )
    
    # File information
    file_path: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
        comment="Video file path"
    )
    file_size: Mapped[Optional[int]] = mapped_column(
        BigInteger,
        nullable=True,
        index=True,
        comment="Video file size in bytes"
    )
    duration: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Video duration in seconds"
    )
    format: Mapped[Optional[str]] = mapped_column(
        String(20),
        nullable=True,
        comment="Video file format"
    )
    resolution: Mapped[Optional[str]] = mapped_column(
        String(20),
        nullable=True,
        comment="Video resolution (e.g., 1920x1080)"
    )
    
    # Processing metrics
    processing_time: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        index=True,
        comment="Total processing time in seconds"
    )
    queue_time: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Time spent in queue"
    )
    start_time: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Processing start time"
    )
    end_time: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Processing end time"
    )
    
    # Metadata and settings
    metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        nullable=True,
        comment="Additional video metadata"
    )
    settings: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        nullable=True,
        comment="Video generation settings"
    )
    
    # Error handling
    error_message: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Error message if processing failed"
    )
    retry_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        comment="Number of processing retries"
    )
    
    # Relationships
    user: Mapped["User"] = relationship(
        "User",
        back_populates="videos",
        lazy="selectin"
    )
    model_usage: Mapped[List["ModelUsage"]] = relationship(
        "ModelUsage",
        back_populates="video",
        cascade="all, delete-orphan",
        lazy="selectin"
    )
    
    # Validators
    @validates('progress')
    def validate_progress(self, key: str, value: Optional[float]) -> Optional[float]:
        """Validate progress value."""
        if value is not None and (value < 0 or value > 100):
            raise ValueError("Progress must be between 0 and 100")
        return value
    
    @validates('processing_time')
    def validate_processing_time(self, key: str, value: Optional[float]) -> Optional[float]:
        """Validate processing time."""
        if value is not None and value < 0:
            raise ValueError("Processing time cannot be negative")
        return value
    
    # Methods
    def start_processing(self) -> None:
        """Mark video as processing."""
        self.status = VideoStatus.PROCESSING
        self.start_time = datetime.now(timezone.utc)
        self.progress = 0.0
    
    def complete_processing(self, file_path: str, file_size: int, duration: float) -> None:
        """Mark video as completed."""
        self.status = VideoStatus.COMPLETED
        self.end_time = datetime.now(timezone.utc)
        self.file_path = file_path
        self.file_size = file_size
        self.duration = duration
        self.progress = 100.0
        
        if self.start_time:
            self.processing_time = (self.end_time - self.start_time).total_seconds()
    
    def fail_processing(self, error_message: str) -> None:
        """Mark video as failed."""
        self.status = VideoStatus.FAILED
        self.end_time = datetime.now(timezone.utc)
        self.error_message = error_message
        self.retry_count += 1
        
        if self.start_time:
            self.processing_time = (self.end_time - self.start_time).total_seconds()
    
    def update_progress(self, progress: float) -> None:
        """Update processing progress."""
        self.progress = progress
    
    def is_completed(self) -> bool:
        """Check if video is completed."""
        return self.status == VideoStatus.COMPLETED
    
    def is_failed(self) -> bool:
        """Check if video failed."""
        return self.status == VideoStatus.FAILED
    
    def can_retry(self) -> bool:
        """Check if video can be retried."""
        return self.status == VideoStatus.FAILED and self.retry_count < 3


# =============================================================================
# Enhanced Model Usage Model
# =============================================================================

class ModelType(str, SQLEnum):
    """Model type enumeration."""
    TRANSFORMER = "transformer"
    DIFFUSION = "diffusion"
    TTS = "tts"
    FACE_DETECTION = "face_detection"
    EMOTION_ANALYSIS = "emotion_analysis"
    AUDIO_PROCESSING = "audio_processing"


class ModelUsage(Base, TimestampMixin):
    """Enhanced model usage tracking for analytics and monitoring."""
    
    __tablename__ = "model_usage"
    __table_args__ = (
        # Indexes for performance
        Index('idx_model_usage_user_id_created', 'user_id', 'created_at'),
        Index('idx_model_usage_video_id_created', 'video_id', 'created_at'),
        Index('idx_model_usage_model_type_created', 'model_type', 'created_at'),
        Index('idx_model_usage_processing_time', 'processing_time'),
        Index('idx_model_usage_memory_usage', 'memory_usage'),
        Index('idx_model_usage_gpu_usage', 'gpu_usage'),
        
        # Check constraints
        CheckConstraint("processing_time >= 0", name='ck_model_usage_processing_time_positive'),
        CheckConstraint("memory_usage >= 0", name='ck_model_usage_memory_positive'),
        CheckConstraint("gpu_usage >= 0", name='ck_model_usage_gpu_positive'),
        CheckConstraint("gpu_usage <= 100", name='ck_model_usage_gpu_max'),
        
        # Table comments
        {'comment': 'Model usage tracking for analytics and monitoring'}
    )
    
    # Primary key
    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
        comment="Usage record unique identifier"
    )
    
    # Foreign keys
    user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="User who triggered the model usage"
    )
    video_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("videos.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Video associated with the model usage"
    )
    
    # Model information
    model_type: Mapped[ModelType] = mapped_column(
        SQLEnum(ModelType),
        nullable=False,
        index=True,
        comment="Type of model used"
    )
    model_name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
        comment="Specific model name/version"
    )
    model_version: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        comment="Model version"
    )
    
    # Performance metrics
    processing_time: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        index=True,
        comment="Model processing time in seconds"
    )
    memory_usage: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        index=True,
        comment="Memory usage in MB"
    )
    gpu_usage: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        index=True,
        comment="GPU usage percentage (0-100)"
    )
    cpu_usage: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="CPU usage percentage (0-100)"
    )
    
    # Input/output metrics
    input_size: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="Input data size in bytes"
    )
    output_size: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="Output data size in bytes"
    )
    
    # Quality metrics
    accuracy: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Model accuracy score (0-1)"
    )
    confidence: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Model confidence score (0-1)"
    )
    
    # Error tracking
    success: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        index=True,
        comment="Whether the model usage was successful"
    )
    error_message: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Error message if usage failed"
    )
    
    # Additional metadata
    metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        nullable=True,
        comment="Additional usage metadata"
    )
    parameters: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        nullable=True,
        comment="Model parameters used"
    )
    
    # Relationships
    user: Mapped["User"] = relationship(
        "User",
        back_populates="model_usage",
        lazy="selectin"
    )
    video: Mapped["Video"] = relationship(
        "Video",
        back_populates="model_usage",
        lazy="selectin"
    )
    
    # Validators
    @validates('processing_time')
    def validate_processing_time(self, key: str, value: float) -> float:
        """Validate processing time."""
        if value < 0:
            raise ValueError("Processing time cannot be negative")
        return value
    
    @validates('gpu_usage')
    def validate_gpu_usage(self, key: str, value: Optional[float]) -> Optional[float]:
        """Validate GPU usage."""
        if value is not None and (value < 0 or value > 100):
            raise ValueError("GPU usage must be between 0 and 100")
        return value
    
    @validates('accuracy')
    def validate_accuracy(self, key: str, value: Optional[float]) -> Optional[float]:
        """Validate accuracy score."""
        if value is not None and (value < 0 or value > 1):
            raise ValueError("Accuracy must be between 0 and 1")
        return value
    
    # Methods
    def mark_success(self, processing_time: float, **kwargs) -> None:
        """Mark usage as successful."""
        self.success = True
        self.processing_time = processing_time
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def mark_failure(self, error_message: str, processing_time: float = 0.0) -> None:
        """Mark usage as failed."""
        self.success = False
        self.error_message = error_message
        self.processing_time = processing_time
    
    def get_efficiency_score(self) -> Optional[float]:
        """Calculate efficiency score based on performance metrics."""
        if not self.success or self.processing_time <= 0:
            return None
        
        # Simple efficiency calculation
        efficiency = 100.0 / self.processing_time
        
        # Adjust for resource usage
        if self.memory_usage:
            efficiency *= (1000 / max(self.memory_usage, 1))  # Prefer lower memory usage
        
        if self.gpu_usage:
            efficiency *= (self.gpu_usage / 100)  # Prefer higher GPU utilization
        
        return min(efficiency, 100.0)  # Cap at 100


# =============================================================================
# Additional Models for Extended Functionality
# =============================================================================

class VideoTemplate(Base, TimestampMixin, SoftDeleteMixin, AuditMixin):
    """Video templates for reusable configurations."""
    
    __tablename__ = "video_templates"
    __table_args__ = (
        Index('idx_video_templates_user_id', 'user_id'),
        Index('idx_video_templates_is_public', 'is_public'),
        UniqueConstraint('name', 'user_id', name='uq_video_templates_name_user'),
        {'comment': 'Reusable video generation templates'}
    )
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    user_id: Mapped[int] = mapped_column(
        Integer, 
        ForeignKey("users.id", ondelete="CASCADE"), 
        nullable=False, 
        index=True
    )
    is_public: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    
    # Template configuration
    default_voice_id: Mapped[str] = mapped_column(String(50), nullable=False)
    default_language: Mapped[str] = mapped_column(String(10), default="en", nullable=False)
    default_quality: Mapped[VideoQuality] = mapped_column(
        SQLEnum(VideoQuality), 
        default=VideoQuality.MEDIUM, 
        nullable=False
    )
    settings: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    
    # Relationships
    user: Mapped["User"] = relationship("User", lazy="selectin")


class UserSubscription(Base, TimestampMixin, AuditMixin):
    """User subscription and billing information."""
    
    __tablename__ = "user_subscriptions"
    __table_args__ = (
        Index('idx_user_subscriptions_user_id', 'user_id'),
        Index('idx_user_subscriptions_status', 'status'),
        Index('idx_user_subscriptions_expires_at', 'expires_at'),
        {'comment': 'User subscription and billing information'}
    )
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        Integer, 
        ForeignKey("users.id", ondelete="CASCADE"), 
        nullable=False, 
        index=True
    )
    
    # Subscription details
    plan_name: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="active", nullable=False)
    starts_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Usage limits
    video_limit: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    videos_used: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    
    # Billing
    amount: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 2), nullable=True)
    currency: Mapped[str] = mapped_column(String(3), default="USD", nullable=False)
    
    # Relationships
    user: Mapped["User"] = relationship("User", lazy="selectin")


# =============================================================================
# Model Registry and Utilities
# =============================================================================

# Model registry for easy access
MODEL_REGISTRY = {
    'user': User,
    'video': Video,
    'model_usage': ModelUsage,
    'video_template': VideoTemplate,
    'user_subscription': UserSubscription,
}

def get_model_by_name(name: str) -> Optional[type]:
    """Get model class by name."""
    return MODEL_REGISTRY.get(name.lower())

def get_all_models() -> List[type]:
    """Get all model classes."""
    return list(MODEL_REGISTRY.values())

def get_table_names() -> List[str]:
    """Get all table names."""
    return [model.__tablename__ for model in MODEL_REGISTRY.values()]

# Export all models
__all__ = [
    'Base',
    'User',
    'Video',
    'ModelUsage',
    'VideoTemplate',
    'UserSubscription',
    'VideoStatus',
    'VideoQuality',
    'ModelType',
    'TimestampMixin',
    'SoftDeleteMixin',
    'UUIDMixin',
    'AuditMixin',
    'MODEL_REGISTRY',
    'get_model_by_name',
    'get_all_models',
    'get_table_names',
] 