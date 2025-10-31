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

import asyncio
import time
from typing import Dict, List, Any, Optional, Union, Literal
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from uuid import UUID
import structlog
from enum import Enum
from pydantic import BaseModel, Field, validator, root_validator, ConfigDict, computed_field
from .pydantic_optimizer import (
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Specialized Pydantic Models for HeyGen AI API
Optimized models for different data types with enhanced serialization.
"""


    OptimizedBaseModel, FastSerializationModel, CompactSerializationModel, ValidatedSerializationModel
)

logger = structlog.get_logger()

# =============================================================================
# Enums and Constants
# =============================================================================

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

class UserRole(str, Enum):
    """User roles."""
    USER = "user"
    ADMIN = "admin"
    MODERATOR = "moderator"
    PREMIUM = "premium"

class TemplateType(str, Enum):
    """Template types."""
    BUSINESS = "business"
    PERSONAL = "personal"
    EDUCATIONAL = "educational"
    ENTERTAINMENT = "entertainment"
    MARKETING = "marketing"

# =============================================================================
# Base Models
# =============================================================================

class TimestampMixin:
    """Mixin for timestamp fields."""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    @validator('updated_at', pre=True, always=True)
    def set_updated_at(cls, v) -> Any:
        return datetime.now(timezone.utc)

class IDMixin:
    """Mixin for ID fields."""
    id: UUID = Field(default_factory=UUID)

class StatusMixin:
    """Mixin for status fields."""
    status: str = Field(default="active")
    is_active: bool = Field(default=True)

# =============================================================================
# User Models
# =============================================================================

class UserBase(OptimizedBaseModel, TimestampMixin, IDMixin, StatusMixin):
    """Base user model with optimized serialization."""
    
    email: str = Field(..., description="User email address")
    first_name: str = Field(..., min_length=1, max_length=50)
    last_name: str = Field(..., min_length=1, max_length=50)
    role: UserRole = Field(default=UserRole.USER)
    
    @computed_field
    @property
    def full_name(self) -> str:
        """Get user's full name."""
        return f"{self.first_name} {self.last_name}"
    
    @computed_field
    @property
    def display_name(self) -> str:
        """Get user's display name."""
        return self.full_name or self.email.split('@')[0]

class UserProfile(FastSerializationModel, TimestampMixin):
    """Fast serialization user profile model."""
    
    user_id: UUID
    bio: Optional[str] = Field(None, max_length=500)
    avatar_url: Optional[str] = None
    website: Optional[str] = None
    location: Optional[str] = None
    company: Optional[str] = None
    job_title: Optional[str] = None
    social_links: Dict[str, str] = Field(default_factory=dict)
    
    @validator('avatar_url')
    def validate_avatar_url(cls, v) -> bool:
        if v and not v.startswith(('http://', 'https://')):
            raise ValueError('Avatar URL must be a valid HTTP/HTTPS URL')
        return v

class UserPreferences(CompactSerializationModel):
    """Compact serialization user preferences model."""
    
    user_id: UUID
    language: str = Field(default="en")
    timezone: str = Field(default="UTC")
    email_notifications: bool = Field(default=True)
    push_notifications: bool = Field(default=True)
    video_quality: VideoQuality = Field(default=VideoQuality.MEDIUM)
    auto_save: bool = Field(default=True)
    theme: str = Field(default="light")
    
    @validator('language')
    def validate_language(cls, v) -> bool:
        valid_languages = ['en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'zh', 'ja', 'ko']
        if v not in valid_languages:
            raise ValueError(f'Language must be one of: {valid_languages}')
        return v

class UserSession(ValidatedSerializationModel, TimestampMixin):
    """Validated serialization user session model."""
    
    session_id: UUID
    user_id: UUID
    token: str
    expires_at: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    is_active: bool = Field(default=True)
    
    @validator('expires_at')
    def validate_expires_at(cls, v) -> bool:
        if v <= datetime.now(timezone.utc):
            raise ValueError('Session must expire in the future')
        return v
    
    @computed_field
    @property
    def is_expired(self) -> bool:
        """Check if session is expired."""
        return datetime.now(timezone.utc) > self.expires_at

# =============================================================================
# Video Models
# =============================================================================

class VideoBase(OptimizedBaseModel, TimestampMixin, IDMixin):
    """Base video model with optimized serialization."""
    
    title: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=1000)
    user_id: UUID
    status: VideoStatus = Field(default=VideoStatus.PENDING)
    duration: Optional[int] = Field(None, ge=0)  # Duration in seconds
    file_size: Optional[int] = Field(None, ge=0)  # File size in bytes
    
    @computed_field
    @property
    def duration_formatted(self) -> Optional[str]:
        """Get formatted duration."""
        if not self.duration:
            return None
        
        minutes = self.duration // 60
        seconds = self.duration % 60
        return f"{minutes:02d}:{seconds:02d}"
    
    @computed_field
    @property
    def file_size_formatted(self) -> Optional[str]:
        """Get formatted file size."""
        if not self.file_size:
            return None
        
        for unit in ['B', 'KB', 'MB', 'GB']:
            if self.file_size < 1024:
                return f"{self.file_size:.1f} {unit}"
            self.file_size /= 1024
        return f"{self.file_size:.1f} TB"

class VideoMetadata(FastSerializationModel, TimestampMixin):
    """Fast serialization video metadata model."""
    
    video_id: UUID
    width: Optional[int] = Field(None, ge=0)
    height: Optional[int] = Field(None, ge=0)
    fps: Optional[float] = Field(None, ge=0)
    bitrate: Optional[int] = Field(None, ge=0)
    codec: Optional[str] = None
    format: Optional[str] = None
    thumbnail_url: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    categories: List[str] = Field(default_factory=list)
    
    @computed_field
    @property
    def resolution(self) -> Optional[str]:
        """Get video resolution."""
        if self.width and self.height:
            return f"{self.width}x{self.height}"
        return None

class VideoProcessing(CompactSerializationModel, TimestampMixin):
    """Compact serialization video processing model."""
    
    video_id: UUID
    progress: float = Field(default=0.0, ge=0.0, le=100.0)
    current_step: str = Field(default="initializing")
    total_steps: int = Field(default=1, ge=1)
    estimated_time: Optional[int] = Field(None, ge=0)  # Estimated time in seconds
    error_message: Optional[str] = None
    processing_log: List[str] = Field(default_factory=list)
    
    @computed_field
    @property
    def is_completed(self) -> bool:
        """Check if processing is completed."""
        return self.progress >= 100.0
    
    @computed_field
    @property
    def is_failed(self) -> bool:
        """Check if processing failed."""
        return bool(self.error_message)

class VideoAnalytics(ValidatedSerializationModel, TimestampMixin):
    """Validated serialization video analytics model."""
    
    video_id: UUID
    views: int = Field(default=0, ge=0)
    likes: int = Field(default=0, ge=0)
    dislikes: int = Field(default=0, ge=0)
    shares: int = Field(default=0, ge=0)
    comments: int = Field(default=0, ge=0)
    watch_time: int = Field(default=0, ge=0)  # Total watch time in seconds
    engagement_rate: float = Field(default=0.0, ge=0.0, le=100.0)
    
    @computed_field
    @property
    def total_engagement(self) -> int:
        """Get total engagement count."""
        return self.likes + self.dislikes + self.shares + self.comments
    
    @computed_field
    @property
    def watch_time_formatted(self) -> str:
        """Get formatted watch time."""
        hours = self.watch_time // 3600
        minutes = (self.watch_time % 3600) // 60
        seconds = self.watch_time % 60
        
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"

# =============================================================================
# Template Models
# =============================================================================

class TemplateBase(OptimizedBaseModel, TimestampMixin, IDMixin, StatusMixin):
    """Base template model with optimized serialization."""
    
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    template_type: TemplateType
    creator_id: UUID
    is_public: bool = Field(default=False)
    tags: List[str] = Field(default_factory=list)
    
    @computed_field
    @property
    def display_name(self) -> str:
        """Get template display name."""
        return self.name

class TemplateConfig(FastSerializationModel):
    """Fast serialization template configuration model."""
    
    template_id: UUID
    width: int = Field(default=1920, ge=1)
    height: int = Field(default=1080, ge=1)
    fps: int = Field(default=30, ge=1, le=60)
    duration: int = Field(default=60, ge=1)  # Duration in seconds
    background_color: str = Field(default="#000000")
    text_color: str = Field(default="#FFFFFF")
    font_family: str = Field(default="Arial")
    font_size: int = Field(default=24, ge=8, le=72)
    transitions: List[str] = Field(default_factory=list)
    effects: List[str] = Field(default_factory=list)
    
    @computed_field
    @property
    def aspect_ratio(self) -> float:
        """Get aspect ratio."""
        return self.width / self.height

class TemplateAssets(CompactSerializationModel):
    """Compact serialization template assets model."""
    
    template_id: UUID
    background_images: List[str] = Field(default_factory=list)
    background_videos: List[str] = Field(default_factory=list)
    audio_tracks: List[str] = Field(default_factory=list)
    fonts: List[str] = Field(default_factory=list)
    overlays: List[str] = Field(default_factory=list)
    icons: List[str] = Field(default_factory=list)
    
    @computed_field
    @property
    def total_assets(self) -> int:
        """Get total number of assets."""
        return (
            len(self.background_images) +
            len(self.background_videos) +
            len(self.audio_tracks) +
            len(self.fonts) +
            len(self.overlays) +
            len(self.icons)
        )

# =============================================================================
# Analytics Models
# =============================================================================

class AnalyticsBase(OptimizedBaseModel, TimestampMixin):
    """Base analytics model with optimized serialization."""
    
    user_id: Optional[UUID] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    referrer: Optional[str] = None

class UserAnalytics(FastSerializationModel, TimestampMixin):
    """Fast serialization user analytics model."""
    
    user_id: UUID
    total_videos: int = Field(default=0, ge=0)
    total_views: int = Field(default=0, ge=0)
    total_likes: int = Field(default=0, ge=0)
    total_shares: int = Field(default=0, ge=0)
    total_watch_time: int = Field(default=0, ge=0)  # Total watch time in seconds
    average_video_duration: float = Field(default=0.0, ge=0.0)
    engagement_rate: float = Field(default=0.0, ge=0.0, le=100.0)
    last_activity: Optional[datetime] = None
    
    @computed_field
    @property
    def average_views_per_video(self) -> float:
        """Get average views per video."""
        return self.total_views / self.total_videos if self.total_videos > 0 else 0.0
    
    @computed_field
    @property
    def total_watch_time_formatted(self) -> str:
        """Get formatted total watch time."""
        hours = self.total_watch_time // 3600
        minutes = (self.total_watch_time % 3600) // 60
        
        if hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m"

class PlatformAnalytics(CompactSerializationModel, TimestampMixin):
    """Compact serialization platform analytics model."""
    
    total_users: int = Field(default=0, ge=0)
    total_videos: int = Field(default=0, ge=0)
    total_views: int = Field(default=0, ge=0)
    total_likes: int = Field(default=0, ge=0)
    total_shares: int = Field(default=0, ge=0)
    total_watch_time: int = Field(default=0, ge=0)
    active_users_today: int = Field(default=0, ge=0)
    active_users_week: int = Field(default=0, ge=0)
    active_users_month: int = Field(default=0, ge=0)
    videos_created_today: int = Field(default=0, ge=0)
    videos_created_week: int = Field(default=0, ge=0)
    videos_created_month: int = Field(default=0, ge=0)
    
    @computed_field
    @property
    def average_engagement_rate(self) -> float:
        """Get average engagement rate."""
        if self.total_views > 0:
            return (self.total_likes + self.total_shares) / self.total_views * 100
        return 0.0

class PerformanceMetrics(ValidatedSerializationModel, TimestampMixin):
    """Validated serialization performance metrics model."""
    
    component: str
    metric_name: str
    value: float
    unit: str
    threshold: Optional[float] = None
    is_alert: bool = Field(default=False)
    tags: Dict[str, str] = Field(default_factory=dict)
    
    @validator('value')
    def validate_value(cls, v) -> bool:
        if not isinstance(v, (int, float)):
            raise ValueError('Value must be a number')
        return float(v)
    
    @computed_field
    @property
    def is_above_threshold(self) -> bool:
        """Check if value is above threshold."""
        if self.threshold is None:
            return False
        return self.value > self.threshold

# =============================================================================
# Search and Filter Models
# =============================================================================

class SearchQuery(OptimizedBaseModel):
    """Optimized search query model."""
    
    query: str = Field(..., min_length=1, max_length=200)
    filters: Dict[str, Any] = Field(default_factory=dict)
    sort_by: str = Field(default="relevance")
    sort_order: Literal["asc", "desc"] = Field(default="desc")
    page: int = Field(default=1, ge=1)
    per_page: int = Field(default=20, ge=1, le=100)
    
    @validator('query')
    def validate_query(cls, v) -> bool:
        return v.strip()
    
    @computed_field
    @property
    def offset(self) -> int:
        """Get offset for pagination."""
        return (self.page - 1) * self.per_page

class SearchResult(CompactSerializationModel):
    """Compact serialization search result model."""
    
    id: UUID
    title: str
    description: Optional[str] = None
    type: str  # video, user, template, etc.
    score: float = Field(ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    highlights: List[str] = Field(default_factory=list)
    
    @computed_field
    @property
    def relevance_percentage(self) -> int:
        """Get relevance as percentage."""
        return int(self.score * 100)

class SearchResponse(FastSerializationModel):
    """Fast serialization search response model."""
    
    query: str
    results: List[SearchResult] = Field(default_factory=list)
    total_count: int = Field(default=0, ge=0)
    page: int = Field(default=1, ge=1)
    per_page: int = Field(default=20, ge=1, le=100)
    total_pages: int = Field(default=0, ge=0)
    search_time_ms: float = Field(default=0.0, ge=0.0)
    
    @computed_field
    @property
    def has_results(self) -> bool:
        """Check if search has results."""
        return len(self.results) > 0
    
    @computed_field
    @property
    def has_more_pages(self) -> bool:
        """Check if there are more pages."""
        return self.page < self.total_pages

# =============================================================================
# Notification Models
# =============================================================================

class NotificationBase(OptimizedBaseModel, TimestampMixin, IDMixin):
    """Base notification model with optimized serialization."""
    
    user_id: UUID
    title: str = Field(..., min_length=1, max_length=200)
    message: str = Field(..., min_length=1, max_length=1000)
    type: str = Field(default="info")  # info, success, warning, error
    is_read: bool = Field(default=False)
    is_important: bool = Field(default=False)
    
    @computed_field
    @property
    def is_unread(self) -> bool:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        """Check if notification is unread."""
        return not self.is_read

class EmailNotification(FastSerializationModel, TimestampMixin):
    """Fast serialization email notification model."""
    
    notification_id: UUID
    recipient_email: str
    subject: str
    html_content: str
    text_content: str
    template_name: Optional[str] = None
    variables: Dict[str, Any] = Field(default_factory=dict)
    scheduled_at: Optional[datetime] = None
    sent_at: Optional[datetime] = None
    status: str = Field(default="pending")  # pending, sent, failed
    
    @computed_field
    @property
    def is_sent(self) -> bool:
        """Check if email is sent."""
        return self.status == "sent"
    
    @computed_field
    @property
    def is_scheduled(self) -> bool:
        """Check if email is scheduled."""
        return self.scheduled_at is not None

class PushNotification(CompactSerializationModel, TimestampMixin):
    """Compact serialization push notification model."""
    
    notification_id: UUID
    user_id: UUID
    device_token: str
    title: str
    body: str
    data: Dict[str, Any] = Field(default_factory=dict)
    badge: Optional[int] = Field(None, ge=0)
    sound: Optional[str] = None
    scheduled_at: Optional[datetime] = None
    sent_at: Optional[datetime] = None
    status: str = Field(default="pending")  # pending, sent, failed, delivered
    
    @computed_field
    @property
    def is_delivered(self) -> bool:
        """Check if push notification is delivered."""
        return self.status == "delivered"

# =============================================================================
# API Response Models
# =============================================================================

class APIResponse(OptimizedBaseModel):
    """Optimized API response model."""
    
    success: bool
    message: str
    data: Optional[Any] = None
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @computed_field
    @property
    def has_errors(self) -> bool:
        """Check if response has errors."""
        return len(self.errors) > 0
    
    @computed_field
    @property
    def has_warnings(self) -> bool:
        """Check if response has warnings."""
        return len(self.warnings) > 0

class PaginatedResponse(CompactSerializationModel):
    """Compact serialization paginated response model."""
    
    data: List[Any] = Field(default_factory=list)
    total_count: int = Field(default=0, ge=0)
    page: int = Field(default=1, ge=1)
    per_page: int = Field(default=20, ge=1, le=100)
    total_pages: int = Field(default=0, ge=0)
    has_next: bool = Field(default=False)
    has_previous: bool = Field(default=False)
    
    @computed_field
    @property
    def start_index(self) -> int:
        """Get start index for current page."""
        return (self.page - 1) * self.per_page + 1
    
    @computed_field
    @property
    def end_index(self) -> int:
        """Get end index for current page."""
        return min(self.page * self.per_page, self.total_count)

class ErrorResponse(ValidatedSerializationModel):
    """Validated serialization error response model."""
    
    error_code: str
    error_message: str
    error_details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    request_id: Optional[str] = None
    user_id: Optional[UUID] = None
    
    @computed_field
    @property
    def is_user_error(self) -> bool:
        """Check if error is user-related."""
        return self.error_code.startswith('USER_')
    
    @computed_field
    @property
    def is_system_error(self) -> bool:
        """Check if error is system-related."""
        return self.error_code.startswith('SYSTEM_')

# =============================================================================
# Model Registry
# =============================================================================

class ModelRegistry:
    """Registry for optimized models."""
    
    def __init__(self) -> Any:
        self.models: Dict[str, Type[OptimizedBaseModel]] = {}
        self.serialization_configs: Dict[str, Dict[str, Any]] = {}
    
    def register_model(
        self,
        model_class: Type[OptimizedBaseModel],
        alias: Optional[str] = None,
        serialization_config: Optional[Dict[str, Any]] = None
    ):
        """Register a model with optional serialization configuration."""
        model_name = alias or model_class.__name__
        self.models[model_name] = model_class
        
        if serialization_config:
            self.serialization_configs[model_name] = serialization_config
    
    def get_model(self, model_name: str) -> Optional[Type[OptimizedBaseModel]]:
        """Get registered model."""
        return self.models.get(model_name)
    
    def get_serialization_config(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get serialization configuration for model."""
        return self.serialization_configs.get(model_name)
    
    def list_models(self) -> List[str]:
        """List all registered models."""
        return list(self.models.keys())
    
    def clear(self) -> Any:
        """Clear all registered models."""
        self.models.clear()
        self.serialization_configs.clear()

# =============================================================================
# Model Factory
# =============================================================================

class ModelFactory:
    """Factory for creating optimized models."""
    
    def __init__(self, registry: ModelRegistry):
        
    """__init__ function."""
self.registry = registry
    
    def create_user_model(self, **kwargs) -> UserBase:
        """Create user model with validation."""
        return UserBase(**kwargs)
    
    def create_video_model(self, **kwargs) -> VideoBase:
        """Create video model with validation."""
        return VideoBase(**kwargs)
    
    def create_template_model(self, **kwargs) -> TemplateBase:
        """Create template model with validation."""
        return TemplateBase(**kwargs)
    
    def create_search_query(self, **kwargs) -> SearchQuery:
        """Create search query with validation."""
        return SearchQuery(**kwargs)
    
    async def create_api_response(self, **kwargs) -> APIResponse:
        """Create API response with validation."""
        return APIResponse(**kwargs)
    
    def create_paginated_response(self, **kwargs) -> PaginatedResponse:
        """Create paginated response with validation."""
        return PaginatedResponse(**kwargs)
    
    def create_error_response(self, **kwargs) -> ErrorResponse:
        """Create error response with validation."""
        return ErrorResponse(**kwargs)

# =============================================================================
# FastAPI Integration
# =============================================================================

def get_model_registry() -> ModelRegistry:
    """Dependency to get model registry."""
    registry = ModelRegistry()
    
    # Register all models
    registry.register_model(UserBase, "user")
    registry.register_model(VideoBase, "video")
    registry.register_model(TemplateBase, "template")
    registry.register_model(SearchQuery, "search_query")
    registry.register_model(APIResponse, "api_response")
    registry.register_model(PaginatedResponse, "paginated_response")
    registry.register_model(ErrorResponse, "error_response")
    
    return registry

def get_model_factory() -> ModelFactory:
    """Dependency to get model factory."""
    registry = get_model_registry()
    return ModelFactory(registry)

# =============================================================================
# Export
# =============================================================================

__all__ = [
    "VideoStatus",
    "VideoQuality",
    "UserRole",
    "TemplateType",
    "TimestampMixin",
    "IDMixin",
    "StatusMixin",
    "UserBase",
    "UserProfile",
    "UserPreferences",
    "UserSession",
    "VideoBase",
    "VideoMetadata",
    "VideoProcessing",
    "VideoAnalytics",
    "TemplateBase",
    "TemplateConfig",
    "TemplateAssets",
    "AnalyticsBase",
    "UserAnalytics",
    "PlatformAnalytics",
    "PerformanceMetrics",
    "SearchQuery",
    "SearchResult",
    "SearchResponse",
    "NotificationBase",
    "EmailNotification",
    "PushNotification",
    "APIResponse",
    "PaginatedResponse",
    "ErrorResponse",
    "ModelRegistry",
    "ModelFactory",
    "get_model_registry",
    "get_model_factory",
] 