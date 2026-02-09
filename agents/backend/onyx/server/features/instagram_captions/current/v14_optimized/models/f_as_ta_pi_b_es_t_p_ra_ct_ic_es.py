from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

from typing import List, Optional, Dict, Any, Union, Literal
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from pydantic import (
import uuid
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
FastAPI Best Practices - Data Models

This module implements data models following FastAPI best practices:
- Pydantic v2 models with proper field validation
- Comprehensive examples and documentation
- Proper type hints and field constraints
- Nested models and relationships
- Response models with computed fields
- Validation and serialization best practices
"""

    BaseModel, Field, ConfigDict, field_validator, 
    model_validator, computed_field, EmailStr, HttpUrl,
    validator, root_validator
)


# =============================================================================
# ENUMS FOR TYPE SAFETY
# =============================================================================

class CaptionStyle(str, Enum):
    """Caption style enumeration"""
    CASUAL = "casual"
    FORMAL = "formal"
    CREATIVE = "creative"
    PROFESSIONAL = "professional"
    FUNNY = "funny"
    INSPIRATIONAL = "inspirational"
    MINIMAL = "minimal"
    DETAILED = "detailed"


class CaptionTone(str, Enum):
    """Caption tone enumeration"""
    FRIENDLY = "friendly"
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    ENTHUSIASTIC = "enthusiastic"
    CALM = "calm"
    ENERGETIC = "energetic"
    SOPHISTICATED = "sophisticated"
    PLAYFUL = "playful"


class LanguageCode(str, Enum):
    """Language code enumeration"""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    RUSSIAN = "ru"
    CHINESE = "zh"
    JAPANESE = "ja"
    KOREAN = "ko"


class OperationStatus(str, Enum):
    """Operation status enumeration"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# =============================================================================
# BASE MODELS WITH COMMON CONFIGURATION
# =============================================================================

class BaseModelWithConfig(BaseModel):
    """Base model with common configuration"""
    
    model_config = ConfigDict(
        # Use alias for field names (camelCase in JSON, snake_case in Python)
        populate_by_name=True,
        # Allow extra fields (useful for API versioning)
        extra="ignore",
        # Validate assignment
        validate_assignment=True,
        # Use enum values
        use_enum_values=True,
        # JSON serialization options
        json_encoders={
            datetime: lambda v: v.isoformat(),
            Decimal: lambda v: float(v),
            uuid.UUID: lambda v: str(v)
        },
        # Example generation
        json_schema_extra={
            "examples": [
                {
                    "name": "Example Model",
                    "description": "This is an example model"
                }
            ]
        }
    )


class TimestampedModel(BaseModelWithConfig):
    """Base model with timestamp fields"""
    
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp in UTC",
        examples=["2024-01-15T10:30:00Z"]
    )
    
    updated_at: Optional[datetime] = Field(
        default=None,
        description="Last update timestamp in UTC",
        examples=["2024-01-15T11:45:00Z"]
    )


# =============================================================================
# REQUEST MODELS
# =============================================================================

class CaptionGenerationRequest(BaseModelWithConfig):
    """
    Caption generation request model
    
    Following FastAPI best practices:
    - Clear field descriptions
    - Proper validation constraints
    - Comprehensive examples
    - Type safety with enums
    """
    
    content_description: str = Field(
        ...,
        min_length=10,
        max_length=1000,
        description="Detailed description of the content for caption generation",
        examples=[
            "Beautiful sunset over mountains with golden light reflecting on a calm lake",
            "Delicious homemade pizza with melted cheese and fresh basil"
        ]
    )
    
    style: CaptionStyle = Field(
        default=CaptionStyle.CASUAL,
        description="Desired caption style",
        examples=["casual", "formal", "creative"]
    )
    
    tone: CaptionTone = Field(
        default=CaptionTone.FRIENDLY,
        description="Desired caption tone",
        examples=["friendly", "professional", "enthusiastic"]
    )
    
    hashtag_count: int = Field(
        default=15,
        ge=0,
        le=30,
        description="Number of hashtags to generate",
        examples=[10, 15, 20]
    )
    
    language: LanguageCode = Field(
        default=LanguageCode.ENGLISH,
        description="Target language for caption generation",
        examples=["en", "es", "fr"]
    )
    
    include_emoji: bool = Field(
        default=True,
        description="Whether to include emojis in the caption",
        examples=[True, False]
    )
    
    max_length: Optional[int] = Field(
        default=None,
        ge=50,
        le=2200,
        description="Maximum caption length (Instagram limit is 2200 characters)",
        examples=[500, 1000, 1500]
    )
    
    # Custom validation
    @field_validator('content_description')
    @classmethod
    def validate_content_description(cls, v: str) -> str:
        """Validate content description"""
        if not v.strip():
            raise ValueError("Content description cannot be empty")
        if len(v.split()) < 3:
            raise ValueError("Content description must have at least 3 words")
        return v.strip()
    
    @field_validator('hashtag_count')
    @classmethod
    def validate_hashtag_count(cls, v: int) -> int:
        """Validate hashtag count"""
        if v > 20 and cls.language == LanguageCode.ENGLISH:
            # English captions work well with more hashtags
            return v
        elif v > 15:
            # Other languages work better with fewer hashtags
            return 15
        return v
    
    # Model validation
    @model_validator(mode='after')
    def validate_model(self) -> 'CaptionGenerationRequest':
        """Validate the entire model"""
        if self.max_length and len(self.content_description) > self.max_length:
            raise ValueError("Content description exceeds maximum length")
        return self
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "content_description": "Beautiful sunset over mountains with golden light reflecting on a calm lake",
                "style": "casual",
                "tone": "friendly",
                "hashtag_count": 15,
                "language": "en",
                "include_emoji": True,
                "max_length": 1000
            }
        }
    )


class BatchCaptionRequest(BaseModelWithConfig):
    """Batch caption generation request model"""
    
    requests: List[CaptionGenerationRequest] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="List of caption generation requests",
        examples=[[
            {
                "content_description": "Beautiful sunset over mountains",
                "style": "casual",
                "tone": "friendly"
            },
            {
                "content_description": "Delicious homemade pizza",
                "style": "creative",
                "tone": "enthusiastic"
            }
        ]]
    )
    
    max_concurrent: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum concurrent processing operations",
        examples=[3, 5, 10]
    )
    
    @field_validator('requests')
    @classmethod
    async def validate_requests(cls, v: List[CaptionGenerationRequest]) -> List[CaptionGenerationRequest]:
        """Validate batch requests"""
        if len(v) > 50:
            raise ValueError("Maximum 50 requests per batch")
        
        # Check for duplicate content descriptions
        descriptions = [req.content_description for req in v]
        if len(descriptions) != len(set(descriptions)):
            raise ValueError("Duplicate content descriptions are not allowed")
        
        return v


class UserPreferences(BaseModelWithConfig):
    """User preferences model"""
    
    user_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Unique user identifier",
        examples=["user_123", "user_abc"]
    )
    
    default_style: CaptionStyle = Field(
        default=CaptionStyle.CASUAL,
        description="Default caption style for the user"
    )
    
    default_tone: CaptionTone = Field(
        default=CaptionTone.FRIENDLY,
        description="Default caption tone for the user"
    )
    
    default_hashtag_count: int = Field(
        default=15,
        ge=0,
        le=30,
        description="Default number of hashtags"
    )
    
    preferred_language: LanguageCode = Field(
        default=LanguageCode.ENGLISH,
        description="User's preferred language"
    )
    
    include_emoji: bool = Field(
        default=True,
        description="Whether user prefers emojis in captions"
    )
    
    # Computed field example
    @computed_field
    @property
    def preferences_summary(self) -> str:
        """Summary of user preferences"""
        return f"{self.default_style.value} {self.default_tone.value} captions in {self.preferred_language.value}"


# =============================================================================
# RESPONSE MODELS
# =============================================================================

class CaptionGenerationResponse(BaseModelWithConfig):
    """
    Caption generation response model
    
    Following FastAPI best practices:
    - Clear response structure
    - Computed fields
    - Proper serialization
    - Performance metrics
    """
    
    caption: str = Field(
        ...,
        description="Generated caption text",
        examples=["Golden hour magic ✨ Nature's perfect lighting show"]
    )
    
    hashtags: List[str] = Field(
        ...,
        description="Generated hashtags",
        examples=[["#sunset", "#mountains", "#goldenhour", "#nature"]]
    )
    
    style: CaptionStyle = Field(
        ...,
        description="Applied caption style"
    )
    
    tone: CaptionTone = Field(
        ...,
        description="Applied caption tone"
    )
    
    language: LanguageCode = Field(
        ...,
        description="Generated language"
    )
    
    processing_time: float = Field(
        ...,
        ge=0,
        description="Processing time in seconds",
        examples=[1.23, 2.45, 0.87]
    )
    
    model_used: str = Field(
        ...,
        description="AI model used for generation",
        examples=["gpt-3.5-turbo", "gpt-4", "claude-3-sonnet"]
    )
    
    confidence_score: float = Field(
        ...,
        ge=0,
        le=1,
        description="Confidence score of the generation",
        examples=[0.95, 0.87, 0.92]
    )
    
    character_count: int = Field(
        ...,
        ge=0,
        description="Total character count of the caption"
    )
    
    word_count: int = Field(
        ...,
        ge=0,
        description="Total word count of the caption"
    )
    
    # Computed fields
    @computed_field
    @property
    def total_length(self) -> int:
        """Total length including hashtags"""
        caption_length = len(self.caption)
        hashtags_length = sum(len(hashtag) for hashtag in self.hashtags)
        return caption_length + hashtags_length
    
    @computed_field
    @property
    def is_within_limits(self) -> bool:
        """Check if caption is within Instagram limits"""
        return self.total_length <= 2200
    
    @computed_field
    @property
    def hashtag_string(self) -> str:
        """Hashtags as a single string"""
        return " ".join(self.hashtags)
    
    @computed_field
    @property
    def full_caption(self) -> str:
        """Complete caption with hashtags"""
        if self.hashtags:
            return f"{self.caption}\n\n{self.hashtag_string}"
        return self.caption
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "caption": "Golden hour magic ✨ Nature's perfect lighting show",
                "hashtags": ["#sunset", "#mountains", "#goldenhour", "#nature"],
                "style": "casual",
                "tone": "friendly",
                "language": "en",
                "processing_time": 1.23,
                "model_used": "gpt-3.5-turbo",
                "confidence_score": 0.95,
                "character_count": 45,
                "word_count": 8
            }
        }
    )


class BatchCaptionResponse(BaseModelWithConfig):
    """Batch caption generation response model"""
    
    results: List[CaptionGenerationResponse] = Field(
        ...,
        description="List of caption generation results"
    )
    
    total_processing_time: float = Field(
        ...,
        ge=0,
        description="Total processing time for all requests"
    )
    
    successful_count: int = Field(
        ...,
        ge=0,
        description="Number of successful generations"
    )
    
    failed_count: int = Field(
        ...,
        ge=0,
        description="Number of failed generations"
    )
    
    @computed_field
    @property
    def success_rate(self) -> float:
        """Success rate percentage"""
        total = len(self.results)
        return (self.successful_count / total * 100) if total > 0 else 0
    
    @computed_field
    @property
    def average_processing_time(self) -> float:
        """Average processing time per request"""
        return self.total_processing_time / len(self.results) if self.results else 0


# =============================================================================
# ERROR MODELS
# =============================================================================

class ErrorDetail(BaseModelWithConfig):
    """Error detail model"""
    
    field: Optional[str] = Field(
        default=None,
        description="Field that caused the error"
    )
    
    message: str = Field(
        ...,
        description="Error message"
    )
    
    code: str = Field(
        ...,
        description="Error code"
    )


class ErrorResponse(BaseModelWithConfig):
    """Standard error response model"""
    
    error: str = Field(
        ...,
        description="Error type"
    )
    
    message: str = Field(
        ...,
        description="Error message"
    )
    
    details: Optional[List[ErrorDetail]] = Field(
        default=None,
        description="Detailed error information"
    )
    
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Error timestamp"
    )
    
    request_id: Optional[str] = Field(
        default=None,
        description="Request identifier for tracking"
    )


# =============================================================================
# STATUS AND HEALTH MODELS
# =============================================================================

class ServiceStatus(BaseModelWithConfig):
    """Service status model"""
    
    service: str = Field(
        ...,
        description="Service name"
    )
    
    status: Literal["healthy", "unhealthy", "degraded"] = Field(
        ...,
        description="Service status"
    )
    
    version: str = Field(
        ...,
        description="Service version"
    )
    
    uptime: float = Field(
        ...,
        description="Service uptime in seconds"
    )
    
    last_check: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last health check timestamp"
    )


class HealthResponse(BaseModelWithConfig):
    """Health check response model"""
    
    status: Literal["healthy", "unhealthy"] = Field(
        ...,
        description="Overall health status"
    )
    
    version: str = Field(
        ...,
        description="API version"
    )
    
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Health check timestamp"
    )
    
    services: Dict[str, ServiceStatus] = Field(
        ...,
        description="Individual service statuses"
    )
    
    @computed_field
    @property
    def healthy_services_count(self) -> int:
        """Number of healthy services"""
        return sum(1 for service in self.services.values() if service.status == "healthy")
    
    @computed_field
    @property
    def total_services_count(self) -> int:
        """Total number of services"""
        return len(self.services)


# =============================================================================
# ANALYTICS AND METRICS MODELS
# =============================================================================

class CaptionAnalytics(BaseModelWithConfig):
    """Caption analytics model"""
    
    total_captions_generated: int = Field(
        ...,
        ge=0,
        description="Total number of captions generated"
    )
    
    average_processing_time: float = Field(
        ...,
        ge=0,
        description="Average processing time in seconds"
    )
    
    most_popular_style: CaptionStyle = Field(
        ...,
        description="Most popular caption style"
    )
    
    most_popular_tone: CaptionTone = Field(
        ...,
        description="Most popular caption tone"
    )
    
    language_distribution: Dict[str, int] = Field(
        ...,
        description="Distribution of languages used"
    )
    
    success_rate: float = Field(
        ...,
        ge=0,
        le=100,
        description="Success rate percentage"
    )
    
    cache_hit_rate: float = Field(
        ...,
        ge=0,
        le=100,
        description="Cache hit rate percentage"
    )


# =============================================================================
# EXPORT ALL MODELS
# =============================================================================

__all__ = [
    # Enums
    "CaptionStyle",
    "CaptionTone", 
    "LanguageCode",
    "OperationStatus",
    
    # Base models
    "BaseModelWithConfig",
    "TimestampedModel",
    
    # Request models
    "CaptionGenerationRequest",
    "BatchCaptionRequest",
    "UserPreferences",
    
    # Response models
    "CaptionGenerationResponse",
    "BatchCaptionResponse",
    
    # Error models
    "ErrorDetail",
    "ErrorResponse",
    
    # Status models
    "ServiceStatus",
    "HealthResponse",
    
    # Analytics models
    "CaptionAnalytics"
] 