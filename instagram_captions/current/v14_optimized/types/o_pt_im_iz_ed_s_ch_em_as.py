from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import re
import hashlib
from typing import Dict, Any, List, Optional, Union, Literal
from datetime import datetime, timezone
from enum import Enum
from pydantic import Field, field_validator, computed_field
from core.optimized_serialization import OptimizedBaseModel, SerializationFormat
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Optimized Pydantic Schemas for Instagram Captions API v14.0

Ultra-fast serialization with:
- OptimizedBaseModel for enhanced performance
- Computed fields for caching
- Advanced validation with custom validators
- Batch processing support
- Memory-efficient serialization
"""


# Import optimized base model


# =============================================================================
# ENUMERATIONS
# =============================================================================

class CaptionStyle(str, Enum):
    """Caption writing styles"""
    CASUAL = "casual"
    PROFESSIONAL = "professional"
    INSPIRATIONAL = "inspirational"
    PLAYFUL = "playful"
    EDUCATIONAL = "educational"
    PROMOTIONAL = "promotional"
    STORYTELLING = "storytelling"
    MINIMALIST = "minimalist"


class AudienceType(str, Enum):
    """Target audience types"""
    GENERAL = "general"
    BUSINESS = "business"
    MILLENNIALS = "millennials"
    GEN_Z = "gen_z"
    CREATORS = "creators"
    LIFESTYLE = "lifestyle"
    FITNESS = "fitness"
    FOOD = "food"
    TRAVEL = "travel"
    FASHION = "fashion"


class ContentType(str, Enum):
    """Instagram content types"""
    POST = "post"
    STORY = "story"
    REEL = "reel"
    IGTV = "igtv"
    CAROUSEL = "carousel"
    HIGHLIGHT = "highlight"


class OptimizationLevel(str, Enum):
    """Performance optimization levels"""
    ULTRA_FAST = "ultra_fast"
    FAST = "fast"
    BALANCED = "balanced"
    QUALITY = "quality"
    MAXIMUM_QUALITY = "maximum_quality"


class PriorityLevel(str, Enum):
    """Request priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class Language(str, Enum):
    """Supported languages"""
    ENGLISH = "english"
    SPANISH = "spanish"
    FRENCH = "french"
    GERMAN = "german"
    ITALIAN = "italian"
    PORTUGUESE = "portuguese"
    RUSSIAN = "russian"
    JAPANESE = "japanese"
    KOREAN = "korean"
    CHINESE = "chinese"


# =============================================================================
# REQUEST SCHEMAS
# =============================================================================

class CaptionGenerationRequest(OptimizedBaseModel):
    """Optimized caption generation request schema"""
    
    # Core content
    content_description: str = Field(
        ...,
        min_length=5,
        max_length=2000,
        description="Content description for caption generation",
        examples=["Beautiful sunset at the beach with golden colors"]
    )
    
    # Style and audience
    style: CaptionStyle = Field(
        default=CaptionStyle.CASUAL,
        description="Caption writing style"
    )
    
    audience: AudienceType = Field(
        default=AudienceType.GENERAL,
        description="Target audience"
    )
    
    # Content type and hashtags
    content_type: ContentType = Field(
        default=ContentType.POST,
        description="Type of Instagram content"
    )
    
    hashtag_count: int = Field(
        default=15,
        ge=1,
        le=50,
        description="Number of hashtags to generate"
    )
    
    include_hashtags: bool = Field(
        default=True,
        description="Whether to include hashtags"
    )
    
    # Performance settings
    optimization_level: OptimizationLevel = Field(
        default=OptimizationLevel.BALANCED,
        description="Performance optimization level"
    )
    
    priority: PriorityLevel = Field(
        default=PriorityLevel.NORMAL,
        description="Request priority level"
    )
    
    # Language and localization
    language: Language = Field(
        default=Language.ENGLISH,
        description="Target language"
    )
    
    timezone: Optional[str] = Field(
        default="UTC",
        pattern=r"^[A-Za-z_/]+$",
        description="Timezone for cultural adaptation"
    )
    
    # Advanced features
    include_emojis: bool = Field(
        default=True,
        description="Whether to include emojis"
    )
    
    include_cta: bool = Field(
        default=True,
        description="Whether to include call-to-action"
    )
    
    max_length: int = Field(
        default=2200,
        ge=50,
        le=2200,
        description="Maximum caption length"
    )
    
    # User context
    user_id: Optional[str] = Field(
        default=None,
        description="User identifier for personalization"
    )
    
    brand_info: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Brand information for customization"
    )
    
    # Metadata
    request_id: Optional[str] = Field(
        default=None,
        description="Request identifier"
    )
    
    @field_validator('content_description')
    @classmethod
    def validate_content_description(cls, v: str) -> str:
        """Validate and sanitize content description"""
        if not v or not v.strip():
            raise ValueError("Content description cannot be empty")
        
        # Check for potentially harmful content
        harmful_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'data:text/html',
            r'vbscript:',
            r'on\w+\s*=',
            r'<iframe[^>]*>',
            r'<object[^>]*>',
            r'<embed[^>]*>'
        ]
        
        for pattern in harmful_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError(f"Potentially harmful content detected: {pattern}")
        
        return v.strip()
    
    @field_validator('hashtag_count')
    @classmethod
    def validate_hashtag_count(cls, v: int) -> int:
        """Validate hashtag count based on content type"""
        if v > 30 and cls.content_type == ContentType.STORY:
            raise ValueError("Stories should have maximum 30 hashtags")
        return v
    
    @computed_field
    @property
    async def request_hash(self) -> str:
        """Generate unique hash for request caching"""
        key_data = {
            "content": self.content_description,
            "style": self.style,
            "audience": self.audience,
            "hashtag_count": self.hashtag_count,
            "language": self.language
        }
        return hashlib.md5(str(key_data).encode()).hexdigest()
    
    @computed_field
    @property
    def estimated_tokens(self) -> int:
        """Estimate token count for AI processing"""
        return len(self.content_description.split()) * 2 + self.hashtag_count * 3
    
    @computed_field
    @property
    def complexity_score(self) -> float:
        """Calculate request complexity score"""
        base_score = 1.0
        
        # Adjust for content length
        if len(self.content_description) > 500:
            base_score += 0.5
        
        # Adjust for optimization level
        if self.optimization_level == OptimizationLevel.MAXIMUM_QUALITY:
            base_score += 1.0
        
        # Adjust for hashtag count
        if self.hashtag_count > 20:
            base_score += 0.3
        
        return min(base_score, 5.0)


class BatchCaptionRequest(OptimizedBaseModel):
    """Optimized batch caption generation request"""
    
    requests: List[CaptionGenerationRequest] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of caption generation requests"
    )
    
    batch_id: Optional[str] = Field(
        default=None,
        description="Batch identifier"
    )
    
    # Batch processing settings
    parallel_processing: bool = Field(
        default=True,
        description="Enable parallel processing"
    )
    
    max_concurrent: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum concurrent requests"
    )
    
    # Quality settings
    quality_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum quality threshold"
    )
    
    @field_validator('requests')
    @classmethod
    async def validate_requests(cls, v: List[CaptionGenerationRequest]) -> List[CaptionGenerationRequest]:
        """Validate batch requests"""
        if len(v) > 100:
            raise ValueError("Batch size cannot exceed 100 requests")
        
        # Check for duplicate requests
        request_hashes = [req.request_hash for req in v]
        if len(request_hashes) != len(set(request_hashes)):
            raise ValueError("Duplicate requests detected in batch")
        
        return v
    
    @computed_field
    @property
    def total_estimated_tokens(self) -> int:
        """Calculate total estimated tokens for batch"""
        return sum(req.estimated_tokens for req in self.requests)
    
    @computed_field
    @property
    def average_complexity(self) -> float:
        """Calculate average complexity score"""
        return sum(req.complexity_score for req in self.requests) / len(self.requests)


# =============================================================================
# RESPONSE SCHEMAS
# =============================================================================

class CaptionVariation(OptimizedBaseModel):
    """Individual caption variation"""
    
    caption: str = Field(
        ...,
        description="Generated caption text"
    )
    
    hashtags: List[str] = Field(
        default_factory=list,
        description="Generated hashtags"
    )
    
    quality_score: float = Field(
        ge=0.0,
        le=100.0,
        description="Caption quality score"
    )
    
    engagement_prediction: float = Field(
        ge=0.0,
        le=1.0,
        description="Predicted engagement rate"
    )
    
    readability_score: float = Field(
        ge=0.0,
        le=100.0,
        description="Readability score"
    )
    
    word_count: int = Field(
        ge=0,
        description="Number of words in caption"
    )
    
    character_count: int = Field(
        ge=0,
        description="Number of characters in caption"
    )
    
    emoji_count: int = Field(
        ge=0,
        description="Number of emojis used"
    )
    
    @computed_field
    @property
    def is_optimal_length(self) -> bool:
        """Check if caption length is optimal"""
        return 50 <= self.character_count <= 2200
    
    @computed_field
    @property
    def hashtag_density(self) -> float:
        """Calculate hashtag density"""
        if self.word_count == 0:
            return 0.0
        return len(self.hashtags) / self.word_count


class CaptionGenerationResponse(OptimizedBaseModel):
    """Optimized caption generation response"""
    
    request_id: str = Field(
        ...,
        description="Request identifier"
    )
    
    variations: List[CaptionVariation] = Field(
        ...,
        min_length=1,
        description="Generated caption variations"
    )
    
    # Performance metrics
    processing_time: float = Field(
        ge=0.0,
        description="Processing time in seconds"
    )
    
    cache_hit: bool = Field(
        default=False,
        description="Whether response was served from cache"
    )
    
    model_used: str = Field(
        description="AI model used for generation"
    )
    
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall confidence score"
    )
    
    # Quality metrics
    best_variation_index: int = Field(
        ge=0,
        description="Index of best variation"
    )
    
    average_quality_score: float = Field(
        ge=0.0,
        le=100.0,
        description="Average quality score across variations"
    )
    
    # Metadata
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Generation timestamp"
    )
    
    api_version: str = Field(
        default="14.0.0",
        description="API version"
    )
    
    optimization_level: OptimizationLevel = Field(
        description="Optimization level used"
    )
    
    @computed_field
    @property
    def best_variation(self) -> CaptionVariation:
        """Get the best variation"""
        return self.variations[self.best_variation_index]
    
    @computed_field
    @property
    def total_variations(self) -> int:
        """Get total number of variations"""
        return len(self.variations)
    
    @computed_field
    @property
    def has_high_quality_variations(self) -> bool:
        """Check if any variation has high quality"""
        return any(var.quality_score >= 80.0 for var in self.variations)


class BatchCaptionResponse(OptimizedBaseModel):
    """Optimized batch caption generation response"""
    
    batch_id: str = Field(
        ...,
        description="Batch identifier"
    )
    
    responses: List[CaptionGenerationResponse] = Field(
        ...,
        description="Individual caption responses"
    )
    
    # Batch metrics
    total_processing_time: float = Field(
        ge=0.0,
        description="Total processing time for batch"
    )
    
    successful_count: int = Field(
        ge=0,
        description="Number of successful generations"
    )
    
    failed_count: int = Field(
        ge=0,
        description="Number of failed generations"
    )
    
    errors: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Error details for failed requests"
    )
    
    # Quality metrics
    average_quality_score: float = Field(
        ge=0.0,
        le=100.0,
        description="Average quality score across all responses"
    )
    
    cache_hit_rate: float = Field(
        ge=0.0,
        le=1.0,
        description="Cache hit rate for batch"
    )
    
    # Metadata
    completed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Batch completion timestamp"
    )
    
    @computed_field
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        total = len(self.responses)
        return self.successful_count / total if total > 0 else 0.0
    
    @computed_field
    @property
    def total_variations(self) -> int:
        """Calculate total variations across all responses"""
        return sum(len(resp.variations) for resp in self.responses)
    
    @computed_field
    @property
    def average_processing_time(self) -> float:
        """Calculate average processing time per request"""
        return self.total_processing_time / len(self.responses) if self.responses else 0.0


# =============================================================================
# ERROR SCHEMAS
# =============================================================================

class ErrorDetail(OptimizedBaseModel):
    """Detailed error information"""
    
    error_code: str = Field(
        ...,
        description="Unique error code"
    )
    
    message: str = Field(
        ...,
        description="Human-readable error message"
    )
    
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional error details"
    )
    
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Error timestamp"
    )
    
    request_id: Optional[str] = Field(
        default=None,
        description="Request identifier"
    )
    
    path: Optional[str] = Field(
        default=None,
        description="Request path"
    )
    
    method: Optional[str] = Field(
        default=None,
        description="HTTP method"
    )


class APIErrorResponse(OptimizedBaseModel):
    """Standardized API error response"""
    
    error: bool = Field(
        default=True,
        description="Error flag"
    )
    
    error_code: str = Field(
        ...,
        description="Unique error code"
    )
    
    message: str = Field(
        ...,
        description="Human-readable error message"
    )
    
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional error details"
    )
    
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Error timestamp"
    )
    
    request_id: Optional[str] = Field(
        default=None,
        description="Request identifier"
    )
    
    path: Optional[str] = Field(
        default=None,
        description="Request path"
    )
    
    method: Optional[str] = Field(
        default=None,
        description="HTTP method"
    )
    
    status_code: int = Field(
        ...,
        description="HTTP status code"
    )


# =============================================================================
# PERFORMANCE SCHEMAS
# =============================================================================

class PerformanceMetrics(OptimizedBaseModel):
    """Performance metrics for monitoring"""
    
    request_count: int = Field(
        ge=0,
        description="Total number of requests"
    )
    
    success_count: int = Field(
        ge=0,
        description="Number of successful requests"
    )
    
    error_count: int = Field(
        ge=0,
        description="Number of failed requests"
    )
    
    average_response_time: float = Field(
        ge=0.0,
        description="Average response time in seconds"
    )
    
    cache_hit_rate: float = Field(
        ge=0.0,
        le=1.0,
        description="Cache hit rate"
    )
    
    throughput_per_second: float = Field(
        ge=0.0,
        description="Requests processed per second"
    )
    
    memory_usage_mb: float = Field(
        ge=0.0,
        description="Memory usage in MB"
    )
    
    cpu_usage_percent: float = Field(
        ge=0.0,
        le=100.0,
        description="CPU usage percentage"
    )
    
    @computed_field
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        return self.success_count / self.request_count if self.request_count > 0 else 0.0
    
    @computed_field
    @property
    def error_rate(self) -> float:
        """Calculate error rate"""
        return self.error_count / self.request_count if self.request_count > 0 else 0.0


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

async def validate_request_data(data: Dict[str, Any], schema_class: type) -> OptimizedBaseModel:
    """Validate request data against schema with optimization"""
    try:
        return schema_class.model_validate(data)
    except Exception as e:
        raise ValueError(f"Validation error: {str(e)}")


def serialize_response(response: OptimizedBaseModel, format: SerializationFormat = SerializationFormat.JSON) -> Union[str, bytes]:
    """Serialize response with optimization"""
    return response.to_json() if format == SerializationFormat.JSON else response.to_dict()


async def deserialize_request(data: Union[str, Dict[str, Any]], schema_class: type) -> OptimizedBaseModel:
    """Deserialize request data with optimization"""
    if isinstance(data, str):
        return schema_class.from_json(data)
    else:
        return schema_class.from_dict(data)


# =============================================================================
# SCHEMA REGISTRY
# =============================================================================

SCHEMA_REGISTRY = {
    "CaptionGenerationRequest": CaptionGenerationRequest,
    "BatchCaptionRequest": BatchCaptionRequest,
    "CaptionGenerationResponse": CaptionGenerationResponse,
    "BatchCaptionResponse": BatchCaptionResponse,
    "APIErrorResponse": APIErrorResponse,
    "PerformanceMetrics": PerformanceMetrics,
}


def get_schema_class(schema_name: str) -> Optional[type]:
    """Get schema class by name"""
    return SCHEMA_REGISTRY.get(schema_name)


def register_schema(schema_name: str, schema_class: type):
    """Register a new schema"""
    SCHEMA_REGISTRY[schema_name] = schema_class 