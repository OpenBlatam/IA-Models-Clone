from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

from typing import Dict, Any, List, Optional, Literal, Union
from pydantic import BaseModel, Field, field_validator, ConfigDict, computed_field
from datetime import datetime, timezone
from enum import Enum
import re
        import hashlib
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Instagram Captions API v14.0 - Pydantic Schemas
Comprehensive input/output validation and response schemas using Pydantic BaseModel
"""



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


class AudienceType(str, Enum):
    """Target audience types"""
    GENERAL = "general"
    BUSINESS = "business"
    MILLENNIALS = "millennials"
    GEN_Z = "gen_z"
    CREATORS = "creators"
    LIFESTYLE = "lifestyle"


class ContentType(str, Enum):
    """Instagram content types"""
    POST = "post"
    STORY = "story"
    REEL = "reel"
    CAROUSEL = "carousel"


class OptimizationLevel(str, Enum):
    """Performance optimization levels"""
    ULTRA_FAST = "ultra_fast"
    BALANCED = "balanced"
    QUALITY = "quality"


class PriorityLevel(str, Enum):
    """Request priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class LanguageCode(str, Enum):
    """Supported language codes"""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    PORTUGUESE = "pt"


# =============================================================================
# BASE CONFIGURATION
# =============================================================================

class BaseSchemaConfig:
    """Base configuration for all schemas"""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=True,
        extra="forbid",
        json_encoders={
            datetime: lambda v: v.isoformat(),
        },
        populate_by_name=True,
        validate_default=True
    )


# =============================================================================
# REQUEST SCHEMAS
# =============================================================================

class CaptionGenerationRequest(BaseModel):
    """Comprehensive caption generation request schema"""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=True,
        extra="forbid",
        json_encoders={
            datetime: lambda v: v.isoformat(),
        },
        populate_by_name=True,
        validate_default=True
    )
    
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
    language: LanguageCode = Field(
        default=LanguageCode.ENGLISH,
        description="Language for caption generation"
    )
    
    timezone: Optional[str] = Field(
        default="UTC",
        pattern=r"^[A-Za-z_/]+$",
        description="Timezone for cultural adaptation"
    )
    
    # Advanced options
    brand_voice: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Brand voice guidelines"
    )
    
    keywords: Optional[List[str]] = Field(
        default=None,
        max_length=20,
        description="Specific keywords to include"
    )
    
    custom_instructions: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Custom instructions for generation"
    )
    
    # Client information
    client_id: str = Field(
        default="v14-client",
        min_length=1,
        max_length=100,
        pattern=r'^[a-zA-Z0-9_-]+$',
        description="Client identifier for tracking"
    )
    
    # Metadata
    campaign_id: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Campaign identifier"
    )
    
    user_id: Optional[str] = Field(
        default=None,
        max_length=100,
        description="User identifier for audit logging"
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
    
    @field_validator('keywords')
    @classmethod
    def validate_keywords(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Validate keywords list"""
        if v is None:
            return v
        
        if len(v) > 20:
            raise ValueError("Maximum 20 keywords allowed")
        
        # Validate each keyword
        for i, keyword in enumerate(v):
            if not keyword or not keyword.strip():
                raise ValueError(f"Keyword {i+1} cannot be empty")
            if len(keyword) > 50:
                raise ValueError(f"Keyword {i+1} too long (max 50 characters)")
        
        return [kw.strip() for kw in v if kw.strip()]
    
    @field_validator('custom_instructions')
    @classmethod
    def validate_custom_instructions(cls, v: Optional[str]) -> Optional[str]:
        """Validate custom instructions"""
        if v is None:
            return v
        
        if len(v) > 1000:
            raise ValueError("Custom instructions too long (max 1000 characters)")
        
        # Check for harmful content in instructions
        harmful_patterns = [r'<script', r'javascript:', r'data:']
        for pattern in harmful_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError(f"Potentially harmful content in instructions: {pattern}")
        
        return v.strip()
    
    @computed_field
    @property
    async def request_hash(self) -> str:
        """Generate unique hash for request caching"""
        content = f"{self.content_description}:{self.style}:{self.audience}:{self.hashtag_count}"
        return hashlib.md5(content.encode()).hexdigest()


class BatchCaptionRequest(BaseModel):
    """Batch caption generation request schema"""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=True,
        extra="forbid"
    )
    
    requests: List[CaptionGenerationRequest] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of caption generation requests"
    )
    
    batch_optimization: bool = Field(
        default=True,
        description="Enable batch optimization"
    )
    
    parallel_processing: bool = Field(
        default=True,
        description="Enable parallel processing"
    )
    
    @field_validator('requests')
    @classmethod
    async def validate_requests(cls, v: List[CaptionGenerationRequest]) -> List[CaptionGenerationRequest]:
        """Validate batch requests"""
        if not v:
            raise ValueError("Batch must contain at least one request")
        
        if len(v) > 100:
            raise ValueError("Batch size cannot exceed 100 requests")
        
        # Check for duplicate requests
        request_hashes = [req.request_hash for req in v]
        if len(request_hashes) != len(set(request_hashes)):
            raise ValueError("Duplicate requests detected in batch")
        
        return v


class CaptionOptimizationRequest(BaseModel):
    """Caption optimization request schema"""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=True,
        extra="forbid"
    )
    
    caption: str = Field(
        ...,
        min_length=5,
        max_length=2200,
        description="Caption text to optimize"
    )
    
    target_style: CaptionStyle = Field(
        default=CaptionStyle.CASUAL,
        description="Target caption style"
    )
    
    target_audience: AudienceType = Field(
        default=AudienceType.GENERAL,
        description="Target audience"
    )
    
    preserve_meaning: bool = Field(
        default=True,
        description="Whether to preserve original meaning"
    )
    
    enhancement_level: Literal["light", "moderate", "aggressive"] = Field(
        default="moderate",
        description="Level of optimization to apply"
    )
    
    @field_validator('caption')
    @classmethod
    def validate_caption(cls, v: str) -> str:
        """Validate caption text"""
        if not v or not v.strip():
            raise ValueError("Caption cannot be empty")
        
        if len(v) > 2200:
            raise ValueError("Caption too long (max 2200 characters)")
        
        return v.strip()


# =============================================================================
# RESPONSE SCHEMAS
# =============================================================================

class CaptionGenerationResponse(BaseModel):
    """Comprehensive caption generation response schema"""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=True,
        extra="forbid"
    )
    
    # Core response data
    request_id: str = Field(
        description="Unique request identifier"
    )
    
    caption: str = Field(
        description="Generated Instagram caption"
    )
    
    hashtags: List[str] = Field(
        description="Generated hashtags"
    )
    
    # Quality metrics
    quality_score: float = Field(
        ge=0.0,
        le=100.0,
        description="Caption quality score (0-100)"
    )
    
    engagement_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Predicted engagement score"
    )
    
    readability_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Readability score"
    )
    
    # Performance metrics
    processing_time: float = Field(
        ge=0.0,
        description="Processing time in seconds"
    )
    
    cache_hit: bool = Field(
        description="Whether response was served from cache"
    )
    
    optimization_level: OptimizationLevel = Field(
        description="Optimization level used"
    )
    
    # Metadata
    api_version: str = Field(
        default="14.0.0",
        description="API version"
    )
    
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Response timestamp"
    )
    
    # Analysis results
    sentiment: Optional[Literal["positive", "neutral", "negative"]] = Field(
        default=None,
        description="Sentiment analysis result"
    )
    
    tone: Optional[str] = Field(
        default=None,
        description="Detected tone of the caption"
    )
    
    word_count: int = Field(
        description="Number of words in caption"
    )
    
    character_count: int = Field(
        description="Number of characters in caption"
    )
    
    @field_validator('caption')
    @classmethod
    def validate_caption(cls, v: str) -> str:
        """Validate generated caption"""
        if not v or not v.strip():
            raise ValueError("Generated caption cannot be empty")
        
        if len(v) > 2200:
            raise ValueError("Generated caption too long")
        
        return v.strip()
    
    @field_validator('hashtags')
    @classmethod
    def validate_hashtags(cls, v: List[str]) -> List[str]:
        """Validate hashtags"""
        if not v:
            return v
        
        # Validate hashtag format
        for i, hashtag in enumerate(v):
            if not hashtag.startswith('#'):
                raise ValueError(f"Hashtag {i+1} must start with #")
            if len(hashtag) > 50:
                raise ValueError(f"Hashtag {i+1} too long (max 50 characters)")
        
        return v
    
    @computed_field
    @property
    def total_hashtags(self) -> int:
        """Get total number of hashtags"""
        return len(self.hashtags)
    
    @computed_field
    @property
    def is_optimized(self) -> bool:
        """Check if response was optimized"""
        return self.optimization_level != OptimizationLevel.ULTRA_FAST


class BatchCaptionResponse(BaseModel):
    """Batch caption generation response schema"""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=True,
        extra="forbid"
    )
    
    batch_id: str = Field(
        description="Unique batch identifier"
    )
    
    total_requests: int = Field(
        ge=1,
        description="Total number of requests"
    )
    
    successful_requests: int = Field(
        ge=0,
        description="Number of successful requests"
    )
    
    failed_requests: int = Field(
        ge=0,
        description="Number of failed requests"
    )
    
    processing_time: float = Field(
        ge=0.0,
        description="Total processing time"
    )
    
    responses: List[CaptionGenerationResponse] = Field(
        description="Generated responses"
    )
    
    errors: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Error details for failed requests"
    )
    
    # Performance metrics
    average_processing_time: float = Field(
        ge=0.0,
        description="Average processing time per request"
    )
    
    cache_hit_rate: float = Field(
        ge=0.0,
        le=100.0,
        description="Cache hit rate percentage"
    )
    
    # Metadata
    api_version: str = Field(
        default="14.0.0",
        description="API version"
    )
    
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Response timestamp"
    )
    
    @field_validator('successful_requests')
    @classmethod
    async def validate_successful_requests(cls, v: int, info) -> int:
        """Validate successful requests count"""
        total_requests = info.data.get('total_requests', 0)
        if v > total_requests:
            raise ValueError("Successful requests cannot exceed total requests")
        return v
    
    @field_validator('failed_requests')
    @classmethod
    async def validate_failed_requests(cls, v: int, info) -> int:
        """Validate failed requests count"""
        total_requests = info.data.get('total_requests', 0)
        successful_requests = info.data.get('successful_requests', 0)
        if v != (total_requests - successful_requests):
            raise ValueError("Failed requests count must equal total minus successful")
        return v
    
    @computed_field
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100


# =============================================================================
# ERROR SCHEMAS
# =============================================================================

class ErrorDetail(BaseModel):
    """Detailed error information schema"""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=True,
        extra="forbid"
    )
    
    error_code: str = Field(
        description="Unique error code"
    )
    
    message: str = Field(
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


class APIErrorResponse(BaseModel):
    """Standardized API error response schema"""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=True,
        extra="forbid"
    )
    
    error: bool = Field(
        default=True,
        description="Error flag"
    )
    
    error_code: str = Field(
        description="Unique error code"
    )
    
    message: str = Field(
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
        description="HTTP status code"
    )


# =============================================================================
# MONITORING SCHEMAS
# =============================================================================

class PerformanceMetrics(BaseModel):
    """Performance metrics schema"""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=True,
        extra="forbid"
    )
    
    total_requests: int = Field(
        ge=0,
        description="Total requests processed"
    )
    
    successful_requests: int = Field(
        ge=0,
        description="Successful requests"
    )
    
    failed_requests: int = Field(
        ge=0,
        description="Failed requests"
    )
    
    cache_hits: int = Field(
        ge=0,
        description="Number of cache hits"
    )
    
    cache_misses: int = Field(
        ge=0,
        description="Number of cache misses"
    )
    
    average_response_time: float = Field(
        ge=0.0,
        description="Average response time in seconds"
    )
    
    p95_response_time: float = Field(
        ge=0.0,
        description="95th percentile response time"
    )
    
    p99_response_time: float = Field(
        ge=0.0,
        description="99th percentile response time"
    )
    
    error_rate: float = Field(
        ge=0.0,
        le=100.0,
        description="Error rate percentage"
    )
    
    cache_hit_rate: float = Field(
        ge=0.0,
        le=100.0,
        description="Cache hit rate percentage"
    )
    
    uptime: float = Field(
        ge=0.0,
        description="System uptime in seconds"
    )
    
    @computed_field
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100


class HealthCheckResponse(BaseModel):
    """Health check response schema"""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=True,
        extra="forbid"
    )
    
    status: Literal["healthy", "degraded", "unhealthy"] = Field(
        description="System health status"
    )
    
    version: str = Field(
        description="API version"
    )
    
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Health check timestamp"
    )
    
    uptime: float = Field(
        ge=0.0,
        description="System uptime in seconds"
    )
    
    components: Dict[str, Literal["healthy", "degraded", "unhealthy"]] = Field(
        description="Individual component health status"
    )
    
    performance: PerformanceMetrics = Field(
        description="Current performance metrics"
    )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_error_response(
    error_code: str,
    message: str,
    status_code: int = 500,
    details: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None,
    path: Optional[str] = None,
    method: Optional[str] = None
) -> APIErrorResponse:
    """Create standardized error response"""
    return APIErrorResponse(
        error_code=error_code,
        message=message,
        details=details,
        request_id=request_id,
        path=path,
        method=method,
        status_code=status_code
    )


async def validate_request_data(data: Dict[str, Any], schema_class: type) -> BaseModel:
    """Validate request data against schema"""
    try:
        return schema_class(**data)
    except Exception as e:
        raise ValueError(f"Validation error: {str(e)}")


def sanitize_input_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize input data"""
    sanitized = {}
    
    for key, value in data.items():
        if isinstance(value, str):
            # Remove null bytes and normalize whitespace
            sanitized_value = value.replace('\x00', '').strip()
            # Basic XSS protection
            sanitized_value = re.sub(r'<script[^>]*>.*?</script>', '', sanitized_value, flags=re.IGNORECASE)
            sanitized_value = re.sub(r'javascript:', '', sanitized_value, flags=re.IGNORECASE)
            sanitized[key] = sanitized_value
        else:
            sanitized[key] = value
    
    return sanitized 