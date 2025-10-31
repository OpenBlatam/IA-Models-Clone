from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime
import uuid
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Instagram Captions API v13.0 - Domain Entities

Clean Architecture domain layer with pure business entities.
No dependencies on external frameworks or infrastructure.
"""



class CaptionStyle(str, Enum):
    """Caption style enumeration for business logic."""
    CASUAL = "casual"
    PROFESSIONAL = "professional" 
    LUXURY = "luxury"
    EDUCATIONAL = "educational"
    STORYTELLING = "storytelling"
    CALL_TO_ACTION = "call_to_action"
    PLAYFUL = "playful"
    INSPIRATIONAL = "inspirational"


class QualityLevel(str, Enum):
    """Quality levels for generated content."""
    EXCELLENT = "excellent"  # 90-100
    GOOD = "good"           # 75-89
    AVERAGE = "average"     # 60-74
    POOR = "poor"          # <60


class CacheStrategy(str, Enum):
    """Caching strategy options."""
    AGGRESSIVE = "aggressive"  # Maximum caching
    BALANCED = "balanced"     # Intelligent caching
    MINIMAL = "minimal"       # Basic caching
    DISABLED = "disabled"     # No caching


@dataclass(frozen=True)
class RequestId:
    """Value object for request identification."""
    value: str = field(default_factory=lambda: f"v13-{uuid.uuid4().hex[:8]}")
    
    def __post_init__(self) -> Any:
        if not self.value or len(self.value) < 3:
            raise ValueError("RequestId must have valid value")


@dataclass(frozen=True)
class Content:
    """Value object for content description."""
    description: str
    
    def __post_init__(self) -> Any:
        if not self.description or len(self.description.strip()) < 3:
            raise ValueError("Content description must be at least 3 characters")
        
        # Sanitize content
        object.__setattr__(self, 'description', self.description.strip()[:2000])
    
    @property
    def is_valid(self) -> bool:
        return len(self.description) >= 3
    
    @property
    def word_count(self) -> int:
        return len(self.description.split())


@dataclass(frozen=True)
class Hashtags:
    """Value object for hashtag collection."""
    tags: List[str]
    
    def __post_init__(self) -> Any:
        # Validate and clean hashtags
        clean_tags = []
        for tag in self.tags:
            if isinstance(tag, str) and len(tag) > 1:
                # Ensure hashtag format
                clean_tag = tag if tag.startswith('#') else f"#{tag}"
                clean_tags.append(clean_tag)
        
        object.__setattr__(self, 'tags', clean_tags[:50])  # Max 50 hashtags
    
    @property
    def count(self) -> int:
        return len(self.tags)
    
    def to_string(self, limit: Optional[int] = None) -> str:
        tags_to_use = self.tags[:limit] if limit else self.tags
        return " ".join(tags_to_use)


@dataclass(frozen=True)
class QualityMetrics:
    """Value object for quality assessment."""
    score: float
    engagement_prediction: float
    virality_score: float
    readability_score: float
    
    def __post_init__(self) -> Any:
        # Validate score ranges
        for field_name in ['score', 'engagement_prediction', 'virality_score', 'readability_score']:
            value = getattr(self, field_name)
            if not 0 <= value <= 100:
                raise ValueError(f"{field_name} must be between 0 and 100")
    
    @property
    def quality_level(self) -> QualityLevel:
        if self.score >= 90:
            return QualityLevel.EXCELLENT
        elif self.score >= 75:
            return QualityLevel.GOOD
        elif self.score >= 60:
            return QualityLevel.AVERAGE
        else:
            return QualityLevel.POOR
    
    @property
    def overall_rating(self) -> float:
        return (self.score + self.engagement_prediction + self.virality_score) / 3


@dataclass(frozen=True)
class PerformanceMetrics:
    """Value object for performance tracking."""
    processing_time: float
    cache_hit: bool
    provider_used: str
    model_used: str
    confidence_score: float
    
    def __post_init__(self) -> Any:
        if self.processing_time < 0:
            raise ValueError("Processing time cannot be negative")
        if not 0 <= self.confidence_score <= 1:
            raise ValueError("Confidence score must be between 0 and 1")
    
    @property
    def processing_time_ms(self) -> float:
        return self.processing_time * 1000
    
    @property
    def is_fast(self) -> bool:
        return self.processing_time < 0.020  # Sub-20ms
    
    @property
    def is_ultra_fast(self) -> bool:
        return self.processing_time < 0.010  # Sub-10ms


@dataclass
class CaptionRequest:
    """Domain entity for caption generation requests."""
    
    # Core request data
    request_id: RequestId
    content: Content
    style: CaptionStyle
    hashtag_count: int = 20
    
    # Optional configuration
    cache_strategy: CacheStrategy = CacheStrategy.AGGRESSIVE
    priority: str = "normal"
    custom_instructions: Optional[str] = None
    
    # Client information
    client_id: str = "modular-v13"
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None
    
    # Advanced options
    enable_advanced_analysis: bool = True
    enable_sentiment_analysis: bool = True
    enable_seo_optimization: bool = False
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self) -> Any:
        # Validate hashtag count
        if not 5 <= self.hashtag_count <= 50:
            self.hashtag_count = max(5, min(50, self.hashtag_count))
        
        # Validate custom instructions
        if self.custom_instructions and len(self.custom_instructions) > 500:
            self.custom_instructions = self.custom_instructions[:500]
    
    @property
    async def is_premium_request(self) -> bool:
        return self.enable_advanced_analysis and self.enable_sentiment_analysis
    
    @property
    def cache_key(self) -> str:
        """Generate cache key for this request."""
        key_parts = [
            self.content.description[:50],
            self.style.value,
            str(self.hashtag_count),
            self.custom_instructions or ""
        ]
        return "|".join(key_parts)


@dataclass
class CaptionResponse:
    """Domain entity for caption generation responses."""
    
    # Core response data
    request_id: RequestId
    caption: str
    hashtags: Hashtags
    
    # Quality assessment
    quality_metrics: QualityMetrics
    performance_metrics: PerformanceMetrics
    
    # Advanced analysis (optional)
    sentiment_analysis: Optional[Dict[str, Any]] = None
    seo_analysis: Optional[Dict[str, Any]] = None
    competitor_insights: Optional[Dict[str, Any]] = None
    
    # Metadata
    api_version: str = "13.0.0"
    generated_at: datetime = field(default_factory=datetime.utcnow)
    
    # Enterprise features
    tenant_id: Optional[str] = None
    audit_id: Optional[str] = None
    
    @property
    def is_high_quality(self) -> bool:
        return self.quality_metrics.quality_level in [QualityLevel.EXCELLENT, QualityLevel.GOOD]
    
    @property
    def performance_grade(self) -> str:
        if self.performance_metrics.is_ultra_fast:
            return "ULTRA_FAST"
        elif self.performance_metrics.is_fast:
            return "FAST"
        else:
            return "NORMAL"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "request_id": self.request_id.value,
            "caption": self.caption,
            "hashtags": self.hashtags.tags,
            "quality_score": self.quality_metrics.score,
            "engagement_prediction": self.quality_metrics.engagement_prediction,
            "virality_score": self.quality_metrics.virality_score,
            "processing_time": self.performance_metrics.processing_time,
            "cache_hit": self.performance_metrics.cache_hit,
            "api_version": self.api_version,
            "generated_at": self.generated_at.isoformat(),
            "performance_grade": self.performance_grade
        }


@dataclass
class BatchRequest:
    """Domain entity for batch processing requests."""
    
    batch_id: RequestId
    requests: List[CaptionRequest]
    batch_strategy: str = "parallel"
    max_concurrent: int = 50
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self) -> Any:
        if not self.requests:
            raise ValueError("Batch requests cannot be empty")
        if len(self.requests) > 100:
            raise ValueError("Batch size cannot exceed 100 requests")
    
    @property
    async def request_count(self) -> int:
        return len(self.requests)
    
    @property
    def estimated_processing_time(self) -> float:
        # Estimate based on request count and strategy
        if self.batch_strategy == "parallel":
            return 0.020 * (self.request_count / self.max_concurrent)
        else:
            return 0.020 * self.request_count


@dataclass
class BatchResponse:
    """Domain entity for batch processing responses."""
    
    batch_id: RequestId
    responses: List[CaptionResponse]
    failed_requests: List[Dict[str, Any]]
    
    total_requests: int
    successful_requests: int
    failed_requests_count: int
    
    processing_time: float
    average_quality: float
    
    generated_at: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def success_rate(self) -> float:
        return self.successful_requests / max(self.total_requests, 1)
    
    @property
    def throughput(self) -> float:
        return self.total_requests / max(self.processing_time, 0.001)
    
    @property
    def is_successful_batch(self) -> bool:
        return self.success_rate >= 0.8


# Domain exceptions
class DomainException(Exception):
    """Base exception for domain errors."""
    pass


class InvalidContentException(DomainException):
    """Raised when content is invalid."""
    pass


class InvalidRequestException(DomainException):
    """Raised when request is invalid."""
    pass


class QualityThresholdException(DomainException):
    """Raised when quality is below threshold."""
    pass 