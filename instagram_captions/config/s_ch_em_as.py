from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

from datetime import datetime
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, ConfigDict, validator
from enum import Enum
from models import CaptionStyle, InstagramTarget, HashtagStrategy, ContentType
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Pydantic v2 schemas for Instagram Captions API.

Optimized request/response models with comprehensive validation.
"""




class CaptionGenerationRequest(BaseModel):
    """Optimized request schema for caption generation."""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=True
    )
    
    content_description: str = Field(
        ...,
        min_length=10,
        max_length=1000,
        description="Content description for caption generation"
    )
    style: CaptionStyle = Field(
        default=CaptionStyle.CASUAL,
        description="Caption writing style"
    )
    audience: InstagramTarget = Field(
        default=InstagramTarget.GENERAL,
        description="Target audience"
    )
    timezone: Optional[str] = Field(
        default="UTC",
        pattern=r"^[A-Za-z_/]+$",
        description="Timezone for cultural adaptation"
    )
    image_description: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Visual content description"
    )
    content_type: ContentType = Field(
        default=ContentType.POST,
        description="Type of Instagram content"
    )
    include_hashtags: bool = Field(
        default=True,
        description="Whether to include hashtags"
    )
    hashtag_strategy: HashtagStrategy = Field(
        default=HashtagStrategy.MIXED,
        description="Hashtag selection strategy"
    )
    hashtag_count: int = Field(
        default=15,
        ge=1,
        le=30,
        description="Number of hashtags to generate"
    )
    brand_context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Brand information for voice consistency"
    )


class QualityMetricsResponse(BaseModel):
    """Response schema for quality metrics."""
    
    model_config = ConfigDict(use_enum_values=True)
    
    overall_score: float = Field(..., ge=0, le=100, description="Overall quality score")
    grade: str = Field(..., description="Quality grade (A+ to F)")
    hook_strength: float = Field(..., ge=0, le=100, description="Hook effectiveness score")
    engagement_potential: float = Field(..., ge=0, le=100, description="Engagement prediction score")
    readability: float = Field(..., ge=0, le=100, description="Readability score")
    cta_effectiveness: float = Field(..., ge=0, le=100, description="Call-to-action effectiveness")
    specificity: float = Field(..., ge=0, le=100, description="Content specificity score")
    issues: List[str] = Field(default_factory=list, description="Identified quality issues")
    suggestions: List[str] = Field(default_factory=list, description="Improvement suggestions")
    performance_expectation: str = Field(..., description="Expected performance description")


class CaptionVariationResponse(BaseModel):
    """Response schema for individual caption variation."""
    
    model_config = ConfigDict(use_enum_values=True)
    
    caption: str = Field(..., description="Generated caption text")
    hashtags: List[str] = Field(default_factory=list, description="Generated hashtags")
    character_count: int = Field(..., ge=0, description="Character count")
    word_count: int = Field(..., ge=0, description="Word count")
    quality_metrics: QualityMetricsResponse = Field(..., description="Quality analysis")
    cultural_adaptations: Optional[List[str]] = Field(
        default=None,
        description="Applied cultural adaptations"
    )


class CaptionGenerationResponse(BaseModel):
    """Complete response schema for caption generation."""
    
    model_config = ConfigDict(use_enum_values=True)
    
    status: Literal["success", "partial", "failed"] = Field(..., description="Generation status")
    variations: List[CaptionVariationResponse] = Field(..., description="Generated caption variations")
    best_variation_id: Optional[int] = Field(default=None, description="Index of best variation")
    timezone_insights: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Timezone-specific insights"
    )
    engagement_recommendations: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Engagement optimization recommendations"
    )
    generation_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Generation process metadata"
    )
    generated_at: datetime = Field(default_factory=datetime.utcnow, description="Generation timestamp")


class QualityAnalysisRequest(BaseModel):
    """Request schema for quality analysis."""
    
    model_config = ConfigDict(str_strip_whitespace=True)
    
    caption: str = Field(
        ...,
        min_length=5,
        max_length=2200,
        description="Caption text to analyze"
    )
    style: CaptionStyle = Field(
        default=CaptionStyle.CASUAL,
        description="Expected caption style"
    )
    audience: InstagramTarget = Field(
        default=InstagramTarget.GENERAL,
        description="Target audience"
    )


class CaptionOptimizationRequest(BaseModel):
    """Request schema for caption optimization."""
    
    model_config = ConfigDict(str_strip_whitespace=True)
    
    caption: str = Field(
        ...,
        min_length=5,
        max_length=2200,
        description="Caption text to optimize"
    )
    style: CaptionStyle = Field(
        default=CaptionStyle.CASUAL,
        description="Desired caption style"
    )
    audience: InstagramTarget = Field(
        default=InstagramTarget.GENERAL,
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


class CaptionOptimizationResponse(BaseModel):
    """Response schema for caption optimization."""
    
    model_config = ConfigDict(use_enum_values=True)
    
    status: Literal["success", "partial", "failed"] = Field(..., description="Optimization status")
    original_caption: str = Field(..., description="Original caption text")
    optimized_caption: str = Field(..., description="Optimized caption text")
    original_quality: QualityMetricsResponse = Field(..., description="Original quality metrics")
    optimized_quality: QualityMetricsResponse = Field(..., description="Optimized quality metrics")
    improvements_applied: List[str] = Field(
        default_factory=list,
        description="List of improvements applied"
    )
    improvement_score: float = Field(
        ...,
        ge=0,
        description="Improvement score (percentage points gained)"
    )
    optimized_at: datetime = Field(default_factory=datetime.utcnow, description="Optimization timestamp")


class BatchOptimizationRequest(BaseModel):
    """Request schema for batch optimization."""
    
    model_config = ConfigDict(str_strip_whitespace=True)
    
    captions: List[str] = Field(
        ...,
        min_length=1,
        max_length=20,
        description="List of captions to optimize"
    )
    style: CaptionStyle = Field(
        default=CaptionStyle.CASUAL,
        description="Desired caption style"
    )
    audience: InstagramTarget = Field(
        default=InstagramTarget.GENERAL,
        description="Target audience"
    )
    enhancement_level: Literal["light", "moderate", "aggressive"] = Field(
        default="moderate",
        description="Level of optimization to apply"
    )
    
    @validator('captions')
    def validate_captions(cls, v) -> bool:
        """Validate individual captions in the list."""
        if not v:
            raise ValueError("At least one caption is required")
        
        for i, caption in enumerate(v):
            if not caption or len(caption.strip()) < 5:
                raise ValueError(f"Caption {i+1} is too short (minimum 5 characters)")
            if len(caption) > 2200:
                raise ValueError(f"Caption {i+1} is too long (maximum 2200 characters)")
        
        return v


class BatchOptimizationResponse(BaseModel):
    """Response schema for batch optimization."""
    
    model_config = ConfigDict(use_enum_values=True)
    
    status: Literal["success", "partial", "failed"] = Field(..., description="Batch processing status")
    results: List[CaptionOptimizationResponse] = Field(..., description="Optimization results")
    batch_statistics: Dict[str, Any] = Field(..., description="Batch processing statistics")
    processed_at: datetime = Field(default_factory=datetime.utcnow, description="Processing timestamp")


class HashtagGenerationRequest(BaseModel):
    """Request schema for hashtag generation."""
    
    model_config = ConfigDict(str_strip_whitespace=True)
    
    content_keywords: List[str] = Field(
        ...,
        min_length=1,
        max_length=20,
        description="Content keywords for hashtag generation"
    )
    audience: InstagramTarget = Field(
        default=InstagramTarget.GENERAL,
        description="Target audience"
    )
    style: CaptionStyle = Field(
        default=CaptionStyle.CASUAL,
        description="Content style"
    )
    strategy: HashtagStrategy = Field(
        default=HashtagStrategy.MIXED,
        description="Hashtag strategy"
    )
    count: int = Field(
        default=20,
        ge=5,
        le=30,
        description="Number of hashtags to generate"
    )


class HashtagGenerationResponse(BaseModel):
    """Response schema for hashtag generation."""
    
    model_config = ConfigDict(use_enum_values=True)
    
    status: Literal["success", "partial", "failed"] = Field(..., description="Generation status")
    hashtags: List[str] = Field(..., description="Generated hashtags")
    strategy_used: HashtagStrategy = Field(..., description="Strategy applied")
    performance_estimate: Dict[str, Any] = Field(
        default_factory=dict,
        description="Estimated performance metrics"
    )
    category_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="Hashtag category distribution"
    )
    generated_at: datetime = Field(default_factory=datetime.utcnow, description="Generation timestamp")


class HealthCheckResponse(BaseModel):
    """Response schema for health check."""
    
    model_config = ConfigDict(use_enum_values=True)
    
    status: Literal["healthy", "degraded", "unhealthy"] = Field(..., description="System health status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Health check timestamp")
    version: str = Field(default="2.0.0", description="API version")
    components: Dict[str, Any] = Field(default_factory=dict, description="Component health status")
    performance_metrics: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Performance metrics"
    )


class ErrorResponse(BaseModel):
    """Standardized error response schema."""
    
    model_config = ConfigDict(use_enum_values=True)
    
    error: bool = Field(default=True, description="Error flag")
    error_code: str = Field(..., description="Error code identifier")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    request_id: Optional[str] = Field(default=None, description="Request identifier for tracking")


class QualityGuidelinesResponse(BaseModel):
    """Response schema for quality guidelines."""
    
    model_config = ConfigDict(use_enum_values=True)
    
    status: Literal["success"] = Field(default="success", description="Response status")
    guidelines: Dict[str, Any] = Field(..., description="Quality guidelines and best practices")
    examples: Dict[str, Any] = Field(..., description="Examples of good vs bad practices")
    metrics_explanation: Dict[str, str] = Field(..., description="Explanation of quality metrics")
    generated_at: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")


class TimezoneInsightsResponse(BaseModel):
    """Response schema for timezone insights."""
    
    model_config = ConfigDict(use_enum_values=True)
    
    timezone: str = Field(..., description="Timezone identifier")
    optimal_posting_times: List[str] = Field(..., description="Optimal posting times")
    peak_engagement_windows: List[Dict[str, Any]] = Field(..., description="Peak engagement windows")
    cultural_context: Dict[str, Any] = Field(..., description="Cultural adaptation context")
    engagement_predictions: Dict[str, float] = Field(..., description="Engagement predictions by hour")
    recommendations: List[str] = Field(..., description="Timezone-specific recommendations")
    confidence_score: float = Field(..., ge=0, le=1, description="Prediction confidence score")
    generated_at: datetime = Field(default_factory=datetime.utcnow, description="Insights timestamp") 