from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field
import uuid
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Instagram Captions Models.

Pydantic models for Instagram caption generation with GMT support.
"""


class CaptionStyle(str, Enum):
    """Caption style options."""
    CASUAL = "casual"
    PROFESSIONAL = "professional"
    PLAYFUL = "playful"
    INSPIRATIONAL = "inspirational"
    EDUCATIONAL = "educational"
    STORYTELLING = "storytelling"
    PROMOTIONAL = "promotional"
    MINIMALIST = "minimalist"
    TRENDY = "trendy"
    AUTHENTIC = "authentic"

class InstagramTarget(str, Enum):
    """Instagram target audience."""
    MILLENNIALS = "millennials"
    GEN_Z = "gen_z"
    BUSINESS = "business"
    CREATORS = "creators"
    LIFESTYLE = "lifestyle"
    FASHION = "fashion"
    FOOD = "food"
    TRAVEL = "travel"
    FITNESS = "fitness"
    TECH = "tech"
    GENERAL = "general"

class HashtagStrategy(str, Enum):
    """Hashtag strategy options."""
    TRENDING = "trending"
    NICHE = "niche"
    BRANDED = "branded"
    LOCATION = "location"
    MIXED = "mixed"
    MINIMAL = "minimal"
    AGGRESSIVE = "aggressive"

class ContentType(str, Enum):
    """Instagram content types."""
    POST = "post"
    STORY = "story"
    REEL = "reel"
    CAROUSEL = "carousel"
    IGTV = "igtv"

class TimeZone(str, Enum):
    """Supported time zones for GMT operations."""
    GMT = "GMT"
    UTC = "UTC"
    EST = "America/New_York"
    PST = "America/Los_Angeles"
    CST = "America/Chicago"
    MST = "America/Denver"
    CET = "Europe/Paris"
    JST = "Asia/Tokyo"
    IST = "Asia/Kolkata"
    AEST = "Australia/Sydney"
    BST = "Europe/London"
    CAT = "Africa/Cairo"
    BRT = "America/Sao_Paulo"
    ART = "America/Argentina/Buenos_Aires"
    COT = "America/Bogota"
    PET = "America/Lima"
    CLT = "America/Santiago"
    VET = "America/Caracas"

class BrandInfo(BaseModel):
    """Brand information for caption generation."""
    name: str = Field(..., description="Brand name")
    voice: str = Field(..., description="Brand voice description")
    values: List[str] = Field(default_factory=list, description="Brand values")
    industry: str = Field(..., description="Industry/sector")
    target_demographics: List[str] = Field(default_factory=list, description="Target demographics")
    keywords: List[str] = Field(default_factory=list, description="Brand keywords")

class PostContent(BaseModel):
    """Post content information."""
    description: str = Field(..., description="Post description or context")
    image_description: Optional[str] = Field(None, description="Image/visual description")
    product_info: Optional[str] = Field(None, description="Product information if applicable")
    occasion: Optional[str] = Field(None, description="Special occasion or event")
    location: Optional[str] = Field(None, description="Location if relevant")

class InstagramCaptionRequest(BaseModel):
    """Instagram caption generation request."""
    content: PostContent = Field(..., description="Post content information")
    brand: Optional[BrandInfo] = Field(None, description="Brand information")
    style: CaptionStyle = Field(CaptionStyle.CASUAL, description="Caption style")
    target_audience: InstagramTarget = Field(InstagramTarget.GENERAL, description="Target audience")
    content_type: ContentType = Field(ContentType.POST, description="Content type")
    hashtag_strategy: HashtagStrategy = Field(HashtagStrategy.MIXED, description="Hashtag strategy")
    
    # GMT/Timezone settings
    target_timezone: TimeZone = Field(TimeZone.GMT, description="Target timezone")
    schedule_time: Optional[datetime] = Field(None, description="Scheduled posting time")
    regional_adaptation: bool = Field(True, description="Enable regional adaptation")
    
    # Generation settings
    max_length: int = Field(2200, ge=50, le=2200, description="Maximum caption length")
    include_hashtags: bool = Field(True, description="Include hashtags")
    hashtag_count: int = Field(10, ge=0, le=30, description="Number of hashtags")
    include_emojis: bool = Field(True, description="Include emojis")
    include_cta: bool = Field(True, description="Include call-to-action")
    
    # AI Provider settings
    use_langchain: bool = Field(True, description="Use LangChain for generation")
    use_openrouter: bool = Field(False, description="Use OpenRouter models")
    use_openai: bool = Field(True, description="Use OpenAI models")
    model_preference: Optional[str] = Field(None, description="Preferred model")
    
    # Variations
    generate_variations: bool = Field(True, description="Generate multiple variations")
    variation_count: int = Field(3, ge=1, le=5, description="Number of variations")
    
    # Metadata
    campaign_id: Optional[str] = Field(None, description="Campaign identifier")
    request_metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class CaptionVariation(BaseModel):
    """Individual caption variation."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Variation ID")
    caption: str = Field(..., description="Generated caption")
    hashtags: List[str] = Field(default_factory=list, description="Hashtags")
    character_count: int = Field(..., description="Character count")
    word_count: int = Field(..., description="Word count")
    emoji_count: int = Field(0, description="Emoji count")
    style_score: float = Field(0.0, ge=0.0, le=1.0, description="Style matching score")
    engagement_prediction: float = Field(0.0, ge=0.0, le=1.0, description="Predicted engagement score")
    readability_score: float = Field(0.0, ge=0.0, le=1.0, description="Readability score")

class TimeZoneInfo(BaseModel):
    """Time zone information."""
    timezone: TimeZone = Field(..., description="Time zone identifier")
    current_time: datetime = Field(..., description="Current time in timezone")
    utc_offset: str = Field(..., description="UTC offset")
    is_dst: bool = Field(False, description="Is daylight saving time active")
    local_hour: int = Field(..., ge=0, le=23, description="Current hour")
    optimal_posting_window: bool = Field(False, description="Is optimal posting time")
    peak_engagement_time: bool = Field(False, description="Is peak engagement time")

class RegionalAdaptation(BaseModel):
    """Regional content adaptations."""
    timezone: TimeZone = Field(..., description="Target timezone")
    cultural_adaptations: List[str] = Field(default_factory=list, description="Cultural adaptations applied")
    local_trends: List[str] = Field(default_factory=list, description="Local trending topics")
    regional_hashtags: List[str] = Field(default_factory=list, description="Regional hashtags")
    time_specific_content: Optional[str] = Field(None, description="Time-specific content additions")

class GenerationMetrics(BaseModel):
    """Generation performance metrics."""
    generation_time: float = Field(..., description="Generation time in seconds")
    model_used: str = Field(..., description="AI model used")
    provider_used: str = Field(..., description="AI provider used")
    token_count: int = Field(0, description="Tokens used")
    cost_estimate: float = Field(0.0, description="Estimated cost")
    quality_score: float = Field(0.0, ge=0.0, le=1.0, description="Quality assessment score")

class InstagramCaptionResponse(BaseModel):
    """Instagram caption generation response."""
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Request ID")
    variations: List[CaptionVariation] = Field(..., description="Generated variations")
    
    # GMT/Timezone info
    timezone_info: TimeZoneInfo = Field(..., description="Timezone information")
    execution_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Execution time")
    
    # Regional adaptations
    regional_adaptations: Optional[RegionalAdaptation] = Field(None, description="Regional adaptations applied")
    
    # Metrics
    generation_metrics: GenerationMetrics = Field(..., description="Generation metrics")
    
    # Recommendations
    best_variation_id: Optional[str] = Field(None, description="Recommended best variation")
    posting_recommendations: List[str] = Field(default_factory=list, description="Posting recommendations")
    optimization_suggestions: List[str] = Field(default_factory=list, description="Content optimization suggestions")
    
    # Metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp")
    expires_at: Optional[datetime] = Field(None, description="Expiration time")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class GlobalCampaignRequest(BaseModel):
    """Global Instagram campaign request."""
    campaign_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Campaign ID")
    base_request: InstagramCaptionRequest = Field(..., description="Base caption request")
    target_timezones: List[TimeZone] = Field(..., description="Target timezones")
    posting_schedule: Dict[str, datetime] = Field(..., description="Posting schedule per timezone")
    sync_posting: bool = Field(False, description="Synchronize posting across timezones")
    regional_customization: bool = Field(True, description="Enable regional customization")

class CampaignStatus(BaseModel):
    """Campaign status information."""
    campaign_id: str = Field(..., description="Campaign ID")
    total_timezones: int = Field(..., description="Total target timezones")
    completed_timezones: int = Field(0, description="Completed timezones")
    pending_timezones: int = Field(..., description="Pending timezones")
    status: str = Field(..., description="Overall status")
    created_at: datetime = Field(..., description="Creation time")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    generated_content: Dict[str, InstagramCaptionResponse] = Field(default_factory=dict, description="Generated content per timezone")

# Error models
class InstagramCaptionError(BaseModel):
    """Error response model."""
    error_code: str = Field(..., description="Error code")
    error_message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Error details")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Failed request ID") 