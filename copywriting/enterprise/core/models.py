from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from dataclasses import dataclass

# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, validator, root_validator
import json
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Enterprise Data Models
=====================

Comprehensive Pydantic models for:
- Request/response validation
- Data serialization and deserialization
- Type safety and documentation
- API schema generation
"""



class LanguageEnum(str, Enum):
    """Supported languages for content generation"""
    SPANISH = "spanish"
    ENGLISH = "english"
    FRENCH = "french"
    PORTUGUESE = "portuguese"
    ITALIAN = "italian"
    GERMAN = "german"
    DUTCH = "dutch"
    RUSSIAN = "russian"
    CHINESE = "chinese"
    JAPANESE = "japanese"
    KOREAN = "korean"
    ARABIC = "arabic"
    HINDI = "hindi"
    TURKISH = "turkish"
    POLISH = "polish"
    SWEDISH = "swedish"
    DANISH = "danish"
    NORWEGIAN = "norwegian"
    FINNISH = "finnish"


class ToneEnum(str, Enum):
    """Supported tones for content generation"""
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    URGENT = "urgent"
    INSPIRATIONAL = "inspirational"
    CONVERSATIONAL = "conversational"
    PERSUASIVE = "persuasive"
    STORYTELLING = "storytelling"
    EDUCATIONAL = "educational"
    MOTIVATIONAL = "motivational"
    EMPATHETIC = "empathetic"
    CONFIDENT = "confident"
    PLAYFUL = "playful"
    AUTHORITATIVE = "authoritative"
    FRIENDLY = "friendly"
    FORMAL = "formal"
    INFORMAL = "informal"
    HUMOROUS = "humorous"
    SERIOUS = "serious"
    OPTIMISTIC = "optimistic"
    DIRECT = "direct"


class UseCaseEnum(str, Enum):
    """Supported use cases for content generation"""
    PRODUCT_LAUNCH = "product_launch"
    BRAND_AWARENESS = "brand_awareness"
    LEAD_GENERATION = "lead_generation"
    SOCIAL_MEDIA = "social_media"
    EMAIL_MARKETING = "email_marketing"
    BLOG_POST = "blog_post"
    WEBSITE_COPY = "website_copy"
    AD_COPY = "ad_copy"
    PRESS_RELEASE = "press_release"
    NEWSLETTER = "newsletter"
    SALES_PAGE = "sales_page"
    LANDING_PAGE = "landing_page"
    CASE_STUDY = "case_study"
    TESTIMONIAL = "testimonial"
    FAQ = "faq"
    PRODUCT_DESCRIPTION = "product_description"
    SERVICE_DESCRIPTION = "service_description"
    COMPANY_BIO = "company_bio"
    TEAM_BIO = "team_bio"
    MISSION_STATEMENT = "mission_statement"
    VALUE_PROPOSITION = "value_proposition"
    CALL_TO_ACTION = "call_to_action"
    HEADLINE = "headline"
    TAGLINE = "tagline"
    SLOGAN = "slogan"


class AIProviderEnum(str, Enum):
    """Supported AI providers"""
    OPENROUTER = "openrouter"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    AUTO = "auto"


class ContentLengthEnum(str, Enum):
    """Content length options"""
    SHORT = "short"      # 50-150 words
    MEDIUM = "medium"    # 150-400 words
    LONG = "long"        # 400+ words
    CUSTOM = "custom"    # Custom word count


class QualityLevelEnum(str, Enum):
    """Content quality levels"""
    DRAFT = "draft"           # Quick generation
    STANDARD = "standard"     # Balanced quality/speed
    PREMIUM = "premium"       # High quality
    ENTERPRISE = "enterprise" # Maximum quality


# ============================================================================
# Core Data Models
# ============================================================================

class WebsiteInfo(BaseModel):
    """Website/company information for context"""
    name: Optional[str] = Field(None, max_length=100, description="Company/website name")
    url: Optional[str] = Field(None, description="Website URL")
    description: Optional[str] = Field(None, max_length=500, description="Company description")
    industry: Optional[str] = Field(None, max_length=100, description="Industry sector")
    target_audience: Optional[str] = Field(None, max_length=300, description="Primary target audience")
    key_features: Optional[List[str]] = Field(None, max_items=10, description="Key features/services")
    value_proposition: Optional[str] = Field(None, max_length=300, description="Main value proposition")
    brand_voice: Optional[str] = Field(None, max_length=200, description="Brand voice description")
    competitors: Optional[List[str]] = Field(None, max_items=5, description="Main competitors")
    unique_selling_points: Optional[List[str]] = Field(None, max_items=5, description="Unique selling points")


class BrandVoice(BaseModel):
    """Brand voice and personality configuration"""
    personality_traits: Optional[List[str]] = Field(
        None, 
        max_items=10,
        description="Brand personality traits (e.g., innovative, trustworthy, friendly)"
    )
    communication_style: Optional[str] = Field(
        None,
        max_length=100,
        description="Communication style (e.g., direct, conversational, formal)"
    )
    values: Optional[List[str]] = Field(
        None,
        max_items=10,
        description="Core brand values"
    )
    avoid_words: Optional[List[str]] = Field(
        None,
        max_items=20,
        description="Words or phrases to avoid"
    )
    preferred_words: Optional[List[str]] = Field(
        None,
        max_items=20,
        description="Preferred words or phrases"
    )
    tone_guidelines: Optional[str] = Field(
        None,
        max_length=500,
        description="Specific tone guidelines"
    )


class ContentRequirements(BaseModel):
    """Specific content requirements and constraints"""
    word_count_min: Optional[int] = Field(None, ge=10, le=10000, description="Minimum word count")
    word_count_max: Optional[int] = Field(None, ge=10, le=10000, description="Maximum word count")
    character_limit: Optional[int] = Field(None, ge=50, le=50000, description="Character limit")
    include_call_to_action: bool = Field(False, description="Include call-to-action")
    include_statistics: bool = Field(False, description="Include relevant statistics")
    include_testimonials: bool = Field(False, description="Include testimonials/social proof")
    seo_optimized: bool = Field(False, description="Optimize for SEO")
    readability_level: Optional[str] = Field(None, description="Target readability level")
    compliance_requirements: Optional[List[str]] = Field(None, description="Compliance requirements")


class TranslationSettings(BaseModel):
    """Translation and localization settings"""
    target_languages: Optional[List[LanguageEnum]] = Field(
        None, 
        max_items=10,
        description="Languages to translate to"
    )
    cultural_adaptation: bool = Field(False, description="Apply cultural adaptation")
    localization_level: str = Field(
        "standard", 
        description="Localization depth: basic, standard, advanced"
    )
    preserve_formatting: bool = Field(True, description="Preserve original formatting")
    translator_notes: Optional[str] = Field(None, max_length=500, description="Notes for translator")


class VariantSettings(BaseModel):
    """Content variant generation settings"""
    count: int = Field(1, ge=1, le=10, description="Number of variants to generate")
    diversity_level: str = Field("medium", description="Diversity level: low, medium, high")
    length_variations: bool = Field(False, description="Generate different length variations")
    tone_variations: bool = Field(False, description="Generate different tone variations")
    style_variations: bool = Field(False, description="Generate different style variations")
    format_variations: bool = Field(False, description="Generate different format variations")


class OptimizationSettings(BaseModel):
    """Performance and optimization settings"""
    cache_enabled: bool = Field(True, description="Enable caching for this request")
    cache_ttl: Optional[int] = Field(None, ge=60, le=86400, description="Cache TTL in seconds")
    priority: str = Field("normal", description="Request priority: low, normal, high, urgent")
    quality_level: QualityLevelEnum = Field(QualityLevelEnum.STANDARD, description="Quality level")
    enable_streaming: bool = Field(False, description="Enable streaming response")
    background_processing: bool = Field(False, description="Process in background")


# ============================================================================
# Request Models
# ============================================================================

class CopywritingRequest(BaseModel):
    """Main copywriting request model"""
    # Required fields
    prompt: str = Field(..., min_length=10, max_length=5000, description="Content generation prompt")
    use_case: UseCaseEnum = Field(..., description="Content use case")
    
    # Basic settings
    language: LanguageEnum = Field(LanguageEnum.ENGLISH, description="Target language")
    tone: ToneEnum = Field(ToneEnum.PROFESSIONAL, description="Content tone")
    length: ContentLengthEnum = Field(ContentLengthEnum.MEDIUM, description="Content length")
    
    # Target audience
    target_audience: Optional[str] = Field(
        None, 
        max_length=500, 
        description="Target audience description"
    )
    
    # Keywords and SEO
    keywords: Optional[List[str]] = Field(
        None, 
        max_items=20,
        description="Keywords to include"
    )
    primary_keyword: Optional[str] = Field(
        None,
        max_length=100,
        description="Primary SEO keyword"
    )
    
    # Context and brand
    website_info: Optional[WebsiteInfo] = Field(None, description="Website/company information")
    brand_voice: Optional[BrandVoice] = Field(None, description="Brand voice configuration")
    
    # Content requirements
    requirements: Optional[ContentRequirements] = Field(None, description="Specific content requirements")
    
    # Advanced options
    translation_settings: Optional[TranslationSettings] = Field(None, description="Translation settings")
    variant_settings: VariantSettings = Field(default_factory=VariantSettings, description="Variant settings")
    optimization_settings: OptimizationSettings = Field(default_factory=OptimizationSettings, description="Optimization settings")
    
    # AI provider options
    ai_provider: Optional[AIProviderEnum] = Field(None, description="Preferred AI provider")
    model: Optional[str] = Field(None, max_length=100, description="Specific model to use")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="Generation temperature")
    max_tokens: Optional[int] = Field(2000, ge=100, le=8000, description="Maximum tokens")
    
    # Metadata
    user_id: Optional[str] = Field(None, description="User identifier")
    session_id: Optional[str] = Field(None, description="Session identifier")
    client_id: Optional[str] = Field(None, description="Client identifier")
    request_id: Optional[str] = Field(default_factory=lambda: str(uuid4()), description="Request ID")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    
    # Validation
    @validator('keywords')
    def validate_keywords(cls, v) -> bool:
        if v and len(v) > 20:
            raise ValueError("Maximum 20 keywords allowed")
        return v
    
    @validator('prompt')
    def validate_prompt(cls, v) -> bool:
        if not v.strip():
            raise ValueError("Prompt cannot be empty")
        return v.strip()
    
    @root_validator
    def validate_word_count(cls, values) -> bool:
        requirements = values.get('requirements')
        if requirements and requirements.word_count_min and requirements.word_count_max:
            if requirements.word_count_min > requirements.word_count_max:
                raise ValueError("Minimum word count cannot be greater than maximum")
        return values
    
    @root_validator
    def validate_translation(cls, values) -> bool:
        translation_settings = values.get('translation_settings')
        if translation_settings and translation_settings.target_languages:
            language = values.get('language')
            if language in translation_settings.target_languages:
                raise ValueError("Target language cannot include source language")
        return values


class BatchRequest(BaseModel):
    """Batch processing request"""
    requests: List[CopywritingRequest] = Field(
        ..., 
        min_items=1, 
        max_items=100, 
        description="List of copywriting requests"
    )
    parallel_processing: bool = Field(True, description="Process requests in parallel")
    fail_fast: bool = Field(False, description="Stop processing on first error")
    batch_id: Optional[str] = Field(default_factory=lambda: str(uuid4()), description="Batch identifier")
    priority: str = Field("normal", description="Batch priority")
    
    @validator('requests')
    def validate_batch_size(cls, v) -> bool:
        if len(v) > 100:
            raise ValueError("Maximum 100 requests per batch")
        return v


# ============================================================================
# Response Models
# ============================================================================

class GenerationMetrics(BaseModel):
    """Metrics for content generation"""
    generation_time: float = Field(..., description="Total generation time in seconds")
    token_count: int = Field(..., description="Number of tokens generated")
    word_count: int = Field(..., description="Number of words generated")
    character_count: int = Field(..., description="Number of characters generated")
    
    # Performance metrics
    cache_hit: bool = Field(False, description="Whether result was from cache")
    cache_level: Optional[str] = Field(None, description="Cache level used (L1, L2, L3)")
    optimization_score: float = Field(0.0, description="Optimization performance score")
    
    # AI metrics
    ai_provider: str = Field(..., description="AI provider used")
    model_used: str = Field(..., description="Specific model used")
    ai_response_time: float = Field(..., description="AI provider response time")
    
    # Quality metrics
    quality_score: Optional[float] = Field(None, ge=0.0, le=100.0, description="Content quality score")
    readability_score: Optional[float] = Field(None, ge=0.0, le=100.0, description="Readability score")
    seo_score: Optional[float] = Field(None, ge=0.0, le=100.0, description="SEO optimization score")
    
    # Usage metrics
    api_calls_made: int = Field(1, description="Number of API calls made")
    retry_count: int = Field(0, description="Number of retries")


class ContentVariant(BaseModel):
    """Individual content variant"""
    content: str = Field(..., description="Generated content")
    variant_id: str = Field(default_factory=lambda: str(uuid4()), description="Variant identifier")
    
    # Variant properties
    tone: ToneEnum = Field(..., description="Tone used for this variant")
    length: ContentLengthEnum = Field(..., description="Content length category")
    style: Optional[str] = Field(None, description="Style variation")
    
    # Metrics
    word_count: int = Field(..., description="Word count")
    character_count: int = Field(..., description="Character count")
    quality_score: Optional[float] = Field(None, description="Quality assessment score")
    uniqueness_score: Optional[float] = Field(None, description="Uniqueness compared to primary")
    
    # Metadata
    generated_at: datetime = Field(default_factory=datetime.utcnow, description="Generation timestamp")


class TranslatedContent(BaseModel):
    """Translated content result"""
    content: str = Field(..., description="Translated content")
    language: LanguageEnum = Field(..., description="Target language")
    translation_id: str = Field(default_factory=lambda: str(uuid4()), description="Translation identifier")
    
    # Translation metrics
    confidence_score: Optional[float] = Field(None, ge=0.0, le=100.0, description="Translation confidence")
    cultural_adaptation_applied: bool = Field(False, description="Whether cultural adaptation was applied")
    
    # Metadata
    cultural_notes: Optional[str] = Field(None, description="Cultural adaptation notes")
    translator_notes: Optional[str] = Field(None, description="Translator notes")
    word_count: int = Field(..., description="Translated content word count")
    character_count: int = Field(..., description="Translated content character count")


class ContentAnalysis(BaseModel):
    """Content analysis and insights"""
    sentiment: Optional[str] = Field(None, description="Content sentiment")
    sentiment_score: Optional[float] = Field(None, ge=-1.0, le=1.0, description="Sentiment score")
    
    # Readability
    readability_grade: Optional[float] = Field(None, description="Reading grade level")
    reading_time_minutes: Optional[float] = Field(None, description="Estimated reading time")
    
    # SEO analysis
    keyword_density: Optional[Dict[str, float]] = Field(None, description="Keyword density analysis")
    seo_recommendations: Optional[List[str]] = Field(None, description="SEO improvement recommendations")
    
    # Content structure
    paragraph_count: int = Field(0, description="Number of paragraphs")
    sentence_count: int = Field(0, description="Number of sentences")
    average_sentence_length: Optional[float] = Field(None, description="Average sentence length")
    
    # Engagement predictions
    engagement_score: Optional[float] = Field(None, ge=0.0, le=100.0, description="Predicted engagement score")
    conversion_potential: Optional[str] = Field(None, description="Conversion potential assessment")


class CopywritingResponse(BaseModel):
    """Main copywriting response model"""
    # Request tracking
    request_id: str = Field(..., description="Request identifier")
    success: bool = Field(True, description="Whether request was successful")
    
    # Generated content
    primary_content: str = Field(..., description="Primary generated content")
    variants: List[ContentVariant] = Field(default_factory=list, description="Content variants")
    translations: List[TranslatedContent] = Field(default_factory=list, description="Translated versions")
    
    # Analysis and insights
    analysis: Optional[ContentAnalysis] = Field(None, description="Content analysis")
    
    # Metadata
    metrics: GenerationMetrics = Field(..., description="Generation metrics")
    suggestions: Optional[List[str]] = Field(None, description="Improvement suggestions")
    keywords_used: Optional[List[str]] = Field(None, description="Keywords actually used")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    expires_at: Optional[datetime] = Field(None, description="Cache expiration timestamp")
    
    # Status and warnings
    warnings: Optional[List[str]] = Field(None, description="Non-critical warnings")
    processing_notes: Optional[List[str]] = Field(None, description="Processing notes")
    
    @dataclass
class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }


class BatchResponse(BaseModel):
    """Batch processing response"""
    batch_id: str = Field(..., description="Batch identifier")
    total_requests: int = Field(..., description="Total number of requests")
    successful_requests: int = Field(..., description="Number of successful requests")
    failed_requests: int = Field(..., description="Number of failed requests")
    
    # Results
    results: List[Union[CopywritingResponse, Dict[str, str]]] = Field(
        ..., 
        description="Results or error information"
    )
    
    # Batch metrics
    batch_metrics: Dict[str, Any] = Field(..., description="Batch processing metrics")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Batch creation time")
    started_at: Optional[datetime] = Field(None, description="Processing start time")
    completed_at: Optional[datetime] = Field(None, description="Processing completion time")
    
    # Status
    status: str = Field("completed", description="Batch status")
    progress_percentage: float = Field(100.0, ge=0.0, le=100.0, description="Completion percentage")
    
    @dataclass
class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }


# ============================================================================
# System Models
# ============================================================================

class HealthCheckResponse(BaseModel):
    """System health check response"""
    status: str = Field(..., description="Overall system status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Check timestamp")
    version: str = Field(..., description="Service version")
    uptime: float = Field(..., description="Uptime in seconds")
    
    # Component health
    database_status: str = Field(..., description="Database status")
    redis_status: str = Field(..., description="Redis status")
    ai_providers_status: Dict[str, str] = Field(..., description="AI providers status")
    cache_status: str = Field(..., description="Cache system status")
    
    # Performance metrics
    memory_usage: Dict[str, float] = Field(..., description="Memory usage statistics")
    cpu_usage: float = Field(..., description="CPU usage percentage")
    disk_usage: float = Field(..., description="Disk usage percentage")
    
    # Service metrics
    active_requests: int = Field(..., description="Number of active requests")
    cache_hit_rate: float = Field(..., description="Current cache hit rate")
    optimization_score: float = Field(..., description="System optimization score")
    
    @dataclass
class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class MetricsResponse(BaseModel):
    """System metrics response"""
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Metrics timestamp")
    
    # Request metrics
    total_requests: int = Field(..., description="Total requests processed")
    successful_requests: int = Field(..., description="Successful requests")
    failed_requests: int = Field(..., description="Failed requests")
    average_response_time: float = Field(..., description="Average response time")
    
    # Performance metrics
    cache_hit_rate: float = Field(..., description="Cache hit rate percentage")
    optimization_score: float = Field(..., description="System optimization score")
    throughput: float = Field(..., description="Requests per second")
    
    # Resource usage
    memory_usage_percentage: float = Field(..., description="Memory usage percentage")
    cpu_usage_percentage: float = Field(..., description="CPU usage percentage")
    
    # AI provider metrics
    ai_provider_usage: Dict[str, int] = Field(..., description="Usage by AI provider")
    model_usage: Dict[str, int] = Field(..., description="Usage by model")
    ai_response_times: Dict[str, float] = Field(..., description="Average response times by provider")
    
    # Content metrics
    language_distribution: Dict[str, int] = Field(..., description="Requests by language")
    tone_distribution: Dict[str, int] = Field(..., description="Requests by tone")
    use_case_distribution: Dict[str, int] = Field(..., description="Requests by use case")
    
    @dataclass
class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        } 