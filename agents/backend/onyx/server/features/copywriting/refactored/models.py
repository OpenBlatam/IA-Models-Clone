from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from dataclasses import dataclass

# Constants
MAX_RETRIES = 100

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator, root_validator
import uuid
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Data Models
===========

Pydantic models for request/response validation and data serialization.
"""



class LanguageEnum(str, Enum):
    """Supported languages"""
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
    """Supported tones"""
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
    """Supported use cases"""
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


class WebsiteInfo(BaseModel):
    """Website information for context"""
    name: Optional[str] = Field(None, description="Website/company name")
    url: Optional[str] = Field(None, description="Website URL")
    description: Optional[str] = Field(None, description="Website description")
    industry: Optional[str] = Field(None, description="Industry/sector")
    target_audience: Optional[str] = Field(None, description="Target audience")
    key_features: Optional[List[str]] = Field(None, description="Key features/services")
    value_proposition: Optional[str] = Field(None, description="Main value proposition")
    brand_voice: Optional[str] = Field(None, description="Brand voice description")


class BrandVoice(BaseModel):
    """Brand voice configuration"""
    personality_traits: Optional[List[str]] = Field(
        None, 
        description="Brand personality traits (e.g., innovative, trustworthy, friendly)"
    )
    communication_style: Optional[str] = Field(
        None,
        description="Communication style (e.g., direct, conversational, formal)"
    )
    values: Optional[List[str]] = Field(
        None,
        description="Brand values (e.g., sustainability, innovation, customer-first)"
    )
    avoid_words: Optional[List[str]] = Field(
        None,
        description="Words or phrases to avoid"
    )
    preferred_words: Optional[List[str]] = Field(
        None,
        description="Preferred words or phrases"
    )


class TranslationSettings(BaseModel):
    """Translation configuration"""
    source_language: Optional[LanguageEnum] = Field(None, description="Source language")
    target_languages: Optional[List[LanguageEnum]] = Field(None, description="Target languages")
    cultural_adaptation: bool = Field(False, description="Enable cultural adaptation")
    localization_level: str = Field("standard", description="Localization level: basic, standard, advanced")


class VariantSettings(BaseModel):
    """Content variant configuration"""
    count: int = Field(1, ge=1, le=10, description="Number of variants to generate")
    diversity_level: str = Field("medium", description="Diversity level: low, medium, high")
    length_variations: bool = Field(False, description="Generate different length variations")
    tone_variations: bool = Field(False, description="Generate different tone variations")


class GenerationMetrics(BaseModel):
    """Metrics for content generation"""
    generation_time: float = Field(..., description="Generation time in seconds")
    token_count: int = Field(..., description="Number of tokens generated")
    cache_hit: bool = Field(False, description="Whether result was from cache")
    ai_provider: str = Field(..., description="AI provider used")
    model_used: str = Field(..., description="Specific model used")
    optimization_score: float = Field(0.0, description="Optimization performance score")
    quality_score: Optional[float] = Field(None, description="Content quality score")


class CopywritingRequest(BaseModel):
    """Main copywriting request model"""
    # Required fields
    prompt: str = Field(..., min_length=10, max_length=5000, description="Main prompt/description")
    use_case: UseCaseEnum = Field(..., description="Use case for the content")
    
    # Optional fields
    language: LanguageEnum = Field(LanguageEnum.ENGLISH, description="Target language")
    tone: ToneEnum = Field(ToneEnum.PROFESSIONAL, description="Desired tone")
    target_audience: Optional[str] = Field(None, max_length=500, description="Target audience description")
    keywords: Optional[List[str]] = Field(None, description="Keywords to include")
    length: Optional[str] = Field("medium", description="Content length: short, medium, long")
    
    # Advanced options
    website_info: Optional[WebsiteInfo] = Field(None, description="Website/company information")
    brand_voice: Optional[BrandVoice] = Field(None, description="Brand voice configuration")
    translation_settings: Optional[TranslationSettings] = Field(None, description="Translation settings")
    variant_settings: VariantSettings = Field(default_factory=VariantSettings, description="Variant settings")
    
    # AI provider options
    ai_provider: Optional[AIProviderEnum] = Field(None, description="Preferred AI provider")
    model: Optional[str] = Field(None, description="Specific model to use")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="Generation temperature")
    max_tokens: Optional[int] = Field(2000, ge=100, le=4000, description="Maximum tokens to generate")
    
    # Metadata
    user_id: Optional[str] = Field(None, description="User identifier")
    session_id: Optional[str] = Field(None, description="Session identifier")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    
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
    def validate_translation(cls, values) -> bool:
        translation_settings = values.get('translation_settings')
        if translation_settings and translation_settings.target_languages:
            language = values.get('language')
            if language in translation_settings.target_languages:
                raise ValueError("Target language cannot be the same as source language")
        return values


class ContentVariant(BaseModel):
    """Individual content variant"""
    content: str = Field(..., description="Generated content")
    tone: ToneEnum = Field(..., description="Tone used for this variant")
    length: str = Field(..., description="Content length category")
    word_count: int = Field(..., description="Word count")
    character_count: int = Field(..., description="Character count")
    quality_score: Optional[float] = Field(None, description="Quality assessment score")


class TranslatedContent(BaseModel):
    """Translated content result"""
    language: LanguageEnum = Field(..., description="Target language")
    content: str = Field(..., description="Translated content")
    cultural_notes: Optional[str] = Field(None, description="Cultural adaptation notes")
    confidence_score: Optional[float] = Field(None, description="Translation confidence")


class CopywritingResponse(BaseModel):
    """Main copywriting response model"""
    # Core response
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique request ID")
    success: bool = Field(True, description="Whether the request was successful")
    
    # Generated content
    primary_content: str = Field(..., description="Primary generated content")
    variants: List[ContentVariant] = Field(default_factory=list, description="Content variants")
    translations: List[TranslatedContent] = Field(default_factory=list, description="Translated versions")
    
    # Metadata
    metrics: GenerationMetrics = Field(..., description="Generation metrics")
    suggestions: Optional[List[str]] = Field(None, description="Improvement suggestions")
    keywords_used: Optional[List[str]] = Field(None, description="Keywords actually used")
    seo_score: Optional[float] = Field(None, description="SEO optimization score")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    expires_at: Optional[datetime] = Field(None, description="Cache expiration timestamp")
    
    # Error handling
    warnings: Optional[List[str]] = Field(None, description="Non-critical warnings")
    
    @dataclass
class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class BatchCopywritingRequest(BaseModel):
    """Batch processing request"""
    requests: List[CopywritingRequest] = Field(..., min_items=1, max_items=50, description="List of requests")
    parallel_processing: bool = Field(True, description="Process requests in parallel")
    fail_fast: bool = Field(False, description="Stop on first error")
    
    @validator('requests')
    async def validate_requests(cls, v) -> bool:
        if len(v) > 50:
            raise ValueError("Maximum 50 requests per batch")
        return v


class BatchCopywritingResponse(BaseModel):
    """Batch processing response"""
    batch_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Batch ID")
    total_requests: int = Field(..., description="Total number of requests")
    successful_requests: int = Field(..., description="Number of successful requests")
    failed_requests: int = Field(..., description="Number of failed requests")
    
    results: List[Union[CopywritingResponse, str]] = Field(..., description="Results or error messages")
    batch_metrics: Dict[str, Any] = Field(..., description="Batch processing metrics")
    
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Batch creation time")
    completed_at: Optional[datetime] = Field(None, description="Batch completion time")
    
    @dataclass
class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Check timestamp")
    version: str = Field(..., description="Service version")
    uptime: float = Field(..., description="Uptime in seconds")
    
    # Service checks
    database_status: str = Field(..., description="Database connectivity status")
    redis_status: str = Field(..., description="Redis connectivity status")
    ai_providers_status: Dict[str, str] = Field(..., description="AI providers status")
    
    # Performance metrics
    memory_usage: Dict[str, float] = Field(..., description="Memory usage statistics")
    cache_stats: Dict[str, Any] = Field(..., description="Cache statistics")
    
    @dataclass
class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class MetricsResponse(BaseModel):
    """Metrics response"""
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Metrics timestamp")
    
    # Request metrics
    total_requests: int = Field(..., description="Total requests processed")
    successful_requests: int = Field(..., description="Successful requests")
    failed_requests: int = Field(..., description="Failed requests")
    average_response_time: float = Field(..., description="Average response time in seconds")
    
    # Performance metrics
    cache_hit_rate: float = Field(..., description="Cache hit rate percentage")
    optimization_score: float = Field(..., description="Overall optimization score")
    
    # AI usage
    ai_provider_usage: Dict[str, int] = Field(..., description="Usage by AI provider")
    model_usage: Dict[str, int] = Field(..., description="Usage by model")
    
    # Language and tone distribution
    language_distribution: Dict[str, int] = Field(..., description="Requests by language")
    tone_distribution: Dict[str, int] = Field(..., description="Requests by tone")
    use_case_distribution: Dict[str, int] = Field(..., description="Requests by use case")
    
    @dataclass
class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        } 