from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from dataclasses import dataclass

# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Union
from enum import Enum
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, validator, root_validator
from pydantic.types import SecretStr
        import re
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
🏗️ Domain Entities
==================

Core business entities for the copywriting system using Pydantic
for validation and serialization.
"""




class ContentType(str, Enum):
    """Types of content that can be generated"""
    BLOG_POST = "blog_post"
    SOCIAL_MEDIA = "social_media"
    EMAIL = "email"
    AD_COPY = "ad_copy"
    PRODUCT_DESCRIPTION = "product_description"
    LANDING_PAGE = "landing_page"
    VIDEO_SCRIPT = "video_script"
    PODCAST_SCRIPT = "podcast_script"
    NEWS_RELEASE = "news_release"
    WHITEPAPER = "whitepaper"


class Tone(str, Enum):
    """Content tone options"""
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    FRIENDLY = "friendly"
    AUTHORITATIVE = "authoritative"
    HUMOROUS = "humorous"
    INSPIRATIONAL = "inspirational"
    CONVERSATIONAL = "conversational"
    FORMAL = "formal"


class Language(str, Enum):
    """Supported languages"""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    PORTUGUESE = "pt"
    ITALIAN = "it"
    DUTCH = "nl"
    RUSSIAN = "ru"
    CHINESE = "zh"
    JAPANESE = "ja"
    KOREAN = "ko"


class Status(str, Enum):
    """Content generation status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class UserRole(str, Enum):
    """User roles in the system"""
    ADMIN = "admin"
    USER = "user"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"


class User(BaseModel):
    """User entity"""
    
    id: UUID = Field(default_factory=uuid4)
    email: str = Field(..., description="User email address")
    username: Optional[str] = Field(None, description="Username")
    full_name: Optional[str] = Field(None, description="Full name")
    role: UserRole = Field(default=UserRole.USER, description="User role")
    credits: int = Field(default=100, description="Available credits")
    max_credits: int = Field(default=1000, description="Maximum credits")
    is_active: bool = Field(default=True, description="Account status")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = Field(None, description="Last login time")
    
    # Preferences
    preferred_language: Language = Field(default=Language.ENGLISH)
    preferred_tone: Tone = Field(default=Tone.PROFESSIONAL)
    content_preferences: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('email')
    def validate_email(cls, v) -> bool:
        if '@' not in v:
            raise ValueError('Invalid email format')
        return v.lower()
    
    @validator('credits')
    def validate_credits(cls, v) -> bool:
        if v < 0:
            raise ValueError('Credits cannot be negative')
        return v
    
    def has_sufficient_credits(self, required: int) -> bool:
        """Check if user has sufficient credits"""
        return self.credits >= required
    
    def deduct_credits(self, amount: int) -> bool:
        """Deduct credits from user account"""
        if self.has_sufficient_credits(amount):
            self.credits -= amount
            self.updated_at = datetime.utcnow()
            return True
        return False
    
    def add_credits(self, amount: int) -> None:
        """Add credits to user account"""
        self.credits = min(self.credits + amount, self.max_credits)
        self.updated_at = datetime.utcnow()
    
    @dataclass
class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }


class ContentRequest(BaseModel):
    """Content generation request"""
    
    id: UUID = Field(default_factory=uuid4)
    user_id: UUID = Field(..., description="User who made the request")
    content_type: ContentType = Field(..., description="Type of content to generate")
    prompt: str = Field(..., min_length=10, max_length=5000, description="Generation prompt")
    
    # Content parameters
    title: Optional[str] = Field(None, max_length=200, description="Content title")
    keywords: List[str] = Field(default_factory=list, description="Target keywords")
    tone: Tone = Field(default=Tone.PROFESSIONAL, description="Content tone")
    language: Language = Field(default=Language.ENGLISH, description="Content language")
    word_count: Optional[int] = Field(None, ge=50, le=5000, description="Target word count")
    
    # Advanced options
    target_audience: Optional[str] = Field(None, description="Target audience")
    brand_voice: Optional[str] = Field(None, description="Brand voice guidelines")
    call_to_action: Optional[str] = Field(None, description="Call to action")
    seo_optimized: bool = Field(default=True, description="SEO optimization")
    
    # Metadata
    tags: List[str] = Field(default_factory=list, description="Content tags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('keywords')
    def validate_keywords(cls, v) -> bool:
        if len(v) > 20:
            raise ValueError('Maximum 20 keywords allowed')
        return [kw.lower().strip() for kw in v if kw.strip()]
    
    @validator('tags')
    def validate_tags(cls, v) -> bool:
        if len(v) > 50:
            raise ValueError('Maximum 50 tags allowed')
        return [tag.lower().strip() for tag in v if tag.strip()]
    
    @root_validator
    def validate_content_parameters(cls, values) -> bool:
        content_type = values.get('content_type')
        word_count = values.get('word_count')
        
        # Set default word counts based on content type
        if word_count is None:
            defaults = {
                ContentType.BLOG_POST: 800,
                ContentType.SOCIAL_MEDIA: 150,
                ContentType.EMAIL: 300,
                ContentType.AD_COPY: 100,
                ContentType.PRODUCT_DESCRIPTION: 200,
                ContentType.LANDING_PAGE: 500,
                ContentType.VIDEO_SCRIPT: 600,
                ContentType.PODCAST_SCRIPT: 800,
                ContentType.NEWS_RELEASE: 400,
                ContentType.WHITEPAPER: 2000
            }
            values['word_count'] = defaults.get(content_type, 500)
        
        return values
    
    def estimate_credits(self) -> int:
        """Estimate credits needed for this request"""
        base_credits = {
            ContentType.BLOG_POST: 10,
            ContentType.SOCIAL_MEDIA: 2,
            ContentType.EMAIL: 5,
            ContentType.AD_COPY: 3,
            ContentType.PRODUCT_DESCRIPTION: 4,
            ContentType.LANDING_PAGE: 8,
            ContentType.VIDEO_SCRIPT: 12,
            ContentType.PODCAST_SCRIPT: 15,
            ContentType.NEWS_RELEASE: 6,
            ContentType.WHITEPAPER: 25
        }
        
        credits = base_credits.get(self.content_type, 5)
        
        # Adjust based on word count
        if self.word_count:
            if self.word_count > 1000:
                credits += 5
            elif self.word_count > 500:
                credits += 3
        
        # Adjust based on language
        if self.language != Language.ENGLISH:
            credits += 2
        
        # Adjust based on SEO optimization
        if self.seo_optimized:
            credits += 1
        
        return credits
    
    @dataclass
class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }


class GeneratedContent(BaseModel):
    """Generated content entity"""
    
    id: UUID = Field(default_factory=uuid4)
    request_id: UUID = Field(..., description="Associated content request")
    user_id: UUID = Field(..., description="User who owns the content")
    
    # Content details
    title: str = Field(..., description="Content title")
    content: str = Field(..., description="Generated content")
    summary: Optional[str] = Field(None, description="Content summary")
    
    # Metadata
    word_count: int = Field(..., description="Actual word count")
    reading_time: Optional[int] = Field(None, description="Estimated reading time in minutes")
    seo_score: Optional[float] = Field(None, ge=0, le=100, description="SEO optimization score")
    readability_score: Optional[float] = Field(None, ge=0, le=100, description="Readability score")
    
    # AI model information
    model_used: str = Field(..., description="AI model used for generation")
    model_version: Optional[str] = Field(None, description="Model version")
    generation_time: Optional[float] = Field(None, description="Generation time in seconds")
    tokens_used: Optional[int] = Field(None, description="Tokens consumed")
    
    # Quality metrics
    confidence_score: Optional[float] = Field(None, ge=0, le=1, description="AI confidence score")
    plagiarism_score: Optional[float] = Field(None, ge=0, le=1, description="Plagiarism detection score")
    
    # Status and timestamps
    status: Status = Field(default=Status.COMPLETED, description="Content status")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Additional data
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    versions: List[Dict[str, Any]] = Field(default_factory=list, description="Content versions")
    
    @validator('content')
    def validate_content(cls, v) -> bool:
        if not v.strip():
            raise ValueError('Content cannot be empty')
        return v
    
    @validator('word_count')
    def validate_word_count(cls, v) -> bool:
        if v <= 0:
            raise ValueError('Word count must be positive')
        return v
    
    def calculate_reading_time(self) -> int:
        """Calculate estimated reading time in minutes"""
        if self.word_count:
            # Average reading speed: 200-250 words per minute
            return max(1, self.word_count // 225)
        return 0
    
    def get_seo_suggestions(self) -> List[str]:
        """Get SEO improvement suggestions"""
        suggestions = []
        
        if self.seo_score is not None:
            if self.seo_score < 70:
                suggestions.append("Consider adding more relevant keywords")
            if self.seo_score < 60:
                suggestions.append("Improve content structure with headings")
            if self.seo_score < 50:
                suggestions.append("Add meta description and title optimization")
        
        return suggestions
    
    def get_readability_suggestions(self) -> List[str]:
        """Get readability improvement suggestions"""
        suggestions = []
        
        if self.readability_score is not None:
            if self.readability_score < 70:
                suggestions.append("Use shorter sentences and simpler words")
            if self.readability_score < 60:
                suggestions.append("Break down complex paragraphs")
            if self.readability_score < 50:
                suggestions.append("Consider your target audience's reading level")
        
        return suggestions
    
    @dataclass
class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }


class ContentTemplate(BaseModel):
    """Content template for reusable prompts"""
    
    id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., description="Template name")
    description: Optional[str] = Field(None, description="Template description")
    user_id: UUID = Field(..., description="Template owner")
    
    # Template content
    prompt_template: str = Field(..., description="Prompt template with placeholders")
    content_type: ContentType = Field(..., description="Content type")
    default_tone: Tone = Field(default=Tone.PROFESSIONAL, description="Default tone")
    default_language: Language = Field(default=Language.ENGLISH, description="Default language")
    
    # Template parameters
    parameters: List[Dict[str, Any]] = Field(default_factory=list, description="Template parameters")
    default_values: Dict[str, Any] = Field(default_factory=dict, description="Default parameter values")
    
    # Usage statistics
    usage_count: int = Field(default=0, description="Number of times used")
    last_used: Optional[datetime] = Field(None, description="Last usage time")
    
    # Metadata
    tags: List[str] = Field(default_factory=list, description="Template tags")
    is_public: bool = Field(default=False, description="Public template flag")
    is_active: bool = Field(default=True, description="Template status")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    def render_prompt(self, **kwargs) -> str:
        """Render the prompt template with provided parameters"""f"
        try:
            return self.prompt_template"
        except KeyError as e:
            raise ValueError(f"Missing required parameter: {e}")
    
    def get_required_parameters(self) -> List[str]:
        """Get list of required parameters"""
        pattern = r'\{(\w+)\}'
        return list(set(re.findall(pattern, self.prompt_template)))
    
    def increment_usage(self) -> None:
        """Increment usage counter"""
        self.usage_count += 1
        self.last_used = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    @dataclass
class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }


class UsageMetrics(BaseModel):
    """Usage metrics for analytics"""
    
    id: UUID = Field(default_factory=uuid4)
    user_id: UUID = Field(..., description="User ID")
    date: datetime = Field(..., description="Metrics date")
    
    # Usage counts
    total_requests: int = Field(default=0, description="Total content requests")
    successful_requests: int = Field(default=0, description="Successful generations")
    failed_requests: int = Field(default=0, description="Failed generations")
    
    # Credit usage
    credits_used: int = Field(default=0, description="Credits consumed")
    credits_earned: int = Field(default=0, description="Credits earned")
    
    # Content type breakdown
    content_type_breakdown: Dict[str, int] = Field(default_factory=dict, description="Requests by content type")
    
    # Performance metrics
    average_generation_time: Optional[float] = Field(None, description="Average generation time")
    total_generation_time: float = Field(default=0.0, description="Total generation time")
    
    # Quality metrics
    average_seo_score: Optional[float] = Field(None, description="Average SEO score")
    average_readability_score: Optional[float] = Field(None, description="Average readability score")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100
    
    @property
    def net_credits(self) -> int:
        """Calculate net credits (used - earned)"""
        return self.credits_used - self.credits_earned
    
    async def add_request(self, content_type: ContentType, successful: bool, 
                   credits: int, generation_time: float = 0.0) -> None:
        """Add a new request to metrics"""
        self.total_requests += 1
        
        if successful:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        
        self.credits_used += credits
        self.total_generation_time += generation_time
        
        # Update content type breakdown
        content_type_str = content_type.value
        self.content_type_breakdown[content_type_str] = \
            self.content_type_breakdown.get(content_type_str, 0) + 1
        
        # Update averages
        if self.successful_requests > 0:
            self.average_generation_time = self.total_generation_time / self.successful_requests
        
        self.updated_at = datetime.utcnow()
    
    @dataclass
class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        } 