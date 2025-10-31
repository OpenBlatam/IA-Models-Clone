"""
Advanced Blog Posts System Schemas
=================================

Pydantic v2 models for blog posts system with comprehensive validation.
"""

from typing import Dict, Any, List, Optional, Union, Literal
from datetime import datetime, timedelta
from enum import Enum
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, validator, root_validator
import json


class BlogPostStatus(str, Enum):
    """Blog post status"""
    DRAFT = "draft"
    PUBLISHED = "published"
    SCHEDULED = "scheduled"
    ARCHIVED = "archived"
    DELETED = "deleted"


class ContentType(str, Enum):
    """Content types"""
    ARTICLE = "article"
    TUTORIAL = "tutorial"
    NEWS = "news"
    REVIEW = "review"
    OPINION = "opinion"
    GUIDE = "guide"
    CASE_STUDY = "case_study"
    INTERVIEW = "interview"
    LISTICLE = "listicle"
    HOW_TO = "how_to"


class ContentFormat(str, Enum):
    """Content formats"""
    MARKDOWN = "markdown"
    HTML = "html"
    RICH_TEXT = "rich_text"
    PLAIN_TEXT = "plain_text"


class SEOStatus(str, Enum):
    """SEO status"""
    OPTIMIZED = "optimized"
    NEEDS_IMPROVEMENT = "needs_improvement"
    NOT_OPTIMIZED = "not_optimized"


class EngagementLevel(str, Enum):
    """Engagement levels"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"


class ContentQuality(str, Enum):
    """Content quality levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    AVERAGE = "average"
    POOR = "poor"
    VERY_POOR = "very_poor"


class BlogPost(BaseModel):
    """Blog post model"""
    post_id: str = Field(default_factory=lambda: str(uuid4()), description="Post ID")
    title: str = Field(..., min_length=1, max_length=200, description="Post title")
    slug: str = Field(..., min_length=1, max_length=200, description="Post slug")
    content: str = Field(..., min_length=1, description="Post content")
    excerpt: Optional[str] = Field(default=None, max_length=500, description="Post excerpt")
    author_id: str = Field(..., description="Author ID")
    status: BlogPostStatus = Field(default=BlogPostStatus.DRAFT, description="Post status")
    content_type: ContentType = Field(default=ContentType.ARTICLE, description="Content type")
    content_format: ContentFormat = Field(default=ContentFormat.MARKDOWN, description="Content format")
    
    # SEO fields
    meta_title: Optional[str] = Field(default=None, max_length=60, description="Meta title")
    meta_description: Optional[str] = Field(default=None, max_length=160, description="Meta description")
    meta_keywords: List[str] = Field(default_factory=list, description="Meta keywords")
    canonical_url: Optional[str] = Field(default=None, description="Canonical URL")
    
    # Content analysis
    word_count: int = Field(default=0, ge=0, description="Word count")
    reading_time: int = Field(default=0, ge=0, description="Reading time in minutes")
    seo_score: float = Field(default=0.0, ge=0.0, le=100.0, description="SEO score")
    readability_score: float = Field(default=0.0, ge=0.0, le=100.0, description="Readability score")
    engagement_score: float = Field(default=0.0, ge=0.0, le=100.0, description="Engagement score")
    quality_score: float = Field(default=0.0, ge=0.0, le=100.0, description="Overall quality score")
    
    # Media
    featured_image: Optional[str] = Field(default=None, description="Featured image URL")
    images: List[str] = Field(default_factory=list, description="Image URLs")
    videos: List[str] = Field(default_factory=list, description="Video URLs")
    
    # Categories and tags
    categories: List[str] = Field(default_factory=list, description="Categories")
    tags: List[str] = Field(default_factory=list, description="Tags")
    
    # Publishing
    published_at: Optional[datetime] = Field(default=None, description="Published timestamp")
    scheduled_at: Optional[datetime] = Field(default=None, description="Scheduled timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Updated timestamp")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Created timestamp")
    
    # Analytics
    views: int = Field(default=0, ge=0, description="View count")
    likes: int = Field(default=0, ge=0, description="Like count")
    shares: int = Field(default=0, ge=0, description="Share count")
    comments: int = Field(default=0, ge=0, description="Comment count")
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @validator('slug')
    def validate_slug(cls, v):
        if not v.replace('-', '').replace('_', '').isalnum():
            raise ValueError('Slug must contain only alphanumeric characters, hyphens, and underscores')
        return v.lower()
    
    @validator('meta_title')
    def validate_meta_title(cls, v):
        if v and len(v) > 60:
            raise ValueError('Meta title should be 60 characters or less for optimal SEO')
        return v
    
    @validator('meta_description')
    def validate_meta_description(cls, v):
        if v and len(v) > 160:
            raise ValueError('Meta description should be 160 characters or less for optimal SEO')
        return v


class BlogPostRequest(BaseModel):
    """Blog post creation/update request"""
    title: str = Field(..., min_length=1, max_length=200, description="Post title")
    content: str = Field(..., min_length=1, description="Post content")
    excerpt: Optional[str] = Field(default=None, max_length=500, description="Post excerpt")
    content_type: ContentType = Field(default=ContentType.ARTICLE, description="Content type")
    content_format: ContentFormat = Field(default=ContentFormat.MARKDOWN, description="Content format")
    
    # SEO fields
    meta_title: Optional[str] = Field(default=None, max_length=60, description="Meta title")
    meta_description: Optional[str] = Field(default=None, max_length=160, description="Meta description")
    meta_keywords: List[str] = Field(default_factory=list, description="Meta keywords")
    
    # Media
    featured_image: Optional[str] = Field(default=None, description="Featured image URL")
    images: List[str] = Field(default_factory=list, description="Image URLs")
    videos: List[str] = Field(default_factory=list, description="Video URLs")
    
    # Categories and tags
    categories: List[str] = Field(default_factory=list, description="Categories")
    tags: List[str] = Field(default_factory=list, description="Tags")
    
    # Publishing
    status: BlogPostStatus = Field(default=BlogPostStatus.DRAFT, description="Post status")
    scheduled_at: Optional[datetime] = Field(default=None, description="Scheduled timestamp")
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class BlogPostResponse(BaseModel):
    """Blog post response"""
    success: bool = Field(..., description="Success status")
    data: Optional[BlogPost] = Field(default=None, description="Blog post data")
    message: str = Field(..., description="Response message")
    processing_time: float = Field(default=0.0, description="Processing time")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")


class BlogPostListResponse(BaseModel):
    """Blog post list response"""
    success: bool = Field(..., description="Success status")
    data: List[BlogPost] = Field(..., description="Blog posts data")
    total: int = Field(..., description="Total count")
    page: int = Field(..., description="Current page")
    per_page: int = Field(..., description="Items per page")
    total_pages: int = Field(..., description="Total pages")
    message: str = Field(..., description="Response message")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")


class ContentAnalysisRequest(BaseModel):
    """Content analysis request"""
    content: str = Field(..., min_length=1, description="Content to analyze")
    content_type: ContentType = Field(default=ContentType.ARTICLE, description="Content type")
    target_audience: Optional[str] = Field(default=None, description="Target audience")
    analysis_type: List[str] = Field(default_factory=lambda: ["seo", "readability", "engagement"], description="Analysis types")
    include_recommendations: bool = Field(default=True, description="Include recommendations")


class ContentAnalysisResponse(BaseModel):
    """Content analysis response"""
    analysis_id: str = Field(default_factory=lambda: str(uuid4()), description="Analysis ID")
    content_hash: int = Field(..., description="Content hash")
    analysis_results: Dict[str, Any] = Field(..., description="Analysis results")
    processing_time: float = Field(default=0.0, description="Processing time")
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence score")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations")
    generated_at: datetime = Field(default_factory=datetime.utcnow, description="Generation timestamp")


class ContentGenerationRequest(BaseModel):
    """Content generation request"""
    topic: str = Field(..., min_length=1, max_length=200, description="Content topic")
    content_type: ContentType = Field(default=ContentType.ARTICLE, description="Content type")
    target_audience: str = Field(..., description="Target audience")
    tone: str = Field(default="professional", description="Content tone")
    length: int = Field(default=1000, ge=100, le=5000, description="Target word count")
    style: str = Field(default="informative", description="Content style")
    keywords: List[str] = Field(default_factory=list, description="Target keywords")
    focus_areas: List[str] = Field(default_factory=list, description="Focus areas")
    avoid_topics: List[str] = Field(default_factory=list, description="Topics to avoid")
    include_seo: bool = Field(default=True, description="Include SEO optimization")
    include_media_suggestions: bool = Field(default=True, description="Include media suggestions")


class ContentGenerationResponse(BaseModel):
    """Content generation response"""
    generation_id: str = Field(default_factory=lambda: str(uuid4()), description="Generation ID")
    generated_content: str = Field(..., description="Generated content")
    word_count: int = Field(..., description="Word count")
    quality_metrics: Dict[str, Any] = Field(..., description="Quality metrics")
    generation_metadata: Dict[str, Any] = Field(..., description="Generation metadata")
    processing_time: float = Field(default=0.0, description="Processing time")
    generated_at: datetime = Field(default_factory=datetime.utcnow, description="Generation timestamp")


class SEOOptimizationRequest(BaseModel):
    """SEO optimization request"""
    content: str = Field(..., min_length=1, description="Content to optimize")
    target_keywords: List[str] = Field(..., min_items=1, description="Target keywords")
    content_type: ContentType = Field(default=ContentType.ARTICLE, description="Content type")
    target_audience: Optional[str] = Field(default=None, description="Target audience")
    competitor_analysis: bool = Field(default=False, description="Include competitor analysis")
    keyword_density_target: float = Field(default=2.0, ge=0.5, le=5.0, description="Target keyword density")
    include_meta_tags: bool = Field(default=True, description="Include meta tag suggestions")


class SEOOptimizationResponse(BaseModel):
    """SEO optimization response"""
    optimization_id: str = Field(default_factory=lambda: str(uuid4()), description="Optimization ID")
    original_content: str = Field(..., description="Original content")
    optimized_content: str = Field(..., description="Optimized content")
    seo_score_before: float = Field(..., ge=0.0, le=100.0, description="SEO score before")
    seo_score_after: float = Field(..., ge=0.0, le=100.0, description="SEO score after")
    recommendations: List[str] = Field(..., description="SEO recommendations")
    keyword_analysis: Dict[str, Any] = Field(..., description="Keyword analysis")
    processing_time: float = Field(default=0.0, description="Processing time")
    optimized_at: datetime = Field(default_factory=datetime.utcnow, description="Optimization timestamp")


class MLPipelineRequest(BaseModel):
    """ML pipeline request"""
    pipeline_type: str = Field(..., description="Pipeline type")
    input_data: Dict[str, Any] = Field(..., description="Input data")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Pipeline parameters")
    priority: str = Field(default="medium", description="Processing priority")
    callback_url: Optional[str] = Field(default=None, description="Callback URL")


class MLPipelineResponse(BaseModel):
    """ML pipeline response"""
    pipeline_id: str = Field(default_factory=lambda: str(uuid4()), description="Pipeline ID")
    status: str = Field(..., description="Pipeline status")
    results: Dict[str, Any] = Field(..., description="Pipeline results")
    processing_time: float = Field(default=0.0, description="Processing time")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Pipeline metadata")
    completed_at: datetime = Field(default_factory=datetime.utcnow, description="Completion timestamp")


class BlogPostSearchRequest(BaseModel):
    """Blog post search request"""
    query: Optional[str] = Field(default=None, description="Search query")
    categories: List[str] = Field(default_factory=list, description="Filter by categories")
    tags: List[str] = Field(default_factory=list, description="Filter by tags")
    content_type: Optional[ContentType] = Field(default=None, description="Filter by content type")
    status: Optional[BlogPostStatus] = Field(default=None, description="Filter by status")
    author_id: Optional[str] = Field(default=None, description="Filter by author")
    date_from: Optional[datetime] = Field(default=None, description="Filter from date")
    date_to: Optional[datetime] = Field(default=None, description="Filter to date")
    sort_by: str = Field(default="created_at", description="Sort field")
    sort_order: str = Field(default="desc", description="Sort order")
    page: int = Field(default=1, ge=1, description="Page number")
    per_page: int = Field(default=10, ge=1, le=100, description="Items per page")


class BlogPostAnalytics(BaseModel):
    """Blog post analytics"""
    post_id: str = Field(..., description="Post ID")
    time_period: str = Field(..., description="Time period")
    views: int = Field(default=0, description="View count")
    unique_views: int = Field(default=0, description="Unique view count")
    likes: int = Field(default=0, description="Like count")
    shares: int = Field(default=0, description="Share count")
    comments: int = Field(default=0, description="Comment count")
    engagement_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Engagement rate")
    bounce_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Bounce rate")
    avg_reading_time: float = Field(default=0.0, ge=0.0, description="Average reading time")
    traffic_sources: Dict[str, int] = Field(default_factory=dict, description="Traffic sources")
    top_keywords: List[str] = Field(default_factory=list, description="Top keywords")
    social_shares: Dict[str, int] = Field(default_factory=dict, description="Social shares")
    generated_at: datetime = Field(default_factory=datetime.utcnow, description="Generation timestamp")


class BlogPostPerformance(BaseModel):
    """Blog post performance metrics"""
    post_id: str = Field(..., description="Post ID")
    performance_score: float = Field(default=0.0, ge=0.0, le=100.0, description="Performance score")
    seo_performance: float = Field(default=0.0, ge=0.0, le=100.0, description="SEO performance")
    engagement_performance: float = Field(default=0.0, ge=0.0, le=100.0, description="Engagement performance")
    content_quality: float = Field(default=0.0, ge=0.0, le=100.0, description="Content quality")
    viral_potential: float = Field(default=0.0, ge=0.0, le=100.0, description="Viral potential")
    recommendations: List[str] = Field(default_factory=list, description="Performance recommendations")
    benchmark_comparison: Dict[str, Any] = Field(default_factory=dict, description="Benchmark comparison")
    generated_at: datetime = Field(default_factory=datetime.utcnow, description="Generation timestamp")


class BlogPostTemplate(BaseModel):
    """Blog post template"""
    template_id: str = Field(default_factory=lambda: str(uuid4()), description="Template ID")
    name: str = Field(..., min_length=1, max_length=100, description="Template name")
    description: str = Field(..., description="Template description")
    content_type: ContentType = Field(..., description="Content type")
    template_content: str = Field(..., description="Template content")
    variables: List[str] = Field(default_factory=list, description="Template variables")
    is_public: bool = Field(default=False, description="Is template public")
    created_by: str = Field(..., description="Created by user ID")
    usage_count: int = Field(default=0, ge=0, description="Usage count")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Created timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Updated timestamp")


class BlogPostWorkflow(BaseModel):
    """Blog post workflow"""
    workflow_id: str = Field(default_factory=lambda: str(uuid4()), description="Workflow ID")
    name: str = Field(..., min_length=1, max_length=100, description="Workflow name")
    description: str = Field(..., description="Workflow description")
    steps: List[Dict[str, Any]] = Field(..., description="Workflow steps")
    triggers: List[Dict[str, Any]] = Field(default_factory=list, description="Workflow triggers")
    is_active: bool = Field(default=True, description="Is workflow active")
    created_by: str = Field(..., description="Created by user ID")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Created timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Updated timestamp")


class BlogPostCollaboration(BaseModel):
    """Blog post collaboration"""
    collaboration_id: str = Field(default_factory=lambda: str(uuid4()), description="Collaboration ID")
    post_id: str = Field(..., description="Post ID")
    user_id: str = Field(..., description="User ID")
    role: str = Field(..., description="Collaboration role")
    permissions: List[str] = Field(..., description="User permissions")
    invited_at: datetime = Field(default_factory=datetime.utcnow, description="Invited timestamp")
    accepted_at: Optional[datetime] = Field(default=None, description="Accepted timestamp")
    status: str = Field(default="pending", description="Collaboration status")


class BlogPostComment(BaseModel):
    """Blog post comment"""
    comment_id: str = Field(default_factory=lambda: str(uuid4()), description="Comment ID")
    post_id: str = Field(..., description="Post ID")
    author_id: str = Field(..., description="Author ID")
    content: str = Field(..., min_length=1, max_length=1000, description="Comment content")
    parent_comment_id: Optional[str] = Field(default=None, description="Parent comment ID")
    is_approved: bool = Field(default=False, description="Is comment approved")
    likes: int = Field(default=0, ge=0, description="Like count")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Created timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Updated timestamp")


class BlogPostCategory(BaseModel):
    """Blog post category"""
    category_id: str = Field(default_factory=lambda: str(uuid4()), description="Category ID")
    name: str = Field(..., min_length=1, max_length=100, description="Category name")
    slug: str = Field(..., min_length=1, max_length=100, description="Category slug")
    description: Optional[str] = Field(default=None, description="Category description")
    parent_category_id: Optional[str] = Field(default=None, description="Parent category ID")
    color: Optional[str] = Field(default=None, description="Category color")
    icon: Optional[str] = Field(default=None, description="Category icon")
    post_count: int = Field(default=0, ge=0, description="Post count")
    is_active: bool = Field(default=True, description="Is category active")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Created timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Updated timestamp")


class BlogPostTag(BaseModel):
    """Blog post tag"""
    tag_id: str = Field(default_factory=lambda: str(uuid4()), description="Tag ID")
    name: str = Field(..., min_length=1, max_length=50, description="Tag name")
    slug: str = Field(..., min_length=1, max_length=50, description="Tag slug")
    description: Optional[str] = Field(default=None, description="Tag description")
    color: Optional[str] = Field(default=None, description="Tag color")
    post_count: int = Field(default=0, ge=0, description="Post count")
    is_active: bool = Field(default=True, description="Is tag active")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Created timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Updated timestamp")


class BlogPostAuthor(BaseModel):
    """Blog post author"""
    author_id: str = Field(default_factory=lambda: str(uuid4()), description="Author ID")
    name: str = Field(..., min_length=1, max_length=100, description="Author name")
    email: str = Field(..., description="Author email")
    bio: Optional[str] = Field(default=None, description="Author bio")
    avatar: Optional[str] = Field(default=None, description="Author avatar URL")
    social_links: Dict[str, str] = Field(default_factory=dict, description="Social media links")
    post_count: int = Field(default=0, ge=0, description="Post count")
    is_active: bool = Field(default=True, description="Is author active")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Created timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Updated timestamp")


class BlogPostSettings(BaseModel):
    """Blog post settings"""
    settings_id: str = Field(default_factory=lambda: str(uuid4()), description="Settings ID")
    site_name: str = Field(..., min_length=1, max_length=100, description="Site name")
    site_description: Optional[str] = Field(default=None, description="Site description")
    site_url: str = Field(..., description="Site URL")
    default_author_id: str = Field(..., description="Default author ID")
    posts_per_page: int = Field(default=10, ge=1, le=100, description="Posts per page")
    enable_comments: bool = Field(default=True, description="Enable comments")
    moderate_comments: bool = Field(default=True, description="Moderate comments")
    enable_seo: bool = Field(default=True, description="Enable SEO")
    enable_analytics: bool = Field(default=True, description="Enable analytics")
    enable_social_sharing: bool = Field(default=True, description="Enable social sharing")
    default_content_type: ContentType = Field(default=ContentType.ARTICLE, description="Default content type")
    default_content_format: ContentFormat = Field(default=ContentFormat.MARKDOWN, description="Default content format")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Created timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Updated timestamp")


class BlogPostSystemStatus(BaseModel):
    """Blog post system status"""
    total_posts: int = Field(default=0, description="Total posts")
    published_posts: int = Field(default=0, description="Published posts")
    draft_posts: int = Field(default=0, description="Draft posts")
    scheduled_posts: int = Field(default=0, description="Scheduled posts")
    total_views: int = Field(default=0, description="Total views")
    total_likes: int = Field(default=0, description="Total likes")
    total_shares: int = Field(default=0, description="Total shares")
    total_comments: int = Field(default=0, description="Total comments")
    system_health: str = Field(default="healthy", description="System health")
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")


# Error response models
class ErrorResponse(BaseModel):
    """Error response"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    request_id: Optional[str] = Field(default=None, description="Request ID")


# Configuration models
class BlogPostSystemConfig(BaseModel):
    """Blog post system configuration"""
    max_posts_per_user: int = Field(default=1000, ge=1, le=10000, description="Max posts per user")
    max_content_length: int = Field(default=100000, ge=1000, le=1000000, description="Max content length")
    max_title_length: int = Field(default=200, ge=10, le=500, description="Max title length")
    max_excerpt_length: int = Field(default=500, ge=50, le=1000, description="Max excerpt length")
    max_tags_per_post: int = Field(default=10, ge=1, le=50, description="Max tags per post")
    max_categories_per_post: int = Field(default=5, ge=1, le=20, description="Max categories per post")
    enable_auto_save: bool = Field(default=True, description="Enable auto save")
    auto_save_interval: int = Field(default=30, ge=5, le=300, description="Auto save interval in seconds")
    enable_content_analysis: bool = Field(default=True, description="Enable content analysis")
    enable_seo_optimization: bool = Field(default=True, description="Enable SEO optimization")
    enable_ml_generation: bool = Field(default=True, description="Enable ML content generation")
    enable_analytics: bool = Field(default=True, description="Enable analytics")
    enable_collaboration: bool = Field(default=True, description="Enable collaboration")
    enable_workflows: bool = Field(default=True, description="Enable workflows")
    enable_templates: bool = Field(default=True, description="Enable templates")


def get_settings() -> BlogPostSystemConfig:
    """Get blog post system settings"""
    return BlogPostSystemConfig()





























