from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

from datetime import datetime
from typing import Dict, List, Optional
from uuid import UUID
from pydantic import BaseModel, Field
from ...core.entities.linkedin_post import PostStatus, PostType, ContentTone
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
LinkedIn Post API Schemas
========================

API schemas for LinkedIn post operations including generation, optimization, and analysis.
"""





class LinkedInPostCreateRequest(BaseModel):
    """Request schema for creating a LinkedIn post."""
    
    title: str = Field(..., min_length=1, max_length=100, description="Post title")
    content: str = Field(..., min_length=10, max_length=3000, description="Post content")
    summary: Optional[str] = Field(None, max_length=500, description="Post summary")
    post_type: PostType = Field(default=PostType.TEXT, description="Type of post")
    tone: ContentTone = Field(default=ContentTone.PROFESSIONAL, description="Content tone")
    keywords: List[str] = Field(default=[], description="Target keywords")
    hashtags: List[str] = Field(default=[], description="Post hashtags")
    category: Optional[str] = Field(None, description="Post category")
    industry: Optional[str] = Field(None, description="Target industry")
    scheduled_at: Optional[datetime] = Field(None, description="Scheduled publish time")


class LinkedInPostUpdateRequest(BaseModel):
    """Request schema for updating a LinkedIn post."""
    
    title: Optional[str] = Field(None, min_length=1, max_length=100, description="Post title")
    content: Optional[str] = Field(None, min_length=10, max_length=3000, description="Post content")
    summary: Optional[str] = Field(None, max_length=500, description="Post summary")
    tone: Optional[ContentTone] = Field(None, description="Content tone")
    keywords: Optional[List[str]] = Field(None, description="Target keywords")
    hashtags: Optional[List[str]] = Field(None, description="Post hashtags")
    category: Optional[str] = Field(None, description="Post category")
    industry: Optional[str] = Field(None, description="Target industry")
    scheduled_at: Optional[datetime] = Field(None, description="Scheduled publish time")


class LinkedInPostResponse(BaseModel):
    """Response schema for LinkedIn post details."""
    
    id: UUID = Field(..., description="Post ID")
    title: str = Field(..., description="Post title")
    content: str = Field(..., description="Post content")
    summary: Optional[str] = Field(None, description="Post summary")
    post_type: PostType = Field(..., description="Type of post")
    status: PostStatus = Field(..., description="Post status")
    tone: ContentTone = Field(..., description="Content tone")
    keywords: List[str] = Field(default=[], description="Target keywords")
    hashtags: List[str] = Field(default=[], description="Post hashtags")
    category: Optional[str] = Field(None, description="Post category")
    industry: Optional[str] = Field(None, description="Target industry")
    
    # Engagement metrics
    likes_count: int = Field(default=0, description="Number of likes")
    comments_count: int = Field(default=0, description="Number of comments")
    shares_count: int = Field(default=0, description="Number of shares")
    views_count: int = Field(default=0, description="Number of views")
    engagement_rate: float = Field(default=0.0, description="Engagement rate")
    
    # Content analysis
    readability_score: Optional[float] = Field(None, description="Readability score")
    sentiment_score: Optional[float] = Field(None, description="Sentiment score")
    engagement_score: Optional[float] = Field(None, description="Engagement score")
    seo_score: Optional[float] = Field(None, description="SEO score")
    
    # Publishing
    scheduled_at: Optional[datetime] = Field(None, description="Scheduled publish time")
    published_at: Optional[datetime] = Field(None, description="Actual publish time")
    linkedin_post_id: Optional[str] = Field(None, description="LinkedIn post ID")
    
    # Metadata
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    
    # A/B testing
    is_ab_test: bool = Field(default=False, description="Is A/B test variant")
    ab_test_id: Optional[UUID] = Field(None, description="A/B test identifier")
    variant_id: Optional[str] = Field(None, description="Test variant ID")
    
    # LangChain data
    langchain_model: Optional[str] = Field(None, description="LangChain model used")
    estimated_read_time: int = Field(..., description="Estimated reading time in minutes")


class LinkedInPostListResponse(BaseModel):
    """Response schema for LinkedIn post list."""
    
    posts: List[LinkedInPostResponse] = Field(..., description="List of posts")
    total_count: int = Field(..., description="Total number of posts")
    has_more: bool = Field(..., description="Whether there are more posts")
    page_info: Dict = Field(default={}, description="Pagination information")


class LinkedInPostGenerateRequest(BaseModel):
    """Request schema for generating a LinkedIn post using AI."""
    
    topic: str = Field(..., min_length=1, description="Main topic of the post")
    key_points: List[str] = Field(..., min_items=1, description="Key points to cover")
    target_audience: str = Field(..., description="Target audience description")
    industry: str = Field(..., description="Industry focus")
    tone: ContentTone = Field(default=ContentTone.PROFESSIONAL, description="Content tone")
    post_type: PostType = Field(default=PostType.TEXT, description="Type of post")
    keywords: Optional[List[str]] = Field(default=[], description="Target keywords")
    additional_context: Optional[str] = Field(None, description="Additional context")
    
    # Generation options
    include_hashtags: bool = Field(default=True, description="Include hashtags in generation")
    include_call_to_action: bool = Field(default=True, description="Include call to action")
    optimize_for_engagement: bool = Field(default=True, description="Optimize for engagement")
    use_storytelling: bool = Field(default=False, description="Use storytelling approach")


class LinkedInPostGenerateResponse(BaseModel):
    """Response schema for LinkedIn post generation."""
    
    post_id: UUID = Field(..., description="Generated post ID")
    title: str = Field(..., description="Generated post title")
    content: str = Field(..., description="Generated post content")
    summary: Optional[str] = Field(None, description="Generated post summary")
    hashtags: List[str] = Field(default=[], description="Generated hashtags")
    keywords: List[str] = Field(default=[], description="Target keywords used")
    call_to_action: Optional[str] = Field(None, description="Generated call to action")
    estimated_engagement: float = Field(..., description="Estimated engagement score")
    status: str = Field(..., description="Post status")
    
    # Generation details
    generation_time: float = Field(..., description="Generation time in seconds")
    langchain_data: Dict = Field(..., description="LangChain generation data")
    optimization_applied: bool = Field(default=False, description="Whether optimization was applied")
    
    # Content analysis
    readability_score: Optional[float] = Field(None, description="Readability score")
    sentiment_score: Optional[float] = Field(None, description="Sentiment score")
    word_count: int = Field(..., description="Word count")
    estimated_read_time: int = Field(..., description="Estimated reading time in minutes")


class LinkedInPostOptimizeRequest(BaseModel):
    """Request schema for optimizing a LinkedIn post."""
    
    post_id: UUID = Field(..., description="Post ID to optimize")
    optimization_type: str = Field(
        default="comprehensive",
        description="Type of optimization: comprehensive, readability, engagement, seo"
    )
    target_audience: Optional[str] = Field(None, description="Target audience for optimization")
    focus_areas: Optional[List[str]] = Field(None, description="Specific areas to focus on")


class LinkedInPostOptimizeResponse(BaseModel):
    """Response schema for LinkedIn post optimization."""
    
    post_id: UUID = Field(..., description="Optimized post ID")
    original_content: str = Field(..., description="Original content")
    optimized_content: str = Field(..., description="Optimized content")
    optimization_type: str = Field(..., description="Type of optimization applied")
    
    # Optimization details
    optimization_details: Dict = Field(..., description="Detailed optimization information")
    improvement_score: Optional[float] = Field(None, description="Improvement score")
    changes_made: List[str] = Field(default=[], description="List of changes made")
    
    # Content analysis comparison
    original_analysis: Dict = Field(default={}, description="Original content analysis")
    optimized_analysis: Dict = Field(default={}, description="Optimized content analysis")
    
    # Performance metrics
    optimization_time: float = Field(..., description="Optimization time in seconds")


class LinkedInPostAnalyzeRequest(BaseModel):
    """Request schema for analyzing LinkedIn post engagement."""
    
    post_id: UUID = Field(..., description="Post ID to analyze")
    analysis_type: str = Field(
        default="comprehensive",
        description="Type of analysis: comprehensive, sentiment, audience, engagement"
    )
    include_recommendations: bool = Field(default=True, description="Include improvement recommendations")


class LinkedInPostAnalyzeResponse(BaseModel):
    """Response schema for LinkedIn post analysis."""
    
    post_id: UUID = Field(..., description="Analyzed post ID")
    analysis_type: str = Field(..., description="Type of analysis performed")
    
    # Analysis results
    overall_score: float = Field(..., description="Overall analysis score")
    sentiment_analysis: Dict = Field(default={}, description="Sentiment analysis results")
    engagement_prediction: Dict = Field(default={}, description="Engagement prediction")
    audience_resonance: Dict = Field(default={}, description="Audience resonance analysis")
    content_quality: Dict = Field(default={}, description="Content quality analysis")
    
    # Detailed breakdown
    factors: Dict = Field(default={}, description="Detailed factor breakdown")
    recommendations: List[str] = Field(default=[], description="Improvement recommendations")
    
    # Performance metrics
    analysis_time: float = Field(..., description="Analysis time in seconds")
    confidence_score: float = Field(..., description="Analysis confidence score")


class LinkedInPostABTestRequest(BaseModel):
    """Request schema for creating A/B test variants."""
    
    post_id: UUID = Field(..., description="Original post ID")
    num_variants: int = Field(default=3, ge=2, le=5, description="Number of variants to create")
    variant_types: Optional[List[str]] = Field(None, description="Types of variants to create")
    test_duration_days: int = Field(default=7, ge=1, le=30, description="Test duration in days")


class LinkedInPostABTestResponse(BaseModel):
    """Response schema for A/B test creation."""
    
    test_id: UUID = Field(..., description="A/B test ID")
    original_post_id: UUID = Field(..., description="Original post ID")
    variant_posts: List[LinkedInPostResponse] = Field(..., description="Created variant posts")
    
    # Test configuration
    num_variants: int = Field(..., description="Number of variants created")
    test_duration_days: int = Field(..., description="Test duration in days")
    test_start_date: datetime = Field(..., description="Test start date")
    test_end_date: datetime = Field(..., description="Test end date")
    
    # Test status
    status: str = Field(..., description="Test status")
    is_active: bool = Field(..., description="Whether test is active")


class LinkedInPostMetricsResponse(BaseModel):
    """Response schema for LinkedIn post metrics."""
    
    post_id: UUID = Field(..., description="Post ID")
    
    # Engagement metrics
    likes_count: int = Field(default=0, description="Number of likes")
    comments_count: int = Field(default=0, description="Number of comments")
    shares_count: int = Field(default=0, description="Number of shares")
    views_count: int = Field(default=0, description="Number of views")
    click_through_rate: float = Field(default=0.0, description="Click-through rate")
    engagement_rate: float = Field(default=0.0, description="Engagement rate")
    
    # Performance metrics
    reach: Optional[int] = Field(None, description="Post reach")
    impressions: Optional[int] = Field(None, description="Post impressions")
    unique_visitors: Optional[int] = Field(None, description="Unique visitors")
    
    # Time-based metrics
    peak_engagement_time: Optional[datetime] = Field(None, description="Peak engagement time")
    average_engagement_time: Optional[float] = Field(None, description="Average engagement time")
    
    # Comparative metrics
    industry_average: Dict = Field(default={}, description="Industry average metrics")
    performance_percentile: float = Field(default=50.0, description="Performance percentile")
    
    # Last updated
    last_updated: datetime = Field(..., description="Last metrics update")


class LinkedInPostBulkOperationRequest(BaseModel):
    """Request schema for bulk LinkedIn post operations."""
    
    operation_type: str = Field(..., description="Type of operation: generate, optimize, analyze")
    posts: List[Dict] = Field(..., description="List of posts for bulk operation")
    options: Dict = Field(default={}, description="Operation options")


class LinkedInPostBulkOperationResponse(BaseModel):
    """Response schema for bulk LinkedIn post operations."""
    
    operation_id: UUID = Field(..., description="Bulk operation ID")
    operation_type: str = Field(..., description="Type of operation performed")
    total_posts: int = Field(..., description="Total number of posts processed")
    successful_posts: int = Field(..., description="Number of successfully processed posts")
    failed_posts: int = Field(..., description="Number of failed posts")
    
    # Results
    results: List[Dict] = Field(..., description="Operation results for each post")
    errors: List[Dict] = Field(default=[], description="Error details for failed posts")
    
    # Performance
    total_time: float = Field(..., description="Total operation time in seconds")
    average_time_per_post: float = Field(..., description="Average time per post")
    
    # Status
    status: str = Field(..., description="Operation status")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp") 