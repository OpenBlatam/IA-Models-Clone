from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
from ...core.domain.entities.linkedin_post import PostStatus, PostType, PostTone
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
LinkedIn Post API Schemas
========================

Pydantic schemas for LinkedIn post API with fast NLP integration.
"""




class LinkedInPostCreate(BaseModel):
    """Schema for creating a LinkedIn post."""
    content: str = Field(..., min_length=1, max_length=3000, description="Post content")
    post_type: PostType = Field(PostType.ARTICLE, description="Type of post")
    tone: PostTone = Field(PostTone.PROFESSIONAL, description="Tone of the post")
    target_audience: Optional[str] = Field(None, max_length=200, description="Target audience")
    industry: Optional[str] = Field(None, max_length=100, description="Industry context")


class LinkedInPostUpdate(BaseModel):
    """Schema for updating a LinkedIn post."""
    content: Optional[str] = Field(None, min_length=1, max_length=3000, description="Post content")
    post_type: Optional[PostType] = Field(None, description="Type of post")
    tone: Optional[PostTone] = Field(None, description="Tone of the post")
    target_audience: Optional[str] = Field(None, max_length=200, description="Target audience")
    industry: Optional[str] = Field(None, max_length=100, description="Industry context")


class LinkedInPostResponse(BaseModel):
    """Schema for LinkedIn post response."""
    id: str = Field(..., description="Post ID")
    content: str = Field(..., description="Post content")
    post_type: PostType = Field(..., description="Type of post")
    tone: PostTone = Field(..., description="Tone of the post")
    target_audience: Optional[str] = Field(None, description="Target audience")
    industry: Optional[str] = Field(None, description="Industry context")
    status: PostStatus = Field(..., description="Post status")
    nlp_enhanced: bool = Field(False, description="Whether post was enhanced with NLP")
    nlp_processing_time: Optional[float] = Field(None, description="NLP processing time in seconds")
    grammar_issues: Optional[int] = Field(None, description="Number of grammar issues detected")
    grammar_suggestions: Optional[List[str]] = Field(None, description="Grammar suggestions list")
    quality_score: Optional[float] = Field(None, description="Overall quality score (0-100)")
    seo_score: Optional[float] = Field(None, description="SEO score (0-100)")
    meta_description: Optional[str] = Field(None, description="Suggested meta description (<=160 chars)")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")


class LinkedInPostListResponse(BaseModel):
    """Schema for LinkedIn post list response."""
    posts: List[LinkedInPostResponse] = Field(..., description="List of posts")
    total: int = Field(..., description="Total number of posts")
    limit: int = Field(..., description="Limit used for pagination")
    offset: int = Field(..., description="Offset used for pagination")


class PostOptimizationRequest(BaseModel):
    """Schema for post optimization request."""
    use_async_nlp: bool = Field(True, description="Use async NLP processor for maximum speed")


class PostAnalysisResponse(BaseModel):
    """Schema for post analysis response."""
    post_id: str = Field(..., description="Post ID")
    sentiment_score: float = Field(..., description="Sentiment score (-1 to 1)")
    readability_score: float = Field(..., description="Readability score")
    keywords: List[str] = Field(..., description="Extracted keywords")
    entities: List[tuple] = Field(..., description="Detected entities")
    processing_time: float = Field(..., description="Processing time in seconds")
    cached: bool = Field(..., description="Whether result was cached")
    async_optimized: bool = Field(..., description="Whether async processing was used")
    analyzed_at: datetime = Field(..., description="Analysis timestamp")


class BatchOptimizationRequest(BaseModel):
    """Schema for batch optimization request."""
    post_ids: List[str] = Field(..., min_items=1, max_items=50, description="List of post IDs to optimize")
    use_async_nlp: bool = Field(True, description="Use async NLP processor for maximum speed")


class NLPPerformanceMetrics(BaseModel):
    """Schema for NLP performance metrics."""
    total_requests: int = Field(..., description="Total number of requests")
    cache_hits: int = Field(..., description="Number of cache hits")
    cache_misses: int = Field(..., description="Number of cache misses")
    cache_hit_rate: float = Field(..., description="Cache hit rate percentage")
    average_processing_time: float = Field(..., description="Average processing time in seconds")
    memory_cache_size: int = Field(..., description="Memory cache size")
    redis_connected: bool = Field(..., description="Whether Redis is connected")
    models_loaded: Optional[bool] = Field(None, description="Whether models are loaded")
    concurrent_operations: Optional[int] = Field(None, description="Number of concurrent operations")
    batch_operations: Optional[int] = Field(None, description="Number of batch operations")


class NLPPerformanceResponse(BaseModel):
    """Schema for NLP performance response."""
    fast_nlp_metrics: NLPPerformanceMetrics = Field(..., description="Fast NLP processor metrics")
    async_nlp_metrics: NLPPerformanceMetrics = Field(..., description="Async NLP processor metrics")
    timestamp: str = Field(..., description="Metrics timestamp")


class HealthCheckResponse(BaseModel):
    """Schema for health check response."""
    status: str = Field(..., description="Service status")
    service: str = Field(..., description="Service name")
    timestamp: str = Field(..., description="Health check timestamp")
    version: str = Field(..., description="Service version")
    features: List[str] = Field(..., description="Available features")


class ErrorResponse(BaseModel):
    """Schema for error response."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Error details")
    timestamp: str = Field(..., description="Error timestamp")


class SuccessResponse(BaseModel):
    """Schema for success response."""
    message: str = Field(..., description="Success message")
    timestamp: str = Field(..., description="Response timestamp")


# Fast NLP specific schemas
class FastNLPOptimizationRequest(BaseModel):
    """Schema for fast NLP optimization request."""
    content: str = Field(..., min_length=1, max_length=3000, description="Content to optimize")
    use_async_processing: bool = Field(True, description="Use async processing for maximum speed")
    enable_caching: bool = Field(True, description="Enable caching for faster subsequent requests")


class FastNLPOptimizationResponse(BaseModel):
    """Schema for fast NLP optimization response."""
    original_content: str = Field(..., description="Original content")
    optimized_content: str = Field(..., description="Optimized content")
    processing_time: float = Field(..., description="Processing time in seconds")
    cached: bool = Field(..., description="Whether result was cached")
    async_optimized: bool = Field(..., description="Whether async processing was used")
    sentiment_analysis: Dict[str, float] = Field(..., description="Sentiment analysis results")
    readability_analysis: Dict[str, float] = Field(..., description="Readability analysis results")
    keywords: List[str] = Field(..., description="Extracted keywords")
    entities: List[tuple] = Field(..., description="Detected entities")
    optimization_score: float = Field(..., description="Overall optimization score")


class BatchFastNLPRequest(BaseModel):
    """Schema for batch fast NLP processing request."""
    contents: List[str] = Field(..., min_items=1, max_items=100, description="List of contents to process")
    use_async_processing: bool = Field(True, description="Use async processing for maximum speed")
    enable_caching: bool = Field(True, description="Enable caching for faster subsequent requests")


class BatchFastNLPResponse(BaseModel):
    """Schema for batch fast NLP processing response."""
    results: List[FastNLPOptimizationResponse] = Field(..., description="Processing results")
    total_processing_time: float = Field(..., description="Total processing time in seconds")
    average_processing_time: float = Field(..., description="Average processing time per content")
    batch_size: int = Field(..., description="Number of contents processed")
    success_count: int = Field(..., description="Number of successful optimizations")
    error_count: int = Field(..., description="Number of failed optimizations")


class CacheManagementRequest(BaseModel):
    """Schema for cache management request."""
    action: str = Field(..., description="Cache action (clear, stats, warm)")
    cache_type: Optional[str] = Field(None, description="Cache type (memory, redis, all)")


class CacheManagementResponse(BaseModel):
    """Schema for cache management response."""
    action: str = Field(..., description="Cache action performed")
    success: bool = Field(..., description="Whether action was successful")
    message: str = Field(..., description="Action result message")
    cache_stats: Optional[Dict[str, Any]] = Field(None, description="Cache statistics")
    timestamp: str = Field(..., description="Action timestamp")


class PerformanceBenchmarkRequest(BaseModel):
    """Schema for performance benchmark request."""
    test_type: str = Field(..., description="Type of benchmark test")
    iterations: int = Field(100, ge=1, le=1000, description="Number of test iterations")
    content_length: str = Field("medium", description="Content length (short, medium, long)")


class PerformanceBenchmarkResponse(BaseModel):
    """Schema for performance benchmark response."""
    test_type: str = Field(..., description="Type of benchmark test")
    iterations: int = Field(..., description="Number of test iterations")
    total_time: float = Field(..., description="Total test time in seconds")
    average_time: float = Field(..., description="Average time per iteration")
    throughput: float = Field(..., description="Throughput (operations per second)")
    cache_hit_rate: float = Field(..., description="Cache hit rate percentage")
    memory_usage: Dict[str, float] = Field(..., description="Memory usage statistics")
    cpu_usage: Dict[str, float] = Field(..., description="CPU usage statistics")
    timestamp: str = Field(..., description="Benchmark timestamp") 