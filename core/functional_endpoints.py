from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

from __future__ import annotations
from typing import Any, Dict, List, Optional, Type
from datetime import datetime
import uuid
import asyncio
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from fastapi.responses import JSONResponse
import structlog
from .functional_components import (
from typing import Any, List, Dict, Optional
import logging
"""
Functional Endpoints with Pydantic Models
========================================

Example FastAPI endpoints demonstrating the use of functional components
with Pydantic models for input validation and response schemas.

This module shows how to:
- Create pure functional components
- Use Pydantic v2 models for validation
- Compose components into pipelines
- Handle errors gracefully
- Monitor performance
- Cache results
- Use async components
"""



    BaseInputModel, BaseOutputModel, ErrorOutputModel,
    component, async_component, compose_components, compose_async_components,
    execute_parallel, conditional_component, retry_component,
    get_component_metrics, log_component_metrics,
    UserInputModel, UserOutputModel,
    validate_user_input, enrich_user_data, async_user_processing
)

logger = structlog.get_logger(__name__)

# Create router
router = APIRouter(prefix="/functional", tags=["functional-components"])

# ============================================================================
# INPUT/OUTPUT MODELS
# ============================================================================

class BlogPostInputModel(BaseInputModel):
    """Input model for blog post operations."""
    
    title: str = Field(..., min_length=1, max_length=200, description="Blog post title")
    content: str = Field(..., min_length=10, description="Blog post content")
    author_id: str = Field(..., description="Author ID")
    tags: List[str] = Field(default_factory=list, description="Blog post tags")
    category: Optional[str] = Field(None, description="Blog post category")
    is_published: bool = Field(default=False, description="Publication status")
    
    @field_validator("title")
    @classmethod
    def validate_title(cls, v: str) -> str:
        """Validate and clean title."""
        return v.strip()
    
    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        """Validate content length."""
        if len(v.strip()) < 10:
            raise ValueError("Content must be at least 10 characters long")
        return v.strip()
    
    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v: List[str]) -> List[str]:
        """Validate and clean tags."""
        return [tag.strip().lower() for tag in v if tag.strip()]

class BlogPostOutputModel(BaseOutputModel):
    """Output model for blog post operations."""
    
    success: bool = Field(default=True)
    post_id: str = Field(..., description="Blog post ID")
    title: str = Field(..., description="Blog post title")
    content: str = Field(..., description="Blog post content")
    author_id: str = Field(..., description="Author ID")
    tags: List[str] = Field(default_factory=list, description="Blog post tags")
    category: Optional[str] = Field(None, description="Blog post category")
    is_published: bool = Field(default=False, description="Publication status")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    word_count: Optional[int] = Field(None, description="Word count")
    reading_time_minutes: Optional[float] = Field(None, description="Estimated reading time")
    
    @computed_field
    @property
    def slug(self) -> str:
        """Generate URL slug from title."""
        return self.title.lower().replace(" ", "-").replace("_", "-")
    
    @computed_field
    @property
    def excerpt(self) -> str:
        """Generate excerpt from content."""
        return self.content[:150] + "..." if len(self.content) > 150 else self.content

class SearchInputModel(BaseInputModel):
    """Input model for search operations."""
    
    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Search filters")
    page: int = Field(default=1, ge=1, description="Page number")
    per_page: int = Field(default=10, ge=1, le=100, description="Items per page")
    sort_by: Optional[str] = Field(None, description="Sort field")
    sort_order: str = Field(default="desc", pattern="^(asc|desc)$", description="Sort order")

class SearchOutputModel(BaseOutputModel):
    """Output model for search operations."""
    
    success: bool = Field(default=True)
    query: str = Field(..., description="Search query")
    results: List[Dict[str, Any]] = Field(default_factory=list, description="Search results")
    total_count: int = Field(default=0, description="Total number of results")
    page: int = Field(default=1, description="Current page")
    per_page: int = Field(default=10, description="Items per page")
    total_pages: int = Field(default=0, description="Total number of pages")
    has_next: bool = Field(default=False, description="Has next page")
    has_prev: bool = Field(default=False, description="Has previous page")
    search_time_ms: Optional[float] = Field(None, description="Search execution time")

# ============================================================================
# FUNCTIONAL COMPONENTS
# ============================================================================

@component(name="validate_blog_post", cache_result=False)
def validate_blog_post(input_data: BlogPostInputModel) -> BlogPostOutputModel:
    """Validate blog post input data."""
    try:
        # Additional validation logic
        if len(input_data.content.split()) < 10:
            raise ValueError("Blog post must have at least 10 words")
        
        # Calculate word count and reading time
        word_count = len(input_data.content.split())
        reading_time_minutes = word_count / 200  # Average reading speed
        
        return BlogPostOutputModel(
            success=True,
            post_id=str(uuid.uuid4()),
            title=input_data.title,
            content=input_data.content,
            author_id=input_data.author_id,
            tags=input_data.tags,
            category=input_data.category,
            is_published=input_data.is_published,
            word_count=word_count,
            reading_time_minutes=reading_time_minutes
        )
    except Exception as e:
        return ErrorOutputModel(
            success=False,
            error_code="BLOG_POST_VALIDATION_ERROR",
            error=str(e),
            error_details={"field": "validation"}
        )

@component(name="enrich_blog_post", cache_result=True, cache_ttl=300)
def enrich_blog_post(input_data: BlogPostOutputModel) -> BlogPostOutputModel:
    """Enrich blog post with additional metadata."""
    try:
        enriched_data = input_data.model_copy()
        
        # Add SEO metadata
        enriched_data.metadata = {
            "seo_title": f"{input_data.title} - Blog Post",
            "seo_description": input_data.excerpt,
            "keywords": input_data.tags,
            "enriched": True,
            "enrichment_timestamp": datetime.utcnow().isoformat()
        }
        
        # Add social media metadata
        enriched_data.metadata.update({
            "og_title": input_data.title,
            "og_description": input_data.excerpt,
            "og_type": "article",
            "twitter_card": "summary"
        })
        
        return enriched_data
    except Exception as e:
        return ErrorOutputModel(
            success=False,
            error_code="BLOG_POST_ENRICHMENT_ERROR",
            error=str(e),
            error_details={"operation": "enrichment"}
        )

@async_component(name="async_blog_post_processing", cache_result=False)
async def async_blog_post_processing(input_data: BlogPostOutputModel) -> BlogPostOutputModel:
    """Async blog post processing (e.g., AI analysis, content optimization)."""
    try:
        # Simulate async processing
        await asyncio.sleep(0.1)
        
        processed_data = input_data.model_copy()
        
        # Simulate AI content analysis
        processed_data.metadata.update({
            "ai_analyzed": True,
            "sentiment_score": 0.85,
            "readability_score": 0.78,
            "processing_timestamp": datetime.utcnow().isoformat()
        })
        
        return processed_data
    except Exception as e:
        return ErrorOutputModel(
            success=False,
            error_code="ASYNC_BLOG_POST_PROCESSING_ERROR",
            error=str(e),
            error_details={"operation": "async_processing"}
        )

@component(name="search_blog_posts", cache_result=True, cache_ttl=60)
def search_blog_posts(input_data: SearchInputModel) -> SearchOutputModel:
    """Search blog posts with filters and pagination."""
    try:
        start_time = datetime.utcnow()
        
        # Simulate search operation
        # In real implementation, this would query a database
        mock_results = [
            {
                "id": str(uuid.uuid4()),
                "title": f"Blog Post {i}",
                "excerpt": f"This is blog post {i} content...",
                "author": f"Author {i}",
                "created_at": datetime.utcnow().isoformat(),
                "tags": ["tag1", "tag2"]
            }
            for i in range(1, 6)
        ]
        
        total_count = 25  # Mock total count
        total_pages = (total_count + input_data.per_page - 1) // input_data.per_page
        
        search_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return SearchOutputModel(
            success=True,
            query=input_data.query,
            results=mock_results,
            total_count=total_count,
            page=input_data.page,
            per_page=input_data.per_page,
            total_pages=total_pages,
            has_next=input_data.page < total_pages,
            has_prev=input_data.page > 1,
            search_time_ms=search_time
        )
    except Exception as e:
        return ErrorOutputModel(
            success=False,
            error_code="SEARCH_ERROR",
            error=str(e),
            error_details={"operation": "search"}
        )

# ============================================================================
# COMPONENT PIPELINES
# ============================================================================

# Blog post processing pipeline
blog_post_pipeline = compose_components(
    validate_blog_post,
    enrich_blog_post
)

# Async blog post processing pipeline
async_blog_post_pipeline = compose_async_components(
    validate_blog_post,
    async_blog_post_processing
)

# Conditional processing based on post length
def is_long_post(input_data: BlogPostInputModel) -> bool:
    """Check if blog post is long (more than 1000 words)."""
    return len(input_data.content.split()) > 1000

long_post_processing = conditional_component(
    condition=is_long_post,
    true_component=async_blog_post_processing,
    false_component=lambda x: x  # No additional processing for short posts
)

# Robust processing with retry logic
robust_blog_post_processing = retry_component(
    component_func=async_blog_post_processing,
    max_retries=3,
    retry_delay=1.0,
    backoff_factor=2.0
)

# ============================================================================
# FASTAPI ENDPOINTS
# ============================================================================

@router.post("/blog-posts", response_model=BlogPostOutputModel)
async def create_blog_post(
    input_data: BlogPostInputModel,
    background_tasks: BackgroundTasks
) -> BlogPostOutputModel:
    """
    Create a new blog post using functional components.
    
    This endpoint demonstrates:
    - Input validation with Pydantic models
    - Functional component pipeline
    - Background task processing
    - Error handling
    """
    try:
        # Process blog post through pipeline
        result = blog_post_pipeline(input_data)
        
        # Check for errors
        if isinstance(result, ErrorOutputModel):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.error
            )
        
        # Add background task for additional processing
        background_tasks.add_task(
            async_blog_post_processing,
            result
        )
        
        logger.info(
            "Blog post created successfully",
            post_id=result.post_id,
            title=result.title,
            author_id=result.author_id
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error creating blog post", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@router.post("/blog-posts/async", response_model=BlogPostOutputModel)
async def create_blog_post_async(input_data: BlogPostInputModel) -> BlogPostOutputModel:
    """
    Create a blog post with async processing.
    
    This endpoint demonstrates:
    - Async functional components
    - Parallel processing
    - Performance monitoring
    """
    try:
        # Process through async pipeline
        result = await async_blog_post_pipeline(input_data)
        
        if isinstance(result, ErrorOutputModel):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.error
            )
        
        logger.info(
            "Async blog post processing completed",
            post_id=result.post_id,
            execution_time_ms=result.execution_time_ms
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error in async blog post processing", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@router.post("/blog-posts/conditional", response_model=BlogPostOutputModel)
async def create_blog_post_conditional(input_data: BlogPostInputModel) -> BlogPostOutputModel:
    """
    Create a blog post with conditional processing.
    
    This endpoint demonstrates:
    - Conditional component execution
    - Different processing paths based on input
    """
    try:
        # Use conditional processing
        result = await long_post_processing(input_data)
        
        if isinstance(result, ErrorOutputModel):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.error
            )
        
        # Add conditional metadata
        if is_long_post(input_data):
            result.metadata["processing_type"] = "long_post_processing"
        else:
            result.metadata["processing_type"] = "standard_processing"
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error in conditional blog post processing", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@router.post("/blog-posts/robust", response_model=BlogPostOutputModel)
async def create_blog_post_robust(input_data: BlogPostInputModel) -> BlogPostOutputModel:
    """
    Create a blog post with robust error handling and retry logic.
    
    This endpoint demonstrates:
    - Retry logic for failed operations
    - Robust error handling
    - Performance monitoring
    """
    try:
        # Use robust processing with retry logic
        result = await robust_blog_post_processing(input_data)
        
        if isinstance(result, ErrorOutputModel):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.error
            )
        
        logger.info(
            "Robust blog post processing completed",
            post_id=result.post_id,
            execution_time_ms=result.execution_time_ms
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error in robust blog post processing", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@router.post("/search", response_model=SearchOutputModel)
async def search_posts(input_data: SearchInputModel) -> SearchOutputModel:
    """
    Search blog posts using functional components.
    
    This endpoint demonstrates:
    - Search functionality with caching
    - Pagination handling
    - Performance monitoring
    """
    try:
        # Execute search component
        result = search_blog_posts(input_data)
        
        if isinstance(result, ErrorOutputModel):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.error
            )
        
        logger.info(
            "Search completed successfully",
            query=input_data.query,
            results_count=len(result.results),
            search_time_ms=result.search_time_ms
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error in search operation", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@router.post("/parallel-processing", response_model=Dict[str, Any])
async def parallel_processing(input_data: BlogPostInputModel) -> Dict[str, Any]:
    """
    Demonstrate parallel processing of multiple components.
    
    This endpoint demonstrates:
    - Parallel component execution
    - Multiple processing paths
    - Result aggregation
    """
    try:
        # Define components to run in parallel
        components_to_run = [
            validate_blog_post,
            enrich_blog_post,
            search_blog_posts
        ]
        
        # Create search input for the search component
        search_input = SearchInputModel(query=input_data.title)
        
        # Execute components in parallel
        results = await execute_parallel(
            components_to_run,
            input_data,
            search_input=search_input
        )
        
        # Aggregate results
        aggregated_result = {
            "validation_result": results[0].model_dump() if results[0].success else None,
            "enrichment_result": results[1].model_dump() if results[1].success else None,
            "search_result": results[2].model_dump() if results[2].success else None,
            "parallel_execution": True,
            "components_executed": len(components_to_run)
        }
        
        logger.info(
            "Parallel processing completed",
            components_executed=len(components_to_run),
            successful_results=sum(1 for r in results if r.success)
        )
        
        return aggregated_result
        
    except Exception as e:
        logger.error("Error in parallel processing", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

# ============================================================================
# MONITORING AND METRICS ENDPOINTS
# ============================================================================

@router.get("/metrics")
async def get_component_metrics() -> Dict[str, Any]:
    """Get performance metrics for all functional components."""
    try:
        metrics = get_all_metrics()
        
        # Convert metrics to serializable format
        serializable_metrics = {}
        for name, metric in metrics.items():
            serializable_metrics[name] = {
                "execution_count": metric.execution_count,
                "total_execution_time": metric.total_execution_time,
                "average_execution_time": metric.average_execution_time,
                "error_count": metric.error_count,
                "error_rate": metric.error_rate,
                "cache_hits": metric.cache_hits,
                "cache_misses": metric.cache_misses,
                "cache_hit_rate": metric.cache_hit_rate
            }
        
        return {
            "component_metrics": serializable_metrics,
            "total_components": len(metrics),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Error getting component metrics", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving metrics"
        )

@router.post("/metrics/reset")
async def reset_component_metrics() -> Dict[str, str]:
    """Reset all component metrics."""
    try:
        reset_metrics()
        logger.info("Component metrics reset successfully")
        return {"message": "Metrics reset successfully"}
        
    except Exception as e:
        logger.error("Error resetting component metrics", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error resetting metrics"
        )

@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint for functional components."""
    try:
        # Log component metrics
        log_component_metrics()
        
        return {
            "status": "healthy",
            "functional_components": "operational",
            "timestamp": datetime.utcnow().isoformat(),
            "component_count": len(get_all_metrics())
        }
        
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unhealthy"
        )

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_blog_post_schema(
    title: str,
    content: str,
    author_id: str,
    tags: Optional[List[str]] = None,
    category: Optional[str] = None,
    is_published: bool = False
) -> BlogPostInputModel:
    """Utility function to create blog post input schema."""
    return BlogPostInputModel(
        title=title,
        content=content,
        author_id=author_id,
        tags=tags or [],
        category=category,
        is_published=is_published
    )

def create_search_schema(
    query: str,
    filters: Optional[Dict[str, Any]] = None,
    page: int = 1,
    per_page: int = 10,
    sort_by: Optional[str] = None,
    sort_order: str = "desc"
) -> SearchInputModel:
    """Utility function to create search input schema."""
    return SearchInputModel(
        query=query,
        filters=filters or {},
        page=page,
        per_page=per_page,
        sort_by=sort_by,
        sort_order=sort_order
    )

# Export main components
__all__ = [
    # Models
    "BlogPostInputModel",
    "BlogPostOutputModel", 
    "SearchInputModel",
    "SearchOutputModel",
    
    # Components
    "validate_blog_post",
    "enrich_blog_post",
    "async_blog_post_processing",
    "search_blog_posts",
    
    # Pipelines
    "blog_post_pipeline",
    "async_blog_post_pipeline",
    "long_post_processing",
    "robust_blog_post_processing",
    
    # Router
    "router",
    
    # Utilities
    "create_blog_post_schema",
    "create_search_schema"
] 