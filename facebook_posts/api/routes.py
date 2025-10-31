"""
FastAPI routes for Facebook Posts API
Following functional programming principles and FastAPI best practices
"""

from typing import List, Dict, Any, Optional, Union
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, Query, Path
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import asyncio
import logging
import time
import uuid
import structlog

from .schemas import (
    PostRequest, PostResponse, BatchPostRequest, BatchPostResponse,
    SystemHealth, PerformanceMetrics, ErrorResponse, OptimizationRequest,
    OptimizationResponse, FacebookPost, PostUpdateRequest
)
from .dependencies import (
    get_facebook_engine, get_current_user, check_rate_limit,
    get_health_status, validate_post_id, get_request_id
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1", tags=["facebook-posts"])


@router.post(
    "/posts/generate",
    response_model=PostResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Generate Facebook Post",
    description="Generate an optimized Facebook post based on the provided request",
    responses={
        201: {"description": "Post generated successfully"},
        400: {"description": "Invalid request data", "model": ErrorResponse},
        422: {"description": "Validation error", "model": ErrorResponse},
        429: {"description": "Rate limit exceeded", "model": ErrorResponse},
        500: {"description": "Internal server error", "model": ErrorResponse}
    }
)
async def generate_post(
    request: PostRequest,
    background_tasks: BackgroundTasks,
    engine: Any = Depends(get_facebook_engine),
    user: Dict[str, Any] = Depends(check_rate_limit),
    request_id: str = Depends(get_request_id)
) -> PostResponse:
    """
    Generate a single Facebook post with AI optimization.
    
    - **topic**: The main topic for the post
    - **audience_type**: Target audience type
    - **content_type**: Type of content to generate
    - **tone**: Desired tone for the post
    - **optimization_level**: Level of AI optimization to apply
    """
    # Early validation with guard clauses
    if not request.topic or len(request.topic.strip()) < 3:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Topic must be at least 3 characters long"
        )
    
    if not is_valid_post_request(request):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid post request parameters"
        )
    
    start_time = time.time()
    
    try:
        # Generate post content
        response = await generate_post_content(request, engine, request_id)
        
        # Add background analytics task if successful
        if response.success and response.post:
            background_tasks.add_task(
                update_analytics_async,
                response.post.id,
                user.get("user_id", "anonymous"),
                request_id
            )
        
        # Log successful generation
        processing_time = time.time() - start_time
        logger.info(
            "Post generated successfully",
            post_id=response.post.id if response.post else None,
            processing_time=processing_time,
            request_id=request_id,
            user_id=user.get("user_id")
        )
        
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(
            "Error generating post",
            error=str(e),
            request_id=request_id,
            processing_time=processing_time,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate post. Please try again later."
        )


@router.post(
    "/posts/generate/batch",
    response_model=BatchPostResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Generate Multiple Posts",
    description="Generate multiple Facebook posts in batch"
)
async def generate_batch_posts(
    request: BatchPostRequest,
    background_tasks: BackgroundTasks,
    engine: Any = Depends(get_facebook_engine),
    user: Dict[str, Any] = Depends(check_rate_limit)
) -> BatchPostResponse:
    """Generate multiple Facebook posts in batch"""
    if not is_valid_batch_request(request):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid batch request"
        )
    
    start_time = time.time()
    batch_id = str(uuid.uuid4())
    
    try:
        results = await generate_posts_batch(request, engine)
        
        # Add background tasks for analytics
        for result in results:
            if result.success and result.post:
                background_tasks.add_task(
                    update_analytics_async,
                    result.post.id,
                    user["user_id"]
                )
        
        return create_batch_response(results, time.time() - start_time, batch_id)
        
    except Exception as e:
        logger.error("Error generating batch posts", error=str(e), batch_id=batch_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate batch posts: {str(e)}"
        )


@router.get(
    "/posts/{post_id}",
    response_model=FacebookPost,
    summary="Get Post",
    description="Retrieve a specific Facebook post by ID",
    responses={
        200: {"description": "Post retrieved successfully"},
        400: {"description": "Invalid post ID", "model": ErrorResponse},
        404: {"description": "Post not found", "model": ErrorResponse},
        429: {"description": "Rate limit exceeded", "model": ErrorResponse},
        500: {"description": "Internal server error", "model": ErrorResponse}
    }
)
async def get_post(
    post_id: str = Path(..., description="The unique identifier of the post", min_length=1),
    engine: Any = Depends(get_facebook_engine),
    user: Dict[str, Any] = Depends(check_rate_limit),
    request_id: str = Depends(get_request_id)
) -> FacebookPost:
    """
    Retrieve a specific Facebook post by its unique identifier.
    
    - **post_id**: The unique identifier of the post to retrieve
    """
    # Early validation with guard clauses
    if not post_id or len(post_id.strip()) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Post ID cannot be empty"
        )
    
    if not is_valid_post_id(post_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid post ID format"
        )
    
    try:
        post = await get_post_by_id(post_id, engine, request_id)
        
        if not post:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Post with ID '{post_id}' not found"
            )
        
        # Log successful retrieval
        logger.info(
            "Post retrieved successfully",
            post_id=post_id,
            request_id=request_id,
            user_id=user.get("user_id")
        )
        
        return post
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(
            "Error retrieving post",
            error=str(e),
            post_id=post_id,
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve post. Please try again later."
        )


@router.get(
    "/posts",
    response_model=List[FacebookPost],
    summary="List Posts",
    description="List Facebook posts with optional filtering and pagination",
    responses={
        200: {"description": "Posts retrieved successfully"},
        400: {"description": "Invalid query parameters", "model": ErrorResponse},
        429: {"description": "Rate limit exceeded", "model": ErrorResponse},
        500: {"description": "Internal server error", "model": ErrorResponse}
    }
)
async def list_posts(
    skip: int = Query(0, ge=0, le=10000, description="Number of posts to skip for pagination"),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of posts to return"),
    status: Optional[str] = Query(None, description="Filter by post status (draft, pending, approved, published, rejected, archived)"),
    content_type: Optional[str] = Query(None, description="Filter by content type (educational, entertainment, promotional, news, personal, technical, inspirational)"),
    audience_type: Optional[str] = Query(None, description="Filter by audience type (general, professionals, entrepreneurs, students, technical, creative, business)"),
    quality_tier: Optional[str] = Query(None, description="Filter by quality tier (basic, good, excellent, exceptional)"),
    engine: Any = Depends(get_facebook_engine),
    user: Dict[str, Any] = Depends(check_rate_limit),
    request_id: str = Depends(get_request_id)
) -> List[FacebookPost]:
    """
    List Facebook posts with optional filtering and pagination.
    
    - **skip**: Number of posts to skip (for pagination)
    - **limit**: Maximum number of posts to return (1-100)
    - **status**: Filter by post status
    - **content_type**: Filter by content type
    - **audience_type**: Filter by audience type
    - **quality_tier**: Filter by quality tier
    """
    # Early validation with guard clauses
    if skip < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Skip parameter must be non-negative"
        )
    
    if limit <= 0 or limit > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Limit must be between 1 and 100"
        )
    
    try:
        # Build filters with validation
        filters = build_post_filters(status, content_type, audience_type, quality_tier)
        
        # Get posts list
        posts = await get_posts_list(skip, limit, filters, engine, request_id)
        
        # Log successful retrieval
        logger.info(
            "Posts listed successfully",
            count=len(posts),
            skip=skip,
            limit=limit,
            filters=filters,
            request_id=request_id,
            user_id=user.get("user_id")
        )
        
        return posts
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(
            "Error listing posts",
            error=str(e),
            skip=skip,
            limit=limit,
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list posts. Please try again later."
        )


@router.put(
    "/posts/{post_id}",
    response_model=FacebookPost,
    summary="Update Post",
    description="Update an existing Facebook post",
    responses={
        200: {"description": "Post updated successfully"},
        400: {"description": "Invalid post ID or request data", "model": ErrorResponse},
        404: {"description": "Post not found", "model": ErrorResponse},
        422: {"description": "Validation error", "model": ErrorResponse},
        429: {"description": "Rate limit exceeded", "model": ErrorResponse},
        500: {"description": "Internal server error", "model": ErrorResponse}
    }
)
async def update_post(
    post_id: str = Path(..., description="The unique identifier of the post to update", min_length=1),
    post_data: PostUpdateRequest = ...,
    engine: Any = Depends(get_facebook_engine),
    user: Dict[str, Any] = Depends(check_rate_limit),
    request_id: str = Depends(get_request_id)
) -> FacebookPost:
    """
    Update an existing Facebook post.
    
    - **post_id**: The unique identifier of the post to update
    - **post_data**: The updated post data
    """
    # Early validation with guard clauses
    if not post_id or len(post_id.strip()) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Post ID cannot be empty"
        )
    
    if not is_valid_post_id(post_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid post ID format"
        )
    
    if not post_data or not hasattr(post_data, 'dict'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid post data provided"
        )
    
    try:
        # Convert Pydantic model to dict for processing
        update_data = post_data.dict(exclude_unset=True)
        
        if not update_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid fields provided for update"
            )
        
        # Update the post
        updated_post = await update_post_by_id(post_id, update_data, engine, request_id)
        
        if not updated_post:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Post with ID '{post_id}' not found"
            )
        
        # Log successful update
        logger.info(
            "Post updated successfully",
            post_id=post_id,
            updated_fields=list(update_data.keys()),
            request_id=request_id,
            user_id=user.get("user_id")
        )
        
        return updated_post
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(
            "Error updating post",
            error=str(e),
            post_id=post_id,
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update post. Please try again later."
        )


@router.delete(
    "/posts/{post_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete Post",
    description="Delete a Facebook post"
)
async def delete_post(
    post_id: str,
    engine: Any = Depends(get_facebook_engine),
    user: Dict[str, Any] = Depends(check_rate_limit)
) -> None:
    """Delete a post"""
    if not is_valid_post_id(post_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid post ID"
        )
    
    try:
        success = await delete_post_by_id(post_id, engine)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Post with ID {post_id} not found"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error deleting post", error=str(e), post_id=post_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete post: {str(e)}"
        )


@router.post(
    "/posts/{post_id}/optimize",
    response_model=OptimizationResponse,
    summary="Optimize Post",
    description="Optimize an existing Facebook post"
)
async def optimize_post(
    post_id: str,
    optimization_request: OptimizationRequest,
    engine: Any = Depends(get_facebook_engine),
    user: Dict[str, Any] = Depends(check_rate_limit)
) -> OptimizationResponse:
    """Optimize an existing post"""
    if not is_valid_post_id(post_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid post ID"
        )
    
    try:
        result = await optimize_post_by_id(post_id, optimization_request, engine)
        return result
        
    except Exception as e:
        logger.error("Error optimizing post", error=str(e), post_id=post_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to optimize post: {str(e)}"
        )


@router.get(
    "/health",
    response_model=SystemHealth,
    summary="Health Check",
    description="Get system health status"
)
async def health_check() -> SystemHealth:
    """Get system health status"""
    try:
        health_data = await get_system_health_status()
        return SystemHealth(**health_data)
        
    except Exception as e:
        logger.error("Error getting health status", error=str(e))
        return create_error_health_response(str(e))


@router.get(
    "/metrics",
    response_model=PerformanceMetrics,
    summary="Performance Metrics",
    description="Get system performance metrics"
)
async def get_metrics(
    engine: Any = Depends(get_facebook_engine),
    user: Dict[str, Any] = Depends(check_rate_limit)
) -> PerformanceMetrics:
    """Get performance metrics"""
    try:
        metrics = await get_system_performance_metrics(engine)
        return PerformanceMetrics(**metrics)
        
    except Exception as e:
        logger.error("Error getting metrics", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get metrics: {str(e)}"
        )


@router.get(
    "/analytics/{post_id}",
    response_model=Dict[str, Any],
    summary="Get Post Analytics",
    description="Get analytics data for a specific post"
)
async def get_post_analytics(
    post_id: str,
    engine: Any = Depends(get_facebook_engine),
    user: Dict[str, Any] = Depends(check_rate_limit)
) -> Dict[str, Any]:
    """Get analytics for a specific post"""
    if not is_valid_post_id(post_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid post ID"
        )
    
    try:
        analytics = await get_analytics_for_post(post_id, engine)
        if not analytics:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Analytics for post {post_id} not found"
            )
        return analytics
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting analytics", error=str(e), post_id=post_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get analytics: {str(e)}"
        )


# Pure functions following functional programming principles

def is_valid_post_request(request: PostRequest) -> bool:
    """Validate post request"""
    return bool(request.topic.strip() and request.max_length >= 50)


def is_valid_batch_request(request: BatchPostRequest) -> bool:
    """Validate batch request"""
    return bool(request.requests and len(request.requests) <= 50)


def is_valid_post_id(post_id: str) -> bool:
    """Validate post ID"""
    return bool(post_id.strip() and len(post_id) > 0)


def build_post_filters(
    status: Optional[str], 
    content_type: Optional[str], 
    audience_type: Optional[str] = None,
    quality_tier: Optional[str] = None
) -> Dict[str, Any]:
    """Build post filters from query parameters with validation"""
    filters = {}
    
    # Validate and add status filter
    if status:
        valid_statuses = ["draft", "pending", "approved", "published", "rejected", "archived"]
        if status.lower() not in valid_statuses:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status. Must be one of: {', '.join(valid_statuses)}"
            )
        filters["status"] = status.lower()
    
    # Validate and add content_type filter
    if content_type:
        valid_content_types = ["educational", "entertainment", "promotional", "news", "personal", "technical", "inspirational"]
        if content_type.lower() not in valid_content_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid content_type. Must be one of: {', '.join(valid_content_types)}"
            )
        filters["content_type"] = content_type.lower()
    
    # Validate and add audience_type filter
    if audience_type:
        valid_audience_types = ["general", "professionals", "entrepreneurs", "students", "technical", "creative", "business"]
        if audience_type.lower() not in valid_audience_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid audience_type. Must be one of: {', '.join(valid_audience_types)}"
            )
        filters["audience_type"] = audience_type.lower()
    
    # Validate and add quality_tier filter
    if quality_tier:
        valid_quality_tiers = ["basic", "good", "excellent", "exceptional"]
        if quality_tier.lower() not in valid_quality_tiers:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid quality_tier. Must be one of: {', '.join(valid_quality_tiers)}"
            )
        filters["quality_tier"] = quality_tier.lower()
    
    return filters


def create_batch_response(results: List[PostResponse], total_time: float, batch_id: str) -> BatchPostResponse:
    """Create batch response from results"""
    successful_posts = sum(1 for r in results if r.success)
    failed_posts = len(results) - successful_posts
    
    return BatchPostResponse(
        success=successful_posts > 0,
        results=results,
        total_processing_time=total_time,
        successful_posts=successful_posts,
        failed_posts=failed_posts,
        batch_id=batch_id
    )


def create_error_health_response(error: str) -> SystemHealth:
    """Create error health response"""
    return SystemHealth(
        status="unhealthy",
        uptime=0.0,
        version="4.0.0",
        components={"error": {"status": "unhealthy", "error": error}},
        performance_metrics={}
    )


# Async business logic functions

async def generate_post_content(request: PostRequest, engine: Any, request_id: str) -> PostResponse:
    """Generate post content using the engine with enhanced error handling"""
    start_time = time.time()
    
    try:
        # Validate request parameters
        if not request.topic or len(request.topic.strip()) < 3:
            raise ValueError("Topic must be at least 3 characters long")
        
        # Mock implementation - replace with actual engine call
        content = f"🚀 Exciting content about: {request.topic}\n\n"
        content += f"This post is tailored for {request.audience_type.value} audience "
        content += f"with {request.content_type.value} content type.\n\n"
        content += f"Tone: {request.tone}\n"
        
        if request.include_hashtags:
            content += f"\n#Innovation #Tech #Future"
        
        # Create post using factory method
        from core.models import FacebookPostFactory, PostStatus
        post = FacebookPostFactory.create_draft(
            content=content,
            content_type=request.content_type,
            audience_type=request.audience_type,
            optimization_level=request.optimization_level,
            tags=request.tags,
            metadata={
                **request.metadata,
                "request_id": request_id,
                "generated_at": time.time()
            }
        )
        
        processing_time = time.time() - start_time
        
        # Log successful generation
        logger.info(
            "Post content generated",
            post_id=post.id,
            topic=request.topic,
            processing_time=processing_time,
            request_id=request_id
        )
        
        return PostResponse(
            success=True,
            post=post,
            processing_time=processing_time,
            optimizations_applied=["ai_generation", "content_optimization", "audience_targeting"]
        )
        
    except ValueError as e:
        processing_time = time.time() - start_time
        logger.warning(
            "Validation error in post generation",
            error=str(e),
            request_id=request_id,
            processing_time=processing_time
        )
        return PostResponse(
            success=False,
            error=str(e),
            processing_time=processing_time
        )
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(
            "Error generating post content",
            error=str(e),
            request_id=request_id,
            processing_time=processing_time,
            exc_info=True
        )
        return PostResponse(
            success=False,
            error="Failed to generate post content",
            processing_time=processing_time
        )


async def generate_posts_batch(request: BatchPostRequest, engine: Any) -> List[PostResponse]:
    """Generate posts in batch"""
    if request.parallel_processing:
        return await generate_posts_parallel(request.requests, engine)
    return await generate_posts_sequential(request.requests, engine)


async def generate_posts_parallel(requests: List[PostRequest], engine: Any) -> List[PostResponse]:
    """Generate posts in parallel"""
    tasks = [generate_post_content(req, engine) for req in requests]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle exceptions
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            processed_results.append(PostResponse(
                success=False,
                error=str(result),
                processing_time=0.0
            ))
        else:
            processed_results.append(result)
    
    return processed_results


async def generate_posts_sequential(requests: List[PostRequest], engine: Any) -> List[PostResponse]:
    """Generate posts sequentially"""
    results = []
    for request in requests:
        result = await generate_post_content(request, engine)
        results.append(result)
    return results


async def get_post_by_id(post_id: str, engine: Any, request_id: str) -> Optional[FacebookPost]:
    """Get post by ID with enhanced error handling"""
    try:
        # Mock implementation - replace with actual engine call
        logger.debug(f"Retrieving post {post_id}", request_id=request_id)
        
        # Simulate database lookup
        if post_id == "not-found":
            return None
        
        # Return mock post for demonstration
        from core.models import FacebookPostFactory, ContentType, AudienceType
        return FacebookPostFactory.create_sample_post()
        
    except Exception as e:
        logger.error(f"Error retrieving post {post_id}", error=str(e), request_id=request_id)
        raise


async def get_posts_list(
    skip: int, 
    limit: int, 
    filters: Dict[str, Any], 
    engine: Any, 
    request_id: str
) -> List[FacebookPost]:
    """Get posts list with filters and enhanced error handling"""
    try:
        logger.debug(
            f"Retrieving posts list",
            skip=skip,
            limit=limit,
            filters=filters,
            request_id=request_id
        )
        
        # Mock implementation - replace with actual engine call
        from core.models import FacebookPostFactory
        
        # Generate mock posts based on filters
        posts = []
        for i in range(min(limit, 5)):  # Return up to 5 mock posts
            post = FacebookPostFactory.create_sample_post()
            posts.append(post)
        
        return posts
        
    except Exception as e:
        logger.error(f"Error retrieving posts list", error=str(e), request_id=request_id)
        raise


async def update_post_by_id(
    post_id: str, 
    post_data: Dict[str, Any], 
    engine: Any, 
    request_id: str
) -> Optional[FacebookPost]:
    """Update post by ID with enhanced error handling"""
    try:
        logger.debug(
            f"Updating post {post_id}",
            update_data=post_data,
            request_id=request_id
        )
        
        # Mock implementation - replace with actual engine call
        if post_id == "not-found":
            return None
        
        # Get existing post and apply updates
        existing_post = await get_post_by_id(post_id, engine, request_id)
        if not existing_post:
            return None
        
        # Apply updates (mock implementation)
        for key, value in post_data.items():
            if hasattr(existing_post, key):
                setattr(existing_post, key, value)
        
        return existing_post
        
    except Exception as e:
        logger.error(f"Error updating post {post_id}", error=str(e), request_id=request_id)
        raise


async def delete_post_by_id(post_id: str, engine: Any, request_id: str) -> bool:
    """Delete post by ID with enhanced error handling"""
    try:
        logger.debug(f"Deleting post {post_id}", request_id=request_id)
        
        # Mock implementation - replace with actual engine call
        if post_id == "not-found":
            return False
        
        # Simulate successful deletion
        return True
        
    except Exception as e:
        logger.error(f"Error deleting post {post_id}", error=str(e), request_id=request_id)
        raise


async def optimize_post_by_id(post_id: str, optimization_request: OptimizationRequest, engine: Any) -> OptimizationResponse:
    """Optimize post by ID"""
    # Mock implementation
    return OptimizationResponse(
        success=True,
        optimized_post=None,
        improvements=["mock_optimization"],
        processing_time=0.1
    )


async def get_system_health_status() -> Dict[str, Any]:
    """Get system health status"""
    return {
        "status": "healthy",
        "uptime": time.time(),
        "version": "4.0.0",
        "components": {
            "database": {"status": "healthy"},
            "cache": {"status": "healthy"},
            "ai_service": {"status": "healthy"}
        },
        "performance_metrics": {}
    }


async def get_system_performance_metrics(engine: Any) -> Dict[str, Any]:
    """Get system performance metrics"""
    return {
        "total_requests": 1000,
        "successful_requests": 950,
        "failed_requests": 50,
        "average_processing_time": 0.5,
        "cache_hit_rate": 0.75,
        "memory_usage": 512.0,
        "cpu_usage": 45.0,
        "active_connections": 10
    }


async def get_analytics_for_post(post_id: str, engine: Any) -> Optional[Dict[str, Any]]:
    """Get analytics for a specific post"""
    # Mock implementation
    return None


async def update_analytics_async(post_id: str, user_id: str, request_id: str) -> None:
    """Update analytics for a post (background task) with enhanced error handling"""
    try:
        logger.info(
            "Updating analytics",
            post_id=post_id,
            user_id=user_id,
            request_id=request_id
        )
        
        # Mock implementation - replace with actual analytics service
        # This would typically:
        # 1. Track post generation metrics
        # 2. Update user activity logs
        # 3. Calculate engagement predictions
        # 4. Store analytics data
        
        # Simulate processing time
        await asyncio.sleep(0.1)
        
        logger.info(
            "Analytics updated successfully",
            post_id=post_id,
            user_id=user_id,
            request_id=request_id
        )
        
    except Exception as e:
        logger.error(
            "Error updating analytics",
            error=str(e),
            post_id=post_id,
            user_id=user_id,
            request_id=request_id,
            exc_info=True
        )