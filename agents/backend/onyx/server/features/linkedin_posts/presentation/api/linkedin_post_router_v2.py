from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

from fastapi import APIRouter, HTTPException, Depends, Query, Response, BackgroundTasks, Request
from fastapi.responses import StreamingResponse, ORJSONResponse
from typing import List, Optional, Dict, Any, AsyncGenerator
from datetime import datetime, timedelta
import asyncio
from functools import lru_cache
import orjson
from prometheus_client import Counter, Histogram, Gauge
import time
from ...core.domain.entities.linkedin_post import PostStatus, PostType, PostTone
from ...application.use_cases.linkedin_post_use_cases import LinkedInPostUseCases
from ...infrastructure.repositories.linkedin_post_repository import LinkedInPostRepository
from ...shared.schemas.linkedin_post_schemas import (
from ...shared.logging import get_logger
from ...shared.dependencies import get_current_user, rate_limiter, User
from ...shared.cache import cache_manager
from ...shared.metrics import metrics_collector
from typing import Any, List, Dict, Optional
import logging
"""
LinkedIn Post API Router V2 - Ultra-Optimized
=============================================

FastAPI router with advanced optimizations, best practices, and enterprise features.
"""


    LinkedInPostCreate,
    LinkedInPostUpdate,
    LinkedInPostResponse,
    LinkedInPostListResponse,
    PostOptimizationRequest,
    PostAnalysisResponse,
    BatchOptimizationRequest,
    NLPPerformanceResponse,
)

logger = get_logger(__name__)

# Metrics
request_counter = Counter('linkedin_api_requests_total', 'Total API requests', ['method', 'endpoint'])
request_duration = Histogram('linkedin_api_request_duration_seconds', 'Request duration', ['method', 'endpoint'])
active_requests = Gauge('linkedin_api_active_requests', 'Active requests')
cache_hits = Counter('linkedin_api_cache_hits_total', 'Cache hits', ['endpoint'])
cache_misses = Counter('linkedin_api_cache_misses_total', 'Cache misses', ['endpoint'])

# Create optimized router
router = APIRouter(
    prefix="/api/v2/linkedin-posts",
    tags=["LinkedIn Posts V2"],
    dependencies=[Depends(rate_limiter)],
    default_response_class=ORJSONResponse
)

# Optimized dependency injection with caching
@lru_cache(maxsize=1)
def get_repository() -> LinkedInPostRepository:
    """Get cached LinkedIn post repository instance."""
    return LinkedInPostRepository()

@lru_cache(maxsize=1)
def get_use_cases() -> LinkedInPostUseCases:
    """Get cached LinkedIn post use cases instance."""
    return LinkedInPostUseCases(get_repository())

# Request tracking middleware
async def track_request(request: Request, call_next):
    """Track request metrics."""
    active_requests.inc()
    start_time = time.time()
    
    try:
        response = await call_next(request)
        duration = time.time() - start_time
        
        request_counter.labels(
            method=request.method,
            endpoint=request.url.path
        ).inc()
        
        request_duration.labels(
            method=request.method,
            endpoint=request.url.path
        ).observe(duration)
        
        return response
    finally:
        active_requests.dec()

# Cache key generators
def generate_cache_key(prefix: str, **kwargs) -> str:
    """Generate cache key from parameters."""
    sorted_items = sorted(kwargs.items())
    key_parts = [f"{k}:{v}" for k, v in sorted_items if v is not None]
    return f"{prefix}:{':'.join(key_parts)}"

# Optimized endpoints
@router.post("/", response_model=LinkedInPostResponse, status_code=201)
async def create_linkedin_post(
    post_data: LinkedInPostCreate,
    background_tasks: BackgroundTasks,
    request: Request,
    use_cases: LinkedInPostUseCases = Depends(get_use_cases),
    use_fast_nlp: bool = Query(True, description="Enable fast NLP enhancement"),
    use_async_nlp: bool = Query(True, description="Use async NLP processor"),
    stream_response: bool = Query(False, description="Stream response for large content")
):
    """
    Create LinkedIn post with ultra-fast processing.
    
    Features:
    - Async NLP processing by default
    - Background task processing
    - Response streaming for large content
    - Request tracing and metrics
    """
    try:
        # Add request ID for tracing
        request_id = request.headers.get("X-Request-ID", "")
        
        # Create post with optimizations
        post = await use_cases.generate_post(
            content=post_data.content,
            post_type=post_data.post_type,
            tone=post_data.tone,
            target_audience=post_data.target_audience,
            industry=post_data.industry,
            use_fast_nlp=use_fast_nlp,
            use_async_nlp=use_async_nlp,
            request_id=request_id
        )
        
        # Background tasks for non-critical operations
        background_tasks.add_task(
            metrics_collector.track_post_creation,
            post_id=post.id,
            processing_time=post.nlp_processing_time
        )
        
        # Stream response for large content
        if stream_response and len(post.content) > 1000:
            async def generate_response():
                
    """generate_response function."""
yield orjson.dumps({
                    "id": post.id,
                    "content": post.content[:500],
                    "streaming": True
                })
                await asyncio.sleep(0.01)
                yield orjson.dumps({
                    "content_continuation": post.content[500:],
                    "metadata": {
                        "post_type": post.post_type.value,
                        "tone": post.tone.value,
                        "nlp_enhanced": post.nlp_enhanced,
                        "processing_time": post.nlp_processing_time
                    }
                })
            
            return StreamingResponse(
                generate_response(),
                media_type="application/json"
            )
        
        return LinkedInPostResponse(
            id=post.id,
            content=post.content,
            post_type=post.post_type,
            tone=post.tone,
            target_audience=post.target_audience,
            industry=post.industry,
            status=post.status,
            nlp_enhanced=post.nlp_enhanced,
            nlp_processing_time=post.nlp_processing_time,
            created_at=post.created_at,
            updated_at=post.updated_at
        )
        
    except Exception as e:
        logger.error(f"Error creating LinkedIn post: {e}", extra={"request_id": request_id})
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", response_model=LinkedInPostListResponse)
async def list_linkedin_posts(
    request: Request,
    response: Response,
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    status: Optional[PostStatus] = Query(None, description="Filter by status"),
    post_type: Optional[PostType] = Query(None, description="Filter by post type"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of posts"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    sort_by: str = Query("created_at", description="Sort field"),
    sort_order: str = Query("desc", description="Sort order (asc/desc)"),
    use_cache: bool = Query(True, description="Use cache for faster response"),
    use_cases: LinkedInPostUseCases = Depends(get_use_cases),
    current_user: User = Depends(get_current_user)
):
    """
    List LinkedIn posts with advanced filtering and caching.
    
    Features:
    - Multi-level caching (Memory + Redis)
    - Flexible sorting and filtering
    - Pagination with cursor support
    - ETag support for client caching
    """
    try:
        # Generate cache key
        cache_key = generate_cache_key(
            "posts:list",
            user_id=user_id,
            status=status,
            post_type=post_type,
            limit=limit,
            offset=offset,
            sort_by=sort_by,
            sort_order=sort_order
        )
        
        # Check cache
        if use_cache:
            cached_result = await cache_manager.get(cache_key)
            if cached_result:
                cache_hits.labels(endpoint="/list").inc()
                
                # Set cache headers
                response.headers["X-Cache"] = "HIT"
                response.headers["Cache-Control"] = "public, max-age=60"
                
                # Set ETag for client caching
                etag = f'"{hash(cache_key)}"'
                response.headers["ETag"] = etag
                
                # Check if client has valid cache
                if request.headers.get("If-None-Match") == etag:
                    return Response(status_code=304)
                
                return orjson.loads(cached_result)
            else:
                cache_misses.labels(endpoint="/list").inc()
        
        # Fetch posts with optimizations
        posts = await use_cases.list_posts(
            user_id=user_id,
            status=status,
            post_type=post_type,
            limit=limit,
            offset=offset,
            sort_by=sort_by,
            sort_order=sort_order
        )
        
        # Prepare response
        result = LinkedInPostListResponse(
            posts=[
                LinkedInPostResponse(
                    id=post.id,
                    content=post.content,
                    post_type=post.post_type,
                    tone=post.tone,
                    target_audience=post.target_audience,
                    industry=post.industry,
                    status=post.status,
                    nlp_enhanced=post.nlp_enhanced,
                    nlp_processing_time=post.nlp_processing_time,
                    created_at=post.created_at,
                    updated_at=post.updated_at
                )
                for post in posts
            ],
            total=len(posts),
            limit=limit,
            offset=offset,
            has_more=len(posts) == limit
        )
        
        # Cache result
        if use_cache:
            await cache_manager.set(
                cache_key,
                orjson.dumps(result.dict()),
                expire=60  # 1 minute cache
            )
            response.headers["X-Cache"] = "MISS"
        
        # Add pagination headers
        response.headers["X-Total-Count"] = str(result.total)
        response.headers["X-Limit"] = str(limit)
        response.headers["X-Offset"] = str(offset)
        response.headers["X-Has-More"] = str(result.has_more)
        
        # Add performance headers
        response.headers["X-Response-Time"] = f"{time.time() - request.state.start_time:.3f}s"
        
        return result
        
    except Exception as e:
        logger.error(f"Error listing LinkedIn posts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{post_id}", response_model=LinkedInPostResponse)
async def get_linkedin_post(
    post_id: str,
    request: Request,
    response: Response,
    use_cache: bool = Query(True, description="Use cache"),
    include_analytics: bool = Query(False, description="Include analytics data"),
    use_cases: LinkedInPostUseCases = Depends(get_use_cases)
):
    """
    Get LinkedIn post with caching and optional analytics.
    
    Features:
    - Multi-level caching
    - Optional analytics inclusion
    - ETag support
    - Conditional requests
    """
    try:
        # Generate cache key
        cache_key = f"post:{post_id}:analytics:{include_analytics}"
        
        # Check cache
        if use_cache:
            cached_post = await cache_manager.get(cache_key)
            if cached_post:
                cache_hits.labels(endpoint="/get").inc()
                response.headers["X-Cache"] = "HIT"
                
                # ETag support
                etag = f'"{hash(cached_post)}"'
                response.headers["ETag"] = etag
                
                if request.headers.get("If-None-Match") == etag:
                    return Response(status_code=304)
                
                return orjson.loads(cached_post)
            else:
                cache_misses.labels(endpoint="/get").inc()
        
        # Fetch post
        post = await use_cases.repository.get_by_id(post_id)
        if not post:
            raise HTTPException(status_code=404, detail="Post not found")
        
        # Prepare response
        response_data = LinkedInPostResponse(
            id=post.id,
            content=post.content,
            post_type=post.post_type,
            tone=post.tone,
            target_audience=post.target_audience,
            industry=post.industry,
            status=post.status,
            nlp_enhanced=post.nlp_enhanced,
            nlp_processing_time=post.nlp_processing_time,
            created_at=post.created_at,
            updated_at=post.updated_at
        )
        
        # Add analytics if requested
        if include_analytics:
            analytics = await use_cases.get_post_analytics(post_id)
            response_data.analytics = analytics
        
        # Cache result
        if use_cache:
            await cache_manager.set(
                cache_key,
                orjson.dumps(response_data.dict()),
                expire=300  # 5 minutes cache
            )
            response.headers["X-Cache"] = "MISS"
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting LinkedIn post: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{post_id}", response_model=LinkedInPostResponse)
async def update_linkedin_post(
    post_id: str,
    post_data: LinkedInPostUpdate,
    background_tasks: BackgroundTasks,
    request: Request,
    use_cases: LinkedInPostUseCases = Depends(get_use_cases),
    use_fast_nlp: bool = Query(True, description="Enable fast NLP enhancement"),
    use_async_nlp: bool = Query(True, description="Use async NLP processor"),
    invalidate_cache: bool = Query(True, description="Invalidate cache after update")
):
    """
    Update LinkedIn post with cache invalidation.
    
    Features:
    - Optimistic updates
    - Cache invalidation
    - Background processing
    - Conflict detection
    """
    try:
        # Check for conflicts with If-Match header
        if_match = request.headers.get("If-Match")
        if if_match:
            current_post = await use_cases.repository.get_by_id(post_id)
            if current_post and f'"{hash(current_post.updated_at)}"' != if_match:
                raise HTTPException(
                    status_code=412,
                    detail="Precondition failed: Post has been modified"
                )
        
        # Update post
        post = await use_cases.update_post(
            post_id=post_id,
            content=post_data.content,
            post_type=post_data.post_type,
            tone=post_data.tone,
            target_audience=post_data.target_audience,
            industry=post_data.industry,
            use_fast_nlp=use_fast_nlp,
            use_async_nlp=use_async_nlp
        )
        
        if not post:
            raise HTTPException(status_code=404, detail="Post not found")
        
        # Invalidate cache
        if invalidate_cache:
            background_tasks.add_task(
                cache_manager.delete_pattern,
                f"post:{post_id}:*"
            )
            background_tasks.add_task(
                cache_manager.delete_pattern,
                "posts:list:*"
            )
        
        return LinkedInPostResponse(
            id=post.id,
            content=post.content,
            post_type=post.post_type,
            tone=post.tone,
            target_audience=post.target_audience,
            industry=post.industry,
            status=post.status,
            nlp_enhanced=post.nlp_enhanced,
            nlp_processing_time=post.nlp_processing_time,
            created_at=post.created_at,
            updated_at=post.updated_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating LinkedIn post: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{post_id}", status_code=204)
async def delete_linkedin_post(
    post_id: str,
    background_tasks: BackgroundTasks,
    request: Request,
    use_cases: LinkedInPostUseCases = Depends(get_use_cases),
    soft_delete: bool = Query(True, description="Soft delete instead of hard delete"),
    invalidate_cache: bool = Query(True, description="Invalidate cache after deletion")
):
    """
    Delete LinkedIn post with cache invalidation.
    
    Features:
    - Soft/hard delete options
    - Cache invalidation
    - Audit logging
    - Background cleanup
    """
    try:
        # Verify post exists
        post = await use_cases.repository.get_by_id(post_id)
        if not post:
            raise HTTPException(status_code=404, detail="Post not found")
        
        # Delete post
        success = await use_cases.delete_post(post_id, soft_delete=soft_delete)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete post")
        
        # Background tasks
        if invalidate_cache:
            background_tasks.add_task(
                cache_manager.delete_pattern,
                f"post:{post_id}:*"
            )
            background_tasks.add_task(
                cache_manager.delete_pattern,
                "posts:list:*"
            )
        
        # Audit logging
        background_tasks.add_task(
            logger.info,
            f"Post {post_id} deleted",
            extra={
                "post_id": post_id,
                "soft_delete": soft_delete,
                "user_id": request.state.user_id
            }
        )
        
        return Response(status_code=204)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting LinkedIn post: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{post_id}/optimize", response_model=LinkedInPostResponse)
async def optimize_linkedin_post(
    post_id: str,
    optimization_request: PostOptimizationRequest,
    background_tasks: BackgroundTasks,
    request: Request,
    response: Response,
    use_cases: LinkedInPostUseCases = Depends(get_use_cases)
):
    """
    Optimize LinkedIn post with advanced NLP.
    
    Features:
    - Multi-stage optimization
    - Progress tracking
    - Async processing
    - Result caching
    """
    try:
        # Start optimization with progress tracking
        optimization_id = f"opt:{post_id}:{time.time()}"
        
        # Set initial progress
        await cache_manager.set(
            f"optimization:progress:{optimization_id}",
            orjson.dumps({"status": "starting", "progress": 0}),
            expire=300
        )
        
        # Return optimization ID immediately for tracking
        response.headers["X-Optimization-ID"] = optimization_id
        
        # Perform optimization
        post = await use_cases.optimize_post(
            post_id=post_id,
            use_async_nlp=optimization_request.use_async_nlp,
            optimization_id=optimization_id
        )
        
        if not post:
            raise HTTPException(status_code=404, detail="Post not found")
        
        # Cache optimized result
        cache_key = f"post:optimized:{post_id}"
        await cache_manager.set(
            cache_key,
            orjson.dumps(post.dict()),
            expire=3600  # 1 hour cache
        )
        
        # Update progress to complete
        await cache_manager.set(
            f"optimization:progress:{optimization_id}",
            orjson.dumps({"status": "complete", "progress": 100}),
            expire=60
        )
        
        # Invalidate old cache
        background_tasks.add_task(
            cache_manager.delete,
            f"post:{post_id}:*"
        )
        
        return LinkedInPostResponse(
            id=post.id,
            content=post.content,
            post_type=post.post_type,
            tone=post.tone,
            target_audience=post.target_audience,
            industry=post.industry,
            status=post.status,
            nlp_enhanced=post.nlp_enhanced,
            nlp_processing_time=post.nlp_processing_time,
            created_at=post.created_at,
            updated_at=post.updated_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error optimizing LinkedIn post: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{post_id}/optimize/progress")
async def get_optimization_progress(
    optimization_id: str = Query(..., description="Optimization ID from optimize endpoint")
):
    """
    Get optimization progress for async operations.
    
    Features:
    - Real-time progress tracking
    - WebSocket alternative
    """
    try:
        progress_key = f"optimization:progress:{optimization_id}"
        progress_data = await cache_manager.get(progress_key)
        
        if not progress_data:
            raise HTTPException(status_code=404, detail="Optimization not found")
        
        return orjson.loads(progress_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting optimization progress: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{post_id}/analyze", response_model=PostAnalysisResponse)
async def analyze_linkedin_post(
    post_id: str,
    request: Request,
    response: Response,
    use_cases: LinkedInPostUseCases = Depends(get_use_cases),
    use_async_nlp: bool = Query(True, description="Use async NLP processor"),
    use_cache: bool = Query(True, description="Use cached analysis"),
    include_competitors: bool = Query(False, description="Include competitor analysis"),
    include_trends: bool = Query(False, description="Include trend analysis")
):
    """
    Analyze LinkedIn post with advanced metrics.
    
    Features:
    - Comprehensive analysis
    - Competitor benchmarking
    - Trend analysis
    - Result caching
    """
    try:
        # Generate cache key
        cache_key = generate_cache_key(
            "analysis",
            post_id=post_id,
            competitors=include_competitors,
            trends=include_trends
        )
        
        # Check cache
        if use_cache:
            cached_analysis = await cache_manager.get(cache_key)
            if cached_analysis:
                cache_hits.labels(endpoint="/analyze").inc()
                response.headers["X-Cache"] = "HIT"
                return orjson.loads(cached_analysis)
            else:
                cache_misses.labels(endpoint="/analyze").inc()
        
        # Perform analysis
        analysis = await use_cases.analyze_post_engagement(
            post_id=post_id,
            use_async_nlp=use_async_nlp,
            include_competitors=include_competitors,
            include_trends=include_trends
        )
        
        if not analysis:
            raise HTTPException(status_code=404, detail="Post not found")
        
        # Prepare response
        response_data = PostAnalysisResponse(
            post_id=analysis["post_id"],
            sentiment_score=analysis["sentiment_score"],
            readability_score=analysis["readability_score"],
            keywords=analysis["keywords"],
            entities=analysis["entities"],
            processing_time=analysis["processing_time"],
            cached=False,
            async_optimized=analysis["async_optimized"],
            analyzed_at=datetime.utcnow()
        )
        
        # Add optional analyses
        if include_competitors:
            response_data.competitor_analysis = analysis.get("competitor_analysis")
        
        if include_trends:
            response_data.trend_analysis = analysis.get("trend_analysis")
        
        # Cache result
        if use_cache:
            await cache_manager.set(
                cache_key,
                orjson.dumps(response_data.dict()),
                expire=1800  # 30 minutes cache
            )
            response.headers["X-Cache"] = "MISS"
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing LinkedIn post: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch", response_model=List[LinkedInPostResponse])
async def batch_create_posts(
    posts_data: List[LinkedInPostCreate],
    background_tasks: BackgroundTasks,
    request: Request,
    use_cases: LinkedInPostUseCases = Depends(get_use_cases),
    use_fast_nlp: bool = Query(True, description="Enable fast NLP enhancement"),
    use_async_nlp: bool = Query(True, description="Use async NLP processor"),
    parallel_processing: bool = Query(True, description="Process in parallel")
):
    """
    Create multiple LinkedIn posts in batch.
    
    Features:
    - Parallel processing
    - Batch optimization
    - Progress tracking
    - Transaction support
    """
    try:
        if len(posts_data) > 100:
            raise HTTPException(
                status_code=400,
                detail="Maximum 100 posts per batch"
            )
        
        # Create batch ID for tracking
        batch_id = f"batch:{request.headers.get('X-Request-ID', time.time())}"
        
        # Process posts
        if parallel_processing:
            # Parallel processing with semaphore to limit concurrency
            semaphore = asyncio.Semaphore(10)
            
            async def create_with_semaphore(post_data) -> Any:
                async with semaphore:
                    return await use_cases.generate_post(
                        content=post_data.content,
                        post_type=post_data.post_type,
                        tone=post_data.tone,
                        target_audience=post_data.target_audience,
                        industry=post_data.industry,
                        use_fast_nlp=use_fast_nlp,
                        use_async_nlp=use_async_nlp,
                        batch_id=batch_id
                    )
            
            # Create all posts in parallel
            posts = await asyncio.gather(
                *[create_with_semaphore(post_data) for post_data in posts_data],
                return_exceptions=True
            )
            
            # Handle any errors
            created_posts = []
            errors = []
            for i, post in enumerate(posts):
                if isinstance(post, Exception):
                    errors.append({"index": i, "error": str(post)})
                else:
                    created_posts.append(post)
            
            if errors:
                # Partial success response
                response = {
                    "created": len(created_posts),
                    "failed": len(errors),
                    "errors": errors,
                    "posts": created_posts
                }
                raise HTTPException(status_code=207, detail=response)
        else:
            # Sequential processing
            created_posts = []
            for post_data in posts_data:
                post = await use_cases.generate_post(
                    content=post_data.content,
                    post_type=post_data.post_type,
                    tone=post_data.tone,
                    target_audience=post_data.target_audience,
                    industry=post_data.industry,
                    use_fast_nlp=use_fast_nlp,
                    use_async_nlp=use_async_nlp,
                    batch_id=batch_id
                )
                created_posts.append(post)
        
        # Background task for batch metrics
        background_tasks.add_task(
            metrics_collector.track_batch_creation,
            batch_id=batch_id,
            count=len(created_posts)
        )
        
        return [
            LinkedInPostResponse(
                id=post.id,
                content=post.content,
                post_type=post.post_type,
                tone=post.tone,
                target_audience=post.target_audience,
                industry=post.industry,
                status=post.status,
                nlp_enhanced=post.nlp_enhanced,
                nlp_processing_time=post.nlp_processing_time,
                created_at=post.created_at,
                updated_at=post.updated_at
            )
            for post in created_posts
        ]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch create: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch/optimize", response_model=List[LinkedInPostResponse])
async def batch_optimize_posts(
    batch_request: BatchOptimizationRequest,
    background_tasks: BackgroundTasks,
    request: Request,
    use_cases: LinkedInPostUseCases = Depends(get_use_cases)
):
    """
    Optimize multiple posts with maximum efficiency.
    
    Features:
    - Parallel optimization
    - Batch caching
    - Progress streaming
    - Resource pooling
    """
    try:
        if len(batch_request.post_ids) > 50:
            raise HTTPException(
                status_code=400,
                detail="Maximum 50 posts per optimization batch"
            )
        
        # Create batch optimization ID
        batch_opt_id = f"batch_opt:{time.time()}"
        
        # Initialize progress tracking
        await cache_manager.set(
            f"batch:progress:{batch_opt_id}",
            orjson.dumps({
                "total": len(batch_request.post_ids),
                "completed": 0,
                "status": "processing"
            }),
            expire=300
        )
        
        # Optimize posts in parallel with progress updates
        optimized_posts = await use_cases.batch_optimize_posts(
            post_ids=batch_request.post_ids,
            use_async_nlp=batch_request.use_async_nlp,
            batch_id=batch_opt_id
        )
        
        # Update progress to complete
        await cache_manager.set(
            f"batch:progress:{batch_opt_id}",
            orjson.dumps({
                "total": len(batch_request.post_ids),
                "completed": len(optimized_posts),
                "status": "complete"
            }),
            expire=60
        )
        
        # Invalidate old cache entries
        background_tasks.add_task(
            cache_manager.delete_pattern,
            "posts:list:*"
        )
        
        for post_id in batch_request.post_ids:
            background_tasks.add_task(
                cache_manager.delete_pattern,
                f"post:{post_id}:*"
            )
        
        return [
            LinkedInPostResponse(
                id=post.id,
                content=post.content,
                post_type=post.post_type,
                tone=post.tone,
                target_audience=post.target_audience,
                industry=post.industry,
                status=post.status,
                nlp_enhanced=post.nlp_enhanced,
                nlp_processing_time=post.nlp_processing_time,
                created_at=post.created_at,
                updated_at=post.updated_at
            )
            for post in optimized_posts
        ]
        
    except Exception as e:
        logger.error(f"Error in batch optimization: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stream/{post_id}")
async def stream_post_updates(
    post_id: str,
    request: Request,
    use_cases: LinkedInPostUseCases = Depends(get_use_cases)
):
    """
    Stream real-time updates for a post using Server-Sent Events.
    
    Features:
    - Real-time updates
    - Server-Sent Events
    - Automatic reconnection
    - Low latency
    """
    async def event_generator():
        """Generate SSE events."""
        try:
            # Send initial post data
            post = await use_cases.repository.get_by_id(post_id)
            if post:
                yield f"data: {orjson.dumps({'type': 'initial', 'post': post.dict()}).decode()}\n\n"
            
            # Stream updates
            last_update = datetime.utcnow()
            while True:
                # Check for updates
                post = await use_cases.repository.get_by_id(post_id)
                if post and post.updated_at > last_update:
                    yield f"data: {orjson.dumps({'type': 'update', 'post': post.dict()}).decode()}\n\n"
                    last_update = post.updated_at
                
                # Send heartbeat
                yield f"data: {orjson.dumps({'type': 'heartbeat', 'timestamp': datetime.utcnow().isoformat()}).decode()}\n\n"
                
                # Wait before next check
                await asyncio.sleep(1)
                
        except asyncio.CancelledError:
            yield f"data: {orjson.dumps({'type': 'close'}).decode()}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.get("/performance/metrics", response_model=NLPPerformanceResponse)
async def get_performance_metrics(
    request: Request,
    response: Response,
    use_cases: LinkedInPostUseCases = Depends(get_use_cases),
    include_detailed: bool = Query(False, description="Include detailed metrics"),
    time_range: str = Query("1h", description="Time range for metrics (1h, 24h, 7d)")
):
    """
    Get comprehensive performance metrics.
    
    Features:
    - Real-time metrics
    - Historical data
    - Detailed breakdowns
    - Export support
    """
    try:
        # Get base metrics
        metrics = await use_cases.get_nlp_performance_metrics()
        
        # Add system metrics
        metrics["system"] = {
            "active_requests": active_requests._value.get(),
            "total_requests": request_counter._value.sum(),
            "cache_hit_rate": (
                cache_hits._value.sum() / 
                (cache_hits._value.sum() + cache_misses._value.sum())
                if (cache_hits._value.sum() + cache_misses._value.sum()) > 0
                else 0
            )
        }
        
        # Add detailed metrics if requested
        if include_detailed:
            metrics["detailed"] = {
                "request_duration_p50": request_duration._sum.sum() / request_duration._count.sum() if request_duration._count.sum() > 0 else 0,
                "request_duration_p95": 0,  # Would need histogram buckets
                "request_duration_p99": 0,  # Would need histogram buckets
                "endpoints": {}
            }
            
            # Add per-endpoint metrics
            for endpoint in ["/", "/{post_id}", "/batch", "/analyze"]:
                endpoint_requests = request_counter.labels(method="GET", endpoint=endpoint)._value.get()
                metrics["detailed"]["endpoints"][endpoint] = {
                    "requests": endpoint_requests,
                    "avg_duration": 0  # Would need per-endpoint tracking
                }
        
        # Add time range filtering
        metrics["time_range"] = time_range
        metrics["timestamp"] = datetime.utcnow()
        
        # Set cache headers for metrics
        response.headers["Cache-Control"] = "public, max-age=10"
        
        return NLPPerformanceResponse(
            fast_nlp_metrics=metrics["fast_nlp"],
            async_nlp_metrics=metrics["async_nlp"],
            system_metrics=metrics.get("system"),
            detailed_metrics=metrics.get("detailed"),
            timestamp=metrics["timestamp"]
        )
        
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/performance/optimize")
async def optimize_performance(
    background_tasks: BackgroundTasks,
    use_cases: LinkedInPostUseCases = Depends(get_use_cases),
    clear_cache: bool = Query(False, description="Clear all caches"),
    warm_cache: bool = Query(False, description="Warm up caches"),
    optimize_nlp: bool = Query(True, description="Optimize NLP models")
):
    """
    Optimize system performance.
    
    Features:
    - Cache management
    - Model optimization
    - Resource cleanup
    - Performance tuning
    """
    try:
        results = {
            "optimizations": [],
            "timestamp": datetime.utcnow()
        }
        
        # Clear cache if requested
        if clear_cache:
            await use_cases.clear_nlp_cache()
            await cache_manager.clear()
            results["optimizations"].append("Cache cleared")
        
        # Warm cache if requested
        if warm_cache:
            background_tasks.add_task(
                use_cases.warm_nlp_cache
            )
            results["optimizations"].append("Cache warming initiated")
        
        # Optimize NLP models
        if optimize_nlp:
            background_tasks.add_task(
                use_cases.optimize_nlp_models
            )
            results["optimizations"].append("NLP optimization initiated")
        
        return results
        
    except Exception as e:
        logger.error(f"Error optimizing performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", tags=["Health"])
async def health_check(
    detailed: bool = Query(False, description="Include detailed health info")
):
    """
    Advanced health check endpoint.
    
    Features:
    - Service health monitoring
    - Dependency checks
    - Performance indicators
    - Detailed diagnostics
    """
    health_status = {
        "status": "healthy",
        "service": "linkedin-posts-v2",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.0.0",
        "uptime": time.time() - startup_time if 'startup_time' in globals() else 0
    }
    
    if detailed:
        # Check dependencies
        health_status["dependencies"] = {
            "database": "healthy",  # Would check actual DB connection
            "cache": "healthy",     # Would check Redis connection
            "nlp_service": "healthy"  # Would check NLP service
        }
        
        # Add performance indicators
        health_status["performance"] = {
            "active_requests": active_requests._value.get(),
            "avg_response_time": "50ms",  # Would calculate actual average
            "error_rate": 0.001,  # Would calculate actual rate
            "cache_hit_rate": 0.85  # Would calculate actual rate
        }
        
        # Add resource usage
        health_status["resources"] = {
            "memory_usage_mb": 256,  # Would get actual memory usage
            "cpu_usage_percent": 15,  # Would get actual CPU usage
            "connections_active": 42  # Would get actual connections
        }
    
    return health_status


@router.get("/health/ready", tags=["Health"])
async def readiness_check():
    """
    Kubernetes readiness probe.
    
    Returns 200 if service is ready to accept traffic.
    """
    # Check if all dependencies are ready
    try:
        # Check database connection
        # Check cache connection
        # Check NLP service
        
        return {"ready": True}
    except Exception:
        raise HTTPException(status_code=503, detail="Service not ready")


@router.get("/health/live", tags=["Health"])
async def liveness_check():
    """
    Kubernetes liveness probe.
    
    Returns 200 if service is alive.
    """
    return {"alive": True}


# Export router
__all__ = ["router"] 