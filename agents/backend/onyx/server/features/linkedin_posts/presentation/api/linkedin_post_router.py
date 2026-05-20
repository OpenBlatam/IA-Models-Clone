from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

from fastapi import APIRouter, HTTPException, Depends, Query, Response
from typing import List, Optional, Dict, Any
from datetime import datetime
from ...core.domain.entities.linkedin_post import PostStatus, PostType, PostTone
from ...application.use_cases.linkedin_post_use_cases import LinkedInPostUseCases
from ...infrastructure.repositories.linkedin_post_repository import LinkedInPostRepository
from ...shared.schemas.linkedin_post_schemas import (
from ...shared.logging import get_logger
from ...shared.dependencies import get_current_user, rate_limiter, User
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
LinkedIn Post API Router
=======================

FastAPI router for LinkedIn post management with fast NLP integration.
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

# Create router with global rate limiter dependency
router = APIRouter(
    prefix="/linkedin-posts",
    tags=["LinkedIn Posts"],
    dependencies=[Depends(rate_limiter)]
)

# Dependency injection
def get_repository() -> LinkedInPostRepository:
    """Get LinkedIn post repository instance."""
    return LinkedInPostRepository()

def get_use_cases(repo: LinkedInPostRepository = Depends(get_repository)) -> LinkedInPostUseCases:
    """Get LinkedIn post use cases instance."""
    return LinkedInPostUseCases(repo)


@router.post("/", response_model=LinkedInPostResponse)
async def create_linkedin_post(
    post_data: LinkedInPostCreate,
    use_cases: LinkedInPostUseCases = Depends(get_use_cases),
    use_fast_nlp: bool = Query(True, description="Enable fast NLP enhancement"),
    use_async_nlp: bool = Query(False, description="Use async NLP processor for maximum speed")
):
    """
    Create a new LinkedIn post with optional fast NLP enhancement.
    
    Features:
    - Fast NLP enhancement for improved content quality
    - Async NLP processing for maximum speed
    - Automatic content optimization
    """
    try:
        post = await use_cases.generate_post(
            content=post_data.content,
            post_type=post_data.post_type,
            tone=post_data.tone,
            target_audience=post_data.target_audience,
            industry=post_data.industry,
            use_fast_nlp=use_fast_nlp,
            use_async_nlp=use_async_nlp
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
        logger.error(f"Error creating LinkedIn post: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", response_model=LinkedInPostListResponse)
async def list_linkedin_posts(
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    status: Optional[PostStatus] = Query(None, description="Filter by status"),
    post_type: Optional[PostType] = Query(None, description="Filter by post type"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of posts"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    use_cases: LinkedInPostUseCases = Depends(get_use_cases),
    current_user: User = Depends(get_current_user),
    response: Response = None
):
    """
    List LinkedIn posts with filtering and pagination.
    
    Features:
    - Flexible filtering by user, status, and type
    - Pagination support
    - Fast response times
    """
    try:
        posts = await use_cases.list_posts(
            user_id=user_id,
            status=status,
            post_type=post_type,
            limit=limit,
            offset=offset
        )
        
        # Add pagination headers
        response.headers["X-Total-Count"] = str(len(posts))
        response.headers["X-Limit"] = str(limit)
        response.headers["X-Offset"] = str(offset)

        return LinkedInPostListResponse(
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
            offset=offset
        )
        
    except Exception as e:
        logger.error(f"Error listing LinkedIn posts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{post_id}", response_model=LinkedInPostResponse)
async def get_linkedin_post(
    post_id: str,
    use_cases: LinkedInPostUseCases = Depends(get_use_cases)
):
    """
    Get a specific LinkedIn post by ID.
    
    Features:
    - Fast retrieval with caching
    - Complete post information
    """
    try:
        post = await use_cases.repository.get_by_id(post_id)
        if not post:
            raise HTTPException(status_code=404, detail="Post not found")
        
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
        logger.error(f"Error getting LinkedIn post: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{post_id}", response_model=LinkedInPostResponse)
async def update_linkedin_post(
    post_id: str,
    post_data: LinkedInPostUpdate,
    use_cases: LinkedInPostUseCases = Depends(get_use_cases),
    use_fast_nlp: bool = Query(False, description="Enable fast NLP enhancement"),
    use_async_nlp: bool = Query(False, description="Use async NLP processor")
):
    """
    Update a LinkedIn post with optional NLP enhancement.
    
    Features:
    - Partial updates supported
    - Optional fast NLP enhancement
    - Async processing for speed
    """
    try:
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


@router.delete("/{post_id}")
async def delete_linkedin_post(
    post_id: str,
    use_cases: LinkedInPostUseCases = Depends(get_use_cases)
):
    """
    Delete a LinkedIn post.
    
    Features:
    - Soft delete support
    - Fast deletion process
    """
    try:
        success = await use_cases.delete_post(post_id)
        if not success:
            raise HTTPException(status_code=404, detail="Post not found")
        
        return {"message": "Post deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting LinkedIn post: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{post_id}/optimize", response_model=LinkedInPostResponse)
async def optimize_linkedin_post(
    post_id: str,
    optimization_request: PostOptimizationRequest,
    use_cases: LinkedInPostUseCases = Depends(get_use_cases)
):
    """
    Optimize a LinkedIn post using fast NLP.
    
    Features:
    - Fast NLP optimization
    - Async processing for maximum speed
    - Content quality improvement
    """
    try:
        post = await use_cases.optimize_post(
            post_id=post_id,
            use_async_nlp=optimization_request.use_async_nlp
        )
        
        if not post:
            raise HTTPException(status_code=404, detail="Post not found")
        
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


@router.get("/{post_id}/analyze", response_model=PostAnalysisResponse)
async def analyze_linkedin_post(
    post_id: str,
    use_cases: LinkedInPostUseCases = Depends(get_use_cases),
    use_async_nlp: bool = Query(True, description="Use async NLP processor")
):
    """
    Analyze LinkedIn post engagement potential using fast NLP.
    
    Features:
    - Sentiment analysis
    - Readability scoring
    - Keyword extraction
    - Entity detection
    - Fast processing with caching
    """
    try:
        analysis = await use_cases.analyze_post_engagement(
            post_id=post_id,
            use_async_nlp=use_async_nlp
        )
        
        if not analysis:
            raise HTTPException(status_code=404, detail="Post not found")
        
        return PostAnalysisResponse(
            post_id=analysis["post_id"],
            sentiment_score=analysis["sentiment_score"],
            readability_score=analysis["readability_score"],
            keywords=analysis["keywords"],
            entities=analysis["entities"],
            processing_time=analysis["processing_time"],
            cached=analysis["cached"],
            async_optimized=analysis["async_optimized"],
            analyzed_at=datetime.utcnow()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing LinkedIn post: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch-optimize", response_model=List[LinkedInPostResponse])
async def batch_optimize_posts(
    batch_request: BatchOptimizationRequest,
    use_cases: LinkedInPostUseCases = Depends(get_use_cases)
):
    """
    Optimize multiple LinkedIn posts using batch processing.
    
    Features:
    - Batch processing for efficiency
    - Fast NLP optimization
    - Async processing for maximum speed
    - Reduced processing time for multiple posts
    """
    try:
        optimized_posts = await use_cases.batch_optimize_posts(
            post_ids=batch_request.post_ids,
            use_async_nlp=batch_request.use_async_nlp
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


@router.get("/nlp/performance", response_model=NLPPerformanceResponse)
async def get_nlp_performance_metrics(
    use_cases: LinkedInPostUseCases = Depends(get_use_cases)
):
    """
    Get NLP performance metrics for monitoring.
    
    Features:
    - Real-time performance metrics
    - Cache hit rates
    - Processing times
    - System health monitoring
    """
    try:
        metrics = await use_cases.get_nlp_performance_metrics()
        
        return NLPPerformanceResponse(
            fast_nlp_metrics=metrics["fast_nlp"],
            async_nlp_metrics=metrics["async_nlp"],
            timestamp=metrics["timestamp"]
        )
        
    except Exception as e:
        logger.error(f"Error getting NLP performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/nlp/clear-cache")
async def clear_nlp_cache(
    use_cases: LinkedInPostUseCases = Depends(get_use_cases)
):
    """
    Clear NLP cache for both processors.
    
    Features:
    - Clear memory and Redis caches
    - Reset performance metrics
    - Fresh start for processing
    """
    try:
        success = await use_cases.clear_nlp_cache()
        
        return {
            "message": "NLP cache cleared successfully",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error clearing NLP cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """
    Health check endpoint for LinkedIn posts service.
    
    Features:
    - Service health monitoring
    - Fast response for load balancers
    """
    return {
        "status": "healthy",
        "service": "linkedin-posts",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "features": [
            "Fast NLP processing",
            "Async optimization",
            "Batch operations",
            "Performance monitoring",
            "Cache management"
        ]
    } 