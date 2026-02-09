"""
Async Facebook Posts Engine
Following functional programming principles and async patterns
"""

from typing import Dict, Any, Optional, List, Tuple
import asyncio
import logging
from datetime import datetime
import uuid
import json

from ..services.async_ai_service import AsyncAIService, generate_post_content_async, batch_generate_content_async
from ..api.schemas import PostRequest, PostResponse, FacebookPost, PostMetrics, BatchPostResponse

logger = logging.getLogger(__name__)


# Pure functions for engine operations

def create_post_id() -> str:
    """Create unique post ID - pure function"""
    return str(uuid.uuid4())


def calculate_processing_time(start_time: float, end_time: float) -> float:
    """Calculate processing time - pure function"""
    return end_time - start_time


def determine_post_status(content: str, is_optimized: bool = False) -> str:
    """Determine post status - pure function"""
    if not content or len(content.strip()) < 10:
        return "draft"
    
    if is_optimized:
        return "optimized"
    
    return "generated"


def build_optimization_metadata(optimizations: List[str]) -> Dict[str, Any]:
    """Build optimization metadata - pure function"""
    return {
        "optimizations_applied": optimizations,
        "optimization_count": len(optimizations),
        "last_optimized": datetime.utcnow().isoformat()
    }


def calculate_quality_score(analysis: Dict[str, Any]) -> float:
    """Calculate quality score - pure function"""
    engagement = analysis.get("engagement_score", 0.5)
    readability = analysis.get("readability_score", 0.5)
    sentiment = abs(analysis.get("sentiment_score", 0.0))
    
    # Weighted average
    return (engagement * 0.4 + readability * 0.4 + sentiment * 0.2)


def build_post_metrics(analysis: Dict[str, Any], content: str) -> PostMetrics:
    """Build post metrics - pure function"""
    return PostMetrics(
        engagement_score=analysis.get("engagement_score", 0.5),
        readability_score=analysis.get("readability_score", 0.5),
        sentiment_score=analysis.get("sentiment_score", 0.0),
        viral_potential=analysis.get("viral_potential", 0.5),
        quality_score=calculate_quality_score(analysis),
        estimated_reach=1000,  # Mock implementation
        estimated_impressions=5000,
        estimated_clicks=100,
        estimated_likes=50,
        estimated_shares=10,
        estimated_comments=5
    )


def create_facebook_post(
    content: str,
    request: PostRequest,
    analysis: Dict[str, Any],
    post_id: Optional[str] = None
) -> FacebookPost:
    """Create Facebook post - pure function"""
    if not post_id:
        post_id = create_post_id()
    
    hashtags = analysis.get("hashtags", [])
    emojis = analysis.get("emojis", [])
    metrics = build_post_metrics(analysis, content)
    
    return FacebookPost(
        id=post_id,
        content=content,
        content_type=request.content_type,
        audience_type=request.audience_type,
        topic=request.topic,
        tone=request.tone,
        language=request.language,
        hashtags=hashtags,
        emojis=emojis,
        call_to_action=request.call_to_action,
        status=determine_post_status(content),
        metrics=metrics,
        optimizations_applied=["ai_generation"]
    )


def create_post_response(
    success: bool,
    post: Optional[FacebookPost] = None,
    error: Optional[str] = None,
    processing_time: float = 0.0,
    optimizations_applied: List[str] = None
) -> PostResponse:
    """Create post response - pure function"""
    return PostResponse(
        success=success,
        post=post,
        error=error,
        processing_time=processing_time,
        optimizations_applied=optimizations_applied or [],
        analytics=post.metrics.dict() if post and post.metrics else None
    )


def create_batch_response(
    results: List[PostResponse],
    total_time: float,
    batch_id: str
) -> BatchPostResponse:
    """Create batch response - pure function"""
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


# Async Engine Class

class AsyncFacebookPostsEngine:
    """Async Facebook Posts Engine following functional principles"""
    
    def __init__(
        self,
        ai_service: AsyncAIService,
        analytics_service: Optional[Any] = None,
        cache_service: Optional[Any] = None,
        post_repository: Optional[Any] = None
    ):
        self.ai_service = ai_service
        self.analytics_service = analytics_service
        self.cache_service = cache_service
        self.post_repository = post_repository
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_processing_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
    
    async def generate_post(self, request: PostRequest) -> PostResponse:
        """Generate single post"""
        start_time = asyncio.get_event_loop().time()
        self.stats["total_requests"] += 1
        
        try:
            # Check cache first
            cache_key = self._build_cache_key(request)
            cached_result = await self._get_from_cache(cache_key)
            
            if cached_result:
                self.stats["cache_hits"] += 1
                return cached_result
            
            self.stats["cache_misses"] += 1
            
            # Generate content
            result = await generate_post_content_async(
                topic=request.topic,
                content_type=request.content_type.value,
                audience_type=request.audience_type.value,
                tone=request.tone,
                language=request.language,
                max_length=request.max_length,
                ai_service=self.ai_service,
                custom_instructions=request.custom_instructions
            )
            
            if not result["success"]:
                self.stats["failed_requests"] += 1
                return create_post_response(
                    success=False,
                    error=result.get("error", "Unknown error"),
                    processing_time=calculate_processing_time(start_time, asyncio.get_event_loop().time())
                )
            
            # Create post
            post = create_facebook_post(
                content=result["content"],
                request=request,
                analysis=result["analysis"]
            )
            
            # Save to repository
            if self.post_repository:
                await self._save_post(post)
            
            # Cache result
            response = create_post_response(
                success=True,
                post=post,
                processing_time=calculate_processing_time(start_time, asyncio.get_event_loop().time()),
                optimizations_applied=["ai_generation"]
            )
            
            await self._save_to_cache(cache_key, response)
            
            self.stats["successful_requests"] += 1
            self.stats["total_processing_time"] += response.processing_time
            
            return response
            
        except Exception as e:
            self.stats["failed_requests"] += 1
            logger.error("Error generating post", error=str(e))
            return create_post_response(
                success=False,
                error=str(e),
                processing_time=calculate_processing_time(start_time, asyncio.get_event_loop().time())
            )
    
    async def generate_batch_posts(self, requests: List[PostRequest]) -> BatchPostResponse:
        """Generate multiple posts in batch"""
        start_time = asyncio.get_event_loop().time()
        batch_id = create_post_id()
        
        try:
            # Convert requests to dict format
            request_dicts = [
                {
                    "topic": req.topic,
                    "content_type": req.content_type.value,
                    "audience_type": req.audience_type.value,
                    "tone": req.tone,
                    "language": req.language,
                    "max_length": req.max_length,
                    "custom_instructions": req.custom_instructions
                }
                for req in requests
            ]
            
            # Generate content in parallel
            results = await batch_generate_content_async(request_dicts, self.ai_service)
            
            # Create responses
            post_responses = []
            for i, result in enumerate(results):
                if result["success"]:
                    post = create_facebook_post(
                        content=result["content"],
                        request=requests[i],
                        analysis=result["analysis"]
                    )
                    
                    if self.post_repository:
                        await self._save_post(post)
                    
                    response = create_post_response(
                        success=True,
                        post=post,
                        processing_time=0.1,  # Mock processing time
                        optimizations_applied=["ai_generation"]
                    )
                else:
                    response = create_post_response(
                        success=False,
                        error=result.get("error", "Unknown error"),
                        processing_time=0.1
                    )
                
                post_responses.append(response)
            
            total_time = calculate_processing_time(start_time, asyncio.get_event_loop().time())
            return create_batch_response(post_responses, total_time, batch_id)
            
        except Exception as e:
            logger.error("Error generating batch posts", error=str(e))
            return create_batch_response(
                [create_post_response(success=False, error=str(e))],
                calculate_processing_time(start_time, asyncio.get_event_loop().time()),
                batch_id
            )
    
    async def get_post(self, post_id: str) -> Optional[FacebookPost]:
        """Get post by ID"""
        try:
            if self.post_repository:
                return await self.post_repository.get_by_id(post_id)
            return None
        except Exception as e:
            logger.error("Error getting post", error=str(e), post_id=post_id)
            return None
    
    async def list_posts(
        self,
        skip: int = 0,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[FacebookPost]:
        """List posts with filters"""
        try:
            if self.post_repository:
                return await self.post_repository.list_posts(skip, limit, filters or {})
            return []
        except Exception as e:
            logger.error("Error listing posts", error=str(e))
            return []
    
    async def update_post(self, post_id: str, data: Dict[str, Any]) -> Optional[FacebookPost]:
        """Update post"""
        try:
            if self.post_repository:
                return await self.post_repository.update(post_id, data)
            return None
        except Exception as e:
            logger.error("Error updating post", error=str(e), post_id=post_id)
            return None
    
    async def delete_post(self, post_id: str) -> bool:
        """Delete post"""
        try:
            if self.post_repository:
                return await self.post_repository.delete(post_id)
            return False
        except Exception as e:
            logger.error("Error deleting post", error=str(e), post_id=post_id)
            return False
    
    async def optimize_post(self, post_id: str, optimization_type: str) -> Optional[FacebookPost]:
        """Optimize existing post"""
        try:
            post = await self.get_post(post_id)
            if not post:
                return None
            
            # Mock optimization - in real implementation, this would use AI
            optimized_content = f"[OPTIMIZED] {post.content}"
            
            optimized_post = create_facebook_post(
                content=optimized_content,
                request=PostRequest(
                    topic=post.topic,
                    content_type=post.content_type,
                    audience_type=post.audience_type,
                    tone=post.tone,
                    language=post.language,
                    max_length=len(optimized_content)
                ),
                analysis={"engagement_score": 0.9, "readability_score": 0.9},
                post_id=post.id
            )
            
            optimized_post.optimizations_applied = post.optimizations_applied + ["optimization"]
            optimized_post.status = "optimized"
            
            if self.post_repository:
                await self._save_post(optimized_post)
            
            return optimized_post
            
        except Exception as e:
            logger.error("Error optimizing post", error=str(e), post_id=post_id)
            return None
    
    async def get_analytics(self, post_id: str) -> Optional[Dict[str, Any]]:
        """Get analytics for post"""
        try:
            if self.analytics_service:
                return await self.analytics_service.get_post_analytics(post_id)
            return None
        except Exception as e:
            logger.error("Error getting analytics", error=str(e), post_id=post_id)
            return None
    
    async def health_check(self) -> Dict[str, Any]:
        """Check engine health"""
        try:
            ai_health = await self.ai_service.health_check()
            
            return {
                "status": "healthy" if ai_health["status"] == "healthy" else "degraded",
                "ai_service": ai_health,
                "cache_service": await self._check_cache_health(),
                "repository_service": await self._check_repository_health(),
                "statistics": self.stats
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "statistics": self.stats
            }
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics"""
        return {
            **self.stats,
            "average_processing_time": (
                self.stats["total_processing_time"] / max(1, self.stats["total_requests"])
            ),
            "success_rate": (
                self.stats["successful_requests"] / max(1, self.stats["total_requests"])
            ),
            "cache_hit_rate": (
                self.stats["cache_hits"] / max(1, self.stats["cache_hits"] + self.stats["cache_misses"])
            )
        }
    
    # Private helper methods
    
    def _build_cache_key(self, request: PostRequest) -> str:
        """Build cache key for request"""
        key_data = {
            "topic": request.topic,
            "content_type": request.content_type.value,
            "audience_type": request.audience_type.value,
            "tone": request.tone,
            "language": request.language,
            "max_length": request.max_length
        }
        return f"post:{hash(json.dumps(key_data, sort_keys=True))}"
    
    async def _get_from_cache(self, key: str) -> Optional[PostResponse]:
        """Get result from cache"""
        if not self.cache_service:
            return None
        
        try:
            cached_data = await self.cache_service.get(key)
            if cached_data:
                return PostResponse(**cached_data)
        except Exception as e:
            logger.warning("Error getting from cache", error=str(e))
        
        return None
    
    async def _save_to_cache(self, key: str, response: PostResponse) -> None:
        """Save result to cache"""
        if not self.cache_service:
            return
        
        try:
            await self.cache_service.set(key, response.dict(), ttl=3600)
        except Exception as e:
            logger.warning("Error saving to cache", error=str(e))
    
    async def _save_post(self, post: FacebookPost) -> None:
        """Save post to repository"""
        if not self.post_repository:
            return
        
        try:
            await self.post_repository.save(post)
        except Exception as e:
            logger.warning("Error saving post", error=str(e))
    
    async def _check_cache_health(self) -> Dict[str, Any]:
        """Check cache service health"""
        if not self.cache_service:
            return {"status": "not_configured"}
        
        try:
            return await self.cache_service.health_check()
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    async def _check_repository_health(self) -> Dict[str, Any]:
        """Check repository service health"""
        if not self.post_repository:
            return {"status": "not_configured"}
        
        try:
            return await self.post_repository.health_check()
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}


# Factory functions

def create_async_engine(
    ai_service: AsyncAIService,
    analytics_service: Optional[Any] = None,
    cache_service: Optional[Any] = None,
    post_repository: Optional[Any] = None
) -> AsyncFacebookPostsEngine:
    """Create async engine instance - pure function"""
    return AsyncFacebookPostsEngine(ai_service, analytics_service, cache_service, post_repository)