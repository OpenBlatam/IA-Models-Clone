"""
Advanced API routes for Facebook Posts API
AI-powered content generation, analytics, and optimization
"""

import asyncio
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, Query, Path
from fastapi.responses import JSONResponse
import structlog

from ..core.config import get_settings
from ..core.models import PostRequest, PostStatus, ContentType, AudienceType, OptimizationLevel
from ..api.schemas import (
    PostUpdateRequest, BatchPostRequest, OptimizationRequest,
    ErrorResponse, SystemHealth, PerformanceMetrics
)
from ..api.dependencies import (
    get_facebook_engine, get_current_user, check_rate_limit,
    get_request_id, validate_post_id
)
from ..services.ai_service import get_ai_service, AIGenerationResult, ContentAnalysis
from ..services.analytics_service import get_analytics_service, PostMetrics, PerformanceReport
from ..infrastructure.cache import get_cache_manager
from ..infrastructure.monitoring import get_monitor, timed
from ..infrastructure.database import get_db_manager, PostRepository

logger = structlog.get_logger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/advanced", tags=["Advanced Features"])


# AI-Powered Content Generation Routes

@router.post(
    "/ai/generate",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Content generated successfully"},
        400: {"description": "Invalid request parameters"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "AI service error"}
    },
    summary="Generate AI-powered content",
    description="Generate Facebook post content using AI with advanced optimization"
)
@timed("ai_content_generation")
async def generate_ai_content(
    request: PostRequest,
    background_tasks: BackgroundTasks,
    request_id: str = Depends(get_request_id),
    user: Dict[str, Any] = Depends(get_current_user),
    rate_limit: bool = Depends(check_rate_limit)
) -> Dict[str, Any]:
    """Generate AI-powered content for Facebook posts"""
    
    # Early validation
    if not request.topic or len(request.topic.strip()) < 3:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Topic must be at least 3 characters long"
        )
    
    try:
        # Get AI service
        ai_service = get_ai_service()
        
        # Generate content
        start_time = time.time()
        result = await ai_service.generate_content(request)
        generation_time = time.time() - start_time
        
        # Create post data
        post_data = {
            "id": f"ai_post_{int(time.time())}",
            "content": result.content,
            "status": "draft",
            "content_type": request.content_type.value,
            "audience_type": request.audience_type.value,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "metadata": {
                "ai_generated": True,
                "model_used": result.model_used,
                "confidence_score": result.confidence_score,
                "generation_time": generation_time,
                "request_id": request_id,
                "topic": request.topic,
                "tone": request.tone
            }
        }
        
        # Save to database in background
        background_tasks.add_task(save_ai_generated_post, post_data, request_id)
        
        # Log successful generation
        logger.info(
            "AI content generated successfully",
            post_id=post_data["id"],
            model=result.model_used,
            confidence=result.confidence_score,
            generation_time=generation_time,
            request_id=request_id
        )
        
        return {
            "success": True,
            "post_id": post_data["id"],
            "content": result.content,
            "metadata": {
                "ai_generated": True,
                "model_used": result.model_used,
                "confidence_score": result.confidence_score,
                "generation_time": generation_time,
                "tokens_used": result.tokens_used,
                "processing_time": result.processing_time
            },
            "request_id": request_id,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(
            "AI content generation failed",
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"AI content generation failed: {str(e)}"
        )


@router.post(
    "/ai/analyze",
    response_model=ContentAnalysis,
    responses={
        200: {"description": "Content analyzed successfully"},
        400: {"description": "Invalid content"},
        500: {"description": "Analysis error"}
    },
    summary="Analyze content with AI",
    description="Analyze Facebook post content for sentiment, engagement, and quality metrics"
)
@timed("ai_content_analysis")
async def analyze_content(
    content: str = Query(..., description="Content to analyze", min_length=10),
    context: Optional[Dict[str, Any]] = None,
    request_id: str = Depends(get_request_id),
    user: Dict[str, Any] = Depends(get_current_user)
) -> ContentAnalysis:
    """Analyze content using AI"""
    
    if len(content) > 5000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Content too long for analysis (max 5000 characters)"
        )
    
    try:
        ai_service = get_ai_service()
        analysis = await ai_service.analyze_content(content, context)
        
        logger.info(
            "Content analysis completed",
            content_length=len(content),
            sentiment_score=analysis.sentiment_score,
            engagement_score=analysis.engagement_score,
            request_id=request_id
        )
        
        return analysis
        
    except Exception as e:
        logger.error(
            "Content analysis failed",
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Content analysis failed: {str(e)}"
        )


@router.post(
    "/ai/optimize",
    response_model=List[Dict[str, Any]],
    responses={
        200: {"description": "Optimization suggestions generated"},
        400: {"description": "Invalid request"},
        500: {"description": "Optimization error"}
    },
    summary="Optimize content with AI",
    description="Generate AI-powered optimization suggestions for content"
)
@timed("ai_content_optimization")
async def optimize_content(
    content: str = Query(..., description="Content to optimize", min_length=10),
    target_audience: AudienceType = Query(..., description="Target audience"),
    optimization_goals: List[str] = Query(..., description="Optimization goals"),
    request_id: str = Depends(get_request_id),
    user: Dict[str, Any] = Depends(get_current_user)
) -> List[Dict[str, Any]]:
    """Optimize content using AI"""
    
    if len(content) > 5000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Content too long for optimization (max 5000 characters)"
        )
    
    valid_goals = ["engagement", "readability", "sentiment", "creativity"]
    invalid_goals = [goal for goal in optimization_goals if goal not in valid_goals]
    
    if invalid_goals:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid optimization goals: {invalid_goals}. Valid goals: {valid_goals}"
        )
    
    try:
        ai_service = get_ai_service()
        suggestions = await ai_service.optimize_content(content, target_audience, optimization_goals)
        
        # Convert to dict format
        result = [
            {
                "type": suggestion.type,
                "priority": suggestion.priority,
                "suggestion": suggestion.suggestion,
                "expected_improvement": suggestion.expected_improvement,
                "implementation": suggestion.implementation
            }
            for suggestion in suggestions
        ]
        
        logger.info(
            "Content optimization completed",
            content_length=len(content),
            suggestions_count=len(result),
            request_id=request_id
        )
        
        return result
        
    except Exception as e:
        logger.error(
            "Content optimization failed",
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Content optimization failed: {str(e)}"
        )


# Advanced Analytics Routes

@router.get(
    "/analytics/trending",
    response_model=List[Dict[str, Any]],
    responses={
        200: {"description": "Trending posts retrieved"},
        500: {"description": "Analytics error"}
    },
    summary="Get trending posts",
    description="Get trending posts based on real-time engagement metrics"
)
@timed("analytics_trending")
async def get_trending_posts(
    limit: int = Query(10, description="Number of trending posts to return", ge=1, le=50),
    request_id: str = Depends(get_request_id),
    user: Dict[str, Any] = Depends(get_current_user)
) -> List[Dict[str, Any]]:
    """Get trending posts based on engagement"""
    
    try:
        analytics_service = get_analytics_service()
        trending_posts = await analytics_service.get_trending_posts(limit)
        
        logger.info(
            "Trending posts retrieved",
            count=len(trending_posts),
            request_id=request_id
        )
        
        return trending_posts
        
    except Exception as e:
        logger.error(
            "Failed to get trending posts",
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get trending posts: {str(e)}"
        )


@router.get(
    "/analytics/audience/{audience_type}",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Audience insights retrieved"},
        400: {"description": "Invalid audience type"},
        500: {"description": "Analytics error"}
    },
    summary="Get audience insights",
    description="Get detailed audience behavior insights and recommendations"
)
@timed("analytics_audience")
async def get_audience_insights(
    audience_type: str = Path(..., description="Audience type to analyze"),
    days: int = Query(30, description="Analysis period in days", ge=1, le=365),
    request_id: str = Depends(get_request_id),
    user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get audience insights"""
    
    # Validate audience type
    valid_audience_types = [audience.value for audience in AudienceType]
    if audience_type not in valid_audience_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid audience type. Valid types: {valid_audience_types}"
        )
    
    try:
        analytics_service = get_analytics_service()
        insight = await analytics_service.get_audience_insights(audience_type, days)
        
        logger.info(
            "Audience insights retrieved",
            audience_type=audience_type,
            days=days,
            request_id=request_id
        )
        
        return {
            "audience_type": insight.audience_type,
            "engagement_rate": insight.engagement_rate,
            "preferred_content_types": insight.preferred_content_types,
            "optimal_posting_times": insight.optimal_posting_times,
            "top_performing_topics": insight.top_performing_topics,
            "demographics": insight.demographics,
            "analysis_period_days": days,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(
            "Failed to get audience insights",
            audience_type=audience_type,
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get audience insights: {str(e)}"
        )


@router.get(
    "/analytics/content/{content_type}",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Content insights retrieved"},
        400: {"description": "Invalid content type"},
        500: {"description": "Analytics error"}
    },
    summary="Get content insights",
    description="Get detailed content performance insights and optimization recommendations"
)
@timed("analytics_content")
async def get_content_insights(
    content_type: str = Path(..., description="Content type to analyze"),
    days: int = Query(30, description="Analysis period in days", ge=1, le=365),
    request_id: str = Depends(get_request_id),
    user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get content insights"""
    
    # Validate content type
    valid_content_types = [content_type.value for content_type in ContentType]
    if content_type not in valid_content_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid content type. Valid types: {valid_content_types}"
        )
    
    try:
        analytics_service = get_analytics_service()
        insight = await analytics_service.get_content_insights(content_type, days)
        
        logger.info(
            "Content insights retrieved",
            content_type=content_type,
            days=days,
            request_id=request_id
        )
        
        return {
            "content_type": insight.content_type,
            "average_engagement": insight.average_engagement,
            "best_performing_length": insight.best_performing_length,
            "optimal_hashtag_count": insight.optimal_hashtag_count,
            "sentiment_impact": insight.sentiment_impact,
            "creativity_score_impact": insight.creativity_score_impact,
            "analysis_period_days": days,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(
            "Failed to get content insights",
            content_type=content_type,
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get content insights: {str(e)}"
        )


@router.get(
    "/analytics/performance",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Performance report generated"},
        500: {"description": "Report generation error"}
    },
    summary="Generate performance report",
    description="Generate comprehensive performance report with insights and recommendations"
)
@timed("analytics_performance")
async def generate_performance_report(
    days: int = Query(30, description="Report period in days", ge=1, le=365),
    request_id: str = Depends(get_request_id),
    user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Generate performance report"""
    
    try:
        analytics_service = get_analytics_service()
        report = await analytics_service.generate_performance_report(days)
        
        logger.info(
            "Performance report generated",
            days=days,
            total_posts=report.total_posts,
            request_id=request_id
        )
        
        return {
            "period_start": report.period_start.isoformat(),
            "period_end": report.period_end.isoformat(),
            "total_posts": report.total_posts,
            "total_engagement": report.total_engagement,
            "average_engagement_rate": report.average_engagement_rate,
            "top_performing_posts": report.top_performing_posts[:5],  # Limit to top 5
            "audience_insights": [
                {
                    "audience_type": insight.audience_type,
                    "engagement_rate": insight.engagement_rate,
                    "preferred_content_types": insight.preferred_content_types,
                    "optimal_posting_times": insight.optimal_posting_times
                }
                for insight in report.audience_insights
            ],
            "content_insights": [
                {
                    "content_type": insight.content_type,
                    "average_engagement": insight.average_engagement,
                    "best_performing_length": insight.best_performing_length,
                    "optimal_hashtag_count": insight.optimal_hashtag_count
                }
                for insight in report.content_insights
            ],
            "recommendations": report.recommendations,
            "generated_at": report.generated_at.isoformat()
        }
        
    except Exception as e:
        logger.error(
            "Failed to generate performance report",
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate performance report: {str(e)}"
        )


# Engagement Tracking Routes

@router.post(
    "/engagement/track",
    responses={
        200: {"description": "Engagement tracked successfully"},
        400: {"description": "Invalid engagement data"},
        500: {"description": "Tracking error"}
    },
    summary="Track post engagement",
    description="Track real-time post engagement metrics"
)
@timed("engagement_tracking")
async def track_engagement(
    post_id: str = Query(..., description="Post ID to track"),
    engagement_type: str = Query(..., description="Type of engagement"),
    value: int = Query(1, description="Engagement value", ge=1),
    request_id: str = Depends(get_request_id),
    user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Track post engagement"""
    
    # Validate engagement type
    valid_engagement_types = ["views", "likes", "shares", "comments", "clicks", "reach", "impressions"]
    if engagement_type not in valid_engagement_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid engagement type. Valid types: {valid_engagement_types}"
        )
    
    try:
        analytics_service = get_analytics_service()
        await analytics_service.track_engagement(post_id, engagement_type, value)
        
        logger.info(
            "Engagement tracked",
            post_id=post_id,
            engagement_type=engagement_type,
            value=value,
            request_id=request_id
        )
        
        return {
            "success": True,
            "post_id": post_id,
            "engagement_type": engagement_type,
            "value": value,
            "tracked_at": datetime.now().isoformat(),
            "request_id": request_id
        }
        
    except Exception as e:
        logger.error(
            "Failed to track engagement",
            post_id=post_id,
            engagement_type=engagement_type,
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to track engagement: {str(e)}"
        )


@router.get(
    "/engagement/{post_id}",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Post engagement metrics retrieved"},
        404: {"description": "Post not found"},
        500: {"description": "Metrics retrieval error"}
    },
    summary="Get post engagement metrics",
    description="Get real-time engagement metrics for a specific post"
)
@timed("engagement_metrics")
async def get_post_engagement(
    post_id: str = Path(..., description="Post ID to get metrics for"),
    request_id: str = Depends(get_request_id),
    user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get post engagement metrics"""
    
    try:
        analytics_service = get_analytics_service()
        metrics = await analytics_service.get_post_analytics(post_id)
        
        if not metrics:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Post not found or no engagement data available"
            )
        
        logger.info(
            "Post engagement metrics retrieved",
            post_id=post_id,
            engagement_rate=metrics.engagement_rate,
            request_id=request_id
        )
        
        return {
            "post_id": metrics.post_id,
            "views": metrics.views,
            "likes": metrics.likes,
            "shares": metrics.shares,
            "comments": metrics.comments,
            "clicks": metrics.clicks,
            "engagement_rate": metrics.engagement_rate,
            "reach": metrics.reach,
            "impressions": metrics.impressions,
            "timestamp": metrics.timestamp.isoformat(),
            "request_id": request_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to get post engagement metrics",
            post_id=post_id,
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get post engagement metrics: {str(e)}"
        )


# Background task functions

async def save_ai_generated_post(post_data: Dict[str, Any], request_id: str):
    """Save AI-generated post to database"""
    try:
        db_manager = get_db_manager()
        post_repo = PostRepository(db_manager)
        
        await post_repo.create_post(post_data)
        
        logger.info(
            "AI-generated post saved to database",
            post_id=post_data["id"],
            request_id=request_id
        )
        
    except Exception as e:
        logger.error(
            "Failed to save AI-generated post",
            post_id=post_data.get("id"),
            error=str(e),
            request_id=request_id,
            exc_info=True
        )


# Export router
__all__ = ["router"]