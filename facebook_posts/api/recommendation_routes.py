"""
Recommendation API routes for Facebook Posts API
Intelligent content recommendations, personalization, and smart suggestions
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
from ..services.recommendation_service import (
    get_recommendation_service, RecommendationType, RecommendationPriority,
    PersonalizedRecommendation, ContentSuggestion
)
from ..infrastructure.cache import get_cache_manager
from ..infrastructure.monitoring import get_monitor, timed
from ..infrastructure.database import get_db_manager, PostRepository

logger = structlog.get_logger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/recommendations", tags=["Recommendations"])


# Content Recommendation Routes

@router.get(
    "/content",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Content recommendations generated successfully"},
        400: {"description": "Invalid request parameters"},
        500: {"description": "Recommendation generation error"}
    },
    summary="Get content recommendations",
    description="Get personalized content topic and suggestion recommendations"
)
@timed("content_recommendations")
async def get_content_recommendations(
    audience_type: str = Query(..., description="Target audience type"),
    content_type: Optional[str] = Query(None, description="Preferred content type"),
    limit: int = Query(5, description="Number of recommendations", ge=1, le=20),
    user_id: str = Query(..., description="User ID for personalization"),
    request_id: str = Depends(get_request_id),
    user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get personalized content recommendations"""
    
    # Validate audience type
    valid_audience_types = [audience_type.value for audience_type in AudienceType]
    if audience_type not in valid_audience_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid audience type. Valid types: {valid_audience_types}"
        )
    
    # Validate content type if provided
    if content_type:
        valid_content_types = [content_type.value for content_type in ContentType]
        if content_type not in valid_content_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid content type. Valid types: {valid_content_types}"
            )
    
    try:
        # Get recommendation service
        recommendation_service = get_recommendation_service()
        
        # Get content recommendations
        content_suggestions = await recommendation_service.content_recommender.recommend_content(
            user_id=user_id,
            audience_type=AudienceType(audience_type),
            content_type=ContentType(content_type) if content_type else None,
            limit=limit
        )
        
        # Convert to response format
        recommendations = []
        for suggestion in content_suggestions:
            recommendations.append({
                "topic": suggestion.topic,
                "content_type": suggestion.content_type.value,
                "audience_type": suggestion.audience_type.value,
                "suggested_content": suggestion.suggested_content,
                "confidence": suggestion.confidence,
                "expected_engagement": suggestion.expected_engagement,
                "tags": suggestion.tags,
                "hashtags": suggestion.hashtags,
                "metadata": suggestion.metadata
            })
        
        logger.info(
            "Content recommendations generated",
            user_id=user_id,
            audience_type=audience_type,
            content_type=content_type,
            recommendations_count=len(recommendations),
            request_id=request_id
        )
        
        return {
            "success": True,
            "recommendations": recommendations,
            "metadata": {
                "user_id": user_id,
                "audience_type": audience_type,
                "content_type": content_type,
                "total_recommendations": len(recommendations),
                "personalization_enabled": True
            },
            "request_id": request_id,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(
            "Content recommendations failed",
            user_id=user_id,
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Content recommendations failed: {str(e)}"
        )


@router.get(
    "/timing",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Timing recommendations generated successfully"},
        400: {"description": "Invalid request parameters"},
        500: {"description": "Timing recommendation error"}
    },
    summary="Get timing recommendations",
    description="Get optimal posting time recommendations based on audience and content type"
)
@timed("timing_recommendations")
async def get_timing_recommendations(
    audience_type: str = Query(..., description="Target audience type"),
    content_type: str = Query(..., description="Content type"),
    user_id: Optional[str] = Query(None, description="User ID for personalization"),
    request_id: str = Depends(get_request_id),
    user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get optimal timing recommendations"""
    
    # Validate audience type
    valid_audience_types = [audience_type.value for audience_type in AudienceType]
    if audience_type not in valid_audience_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid audience type. Valid types: {valid_audience_types}"
        )
    
    # Validate content type
    valid_content_types = [content_type.value for content_type in ContentType]
    if content_type not in valid_content_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid content type. Valid types: {valid_content_types}"
        )
    
    try:
        # Get recommendation service
        recommendation_service = get_recommendation_service()
        
        # Get timing recommendations
        timing_recommendations = await recommendation_service.timing_recommender.recommend_timing(
            audience_type=AudienceType(audience_type),
            content_type=ContentType(content_type),
            user_id=user_id
        )
        
        # Convert to response format
        recommendations = []
        for rec in timing_recommendations:
            recommendations.append({
                "id": rec.id,
                "type": rec.type.value,
                "priority": rec.priority.value,
                "title": rec.title,
                "description": rec.description,
                "suggestion": rec.suggestion,
                "expected_impact": rec.expected_impact,
                "confidence": rec.confidence,
                "implementation": rec.implementation,
                "metadata": rec.metadata
            })
        
        logger.info(
            "Timing recommendations generated",
            audience_type=audience_type,
            content_type=content_type,
            user_id=user_id,
            recommendations_count=len(recommendations),
            request_id=request_id
        )
        
        return {
            "success": True,
            "recommendations": recommendations,
            "metadata": {
                "audience_type": audience_type,
                "content_type": content_type,
                "user_id": user_id,
                "total_recommendations": len(recommendations),
                "personalization_enabled": user_id is not None
            },
            "request_id": request_id,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(
            "Timing recommendations failed",
            audience_type=audience_type,
            content_type=content_type,
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Timing recommendations failed: {str(e)}"
        )


@router.post(
    "/hashtags",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Hashtag recommendations generated successfully"},
        400: {"description": "Invalid request parameters"},
        500: {"description": "Hashtag recommendation error"}
    },
    summary="Get hashtag recommendations",
    description="Get relevant hashtag recommendations for content"
)
@timed("hashtag_recommendations")
async def get_hashtag_recommendations(
    content: str = Query(..., description="Content to analyze for hashtags", min_length=10),
    audience_type: str = Query(..., description="Target audience type"),
    content_type: str = Query(..., description="Content type"),
    limit: int = Query(5, description="Number of hashtag recommendations", ge=1, le=15),
    request_id: str = Depends(get_request_id),
    user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get hashtag recommendations for content"""
    
    if len(content) > 5000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Content too long for hashtag analysis (max 5000 characters)"
        )
    
    # Validate audience type
    valid_audience_types = [audience_type.value for audience_type in AudienceType]
    if audience_type not in valid_audience_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid audience type. Valid types: {valid_audience_types}"
        )
    
    # Validate content type
    valid_content_types = [content_type.value for content_type in ContentType]
    if content_type not in valid_content_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid content type. Valid types: {valid_content_types}"
        )
    
    try:
        # Get recommendation service
        recommendation_service = get_recommendation_service()
        
        # Get hashtag recommendations
        hashtag_recommendations = await recommendation_service.hashtag_recommender.recommend_hashtags(
            content=content,
            audience_type=AudienceType(audience_type),
            content_type=ContentType(content_type),
            limit=limit
        )
        
        # Convert to response format
        recommendations = []
        for rec in hashtag_recommendations:
            recommendations.append({
                "id": rec.id,
                "type": rec.type.value,
                "priority": rec.priority.value,
                "title": rec.title,
                "description": rec.description,
                "suggestion": rec.suggestion,
                "expected_impact": rec.expected_impact,
                "confidence": rec.confidence,
                "implementation": rec.implementation,
                "metadata": rec.metadata
            })
        
        logger.info(
            "Hashtag recommendations generated",
            content_length=len(content),
            audience_type=audience_type,
            content_type=content_type,
            recommendations_count=len(recommendations),
            request_id=request_id
        )
        
        return {
            "success": True,
            "recommendations": recommendations,
            "metadata": {
                "content_length": len(content),
                "audience_type": audience_type,
                "content_type": content_type,
                "total_recommendations": len(recommendations)
            },
            "request_id": request_id,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(
            "Hashtag recommendations failed",
            content_length=len(content),
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hashtag recommendations failed: {str(e)}"
        )


@router.get(
    "/comprehensive",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Comprehensive recommendations generated successfully"},
        400: {"description": "Invalid request parameters"},
        500: {"description": "Comprehensive recommendation error"}
    },
    summary="Get comprehensive recommendations",
    description="Get comprehensive personalized recommendations including content, timing, and hashtags"
)
@timed("comprehensive_recommendations")
async def get_comprehensive_recommendations(
    user_id: str = Query(..., description="User ID for personalization"),
    audience_type: str = Query(..., description="Target audience type"),
    content_type: Optional[str] = Query(None, description="Preferred content type"),
    limit: int = Query(10, description="Number of recommendations", ge=1, le=25),
    request_id: str = Depends(get_request_id),
    user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get comprehensive personalized recommendations"""
    
    # Validate audience type
    valid_audience_types = [audience_type.value for audience_type in AudienceType]
    if audience_type not in valid_audience_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid audience type. Valid types: {valid_audience_types}"
        )
    
    # Validate content type if provided
    if content_type:
        valid_content_types = [content_type.value for content_type in ContentType]
        if content_type not in valid_content_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid content type. Valid types: {valid_content_types}"
            )
    
    try:
        # Get recommendation service
        recommendation_service = get_recommendation_service()
        
        # Get comprehensive recommendations
        personalized_recommendations = await recommendation_service.get_comprehensive_recommendations(
            user_id=user_id,
            audience_type=AudienceType(audience_type),
            content_type=ContentType(content_type) if content_type else None,
            limit=limit
        )
        
        # Convert to response format
        recommendations = []
        for rec in personalized_recommendations.recommendations:
            recommendations.append({
                "id": rec.id,
                "type": rec.type.value,
                "priority": rec.priority.value,
                "title": rec.title,
                "description": rec.description,
                "suggestion": rec.suggestion,
                "expected_impact": rec.expected_impact,
                "confidence": rec.confidence,
                "implementation": rec.implementation,
                "metadata": rec.metadata,
                "created_at": rec.created_at.isoformat()
            })
        
        logger.info(
            "Comprehensive recommendations generated",
            user_id=user_id,
            audience_type=audience_type,
            content_type=content_type,
            recommendations_count=len(recommendations),
            personalization_score=personalized_recommendations.personalization_score,
            request_id=request_id
        )
        
        return {
            "success": True,
            "user_id": personalized_recommendations.user_id,
            "personalization_score": personalized_recommendations.personalization_score,
            "recommendations": recommendations,
            "based_on": personalized_recommendations.based_on,
            "metadata": {
                "audience_type": audience_type,
                "content_type": content_type,
                "total_recommendations": len(recommendations),
                "personalization_enabled": True,
                "generation_time": datetime.now().isoformat()
            },
            "request_id": request_id,
            "generated_at": personalized_recommendations.generated_at.isoformat()
        }
        
    except Exception as e:
        logger.error(
            "Comprehensive recommendations failed",
            user_id=user_id,
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Comprehensive recommendations failed: {str(e)}"
        )


@router.get(
    "/trending",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Trending topics retrieved successfully"},
        400: {"description": "Invalid request parameters"},
        500: {"description": "Trending topics error"}
    },
    summary="Get trending topics",
    description="Get trending topics and hashtags for content inspiration"
)
@timed("trending_topics")
async def get_trending_topics(
    category: Optional[str] = Query(None, description="Topic category filter"),
    audience_type: Optional[str] = Query(None, description="Audience type filter"),
    limit: int = Query(10, description="Number of trending topics", ge=1, le=50),
    request_id: str = Depends(get_request_id),
    user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get trending topics and hashtags"""
    
    # Validate category if provided
    valid_categories = ["technology", "business", "lifestyle", "education"]
    if category and category not in valid_categories:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid category. Valid categories: {valid_categories}"
        )
    
    # Validate audience type if provided
    if audience_type:
        valid_audience_types = [audience_type.value for audience_type in AudienceType]
        if audience_type not in valid_audience_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid audience type. Valid types: {valid_audience_types}"
            )
    
    try:
        # Get recommendation service
        recommendation_service = get_recommendation_service()
        
        # Get trending topics
        trending_topics = recommendation_service.content_recommender.trending_topics
        
        # Filter by category if specified
        if category:
            topics = trending_topics.get(category, [])
        else:
            # Combine all categories
            topics = []
            for cat_topics in trending_topics.values():
                topics.extend(cat_topics)
        
        # Filter by audience type if specified
        if audience_type:
            audience_topics = recommendation_service.content_recommender._get_trending_topics_for_audience(
                AudienceType(audience_type)
            )
            topics = [topic for topic in topics if topic in audience_topics]
        
        # Limit results
        topics = topics[:limit]
        
        # Get trending hashtags
        trending_hashtags = recommendation_service.hashtag_recommender.trending_hashtags
        
        # Filter hashtags by category if specified
        if category:
            hashtags = trending_hashtags.get(category, [])
        else:
            hashtags = []
            for cat_hashtags in trending_hashtags.values():
                hashtags.extend(cat_hashtags)
        
        # Limit hashtags
        hashtags = hashtags[:limit]
        
        logger.info(
            "Trending topics retrieved",
            category=category,
            audience_type=audience_type,
            topics_count=len(topics),
            hashtags_count=len(hashtags),
            request_id=request_id
        )
        
        return {
            "success": True,
            "trending_topics": topics,
            "trending_hashtags": hashtags,
            "metadata": {
                "category": category,
                "audience_type": audience_type,
                "topics_count": len(topics),
                "hashtags_count": len(hashtags),
                "last_updated": datetime.now().isoformat()
            },
            "request_id": request_id,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(
            "Trending topics retrieval failed",
            category=category,
            audience_type=audience_type,
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Trending topics retrieval failed: {str(e)}"
        )


@router.get(
    "/personalization/{user_id}",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Personalization data retrieved successfully"},
        404: {"description": "User not found"},
        500: {"description": "Personalization data error"}
    },
    summary="Get user personalization data",
    description="Get user's personalization preferences and history for recommendations"
)
@timed("personalization_data")
async def get_personalization_data(
    user_id: str = Path(..., description="User ID to get personalization data for"),
    request_id: str = Depends(get_request_id),
    user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get user personalization data"""
    
    try:
        # Get recommendation service
        recommendation_service = get_recommendation_service()
        
        # Get user preferences
        user_preferences = await recommendation_service.content_recommender._get_user_preferences(user_id)
        
        # Get user content history
        user_history = await recommendation_service.content_recommender._get_user_content_history(user_id)
        
        # Get personalized timing data
        personalized_timing = await recommendation_service.timing_recommender._get_user_posting_history(user_id)
        
        logger.info(
            "Personalization data retrieved",
            user_id=user_id,
            preferences_count=len(user_preferences),
            history_count=len(user_history),
            timing_count=len(personalized_timing),
            request_id=request_id
        )
        
        return {
            "success": True,
            "user_id": user_id,
            "preferences": user_preferences,
            "content_history": user_history,
            "posting_history": personalized_timing,
            "metadata": {
                "preferences_available": len(user_preferences) > 0,
                "history_available": len(user_history) > 0,
                "timing_data_available": len(personalized_timing) > 0,
                "personalization_score": len(user_preferences) / 10.0  # Simple score
            },
            "request_id": request_id,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(
            "Personalization data retrieval failed",
            user_id=user_id,
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Personalization data retrieval failed: {str(e)}"
        )


# Export router
__all__ = ["router"]






























