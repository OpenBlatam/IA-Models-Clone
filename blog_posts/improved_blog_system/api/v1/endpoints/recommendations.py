"""
Recommendation API endpoints
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel, Field

from ....services.recommendation_service import RecommendationService
from ....api.dependencies import CurrentUserDep, DatabaseSessionDep
from ....core.exceptions import DatabaseError

router = APIRouter()


class RecommendationRequest(BaseModel):
    """Request model for recommendations."""
    limit: int = Field(default=10, ge=1, le=50, description="Number of recommendations")
    algorithm: str = Field(default="hybrid", description="Recommendation algorithm")


class SimilarPostsRequest(BaseModel):
    """Request model for similar posts."""
    post_id: int = Field(..., description="Post ID to find similar posts for")
    limit: int = Field(default=5, ge=1, le=20, description="Number of similar posts")


class TrendingRequest(BaseModel):
    """Request model for trending posts."""
    limit: int = Field(default=10, ge=1, le=50, description="Number of trending posts")
    days: int = Field(default=7, ge=1, le=30, description="Number of days to look back")


async def get_recommendation_service(session: DatabaseSessionDep) -> RecommendationService:
    """Get recommendation service instance."""
    return RecommendationService(session)


@router.get("/personalized", response_model=Dict[str, Any])
async def get_personalized_recommendations(
    request: RecommendationRequest = Depends(),
    recommendation_service: RecommendationService = Depends(get_recommendation_service),
    current_user: CurrentUserDep = Depends()
):
    """Get personalized content recommendations for the current user."""
    try:
        recommendations = await recommendation_service.get_personalized_recommendations(
            user_id=str(current_user.id),
            limit=request.limit,
            algorithm=request.algorithm
        )
        
        return {
            "success": True,
            "data": {
                "recommendations": recommendations,
                "algorithm": request.algorithm,
                "total": len(recommendations)
            },
            "message": "Personalized recommendations retrieved successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get personalized recommendations"
        )


@router.get("/similar/{post_id}", response_model=Dict[str, Any])
async def get_similar_posts(
    post_id: int,
    limit: int = Query(default=5, ge=1, le=20, description="Number of similar posts"),
    recommendation_service: RecommendationService = Depends(get_recommendation_service),
    current_user: CurrentUserDep = Depends()
):
    """Get posts similar to a specific post."""
    try:
        similar_posts = await recommendation_service.get_similar_posts(
            post_id=post_id,
            limit=limit
        )
        
        return {
            "success": True,
            "data": {
                "similar_posts": similar_posts,
                "original_post_id": post_id,
                "total": len(similar_posts)
            },
            "message": "Similar posts retrieved successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get similar posts"
        )


@router.get("/trending", response_model=Dict[str, Any])
async def get_trending_posts(
    limit: int = Query(default=10, ge=1, le=50, description="Number of trending posts"),
    days: int = Query(default=7, ge=1, le=30, description="Number of days to look back"),
    recommendation_service: RecommendationService = Depends(get_recommendation_service),
    current_user: CurrentUserDep = Depends()
):
    """Get trending posts based on recent activity."""
    try:
        trending_posts = await recommendation_service.get_trending_posts(
            limit=limit,
            days=days
        )
        
        return {
            "success": True,
            "data": {
                "trending_posts": trending_posts,
                "period_days": days,
                "total": len(trending_posts)
            },
            "message": "Trending posts retrieved successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get trending posts"
        )


@router.get("/authors", response_model=Dict[str, Any])
async def get_author_recommendations(
    limit: int = Query(default=5, ge=1, le=20, description="Number of author recommendations"),
    recommendation_service: RecommendationService = Depends(get_recommendation_service),
    current_user: CurrentUserDep = Depends()
):
    """Get author recommendations based on user preferences."""
    try:
        author_recommendations = await recommendation_service.get_author_recommendations(
            user_id=str(current_user.id),
            limit=limit
        )
        
        return {
            "success": True,
            "data": {
                "author_recommendations": author_recommendations,
                "total": len(author_recommendations)
            },
            "message": "Author recommendations retrieved successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get author recommendations"
        )


@router.get("/categories", response_model=Dict[str, Any])
async def get_category_recommendations(
    limit: int = Query(default=5, ge=1, le=20, description="Number of category recommendations"),
    recommendation_service: RecommendationService = Depends(get_recommendation_service),
    current_user: CurrentUserDep = Depends()
):
    """Get category recommendations based on user preferences."""
    try:
        category_recommendations = await recommendation_service.get_category_recommendations(
            user_id=str(current_user.id),
            limit=limit
        )
        
        return {
            "success": True,
            "data": {
                "category_recommendations": category_recommendations,
                "total": len(category_recommendations)
            },
            "message": "Category recommendations retrieved successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get category recommendations"
        )


@router.get("/feed", response_model=Dict[str, Any])
async def get_personalized_feed(
    limit: int = Query(default=20, ge=1, le=100, description="Number of posts in feed"),
    algorithm: str = Query(default="hybrid", description="Recommendation algorithm"),
    recommendation_service: RecommendationService = Depends(get_recommendation_service),
    current_user: CurrentUserDep = Depends()
):
    """Get personalized content feed for the user."""
    try:
        # Get personalized recommendations
        recommendations = await recommendation_service.get_personalized_recommendations(
            user_id=str(current_user.id),
            limit=limit,
            algorithm=algorithm
        )
        
        # Get trending posts as fallback
        trending_posts = await recommendation_service.get_trending_posts(
            limit=limit // 2,
            days=7
        )
        
        # Combine and deduplicate
        feed_posts = []
        seen_ids = set()
        
        # Add recommendations first
        for rec in recommendations:
            if rec["id"] not in seen_ids:
                feed_posts.append(rec)
                seen_ids.add(rec["id"])
        
        # Add trending posts if needed
        for post in trending_posts:
            if post["id"] not in seen_ids and len(feed_posts) < limit:
                feed_posts.append(post)
                seen_ids.add(post["id"])
        
        return {
            "success": True,
            "data": {
                "feed_posts": feed_posts,
                "algorithm": algorithm,
                "total": len(feed_posts),
                "recommendations_count": len(recommendations),
                "trending_count": len(trending_posts)
            },
            "message": "Personalized feed retrieved successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get personalized feed"
        )


@router.get("/algorithms", response_model=Dict[str, Any])
async def get_available_algorithms():
    """Get list of available recommendation algorithms."""
    algorithms = [
        {
            "name": "hybrid",
            "description": "Combines collaborative filtering and content-based filtering",
            "recommended": True
        },
        {
            "name": "collaborative",
            "description": "Based on user behavior similarity",
            "recommended": False
        },
        {
            "name": "content_based",
            "description": "Based on content similarity",
            "recommended": False
        },
        {
            "name": "popularity",
            "description": "Based on overall popularity",
            "recommended": False
        }
    ]
    
    return {
        "success": True,
        "data": {
            "algorithms": algorithms,
            "total": len(algorithms)
        },
        "message": "Available algorithms retrieved successfully"
    }


@router.get("/stats", response_model=Dict[str, Any])
async def get_recommendation_stats(
    recommendation_service: RecommendationService = Depends(get_recommendation_service),
    current_user: CurrentUserDep = Depends()
):
    """Get recommendation statistics for the current user."""
    try:
        # Get user's interaction history
        user_interactions = await recommendation_service._get_user_interactions(str(current_user.id))
        
        # Get user's liked posts count
        liked_posts_count = len([i for i in user_interactions if i["type"] == "like"])
        
        # Get user's commented posts count
        commented_posts_count = len([i for i in user_interactions if i["type"] == "comment"])
        
        # Get category preferences
        category_preferences = {}
        for interaction in user_interactions:
            category = interaction.get("category")
            if category:
                category_preferences[category] = category_preferences.get(category, 0) + 1
        
        # Get tag preferences
        tag_preferences = {}
        for interaction in user_interactions:
            tags = interaction.get("tags", [])
            for tag in tags:
                tag_preferences[tag] = tag_preferences.get(tag, 0) + 1
        
        stats = {
            "total_interactions": len(user_interactions),
            "liked_posts": liked_posts_count,
            "commented_posts": commented_posts_count,
            "category_preferences": category_preferences,
            "top_tags": dict(sorted(tag_preferences.items(), key=lambda x: x[1], reverse=True)[:10]),
            "recommendation_quality": "high" if len(user_interactions) > 10 else "medium" if len(user_interactions) > 5 else "low"
        }
        
        return {
            "success": True,
            "data": stats,
            "message": "Recommendation statistics retrieved successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get recommendation statistics"
        )






























