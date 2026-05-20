"""
Analytics API Routes
====================

Endpoints para analytics y métricas.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Optional
from pydantic import BaseModel

router = APIRouter(prefix="/analytics", tags=["analytics"])


class AnalyticsResponse(BaseModel):
    platform: str
    total_posts: int
    total_engagement: int
    average_engagement_rate: float


def get_analytics_service():
    """Dependency para obtener AnalyticsService"""
    from ...services.analytics_service import AnalyticsService
    return AnalyticsService()


@router.get("/platform/{platform}", response_model=dict)
async def get_platform_analytics(
    platform: str,
    days: int = Query(7, ge=1, le=365),
    service = Depends(get_analytics_service)
):
    """Obtener analytics de una plataforma"""
    try:
        analytics = service.get_platform_analytics(platform, days)
        return analytics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/post/{post_id}", response_model=dict)
async def get_post_analytics(
    post_id: str,
    platform: Optional[str] = Query(None),
    service = Depends(get_analytics_service)
):
    """Obtener analytics de un post específico"""
    try:
        analytics = service.get_post_analytics(post_id, platform)
        return analytics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/best-performing", response_model=list)
async def get_best_performing_posts(
    platform: Optional[str] = Query(None),
    limit: int = Query(10, ge=1, le=100),
    service = Depends(get_analytics_service)
):
    """Obtener posts con mejor performance"""
    try:
        posts = service.get_best_performing_posts(platform, limit)
        return posts
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trends/{platform}", response_model=dict)
async def get_engagement_trends(
    platform: str,
    days: int = Query(30, ge=1, le=365),
    service = Depends(get_analytics_service)
):
    """Obtener tendencias de engagement"""
    try:
        trends = service.get_engagement_trends(platform, days)
        return trends
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




