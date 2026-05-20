"""
Tag routes for Lovable Community SAM3.
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from sqlalchemy.orm import Session
from typing import Dict, Any
import logging

from ...database import get_db_session
from ...services.tag_service import TagService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/tags", tags=["tags"])


@router.get("/popular", response_model=Dict[str, Any])
async def get_popular_tags(
    limit: int = Query(50, ge=1, le=200, description="Limit results"),
    min_usage: int = Query(1, ge=1, description="Minimum usage count"),
    db: Session = Depends(get_db_session)
):
    """Get popular tags with usage statistics."""
    tag_service = TagService(db)
    return tag_service.get_popular_tags(limit=limit, min_usage=min_usage)


@router.get("/trending", response_model=Dict[str, Any])
async def get_trending_tags(
    period: str = Query("day", pattern="^(hour|day|week|month)$", description="Trending period"),
    limit: int = Query(20, ge=1, le=100, description="Limit results"),
    db: Session = Depends(get_db_session)
):
    """Get trending tags for a specific period."""
    tag_service = TagService(db)
    return tag_service.get_trending_tags(period=period, limit=limit)


@router.get("/{tag_name}/stats", response_model=Dict[str, Any])
async def get_tag_stats(
    tag_name: str,
    db: Session = Depends(get_db_session)
):
    """Get statistics for a specific tag."""
    from ...exceptions import NotFoundError
    
    tag_service = TagService(db)
    stats = tag_service.get_tag_stats(tag_name)
    
    if not stats:
        raise NotFoundError("Tag", tag_name)
    
    return stats


@router.get("/{tag_name}/chats", response_model=Dict[str, Any])
async def get_tag_chats(
    tag_name: str,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Page size"),
    sort_by: str = Query("score", pattern="^(score|created_at|vote_count)$", description="Sort by"),
    db: Session = Depends(get_db_session)
):
    """Get chats with a specific tag."""
    tag_service = TagService(db)
    chats, total = tag_service.get_tag_chats(
        tag_name=tag_name,
        page=page,
        page_size=page_size,
        sort_by=sort_by
    )
    
    from ...utils.serializers import serialize_list
    
    return {
        "chats": serialize_list(chats),
        "tag": tag_name,
        "page": page,
        "page_size": page_size,
        "total": total,
        "total_pages": (total + page_size - 1) // page_size if total > 0 else 0
    }







