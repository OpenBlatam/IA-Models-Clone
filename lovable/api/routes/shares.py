"""
Share routes for Lovable Community SAM3.
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from sqlalchemy.orm import Session
from typing import Dict, Any
import logging

from ...database import get_db_session
from ...services.share_service import ShareService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/shares", tags=["shares"])


@router.post("/content")
async def share_content(
    content_type: str = Query(..., pattern="^(chat|comment)$", description="Type of content"),
    content_id: str = Query(..., description="ID of content to share"),
    platform: str = Query(..., description="Sharing platform"),
    user_id: str = Query(..., description="User ID sharing"),
    db: Session = Depends(get_db_session)
):
    """Share content on a platform."""
    share_service = ShareService(db)
    result = share_service.share_content(user_id, content_type, content_id, platform)
    
    from ...utils.serializers import serialize_model
    
    return {
        "message": "Content shared successfully",
        "share": serialize_model(result["share"])
    }


@router.get("/content/{content_type}/{content_id}", response_model=Dict[str, Any])
async def get_content_shares(
    content_type: str,
    content_id: str,
    db: Session = Depends(get_db_session)
):
    """Get shares for specific content."""
    share_service = ShareService(db)
    result = share_service.get_content_shares(content_type, content_id)
    
    return {
        **result,
        "content_type": content_type,
        "content_id": content_id
    }


@router.get("/users/{user_id}", response_model=Dict[str, Any])
async def get_user_shares(
    user_id: str,
    limit: int = Query(50, ge=1, le=100, description="Limit results"),
    db: Session = Depends(get_db_session)
):
    """Get shares by a user."""
    share_service = ShareService(db)
    shares = share_service.get_user_shares(user_id, limit=limit)
    
    return {
        "shares": shares,
        "user_id": user_id,
        "count": len(shares)
    }


@router.get("/content/{content_type}/{content_id}/stats", response_model=Dict[str, Any])
async def get_share_stats(
    content_type: str,
    content_id: str,
    db: Session = Depends(get_db_session)
):
    """Get share statistics for content."""
    share_service = ShareService(db)
    stats = share_service.get_share_stats(content_type, content_id)
    
    return {
        **stats,
        "content_type": content_type,
        "content_id": content_id
    }







