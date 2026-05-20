"""
Bookmark routes for Lovable Community SAM3.
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from sqlalchemy.orm import Session
from typing import Dict, Any
import logging

from ...database import get_db_session
from ...services.bookmark_service import BookmarkService
from ...utils.pagination import calculate_pagination_metadata

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/bookmarks", tags=["bookmarks"])


@router.post("/chats/{chat_id}")
async def create_bookmark(
    chat_id: str,
    user_id: str = Query(..., description="User ID"),
    db: Session = Depends(get_db_session)
):
    """Bookmark a chat."""
    from ...exceptions import NotFoundError, ConflictError
    
    bookmark_service = BookmarkService(db)
    result = bookmark_service.create_bookmark(user_id, chat_id)
    
    from ...utils.serializers import serialize_model
    
    return {
        "message": "Chat bookmarked successfully",
        "bookmark": serialize_model(result["bookmark"])
    }


@router.delete("/chats/{chat_id}")
async def delete_bookmark(
    chat_id: str,
    user_id: str = Query(..., description="User ID"),
    db: Session = Depends(get_db_session)
):
    """Remove a bookmark."""
    from ...exceptions import NotFoundError
    
    bookmark_service = BookmarkService(db)
    success = bookmark_service.delete_bookmark(user_id, chat_id)
    
    if not success:
        raise NotFoundError("Bookmark", f"{user_id}:{chat_id}")
    
    return {"message": "Bookmark removed successfully", "chat_id": chat_id}


@router.get("/users/{user_id}", response_model=Dict[str, Any])
async def get_user_bookmarks(
    user_id: str,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Page size"),
    db: Session = Depends(get_db_session)
):
    """Get bookmarks for a user."""
    bookmark_service = BookmarkService(db)
    bookmarks, total = bookmark_service.get_user_bookmarks(user_id, page=page, page_size=page_size)
    
    pagination = calculate_pagination_metadata(page, page_size, total)
    
    return {
        "bookmarks": bookmarks,
        **pagination
    }


@router.get("/chats/{chat_id}/check")
async def check_bookmark(
    chat_id: str,
    user_id: str = Query(..., description="User ID"),
    db: Session = Depends(get_db_session)
):
    """Check if a chat is bookmarked by user."""
    bookmark_service = BookmarkService(db)
    is_bookmarked = bookmark_service.is_bookmarked(user_id, chat_id)
    
    return {
        "is_bookmarked": is_bookmarked,
        "chat_id": chat_id,
        "user_id": user_id
    }


@router.get("/chats/{chat_id}/count")
async def get_bookmark_count(
    chat_id: str,
    db: Session = Depends(get_db_session)
):
    """Get bookmark count for a chat."""
    bookmark_service = BookmarkService(db)
    count = bookmark_service.get_bookmark_count(chat_id)
    
    return {
        "chat_id": chat_id,
        "bookmark_count": count
    }







