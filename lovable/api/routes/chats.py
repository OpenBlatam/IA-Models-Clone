"""
Chat routes for Lovable Community SAM3
========================================
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
import logging

from ...schemas.responses import ChatResponse
from ...services.chat_service import ChatService
from ...services.ranking_service import RankingService
from ...database import get_db_session
from ...repositories.chat_repository import ChatRepository
from ...models.published_chat import PublishedChat
from ...exceptions import NotFoundError, AuthorizationError
from ...utils.pagination import calculate_pagination_metadata

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/chats", tags=["chats"])


@router.get("/", response_model=Dict[str, Any])
async def list_chats(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Page size"),
    sort_by: str = Query("score", description="Sort by: score, created_at, vote_count"),
    order: str = Query("desc", pattern="^(asc|desc)$", description="Sort order"),
    category: Optional[str] = Query(None, description="Filter by category"),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    featured: Optional[bool] = Query(None, description="Filter featured chats"),
    db: Session = Depends(get_db_session)
):
    """List chats with pagination and filtering."""
    chat_service = ChatService(db)
    return chat_service.list_chats(
        page=page,
        page_size=page_size,
        sort_by=sort_by,
        order=order,
        category=category,
        user_id=user_id,
        featured=featured
    )


@router.get("/{chat_id}", response_model=Dict[str, Any])
async def get_chat(
    chat_id: str,
    db: Session = Depends(get_db_session)
):
    """Get a specific chat by ID."""
    chat_service = ChatService(db)
    return chat_service.get_chat(chat_id)


@router.get("/{chat_id}/stats", response_model=Dict[str, Any])
async def get_chat_stats(
    chat_id: str,
    detailed: bool = Query(False, description="Include detailed statistics"),
    db: Session = Depends(get_db_session)
):
    """Get statistics for a specific chat."""
    chat_service = ChatService(db)
    return chat_service.get_chat_stats(chat_id, detailed=detailed)


@router.get("/{chat_id}/stats/detailed", response_model=Dict[str, Any])
async def get_chat_stats_detailed(
    chat_id: str,
    db: Session = Depends(get_db_session)
):
    """Get detailed statistics for a specific chat."""
    chat_service = ChatService(db)
    stats = chat_service.get_chat_with_stats(chat_id)
    
    if not stats:
        raise NotFoundError("Chat", chat_id)
    
    return stats


@router.get("/{chat_id}/remixes", response_model=List[Dict[str, Any]])
async def get_chat_remixes(
    chat_id: str,
    limit: int = Query(20, ge=1, le=100, description="Limit results"),
    db: Session = Depends(get_db_session)
):
    """Get all remixes of a chat."""
    chat_service = ChatService(db)
    return chat_service.get_chat_remixes(chat_id, limit=limit)


@router.get("/search/query", response_model=Dict[str, Any])
async def search_chats(
    q: str = Query(..., min_length=1, description="Search query"),
    tags: Optional[str] = Query(None, description="Comma-separated tags"),
    category: Optional[str] = Query(None, description="Filter by category"),
    sort_by: str = Query("relevance", pattern="^(relevance|score|created_at|trending)$", description="Sort by"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db_session)
):
    """Search chats by query, tags, or category - Enhanced with relevance scoring."""
    from ...services.search_service import SearchService
    from ...utils.security import sanitize_input
    
    # Sanitize search query
    q = sanitize_input(q) if q else ""
    
    search_service = SearchService(db)
    
    # Parse tags
    tag_list = [t.strip() for t in tags.split(",")] if tags else None
    
    result = search_service.search(
        query=q,
        tags=tag_list,
        category=category,
        sort_by=sort_by,
        page=page,
        page_size=page_size
    )
    
    return result


@router.get("/top/ranked", response_model=List[Dict[str, Any]])
async def get_top_chats(
    limit: int = Query(20, ge=1, le=100, description="Number of top chats"),
    category: Optional[str] = Query(None, description="Filter by category"),
    db: Session = Depends(get_db_session)
):
    """Get top ranked chats."""
    chat_service = ChatService(db)
    return chat_service.get_top_chats(limit=limit, category=category)


@router.get("/trending/now", response_model=List[Dict[str, Any]])
async def get_trending_chats(
    period: str = Query("day", pattern="^(hour|day|week|month)$", description="Trending period"),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db_session)
):
    """Get trending chats for a specific period."""
    chat_service = ChatService(db)
    return chat_service.get_trending_chats(period=period, limit=limit)


@router.get("/featured", response_model=Dict[str, Any])
async def get_featured_chats(
    limit: int = Query(50, ge=1, le=100, description="Limit results"),
    db: Session = Depends(get_db_session)
):
    """Get all featured chats ordered by score."""
    chat_service = ChatService(db)
    return chat_service.get_featured_chats(limit=limit)


@router.put("/{chat_id}", response_model=Dict[str, Any])
async def update_chat(
    chat_id: str,
    request: "UpdateChatRequest",
    db: Session = Depends(get_db_session)
):
    """Update a chat."""
    from ...schemas.requests import UpdateChatRequest
    from ...utils.security import sanitize_input
    from ...constants import MAX_TITLE_LENGTH, MAX_DESCRIPTION_LENGTH
    
    # Build update data with sanitization
    update_data = {}
    if request.title is not None:
        update_data["title"] = sanitize_input(request.title, max_length=MAX_TITLE_LENGTH)
    if request.content is not None:
        update_data["chat_content"] = sanitize_input(request.content)
    if request.description is not None:
        update_data["description"] = sanitize_input(request.description, max_length=MAX_DESCRIPTION_LENGTH)
    if request.tags is not None:
        # Sanitize each tag
        if isinstance(request.tags, list):
            sanitized_tags = [sanitize_input(tag, max_length=50) for tag in request.tags if tag]
            update_data["tags"] = ",".join(sanitized_tags) if sanitized_tags else None
        else:
            update_data["tags"] = sanitize_input(str(request.tags))
    if request.category is not None:
        update_data["category"] = sanitize_input(request.category, max_length=50)
    if request.is_public is not None:
        update_data["is_public"] = request.is_public
    
    chat_service = ChatService(db)
    return chat_service.update_chat(chat_id, update_data)


@router.delete("/{chat_id}")
async def delete_chat(
    chat_id: str,
    user_id: str = Query(..., description="User ID (for authorization)"),
    db: Session = Depends(get_db_session)
):
    """Delete a chat (only by owner)."""
    chat_service = ChatService(db)
    return chat_service.delete_chat(chat_id, user_id)


@router.post("/{chat_id}/feature", response_model=Dict[str, Any])
async def feature_chat(
    chat_id: str,
    featured: bool = Query(..., description="Whether to feature the chat"),
    db: Session = Depends(get_db_session)
):
    """Feature or unfeature a chat."""
    chat_service = ChatService(db)
    result = chat_service.feature_chat(chat_id, featured)
    
    # Send notification if featured (if NotificationService exists)
    if featured and result:
        try:
            from ...services.notification_service import NotificationService
            notification_service = NotificationService(db=db)
            # Get chat to extract user_id and title
            chat = chat_service.chat_repo.get_by_id(chat_id)
            if chat:
                # Check if method is async
                import inspect
                if inspect.iscoroutinefunction(notification_service.notify_chat_featured):
                    await notification_service.notify_chat_featured(
                        chat_id=chat_id,
                        user_id=chat.user_id,
                        title=chat.title
                    )
                else:
                    notification_service.notify_chat_featured(
                        chat_id=chat_id,
                        user_id=chat.user_id,
                        title=chat.title
                    )
        except (ImportError, AttributeError):
            # NotificationService not available, skip notification
            pass
    
    return result


# Keep PATCH endpoint for backward compatibility
@router.patch("/{chat_id}/feature", response_model=Dict[str, Any])
async def feature_chat_patch(
    chat_id: str,
    request: "FeatureChatRequest",
    db: Session = Depends(get_db_session)
):
    """Feature or unfeature a chat - PATCH version."""
    from ...schemas.requests import FeatureChatRequest
    
    chat_service = ChatService(db)
    return chat_service.feature_chat(chat_id, request.is_featured)


@router.get("/users/{user_id}", response_model=Dict[str, Any])
async def get_user_chats(
    user_id: str,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Page size"),
    db: Session = Depends(get_db_session)
):
    """Get all chats by a user."""
    chat_service = ChatService(db)
    return chat_service.get_user_chats(user_id, page=page, page_size=page_size)


@router.post("/batch", response_model=Dict[str, Any])
async def batch_operations(
    request: "BatchOperationRequest",
    user_id: str = Query(..., description="User ID (for authorization)"),
    db: Session = Depends(get_db_session)
):
    """Perform batch operations on multiple chats."""
    from ...schemas.requests import BatchOperationRequest
    
    chat_repo = ChatRepository(db)
    results = {"success": 0, "failed": 0, "errors": []}
    
    # Verify ownership for all chats
    for chat_id in request.chat_ids:
        chat = chat_repo.get_by_id(chat_id)
        if not chat:
            results["failed"] += 1
            results["errors"].append(f"Chat {chat_id} not found")
            continue
        
        if chat.user_id != user_id:
            results["failed"] += 1
            results["errors"].append(f"Not authorized for chat {chat_id}")
            continue
    
    if results["failed"] > 0:
        raise AuthorizationError(results["errors"][0])
    
    # Perform operation
    if request.operation == "delete":
        deleted = chat_repo.batch_delete(request.chat_ids)
        results["success"] = deleted
    elif request.operation == "feature":
        updated = chat_repo.batch_feature(request.chat_ids, True)
        results["success"] = updated
    elif request.operation == "unfeature":
        updated = chat_repo.batch_feature(request.chat_ids, False)
        results["success"] = updated
    elif request.operation == "publish":
        updated = chat_repo.db.query(PublishedChat).filter(
            PublishedChat.id.in_(request.chat_ids)
        ).update({"is_public": True}, synchronize_session=False)
        chat_repo.db.commit()
        results["success"] = updated
    elif request.operation == "unpublish":
        updated = chat_repo.db.query(PublishedChat).filter(
            PublishedChat.id.in_(request.chat_ids)
        ).update({"is_public": False}, synchronize_session=False)
        chat_repo.db.commit()
        results["success"] = updated
    
    return {
        "operation": request.operation,
        "total": len(request.chat_ids),
        **results
    }


@router.get("/feed/personalized", response_model=Dict[str, Any])
async def get_personalized_feed(
    user_id: str = Query(..., description="User ID"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Page size"),
    db: Session = Depends(get_db_session)
):
    """Get personalized feed based on followed users."""
    chat_service = ChatService(db)
    return chat_service.get_personalized_feed(user_id, page=page, page_size=page_size)








