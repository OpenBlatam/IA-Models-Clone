"""
Response helper functions

Utility functions for building API responses with common patterns.
"""

from typing import List, Optional, Dict
from ..models import PublishedChat, ChatVote
from ..schemas import ChatListResponse, PublishedChatResponse
from ..services import ChatService
from .converters import chats_to_responses
from .pagination import calculate_pagination_metadata
from .validation_common import (
    validate_list_not_none,
    validate_required_not_none,
    filter_none_values,
    is_empty_list
)

__all__ = [
    "build_chat_list_response",
    "get_chats_with_votes",
    "get_user_votes_for_chats",
]


def build_chat_list_response(
    chats: List[PublishedChat],
    total: int,
    page: int,
    page_size: int,
    current_user_id: Optional[str] = None,
    service: Optional[ChatService] = None
) -> ChatListResponse:
    """
    Build a ChatListResponse with pagination and user votes.
    
    Optimized to batch user votes query when service is provided.
    Handles None values gracefully.
    
    Args:
        chats: List of chat models
        total: Total count of chats
        page: Current page number
        page_size: Page size
        current_user_id: Optional current user ID
        service: Optional ChatService for getting user votes
        
    Returns:
        ChatListResponse with pagination metadata
        
    Raises:
        ValueError: If chats is None or total/page/page_size are invalid
    """
    chats = validate_list_not_none(chats, "chats")
    
    if total < 0:
        raise ValueError("total cannot be negative")
    
    if page < 1:
        raise ValueError("page must be >= 1")
    
    if page_size < 1:
        raise ValueError("page_size must be >= 1")
    
    user_votes = get_user_votes_for_chats(chats, current_user_id, service) if service else {}
    
    chat_responses = chats_to_responses(chats, current_user_id, user_votes)
    pagination_meta = calculate_pagination_metadata(total, page, page_size)
    
    return ChatListResponse(
        chats=chat_responses,
        total=pagination_meta["total"],
        page=pagination_meta["page"],
        page_size=pagination_meta["page_size"],
        has_more=pagination_meta["has_more"]
    )


def get_chats_with_votes(
    chats: List[PublishedChat],
    current_user_id: Optional[str],
    service: ChatService
) -> tuple[List[PublishedChatResponse], Dict[str, ChatVote]]:
    """
    Get chats converted to responses with user votes.
    
    Optimized to batch user votes query and convert chats in one pass.
    
    Args:
        chats: List of chat models
        current_user_id: Optional current user ID
        service: ChatService instance
        
    Returns:
        Tuple of (chat_responses, user_votes_dict)
        
    Raises:
        ValueError: If chats is None or service is None
    """
    chats = validate_list_not_none(chats, "chats")
    validate_required_not_none(service, "service")
    
    user_votes = get_user_votes_for_chats(chats, current_user_id, service)
    chat_responses = chats_to_responses(chats, current_user_id, user_votes)
    return chat_responses, user_votes


def get_user_votes_for_chats(
    chats: List[PublishedChat],
    current_user_id: Optional[str],
    service: ChatService
) -> Dict[str, ChatVote]:
    """
    Get user votes for a list of chats in a single batch query.
    
    Optimized to use batch query instead of individual queries.
    Handles None values gracefully.
    
    Args:
        chats: List of chat models
        current_user_id: Optional current user ID
        service: ChatService instance
        
    Returns:
        Dictionary mapping chat_id to ChatVote
        
    Raises:
        ValueError: If chats is None or service is None
    """
    chats = validate_list_not_none(chats, "chats")
    validate_required_not_none(service, "service")
    
    if not current_user_id or is_empty_list(chats):
        return {}
    
    valid_chats = filter_none_values(chats)
    if is_empty_list(valid_chats):
        return {}
    
    chat_ids = [chat.id for chat in valid_chats]
    return service.get_user_votes_batch(chat_ids, current_user_id)






