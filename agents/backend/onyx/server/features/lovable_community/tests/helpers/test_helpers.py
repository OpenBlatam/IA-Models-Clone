"""
Helpers generales para tests de Lovable Community
"""

import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional


def generate_chat_id() -> str:
    """
    Generate a unique chat ID.
    
    Returns:
        UUID string for chat ID
    """
    return str(uuid.uuid4())


def generate_user_id() -> str:
    """
    Generate a unique user ID.
    
    Returns:
        User ID string in format "user-{hex}"
    """
    return f"user-{uuid.uuid4().hex[:8]}"


def create_chat_dict(
    chat_id: Optional[str] = None,
    user_id: Optional[str] = None,
    title: str = "Test Chat",
    description: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Create a chat dictionary for testing.
    
    Args:
        chat_id: Optional chat ID (generated if not provided)
        user_id: Optional user ID (generated if not provided)
        title: Chat title (default: "Test Chat")
        description: Optional description
        **kwargs: Additional chat fields
        
    Returns:
        Dictionary with chat data
        
    Raises:
        ValueError: If title is None or empty
    """
    if not title or not title.strip():
        raise ValueError("title cannot be None or empty")
    
    return {
        "id": chat_id or generate_chat_id(),
        "user_id": user_id or generate_user_id(),
        "title": title.strip(),
        "description": description.strip() if description else "Test description",
        "chat_content": kwargs.get("chat_content", '{"messages": []}'),
        "tags": kwargs.get("tags", ["test"]),
        "vote_count": max(0, kwargs.get("vote_count", 0)),
        "remix_count": max(0, kwargs.get("remix_count", 0)),
        "view_count": max(0, kwargs.get("view_count", 0)),
        "score": max(0.0, kwargs.get("score", 0.0)),
        "is_public": kwargs.get("is_public", True),
        "is_featured": kwargs.get("is_featured", False),
        "created_at": kwargs.get("created_at", datetime.utcnow()),
        "updated_at": kwargs.get("updated_at", datetime.utcnow()),
        **kwargs
    }


def create_publish_request(
    title: str = "Test Chat",
    description: Optional[str] = None,
    tags: Optional[List[str]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Create a publish request for testing.
    
    Args:
        title: Chat title (default: "Test Chat")
        description: Optional description
        tags: Optional list of tags
        **kwargs: Additional request fields
        
    Returns:
        Dictionary with publish request data
        
    Raises:
        ValueError: If title is None or empty
    """
    if not title or not title.strip():
        raise ValueError("title cannot be None or empty")
    
    return {
        "title": title.strip(),
        "description": description.strip() if description else "Test description",
        "chat_content": kwargs.get("chat_content", '{"messages": []}'),
        "tags": tags or ["test"],
        "is_public": kwargs.get("is_public", True),
        **kwargs
    }


def create_remix_request(
    original_chat_id: str,
    title: str = "Remix: Test Chat",
    **kwargs
) -> Dict[str, Any]:
    """
    Create a remix request for testing.
    
    Args:
        original_chat_id: Original chat ID (required)
        title: Remix title (default: "Remix: Test Chat")
        **kwargs: Additional request fields
        
    Returns:
        Dictionary with remix request data
        
    Raises:
        ValueError: If original_chat_id or title is None or empty
    """
    if not original_chat_id or not original_chat_id.strip():
        raise ValueError("original_chat_id cannot be None or empty")
    
    if not title or not title.strip():
        raise ValueError("title cannot be None or empty")
    
    return {
        "original_chat_id": original_chat_id.strip(),
        "title": title.strip(),
        "description": kwargs.get("description", "Remix description"),
        "chat_content": kwargs.get("chat_content", '{"messages": []}'),
        "tags": kwargs.get("tags", ["remix", "test"]),
        **kwargs
    }


def create_vote_request(
    chat_id: str,
    vote_type: str = "upvote"
) -> Dict[str, Any]:
    """
    Create a vote request for testing.
    
    Args:
        chat_id: Chat ID to vote on (required)
        vote_type: Vote type - "upvote" or "downvote" (default: "upvote")
        
    Returns:
        Dictionary with vote request data
        
    Raises:
        ValueError: If chat_id is None or empty, or vote_type is invalid
    """
    if not chat_id or not chat_id.strip():
        raise ValueError("chat_id cannot be None or empty")
    
    vote_type = vote_type.strip().lower() if vote_type else "upvote"
    if vote_type not in ("upvote", "downvote"):
        raise ValueError(f"vote_type must be 'upvote' or 'downvote', got '{vote_type}'")
    
    return {
        "chat_id": chat_id.strip(),
        "vote_type": vote_type
    }


def create_search_request(
    query: Optional[str] = None,
    tags: Optional[List[str]] = None,
    **kwargs
) -> Dict[str, Any]:
    """Crea un request de búsqueda para testing"""
    return {
        "query": query,
        "tags": tags,
        "user_id": kwargs.get("user_id"),
        "sort_by": kwargs.get("sort_by", "score"),
        "order": kwargs.get("order", "desc"),
        "page": kwargs.get("page", 1),
        "page_size": kwargs.get("page_size", 20),
        **kwargs
    }

