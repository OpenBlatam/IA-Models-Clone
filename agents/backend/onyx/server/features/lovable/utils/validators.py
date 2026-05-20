"""
Validation utilities for common data validation patterns.
"""

from typing import Any, Optional, List, Dict
import re
import logging

from ..constants import (
    MAX_TITLE_LENGTH,
    MAX_DESCRIPTION_LENGTH,
    MAX_TAG_LENGTH,
    MAX_USER_ID_LENGTH,
    MAX_CHAT_ID_LENGTH,
    MAX_COMMENT_LENGTH
)

logger = logging.getLogger(__name__)


def validate_string_length(
    value: str,
    field_name: str,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    allow_empty: bool = False
) -> str:
    """
    Validate string length.
    
    Args:
        value: String to validate
        field_name: Name of the field for error messages
        min_length: Minimum length
        max_length: Maximum length
        allow_empty: Whether to allow empty strings
        
    Returns:
        Validated string
        
    Raises:
        ValueError: If validation fails
    """
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    
    if not allow_empty and not value.strip():
        raise ValueError(f"{field_name} cannot be empty")
    
    if min_length is not None and len(value) < min_length:
        raise ValueError(f"{field_name} must be at least {min_length} characters")
    
    if max_length is not None and len(value) > max_length:
        raise ValueError(f"{field_name} must be at most {max_length} characters")
    
    return value.strip()


def validate_title(title: str) -> str:
    """
    Validate chat title.
    
    Args:
        title: Title to validate
        
    Returns:
        Validated title
        
    Raises:
        ValueError: If validation fails
    """
    return validate_string_length(
        title,
        "Title",
        min_length=1,
        max_length=MAX_TITLE_LENGTH,
        allow_empty=False
    )


def validate_description(description: Optional[str]) -> Optional[str]:
    """
    Validate chat description.
    
    Args:
        description: Description to validate
        
    Returns:
        Validated description or None
        
    Raises:
        ValueError: If validation fails
    """
    if description is None:
        return None
    
    return validate_string_length(
        description,
        "Description",
        min_length=1,
        max_length=MAX_DESCRIPTION_LENGTH,
        allow_empty=True
    )


def validate_tags(tags: Optional[List[str]]) -> Optional[List[str]]:
    """
    Validate tags list.
    
    Args:
        tags: List of tags to validate
        
    Returns:
        Validated tags list or None
        
    Raises:
        ValueError: If validation fails
    """
    if tags is None:
        return None
    
    if not isinstance(tags, list):
        raise ValueError("Tags must be a list")
    
    validated_tags = []
    for tag in tags:
        if not isinstance(tag, str):
            raise ValueError("All tags must be strings")
        
        tag = tag.strip()
        if not tag:
            continue
        
        if len(tag) > MAX_TAG_LENGTH:
            raise ValueError(f"Tag '{tag}' exceeds maximum length of {MAX_TAG_LENGTH} characters")
        
        # Validate tag format (alphanumeric, spaces, hyphens, underscores)
        if not re.match(r'^[a-zA-Z0-9\s\-_]+$', tag):
            raise ValueError(f"Tag '{tag}' contains invalid characters. Only alphanumeric, spaces, hyphens, and underscores are allowed")
        
        validated_tags.append(tag)
    
    return validated_tags if validated_tags else None


def validate_user_id(user_id: str) -> str:
    """
    Validate user ID.
    
    Args:
        user_id: User ID to validate
        
    Returns:
        Validated user ID
        
    Raises:
        ValueError: If validation fails
    """
    return validate_string_length(
        user_id,
        "User ID",
        min_length=1,
        max_length=MAX_USER_ID_LENGTH,
        allow_empty=False
    )


def validate_chat_id(chat_id: str) -> str:
    """
    Validate chat ID.
    
    Args:
        chat_id: Chat ID to validate
        
    Returns:
        Validated chat ID
        
    Raises:
        ValueError: If validation fails
    """
    return validate_string_length(
        chat_id,
        "Chat ID",
        min_length=1,
        max_length=MAX_CHAT_ID_LENGTH,
        allow_empty=False
    )


def validate_pagination(
    page: int,
    page_size: int,
    max_page_size: int = 100
) -> tuple[int, int]:
    """
    Validate pagination parameters.
    
    Args:
        page: Page number
        page_size: Items per page
        max_page_size: Maximum allowed page size
        
    Returns:
        Tuple of (validated page, validated page_size)
        
    Raises:
        ValueError: If validation fails
    """
    if page < 1:
        raise ValueError("Page must be at least 1")
    
    if page_size < 1:
        raise ValueError("Page size must be at least 1")
    
    if page_size > max_page_size:
        raise ValueError(f"Page size cannot exceed {max_page_size}")
    
    return page, page_size


def validate_limit(limit: int, max_limit: int = 1000, min_limit: int = 1) -> int:
    """
    Validate limit parameter.
    
    Args:
        limit: Limit value
        max_limit: Maximum allowed limit
        min_limit: Minimum allowed limit
        
    Returns:
        Validated limit
        
    Raises:
        ValueError: If validation fails
    """
    if limit < min_limit:
        raise ValueError(f"Limit must be at least {min_limit}")
    
    if limit > max_limit:
        raise ValueError(f"Limit cannot exceed {max_limit}")
    
    return limit


def validate_vote_type(vote_type: str) -> str:
    """
    Validate vote type.
    
    Args:
        vote_type: Vote type to validate
        
    Returns:
        Validated vote type
        
    Raises:
        ValueError: If validation fails
    """
    valid_types = ["upvote", "downvote"]
    vote_type = vote_type.lower().strip()
    
    if vote_type not in valid_types:
        raise ValueError(f"Vote type must be one of: {', '.join(valid_types)}")
    
    return vote_type


def validate_category(category: Optional[str]) -> Optional[str]:
    """
    Validate category.
    
    Args:
        category: Category to validate
        
    Returns:
        Validated category or None
        
    Raises:
        ValueError: If validation fails
    """
    if category is None:
        return None
    
    if not isinstance(category, str):
        raise ValueError("Category must be a string")
    
    category = category.strip()
    if not category:
        return None
    
    if len(category) > 50:
        raise ValueError("Category must be at most 50 characters")
    
    return category


def validate_comment(comment: str) -> str:
    """
    Validate comment content.
    
    Args:
        comment: Comment to validate
        
    Returns:
        Validated comment
        
    Raises:
        ValueError: If validation fails
    """
    return validate_string_length(
        comment,
        "Comment",
        min_length=1,
        max_length=MAX_COMMENT_LENGTH,
        allow_empty=False
    )


def validate_sort(
    sort_by: str,
    order: str = "desc",
    allowed_fields: Optional[List[str]] = None
) -> tuple[str, str]:
    """
    Validate sort parameters.
    
    Args:
        sort_by: Field to sort by
        order: Sort order ('asc' or 'desc')
        allowed_fields: List of allowed sort fields
        
    Returns:
        Tuple of (validated sort_by, validated order)
        
    Raises:
        ValueError: If validation fails
    """
    from ..constants import SORT_OPTIONS
    
    if allowed_fields is None:
        allowed_fields = list(SORT_OPTIONS.values())
    
    if sort_by not in allowed_fields:
        raise ValueError(f"Invalid sort field. Must be one of: {', '.join(allowed_fields)}")
    
    order = order.lower()
    if order not in ["asc", "desc"]:
        raise ValueError("Sort order must be 'asc' or 'desc'")
    
    return sort_by, order




