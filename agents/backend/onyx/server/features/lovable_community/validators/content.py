"""
Content validation functions

Functions for validating chat content, titles, and descriptions.
"""

from typing import Optional


def validate_title(title: str, max_length: int = 200) -> str:
    """
    Validates a chat title.
    
    Args:
        title: Title to validate
        max_length: Maximum allowed length
        
    Returns:
        Sanitized title
        
    Raises:
        ValueError: If the title is invalid
    """
    if not title or not isinstance(title, str):
        raise ValueError("Title is required and must be a string")
    
    title = title.strip()
    
    if not title:
        raise ValueError("Title cannot be empty")
    
    if len(title) > max_length:
        raise ValueError(f"Title cannot exceed {max_length} characters")
    
    return title


def validate_description(description: Optional[str], max_length: int = 1000) -> Optional[str]:
    """
    Validates a chat description.
    
    Args:
        description: Description to validate
        max_length: Maximum allowed length
        
    Returns:
        Sanitized description or None
        
    Raises:
        ValueError: If the description is invalid
    """
    if description is None:
        return None
    
    if not isinstance(description, str):
        raise ValueError("Description must be a string")
    
    description = description.strip()
    
    if not description:
        return None
    
    if len(description) > max_length:
        raise ValueError(f"Description cannot exceed {max_length} characters")
    
    return description


def validate_chat_content(content: str, max_length: int = 50000) -> str:
    """
    Validates chat content.
    
    Args:
        content: Content to validate
        max_length: Maximum allowed length
        
    Returns:
        Sanitized content
        
    Raises:
        ValueError: If the content is invalid
    """
    if not content or not isinstance(content, str):
        raise ValueError("Chat content is required and must be a string")
    
    content = content.strip()
    
    if not content:
        raise ValueError("Chat content cannot be empty")
    
    if len(content) > max_length:
        raise ValueError(f"Chat content cannot exceed {max_length} characters")
    
    return content













