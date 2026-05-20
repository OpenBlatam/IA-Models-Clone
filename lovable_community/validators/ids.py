"""
ID validation functions

Functions for validating chat IDs, user IDs, and other identifiers.
"""


def validate_chat_id(chat_id: str) -> str:
    """
    Validates a chat ID.
    
    Args:
        chat_id: Chat ID to validate
        
    Returns:
        Sanitized chat ID
        
    Raises:
        ValueError: If the ID is invalid
    """
    if not chat_id or not isinstance(chat_id, str):
        raise ValueError("Chat ID is required and must be a string")
    
    chat_id = chat_id.strip()
    
    if not chat_id:
        raise ValueError("Chat ID cannot be empty")
    
    if len(chat_id) > 100:
        raise ValueError("Chat ID cannot exceed 100 characters")
    
    return chat_id


def validate_user_id(user_id: str) -> str:
    """
    Validates a user ID.
    
    Args:
        user_id: User ID to validate
        
    Returns:
        Sanitized user ID
        
    Raises:
        ValueError: If the ID is invalid
    """
    if not user_id or not isinstance(user_id, str):
        raise ValueError("User ID is required and must be a string")
    
    user_id = user_id.strip()
    
    if not user_id:
        raise ValueError("User ID cannot be empty")
    
    if len(user_id) > 100:
        raise ValueError("User ID cannot exceed 100 characters")
    
    return user_id













