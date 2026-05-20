"""
Operation validation functions

Functions for validating bulk operations and operation-related data.
"""

from typing import List


def validate_operation(operation: str) -> str:
    """
    Validates a bulk operation.
    
    Args:
        operation: Operation to validate
        
    Returns:
        Validated operation
        
    Raises:
        ValueError: If the operation is invalid
    """
    if not operation or not isinstance(operation, str):
        raise ValueError("Operation is required and must be a string")
    
    operation = operation.strip().lower()
    
    valid_operations = ("delete", "feature", "unfeature", "make_public", "make_private")
    if operation not in valid_operations:
        raise ValueError(f"Operation must be one of: {', '.join(valid_operations)}")
    
    return operation


def validate_chat_ids(chat_ids: List[str], max_count: int = 100) -> List[str]:
    """
    Validates a list of chat IDs.
    
    Args:
        chat_ids: List of IDs to validate
        max_count: Maximum number of IDs allowed
        
    Returns:
        List of validated and sanitized IDs
        
    Raises:
        ValueError: If the list is invalid
    """
    if not chat_ids:
        raise ValueError("Chat IDs list cannot be empty")
    
    if len(chat_ids) > max_count:
        raise ValueError(f"Maximum {max_count} chat IDs allowed")
    
    sanitized = []
    seen = set()
    
    for chat_id in chat_ids:
        if chat_id and isinstance(chat_id, str):
            chat_id_clean = chat_id.strip()
            if chat_id_clean and chat_id_clean not in seen:
                sanitized.append(chat_id_clean)
                seen.add(chat_id_clean)
    
    if not sanitized:
        raise ValueError("No valid chat IDs provided")
    
    return sanitized



