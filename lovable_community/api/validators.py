"""
API Validators (backward compatibility)

This file maintains backward compatibility by importing from validators/ module
and providing API-specific wrappers that convert ValueError to InvalidChatError
and support raise_on_invalid parameter.

For new imports, use:
    from ..validators import validate_chat_id, validate_user_id, etc.
"""

from typing import Optional, List

from ..exceptions import InvalidChatError
from ..utils import sanitize_string
from ..validators import (
    validate_chat_id as _validate_chat_id_base,
    validate_user_id as _validate_user_id_base,
    validate_vote_type as _validate_vote_type_base,
    validate_period as _validate_period_base,
    validate_sort_by as _validate_sort_by_base,
    validate_order as _validate_order_base,
    validate_operation as _validate_operation_base,
    validate_chat_ids as _validate_chat_ids_base,
)


def validate_chat_id(chat_id: str, raise_on_invalid: bool = True) -> Optional[str]:
    """
    Validates a chat ID (API wrapper).
    
    Args:
        chat_id: Chat ID to validate
        raise_on_invalid: Whether to raise exception or return None
        
    Returns:
        Sanitized chat ID or None
        
    Raises:
        InvalidChatError: If the ID is invalid and raise_on_invalid=True
    """
    if not chat_id:
        if raise_on_invalid:
            raise InvalidChatError("Chat ID cannot be empty")
        return None
    
    try:
        chat_id = sanitize_string(chat_id)
        if not chat_id:
            if raise_on_invalid:
                raise InvalidChatError("Chat ID cannot be empty")
            return None
        return _validate_chat_id_base(chat_id)
    except ValueError as e:
        if raise_on_invalid:
            raise InvalidChatError(str(e)) from e
        return None


def validate_user_id(user_id: str, raise_on_invalid: bool = True) -> Optional[str]:
    """
    Validates a user ID (API wrapper).
    
    Args:
        user_id: User ID to validate
        raise_on_invalid: Whether to raise exception or return None
        
    Returns:
        Sanitized user ID or None
        
    Raises:
        InvalidChatError: If the ID is invalid and raise_on_invalid=True
    """
    if not user_id:
        if raise_on_invalid:
            raise InvalidChatError("User ID cannot be empty")
        return None
    
    try:
        user_id = sanitize_string(user_id)
        if not user_id:
            if raise_on_invalid:
                raise InvalidChatError("User ID cannot be empty")
            return None
        return _validate_user_id_base(user_id)
    except ValueError as e:
        if raise_on_invalid:
            raise InvalidChatError(str(e)) from e
        return None


def validate_vote_type(vote_type: str) -> str:
    """
    Validates a vote type (API wrapper).
    
    Args:
        vote_type: Vote type to validate
        
    Returns:
        Validated vote type
        
    Raises:
        InvalidChatError: If the vote type is invalid
    """
    try:
        return _validate_vote_type_base(vote_type)
    except ValueError as e:
        raise InvalidChatError(str(e)) from e


def validate_period(period: str) -> str:
    """
    Validates a time period (API wrapper).
    
    Args:
        period: Period to validate (hour, day, week, month)
        
    Returns:
        Validated period
        
    Raises:
        InvalidChatError: If the period is invalid
    """
    try:
        return _validate_period_base(period)
    except ValueError as e:
        raise InvalidChatError(str(e)) from e


def validate_operation(operation: str) -> str:
    """
    Validates a bulk operation (API wrapper).
    
    Args:
        operation: Operation to validate
        
    Returns:
        Validated operation
        
    Raises:
        InvalidChatError: If the operation is invalid
    """
    try:
        return _validate_operation_base(operation)
    except ValueError as e:
        raise InvalidChatError(str(e)) from e


def validate_chat_ids(chat_ids: List[str], max_count: int = 100) -> List[str]:
    """
    Validates a list of chat IDs (API wrapper).
    
    Args:
        chat_ids: List of IDs to validate
        max_count: Maximum number of IDs allowed
        
    Returns:
        List of validated and sanitized IDs
        
    Raises:
        InvalidChatError: If the list is invalid
    """
    try:
        return _validate_chat_ids_base(chat_ids, max_count)
    except ValueError as e:
        raise InvalidChatError(str(e)) from e


def validate_sort_by(sort_by: str) -> str:
    """
    Validates a sort field (API wrapper).
    
    Args:
        sort_by: Sort field to validate
        
    Returns:
        Validated sort field (defaults to "score")
        
    Raises:
        InvalidChatError: If the sort field is invalid
    """
    if not sort_by:
        return "score"
    
    try:
        allowed_fields = ("score", "created_at", "vote_count", "remix_count")
        return _validate_sort_by_base(sort_by, allowed_fields)
    except ValueError as e:
        raise InvalidChatError(str(e)) from e


def validate_order(order: str) -> str:
    """
    Validates an order (asc/desc) (API wrapper).
    
    Args:
        order: Order to validate
        
    Returns:
        Validated order (defaults to "desc")
        
    Raises:
        InvalidChatError: If the order is invalid
    """
    if not order:
        return "desc"
    
    try:
        return _validate_order_base(order)
    except ValueError as e:
        raise InvalidChatError(str(e)) from e

