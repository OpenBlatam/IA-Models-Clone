"""
Request context utilities for tracking request information.
"""

from typing import Optional
from contextvars import ContextVar
import uuid

# Context variable for request ID
_request_id: ContextVar[Optional[str]] = ContextVar('request_id', default=None)


def get_request_id() -> Optional[str]:
    """
    Get current request ID from context.
    
    Returns:
        Request ID or None if not set
    """
    return _request_id.get()


def set_request_id(request_id: Optional[str] = None) -> str:
    """
    Set request ID in context.
    
    Args:
        request_id: Optional request ID (generates one if not provided)
        
    Returns:
        Request ID
    """
    if request_id is None:
        request_id = str(uuid.uuid4())[:8]
    
    _request_id.set(request_id)
    return request_id


def generate_request_id() -> str:
    """
    Generate a new request ID.
    
    Returns:
        Short request ID (8 characters)
    """
    return str(uuid.uuid4())[:8]




