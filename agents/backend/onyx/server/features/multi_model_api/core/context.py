"""
Context management for Multi-Model API
Request context and resource management
"""

import contextvars
import uuid
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime

# Request context variable
request_context: contextvars.ContextVar[Optional['RequestContext']] = contextvars.ContextVar(
    'request_context',
    default=None
)


@dataclass
class RequestContext:
    """Context for a single request"""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    start_time: float = field(default_factory=time.time)
    user_id: Optional[str] = None
    api_key: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds"""
        return (time.time() - self.start_time) * 1000
    
    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to context"""
        self.metadata[key] = value
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata from context"""
        return self.metadata.get(key, default)


def get_request_context() -> Optional[RequestContext]:
    """
    Get current request context
    
    Returns:
        RequestContext or None if not set
    """
    return request_context.get()


def set_request_context(context: RequestContext) -> None:
    """
    Set current request context
    
    Args:
        context: RequestContext instance
    """
    request_context.set(context)


def create_request_context(
    request_id: Optional[str] = None,
    user_id: Optional[str] = None,
    api_key: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> RequestContext:
    """
    Create and set a new request context
    
    Args:
        request_id: Optional request ID (generated if not provided)
        user_id: Optional user ID
        api_key: Optional API key
        metadata: Optional metadata dictionary
        
    Returns:
        RequestContext instance
    """
    context = RequestContext(
        request_id=request_id or str(uuid.uuid4()),
        user_id=user_id,
        api_key=api_key,
        metadata=metadata or {}
    )
    set_request_context(context)
    return context


def clear_request_context() -> None:
    """Clear current request context"""
    request_context.set(None)




