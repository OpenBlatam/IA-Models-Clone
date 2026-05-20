"""
Request Context
Provides request-scoped context for tracing and logging
"""

import uuid
from typing import Dict, Any, Optional
from contextvars import ContextVar
import logging

logger = logging.getLogger(__name__)

# Context variables for request-scoped data
_request_id: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
_user_id: ContextVar[Optional[str]] = ContextVar('user_id', default=None)
_correlation_id: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)
_request_metadata: ContextVar[Dict[str, Any]] = ContextVar('request_metadata', default_factory=dict)


class RequestContext:
    """Request context manager"""
    
    @staticmethod
    def set_request_id(request_id: str):
        """Set request ID"""
        _request_id.set(request_id)
    
    @staticmethod
    def get_request_id() -> Optional[str]:
        """Get current request ID"""
        return _request_id.get()
    
    @staticmethod
    def set_user_id(user_id: str):
        """Set user ID"""
        _user_id.set(user_id)
    
    @staticmethod
    def get_user_id() -> Optional[str]:
        """Get current user ID"""
        return _user_id.get()
    
    @staticmethod
    def set_correlation_id(correlation_id: str):
        """Set correlation ID for distributed tracing"""
        _correlation_id.set(correlation_id)
    
    @staticmethod
    def get_correlation_id() -> Optional[str]:
        """Get current correlation ID"""
        return _correlation_id.get()
    
    @staticmethod
    def set_metadata(key: str, value: Any):
        """Set metadata value"""
        metadata = _request_metadata.get()
        metadata[key] = value
        _request_metadata.set(metadata)
    
    @staticmethod
    def get_metadata(key: str, default: Any = None) -> Any:
        """Get metadata value"""
        metadata = _request_metadata.get()
        return metadata.get(key, default)
    
    @staticmethod
    def get_all_metadata() -> Dict[str, Any]:
        """Get all metadata"""
        return _request_metadata.get()
    
    @staticmethod
    def clear():
        """Clear all context"""
        _request_id.set(None)
        _user_id.set(None)
        _correlation_id.set(None)
        _request_metadata.set({})
    
    @staticmethod
    def get_context_dict() -> Dict[str, Any]:
        """Get all context as dictionary"""
        return {
            "request_id": RequestContext.get_request_id(),
            "user_id": RequestContext.get_user_id(),
            "correlation_id": RequestContext.get_correlation_id(),
            "metadata": RequestContext.get_all_metadata(),
        }
    
    @staticmethod
    def generate_request_id() -> str:
        """Generate a new request ID"""
        return str(uuid.uuid4())


def with_request_context(request_id: Optional[str] = None, **metadata):
    """
    Decorator to set request context for a function
    
    Args:
        request_id: Request ID (generated if not provided)
        **metadata: Additional metadata to set
    """
    def decorator(func):
        import functools
        import asyncio
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            req_id = request_id or RequestContext.generate_request_id()
            RequestContext.set_request_id(req_id)
            
            for key, value in metadata.items():
                RequestContext.set_metadata(key, value)
            
            try:
                return await func(*args, **kwargs)
            finally:
                RequestContext.clear()
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            req_id = request_id or RequestContext.generate_request_id()
            RequestContext.set_request_id(req_id)
            
            for key, value in metadata.items():
                RequestContext.set_metadata(key, value)
            
            try:
                return func(*args, **kwargs)
            finally:
                RequestContext.clear()
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator















