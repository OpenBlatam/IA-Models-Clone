"""
Context Manager for Color Grading AI
=====================================

Request context management with tracing and correlation.
"""

import logging
import uuid
from typing import Dict, Any, Optional
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

# Context variables
_request_context: ContextVar[Optional[Dict[str, Any]]] = ContextVar('request_context', default=None)
_correlation_id: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)


@dataclass
class RequestContext:
    """Request context."""
    request_id: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request_id": self.request_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "metadata": self.metadata,
            "start_time": self.start_time.isoformat(),
        }


class ContextManager:
    """
    Request context manager.
    
    Features:
    - Request context tracking
    - Correlation IDs
    - Context propagation
    - Metadata management
    """
    
    def __init__(self):
        """Initialize context manager."""
        self._contexts: Dict[str, RequestContext] = {}
    
    def create_context(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> RequestContext:
        """
        Create new request context.
        
        Args:
            user_id: User ID
            session_id: Session ID
            ip_address: IP address
            user_agent: User agent
            metadata: Additional metadata
            
        Returns:
            Request context
        """
        request_id = str(uuid.uuid4())
        
        context = RequestContext(
            request_id=request_id,
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address,
            user_agent=user_agent,
            metadata=metadata or {}
        )
        
        self._contexts[request_id] = context
        _request_context.set(context.to_dict())
        _correlation_id.set(request_id)
        
        logger.debug(f"Created request context: {request_id}")
        
        return context
    
    def get_context(self, request_id: Optional[str] = None) -> Optional[RequestContext]:
        """
        Get request context.
        
        Args:
            request_id: Optional request ID (uses current if not provided)
            
        Returns:
            Request context or None
        """
        if request_id:
            return self._contexts.get(request_id)
        
        # Get from context variable
        ctx_dict = _request_context.get()
        if ctx_dict:
            return RequestContext(**ctx_dict)
        
        return None
    
    def get_current_context(self) -> Optional[RequestContext]:
        """Get current request context from context variable."""
        ctx_dict = _request_context.get()
        if ctx_dict:
            return RequestContext(**ctx_dict)
        return None
    
    def get_correlation_id(self) -> Optional[str]:
        """Get current correlation ID."""
        return _correlation_id.get()
    
    def update_context(
        self,
        request_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Update request context.
        
        Args:
            request_id: Request ID
            metadata: Additional metadata
            **kwargs: Additional context fields
        """
        context = self._contexts.get(request_id)
        if not context:
            return
        
        if metadata:
            context.metadata.update(metadata)
        
        for key, value in kwargs.items():
            if hasattr(context, key):
                setattr(context, key, value)
    
    def clear_context(self, request_id: Optional[str] = None):
        """Clear request context."""
        if request_id:
            if request_id in self._contexts:
                del self._contexts[request_id]
        else:
            _request_context.set(None)
            _correlation_id.set(None)
    
    def get_all_contexts(self) -> Dict[str, RequestContext]:
        """Get all contexts."""
        return self._contexts.copy()




