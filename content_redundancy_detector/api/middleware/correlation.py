"""
Advanced Correlation ID Middleware
Handles X-Request-ID and X-Correlation-ID propagation for distributed tracing
"""

import uuid
from typing import Callable, Optional
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from contextvars import ContextVar

# Context variables for async context propagation
request_id_var: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
correlation_id_var: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)


class CorrelationMiddleware(BaseHTTPMiddleware):
    """
    Middleware for request correlation and tracing
    
    Features:
    - Generates unique request IDs if not provided
    - Propagates correlation IDs across service boundaries
    - Stores IDs in request state and context vars
    - Adds headers to all responses
    - Supports distributed tracing patterns
    """
    
    def __init__(self, app, header_request_id: str = "X-Request-ID", 
                 header_correlation: str = "X-Correlation-ID",
                 generate_if_missing: bool = True):
        super().__init__(app)
        self.header_request_id = header_request_id
        self.header_correlation = header_correlation
        self.generate_if_missing = generate_if_missing
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Extract or generate request ID
        request_id = request.headers.get(self.header_request_id)
        if not request_id and self.generate_if_missing:
            request_id = str(uuid.uuid4())
        
        # Extract or propagate correlation ID
        correlation_id = request.headers.get(self.header_correlation)
        if not correlation_id:
            # If no correlation ID, use request ID as correlation ID
            correlation_id = request_id
        
        # Store in request state
        request.state.request_id = request_id
        request.state.correlation_id = correlation_id
        
        # Store in context vars for async context propagation
        if request_id:
            request_id_var.set(request_id)
        if correlation_id:
            correlation_id_var.set(correlation_id)
        
        # Process request
        response = await call_next(request)
        
        # Add correlation headers to response
        if request_id:
            response.headers[self.header_request_id] = request_id
        if correlation_id:
            response.headers[self.header_correlation] = correlation_id
        
        # Clean up context vars (optional, for explicit cleanup)
        # Context vars are automatically cleaned up when the task completes
        
        return response


def get_request_id() -> Optional[str]:
    """Get current request ID from context"""
    return request_id_var.get(None)


def get_correlation_id() -> Optional[str]:
    """Get current correlation ID from context"""
    return correlation_id_var.get(None)


def set_request_id(value: str) -> None:
    """Set request ID in context"""
    request_id_var.set(value)


def set_correlation_id(value: str) -> None:
    """Set correlation ID in context"""
    correlation_id_var.set(value)


def get_correlation_context() -> dict:
    """Get full correlation context"""
    return {
        "request_id": get_request_id(),
        "correlation_id": get_correlation_id()
    }


