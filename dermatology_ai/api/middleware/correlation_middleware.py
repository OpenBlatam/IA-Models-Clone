"""
Correlation ID Middleware
Adds correlation ID to requests for distributed tracing
"""

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import uuid
import logging

from ...core.infrastructure.request_context import RequestContext

logger = logging.getLogger(__name__)


class CorrelationIDMiddleware(BaseHTTPMiddleware):
    """Middleware to add correlation ID to requests"""
    
    def __init__(self, app, header_name: str = "X-Correlation-ID"):
        super().__init__(app)
        self.header_name = header_name
    
    async def dispatch(self, request: Request, call_next):
        # Get correlation ID from header or generate new one
        correlation_id = request.headers.get(self.header_name.lower())
        
        if not correlation_id:
            correlation_id = str(uuid.uuid4())
        
        # Set in request context
        RequestContext.set_correlation_id(correlation_id)
        RequestContext.set_request_id(correlation_id)
        
        # Add to request state
        request.state.correlation_id = correlation_id
        request.state.request_id = correlation_id
        
        # Process request
        response = await call_next(request)
        
        # Add correlation ID to response headers
        if isinstance(response, Response):
            response.headers[self.header_name] = correlation_id
        
        return response















