"""
Gateway Middleware
=================

Middleware for API Gateway integration.
"""

import logging
from typing import Dict, Any, Optional
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class GatewayMiddleware(BaseHTTPMiddleware):
    """Middleware for API Gateway."""
    
    def __init__(self, app, gateway_type: str = "aws_api_gateway"):
        super().__init__(app)
        self.gateway_type = gateway_type
    
    async def dispatch(self, request: Request, call_next):
        """Process request with gateway context."""
        # Extract gateway-specific headers
        request_id = request.headers.get("X-Request-ID")
        api_key = request.headers.get("X-API-Key")
        user_id = request.headers.get("X-User-ID")
        
        # Add to request state
        request.state.gateway_type = self.gateway_type
        request.state.request_id = request_id
        request.state.api_key = api_key
        request.state.user_id = user_id
        
        # Process request
        response = await call_next(request)
        
        # Add gateway-specific headers
        if request_id:
            response.headers["X-Request-ID"] = request_id
        
        return response















