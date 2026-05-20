"""
Security Middleware
==================

Security-related middleware components.
"""

from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class SecurityMiddleware(BaseHTTPMiddleware):
    """Security middleware for request validation and protection."""
    
    def __init__(self, app, rate_limit_config: Dict[str, Any] = None):
        super().__init__(app)
        self.rate_limit_config = rate_limit_config or {}
        self.request_counts: Dict[str, int] = {}
    
    async def dispatch(self, request: Request, call_next):
        """Apply security checks to requests."""
        
        # Basic security headers
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        return response
