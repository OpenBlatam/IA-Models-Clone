"""
Security Headers Middleware
===========================

Advanced security headers for FastAPI.
"""

import logging
from typing import Dict, List, Optional
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware for adding security headers."""
    
    def __init__(
        self,
        app,
        csp: Optional[str] = None,
        hsts_max_age: int = 31536000,
        frame_options: str = "DENY",
        content_type_nosniff: bool = True,
        xss_protection: bool = True,
        referrer_policy: str = "strict-origin-when-cross-origin"
    ):
        super().__init__(app)
        self.csp = csp or "default-src 'self'"
        self.hsts_max_age = hsts_max_age
        self.frame_options = frame_options
        self.content_type_nosniff = content_type_nosniff
        self.xss_protection = xss_protection
        self.referrer_policy = referrer_policy
    
    async def dispatch(self, request: Request, call_next):
        """Add security headers to response."""
        response = await call_next(request)
        
        # Content Security Policy
        response.headers["Content-Security-Policy"] = self.csp
        
        # HSTS
        if request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = f"max-age={self.hsts_max_age}; includeSubDomains"
        
        # X-Frame-Options
        response.headers["X-Frame-Options"] = self.frame_options
        
        # X-Content-Type-Options
        if self.content_type_nosniff:
            response.headers["X-Content-Type-Options"] = "nosniff"
        
        # X-XSS-Protection
        if self.xss_protection:
            response.headers["X-XSS-Protection"] = "1; mode=block"
        
        # Referrer-Policy
        response.headers["Referrer-Policy"] = self.referrer_policy
        
        # Permissions-Policy
        response.headers["Permissions-Policy"] = (
            "geolocation=(), microphone=(), camera=()"
        )
        
        return response

