"""
Security Headers Plugin
=======================
"""

import logging
from typing import Dict, Any
from fastapi import FastAPI, Request
from fastapi.middleware.base import BaseHTTPMiddleware
from aws.core.interfaces import MiddlewarePlugin

logger = logging.getLogger(__name__)


class SecurityHeadersMiddlewarePlugin(MiddlewarePlugin):
    """Security headers middleware plugin."""
    
    def get_name(self) -> str:
        return "security_headers"
    
    def is_enabled(self, config: Dict[str, Any]) -> bool:
        middleware_config = config.get("middleware", {})
        return middleware_config.get("enable_security_headers", True)
    
    def setup(self, app: FastAPI, config: Dict[str, Any]) -> FastAPI:
        """Setup security headers middleware."""
        class SecurityHeadersMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request: Request, call_next):
                response = await call_next(request)
                
                # Security headers
                response.headers["X-Content-Type-Options"] = "nosniff"
                response.headers["X-Frame-Options"] = "DENY"
                response.headers["X-XSS-Protection"] = "1; mode=block"
                response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
                response.headers["Content-Security-Policy"] = "default-src 'self'"
                response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
                response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
                
                return response
        
        app.add_middleware(SecurityHeadersMiddleware)
        logger.info("Security headers enabled")
        
        return app















