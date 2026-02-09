"""
Security Headers Middleware
Implementa headers de seguridad avanzados (OWASP best practices)
"""

import logging
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import Message

logger = logging.getLogger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware para agregar headers de seguridad
    Implementa OWASP best practices
    """
    
    def __init__(self, app, **kwargs):
        super().__init__(app)
        self.strict_transport_security = kwargs.get(
            'strict_transport_security',
            'max-age=31536000; includeSubDomains; preload'
        )
        self.content_security_policy = kwargs.get(
            'content_security_policy',
            "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'"
        )
        self.x_content_type_options = kwargs.get('x_content_type_options', 'nosniff')
        self.x_frame_options = kwargs.get('x_frame_options', 'DENY')
        self.x_xss_protection = kwargs.get('x_xss_protection', '1; mode=block')
        self.referrer_policy = kwargs.get('referrer_policy', 'strict-origin-when-cross-origin')
        self.permissions_policy = kwargs.get(
            'permissions_policy',
            'geolocation=(), microphone=(), camera=()'
        )
        self.enable_cors = kwargs.get('enable_cors', True)
    
    async def dispatch(self, request: Request, call_next):
        """Procesa request y agrega security headers"""
        response = await call_next(request)
        
        # Security Headers
        response.headers["Strict-Transport-Security"] = self.strict_transport_security
        response.headers["Content-Security-Policy"] = self.content_security_policy
        response.headers["X-Content-Type-Options"] = self.x_content_type_options
        response.headers["X-Frame-Options"] = self.x_frame_options
        response.headers["X-XSS-Protection"] = self.x_xss_protection
        response.headers["Referrer-Policy"] = self.referrer_policy
        response.headers["Permissions-Policy"] = self.permissions_policy
        
        # Remove server header (security through obscurity)
        if "server" in response.headers:
            del response.headers["server"]
        
        # Add custom security headers
        response.headers["X-Request-ID"] = request.headers.get(
            "X-Request-ID",
            str(id(request))
        )
        
        return response















