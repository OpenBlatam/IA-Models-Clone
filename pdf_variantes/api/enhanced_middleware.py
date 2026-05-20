"""
PDF Variantes API - Enhanced Middleware
Additional middleware for security headers and request tracking
"""

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from ..utils.response_helpers import set_request_id
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses"""
    
    async def dispatch(self, request: Request, call_next):
        """Add security headers"""
        response = await call_next(request)
        
        # Security headers
        security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
        }
        
        # Add HSTS only for HTTPS
        if request.url.scheme == "https":
            security_headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        # Add security headers to response
        for header, value in security_headers.items():
            response.headers[header] = value
        
        return response


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Ensure Request ID is set for all requests"""
    
    async def dispatch(self, request: Request, call_next):
        """Set request ID if not already set"""
        # Set request ID if not already present
        if not hasattr(request.state, "request_id"):
            set_request_id(request)
        
        response = await call_next(request)
        
        # Ensure request ID is in response headers
        if hasattr(request.state, "request_id"):
            response.headers["X-Request-ID"] = request.state.request_id
        
        return response


class SlowRequestMiddleware(BaseHTTPMiddleware):
    """Log and alert on slow requests"""
    
    def __init__(self, app: ASGIApp, threshold: float = 2.0):
        super().__init__(app)
        self.threshold = threshold
    
    async def dispatch(self, request: Request, call_next):
        """Monitor request duration"""
        import time
        from ..utils.response_helpers import get_request_id
        
        start_time = time.time()
        response = await call_next(request)
        duration = time.time() - start_time
        
        if duration > self.threshold:
            request_id = get_request_id(request)
            logger.warning(
                f"[{request_id}] Slow request detected: {duration:.3f}s "
                f"for {request.method} {request.url.path}"
            )
            
            # Add alert header
            response.headers["X-Slow-Request"] = "true"
            response.headers["X-Duration"] = f"{duration:.3f}"
        
        return response


def setup_enhanced_middleware(app: ASGIApp):
    """Setup enhanced middleware stack"""
    # Security headers (should be early)
    app.add_middleware(SecurityHeadersMiddleware)
    
    # Request ID (should be early, before logging)
    app.add_middleware(RequestIDMiddleware)
    
    # Slow request monitoring (can be later)
    app.add_middleware(SlowRequestMiddleware, threshold=2.0)

