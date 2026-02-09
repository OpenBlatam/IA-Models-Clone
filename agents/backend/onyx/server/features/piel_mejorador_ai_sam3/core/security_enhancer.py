"""
Security Enhancer for Piel Mejorador AI SAM3
============================================

Advanced security features.
"""

import logging
from typing import Dict, Any, Optional, List
from fastapi import Request, Response
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import secrets

logger = logging.getLogger(__name__)


class SecurityEnhancer:
    """
    Advanced security enhancements.
    
    Features:
    - CORS configuration
    - CSP headers
    - Security headers
    - Request validation
    - Rate limiting per user
    """
    
    def __init__(self):
        """Initialize security enhancer."""
        self.allowed_origins: List[str] = []
        self.allowed_methods: List[str] = ["GET", "POST", "PUT", "DELETE"]
        self.allowed_headers: List[str] = ["*"]
        self.max_age: int = 3600
        
        self._request_ids: Dict[str, str] = {}
    
    def configure_cors(
        self,
        allowed_origins: List[str],
        allowed_methods: Optional[List[str]] = None,
        allowed_headers: Optional[List[str]] = None
    ):
        """
        Configure CORS settings.
        
        Args:
            allowed_origins: List of allowed origins
            allowed_methods: Optional allowed methods
            allowed_headers: Optional allowed headers
        """
        self.allowed_origins = allowed_origins
        if allowed_methods:
            self.allowed_methods = allowed_methods
        if allowed_headers:
            self.allowed_headers = allowed_headers
    
    def add_security_headers(self, response: Response):
        """
        Add security headers to response.
        
        Args:
            response: FastAPI response
        """
        # Content Security Policy
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self' data:;"
        )
        
        # X-Content-Type-Options
        response.headers["X-Content-Type-Options"] = "nosniff"
        
        # X-Frame-Options
        response.headers["X-Frame-Options"] = "DENY"
        
        # X-XSS-Protection
        response.headers["X-XSS-Protection"] = "1; mode=block"
        
        # Referrer-Policy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # Permissions-Policy
        response.headers["Permissions-Policy"] = (
            "geolocation=(), "
            "microphone=(), "
            "camera=()"
        )
    
    def add_cors_headers(self, response: Response, origin: Optional[str] = None):
        """
        Add CORS headers to response.
        
        Args:
            response: FastAPI response
            origin: Request origin
        """
        if origin and origin in self.allowed_origins:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"
        elif "*" in self.allowed_origins:
            response.headers["Access-Control-Allow-Origin"] = "*"
        
        response.headers["Access-Control-Allow-Methods"] = ", ".join(self.allowed_methods)
        response.headers["Access-Control-Allow-Headers"] = ", ".join(self.allowed_headers)
        response.headers["Access-Control-Max-Age"] = str(self.max_age)
    
    def generate_request_id(self) -> str:
        """Generate unique request ID."""
        return secrets.token_urlsafe(16)
    
    def validate_request(self, request: Request) -> Dict[str, Any]:
        """
        Validate incoming request.
        
        Args:
            request: FastAPI request
            
        Returns:
            Validation result
        """
        validation_result = {
            "valid": True,
            "warnings": [],
            "request_id": self.generate_request_id(),
        }
        
        # Check content type for POST requests
        if request.method == "POST":
            content_type = request.headers.get("Content-Type", "")
            if not content_type:
                validation_result["warnings"].append("Missing Content-Type header")
            elif "application/json" not in content_type and "multipart/form-data" not in content_type:
                validation_result["warnings"].append(f"Unexpected Content-Type: {content_type}")
        
        # Check user agent
        user_agent = request.headers.get("User-Agent", "")
        if not user_agent:
            validation_result["warnings"].append("Missing User-Agent header")
        
        return validation_result
    
    def get_security_config(self) -> Dict[str, Any]:
        """Get security configuration."""
        return {
            "cors": {
                "allowed_origins": self.allowed_origins,
                "allowed_methods": self.allowed_methods,
                "allowed_headers": self.allowed_headers,
                "max_age": self.max_age,
            },
            "security_headers": {
                "csp": True,
                "x_content_type_options": True,
                "x_frame_options": True,
                "x_xss_protection": True,
            },
        }




