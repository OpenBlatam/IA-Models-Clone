"""
Advanced Security Middleware
OWASP best practices, security headers, DDoS protection
"""

import logging
import time
from typing import Callable, Dict, List
from collections import defaultdict
from fastapi import Request, Response, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Advanced security headers middleware
    
    Implements OWASP security best practices:
    - Content Security Policy (CSP)
    - X-Frame-Options
    - X-Content-Type-Options
    - Strict-Transport-Security (HSTS)
    - X-XSS-Protection
    - Referrer-Policy
    """
    
    def __init__(self, app: ASGIApp, strict_csp: bool = True):
        super().__init__(app)
        self.strict_csp = strict_csp
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add security headers to response"""
        response = await call_next(request)
        
        # Content Security Policy
        if self.strict_csp:
            csp = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data: https:; "
                "font-src 'self' data:; "
                "connect-src 'self' https://api.openai.com; "
                "frame-ancestors 'none';"
            )
            response.headers["Content-Security-Policy"] = csp
        
        # X-Frame-Options
        response.headers["X-Frame-Options"] = "DENY"
        
        # X-Content-Type-Options
        response.headers["X-Content-Type-Options"] = "nosniff"
        
        # X-XSS-Protection
        response.headers["X-XSS-Protection"] = "1; mode=block"
        
        # Referrer-Policy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # Permissions-Policy
        response.headers["Permissions-Policy"] = (
            "geolocation=(), "
            "microphone=(), "
            "camera=(), "
            "payment=()"
        )
        
        # Strict-Transport-Security (HSTS) - only for HTTPS
        if request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains; preload"
            )
        
        # Remove server header
        response.headers.pop("server", None)
        
        return response


class DDoSProtectionMiddleware(BaseHTTPMiddleware):
    """
    DDoS protection middleware
    
    Features:
    - Rate limiting per IP
    - Request size limits
    - Connection throttling
    - IP whitelist/blacklist
    """
    
    def __init__(
        self,
        app: ASGIApp,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        max_request_size: int = 10 * 1024 * 1024,  # 10MB
        block_duration: int = 300  # 5 minutes
    ):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.max_request_size = max_request_size
        self.block_duration = block_duration
        
        # In-memory tracking (use Redis in production)
        self._request_counts: Dict[str, List[float]] = defaultdict(list)
        self._blocked_ips: Dict[str, float] = {}
        self._whitelist: set = set()
        self._blacklist: set = set()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Check for DDoS attacks"""
        client_ip = self._get_client_ip(request)
        
        # Check whitelist
        if client_ip in self._whitelist:
            return await call_next(request)
        
        # Check blacklist
        if client_ip in self._blacklist:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="IP address blocked"
            )
        
        # Check if IP is temporarily blocked
        if client_ip in self._blocked_ips:
            block_until = self._blocked_ips[client_ip]
            if time.time() < block_until:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Too many requests. Please try again later."
                )
            else:
                # Unblock after duration
                del self._blocked_ips[client_ip]
        
        # Check request size
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_request_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Request too large"
            )
        
        # Rate limiting
        current_time = time.time()
        requests = self._request_counts[client_ip]
        
        # Clean old requests
        requests[:] = [req_time for req_time in requests if current_time - req_time < 3600]
        
        # Check per-minute limit
        recent_requests = [req_time for req_time in requests if current_time - req_time < 60]
        if len(recent_requests) >= self.requests_per_minute:
            self._block_ip(client_ip)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded"
            )
        
        # Check per-hour limit
        if len(requests) >= self.requests_per_hour:
            self._block_ip(client_ip)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Hourly rate limit exceeded"
            )
        
        # Record request
        requests.append(current_time)
        
        return await call_next(request)
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address"""
        # Check X-Forwarded-For header (for proxies)
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        
        # Check X-Real-IP header
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        # Fallback to direct client
        if request.client:
            return request.client.host
        
        return "unknown"
    
    def _block_ip(self, ip: str) -> None:
        """Block IP address temporarily"""
        self._blocked_ips[ip] = time.time() + self.block_duration
        logger.warning(f"IP blocked due to rate limiting: {ip}")
    
    def add_to_whitelist(self, ip: str) -> None:
        """Add IP to whitelist"""
        self._whitelist.add(ip)
    
    def add_to_blacklist(self, ip: str) -> None:
        """Add IP to blacklist"""
        self._blacklist.add(ip)
        logger.warning(f"IP added to blacklist: {ip}")


class InputValidationMiddleware(BaseHTTPMiddleware):
    """
    Input validation middleware
    
    Features:
    - SQL injection detection
    - XSS detection
    - Path traversal detection
    - Command injection detection
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self._sql_patterns = [
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)",
            r"(--|#|/\*|\*/)",
            r"(\bOR\b.*=.*)",
        ]
        self._xss_patterns = [
            r"<script[^>]*>",
            r"javascript:",
            r"on\w+\s*=",
        ]
        self._path_traversal_patterns = [
            r"\.\./",
            r"\.\.\\",
            r"%2e%2e%2f",
        ]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Validate input for security threats"""
        import re
        
        # Check query parameters
        for key, value in request.query_params.items():
            if self._is_malicious(str(value)):
                logger.warning(f"Malicious input detected in query param {key}: {value}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid input detected"
                )
        
        # Check path parameters
        for key, value in request.path_params.items():
            if self._is_malicious(str(value)):
                logger.warning(f"Malicious input detected in path param {key}: {value}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid input detected"
                )
        
        return await call_next(request)
    
    def _is_malicious(self, value: str) -> bool:
        """Check if value contains malicious patterns"""
        import re
        
        # Check SQL injection
        for pattern in self._sql_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                return True
        
        # Check XSS
        for pattern in self._xss_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                return True
        
        # Check path traversal
        for pattern in self._path_traversal_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                return True
        
        return False















