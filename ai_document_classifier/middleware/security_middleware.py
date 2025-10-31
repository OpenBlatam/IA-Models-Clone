"""
Security Middleware
==================

Advanced security middleware for the AI Document Classifier including
rate limiting, request validation, and security headers.
"""

import time
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import hashlib
import re
from pathlib import Path
import json

logger = logging.getLogger(__name__)

@dataclass
class RateLimit:
    """Rate limit configuration"""
    requests: int
    window_seconds: int
    burst_limit: int = 0  # Additional burst requests allowed

@dataclass
class SecurityConfig:
    """Security configuration"""
    enable_rate_limiting: bool = True
    enable_request_validation: bool = True
    enable_security_headers: bool = True
    enable_cors: bool = True
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    max_query_length: int = 1000
    allowed_origins: List[str] = field(default_factory=lambda: ["*"])
    blocked_ips: List[str] = field(default_factory=list)
    rate_limits: Dict[str, RateLimit] = field(default_factory=dict)

class RateLimiter:
    """
    Advanced rate limiter with sliding window and burst support
    """
    
    def __init__(self, config: SecurityConfig):
        """
        Initialize rate limiter
        
        Args:
            config: Security configuration
        """
        self.config = config
        self.requests = defaultdict(lambda: deque())
        self.burst_usage = defaultdict(int)
        self.blocked_ips = set(config.blocked_ips)
        
        # Default rate limits
        self.default_limits = {
            "default": RateLimit(requests=100, window_seconds=3600, burst_limit=20),
            "classification": RateLimit(requests=50, window_seconds=3600, burst_limit=10),
            "batch": RateLimit(requests=10, window_seconds=3600, burst_limit=2),
            "admin": RateLimit(requests=1000, window_seconds=3600, burst_limit=100)
        }
        
        # Merge with config
        self.rate_limits = {**self.default_limits, **config.rate_limits}
    
    def is_allowed(
        self, 
        identifier: str, 
        endpoint: str = "default",
        ip_address: Optional[str] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if request is allowed based on rate limits
        
        Args:
            identifier: Unique identifier (user_id, api_key, ip, etc.)
            endpoint: Endpoint type for specific rate limits
            ip_address: IP address for IP-based blocking
            
        Returns:
            Tuple of (is_allowed, rate_limit_info)
        """
        # Check if IP is blocked
        if ip_address and ip_address in self.blocked_ips:
            return False, {
                "reason": "blocked_ip",
                "message": "IP address is blocked",
                "retry_after": None
            }
        
        # Get rate limit for endpoint
        rate_limit = self.rate_limits.get(endpoint, self.rate_limits["default"])
        
        now = time.time()
        window_start = now - rate_limit.window_seconds
        
        # Clean old requests
        requests_queue = self.requests[identifier]
        while requests_queue and requests_queue[0] < window_start:
            requests_queue.popleft()
        
        # Check burst limit
        burst_used = self.burst_usage.get(identifier, 0)
        if burst_used > 0:
            # Reset burst usage if window has passed
            if requests_queue and requests_queue[0] > window_start:
                self.burst_usage[identifier] = 0
                burst_used = 0
        
        # Check if within limits
        current_requests = len(requests_queue)
        total_allowed = rate_limit.requests + rate_limit.burst_limit
        
        if current_requests >= total_allowed:
            # Calculate retry after
            oldest_request = requests_queue[0] if requests_queue else now
            retry_after = int(oldest_request + rate_limit.window_seconds - now)
            
            return False, {
                "reason": "rate_limit_exceeded",
                "message": f"Rate limit exceeded for {endpoint}",
                "current_requests": current_requests,
                "limit": total_allowed,
                "retry_after": max(0, retry_after),
                "window_seconds": rate_limit.window_seconds
            }
        
        # Check burst limit
        if current_requests >= rate_limit.requests and burst_used >= rate_limit.burst_limit:
            return False, {
                "reason": "burst_limit_exceeded",
                "message": f"Burst limit exceeded for {endpoint}",
                "current_requests": current_requests,
                "limit": rate_limit.requests,
                "burst_used": burst_used,
                "burst_limit": rate_limit.burst_limit,
                "retry_after": None
            }
        
        # Record request
        requests_queue.append(now)
        
        # Update burst usage if over normal limit
        if current_requests >= rate_limit.requests:
            self.burst_usage[identifier] += 1
        
        return True, {
            "reason": "allowed",
            "current_requests": current_requests + 1,
            "limit": total_allowed,
            "burst_used": self.burst_usage[identifier],
            "window_seconds": rate_limit.window_seconds
        }
    
    def get_rate_limit_info(self, identifier: str, endpoint: str = "default") -> Dict[str, Any]:
        """Get current rate limit information for identifier"""
        rate_limit = self.rate_limits.get(endpoint, self.rate_limits["default"])
        now = time.time()
        window_start = now - rate_limit.window_seconds
        
        # Clean old requests
        requests_queue = self.requests[identifier]
        while requests_queue and requests_queue[0] < window_start:
            requests_queue.popleft()
        
        current_requests = len(requests_queue)
        burst_used = self.burst_usage.get(identifier, 0)
        
        return {
            "current_requests": current_requests,
            "limit": rate_limit.requests,
            "burst_limit": rate_limit.burst_limit,
            "burst_used": burst_used,
            "window_seconds": rate_limit.window_seconds,
            "remaining": max(0, rate_limit.requests - current_requests),
            "burst_remaining": max(0, rate_limit.burst_limit - burst_used)
        }
    
    def reset_rate_limit(self, identifier: str):
        """Reset rate limit for identifier"""
        if identifier in self.requests:
            del self.requests[identifier]
        if identifier in self.burst_usage:
            del self.burst_usage[identifier]
    
    def add_blocked_ip(self, ip_address: str):
        """Add IP address to blocked list"""
        self.blocked_ips.add(ip_address)
    
    def remove_blocked_ip(self, ip_address: str):
        """Remove IP address from blocked list"""
        self.blocked_ips.discard(ip_address)

class RequestValidator:
    """
    Request validation and sanitization
    """
    
    def __init__(self, config: SecurityConfig):
        """
        Initialize request validator
        
        Args:
            config: Security configuration
        """
        self.config = config
        
        # Malicious patterns
        self.malicious_patterns = [
            r'<script[^>]*>.*?</script>',  # XSS
            r'javascript:',  # JavaScript injection
            r'vbscript:',  # VBScript injection
            r'onload\s*=',  # Event handler injection
            r'onerror\s*=',  # Event handler injection
            r'<iframe[^>]*>',  # Iframe injection
            r'<object[^>]*>',  # Object injection
            r'<embed[^>]*>',  # Embed injection
            r'<link[^>]*>',  # Link injection
            r'<meta[^>]*>',  # Meta injection
            r'<style[^>]*>.*?</style>',  # CSS injection
            r'expression\s*\(',  # CSS expression
            r'url\s*\(',  # CSS url
            r'@import',  # CSS import
            r'\.\./',  # Path traversal
            r'\.\.\\',  # Path traversal (Windows)
            r'%2e%2e%2f',  # URL encoded path traversal
            r'%2e%2e%5c',  # URL encoded path traversal (Windows)
            r'%00',  # Null byte injection
            r'\x00',  # Null byte injection
            r'<[^>]*>',  # HTML tags (basic check)
        ]
        
        # Compile patterns for better performance
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.malicious_patterns]
    
    def validate_request(self, request_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate request data for security issues
        
        Args:
            request_data: Request data to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check request size
        request_size = len(json.dumps(request_data))
        if request_size > self.config.max_request_size:
            issues.append(f"Request size too large: {request_size} bytes")
        
        # Validate string fields
        self._validate_string_fields(request_data, issues)
        
        # Check for malicious patterns
        self._check_malicious_patterns(request_data, issues)
        
        return len(issues) == 0, issues
    
    def _validate_string_fields(self, data: Any, issues: List[str], path: str = ""):
        """Recursively validate string fields"""
        if isinstance(data, dict):
            for key, value in data.items():
                current_path = f"{path}.{key}" if path else key
                self._validate_string_fields(value, issues, current_path)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                current_path = f"{path}[{i}]" if path else f"[{i}]"
                self._validate_string_fields(item, issues, current_path)
        elif isinstance(data, str):
            # Check string length
            if len(data) > self.config.max_query_length:
                issues.append(f"String too long at {path}: {len(data)} characters")
            
            # Check for null bytes
            if '\x00' in data:
                issues.append(f"Null byte found at {path}")
    
    def _check_malicious_patterns(self, data: Any, issues: List[str], path: str = ""):
        """Recursively check for malicious patterns"""
        if isinstance(data, dict):
            for key, value in data.items():
                current_path = f"{path}.{key}" if path else key
                self._check_malicious_patterns(value, issues, current_path)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                current_path = f"{path}[{i}]" if path else f"[{i}]"
                self._check_malicious_patterns(item, issues, current_path)
        elif isinstance(data, str):
            for pattern in self.compiled_patterns:
                if pattern.search(data):
                    issues.append(f"Malicious pattern detected at {path}: {pattern.pattern}")
                    break
    
    def sanitize_input(self, data: Any) -> Any:
        """
        Sanitize input data
        
        Args:
            data: Data to sanitize
            
        Returns:
            Sanitized data
        """
        if isinstance(data, dict):
            return {key: self.sanitize_input(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self.sanitize_input(item) for item in data]
        elif isinstance(data, str):
            # Remove null bytes
            sanitized = data.replace('\x00', '')
            
            # Truncate if too long
            if len(sanitized) > self.config.max_query_length:
                sanitized = sanitized[:self.config.max_query_length]
            
            return sanitized
        else:
            return data

class SecurityHeaders:
    """
    Security headers management
    """
    
    def __init__(self, config: SecurityConfig):
        """
        Initialize security headers
        
        Args:
            config: Security configuration
        """
        self.config = config
        
        # Default security headers
        self.default_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
            "Cross-Origin-Embedder-Policy": "require-corp",
            "Cross-Origin-Opener-Policy": "same-origin",
            "Cross-Origin-Resource-Policy": "same-origin"
        }
    
    def get_headers(self, request_origin: Optional[str] = None) -> Dict[str, str]:
        """
        Get security headers for response
        
        Args:
            request_origin: Request origin for CORS
            
        Returns:
            Dictionary of security headers
        """
        headers = self.default_headers.copy()
        
        # Add CORS headers if enabled
        if self.config.enable_cors:
            if request_origin and self._is_origin_allowed(request_origin):
                headers["Access-Control-Allow-Origin"] = request_origin
            elif "*" in self.config.allowed_origins:
                headers["Access-Control-Allow-Origin"] = "*"
            
            headers.update({
                "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, Authorization, X-API-Key",
                "Access-Control-Max-Age": "86400"
            })
        
        return headers
    
    def _is_origin_allowed(self, origin: str) -> bool:
        """Check if origin is allowed"""
        return origin in self.config.allowed_origins or "*" in self.config.allowed_origins

class SecurityMiddleware:
    """
    Main security middleware class
    """
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        """
        Initialize security middleware
        
        Args:
            config: Security configuration
        """
        self.config = config or SecurityConfig()
        self.rate_limiter = RateLimiter(self.config)
        self.request_validator = RequestValidator(self.config)
        self.security_headers = SecurityHeaders(self.config)
        
        # Request tracking
        self.request_count = 0
        self.blocked_requests = 0
        self.start_time = time.time()
    
    def process_request(
        self, 
        request_data: Dict[str, Any],
        identifier: str,
        endpoint: str = "default",
        ip_address: Optional[str] = None,
        origin: Optional[str] = None
    ) -> Tuple[bool, Dict[str, Any], Dict[str, str]]:
        """
        Process incoming request through security checks
        
        Args:
            request_data: Request data
            identifier: Unique identifier for rate limiting
            endpoint: Endpoint type
            ip_address: Client IP address
            origin: Request origin
            
        Returns:
            Tuple of (is_allowed, response_data, headers)
        """
        self.request_count += 1
        
        # Rate limiting check
        if self.config.enable_rate_limiting:
            is_allowed, rate_info = self.rate_limiter.is_allowed(identifier, endpoint, ip_address)
            if not is_allowed:
                self.blocked_requests += 1
                return False, {
                    "error": "rate_limit_exceeded",
                    "message": rate_info["message"],
                    "retry_after": rate_info.get("retry_after"),
                    "details": rate_info
                }, self.security_headers.get_headers(origin)
        
        # Request validation
        if self.config.enable_request_validation:
            is_valid, issues = self.request_validator.validate_request(request_data)
            if not is_valid:
                self.blocked_requests += 1
                return False, {
                    "error": "invalid_request",
                    "message": "Request validation failed",
                    "issues": issues
                }, self.security_headers.get_headers(origin)
        
        # Get security headers
        headers = self.security_headers.get_headers(origin) if self.config.enable_security_headers else {}
        
        return True, {}, headers
    
    def get_security_stats(self) -> Dict[str, Any]:
        """Get security statistics"""
        uptime = time.time() - self.start_time
        
        return {
            "total_requests": self.request_count,
            "blocked_requests": self.blocked_requests,
            "block_rate": self.blocked_requests / max(self.request_count, 1),
            "uptime_seconds": uptime,
            "requests_per_second": self.request_count / max(uptime, 1),
            "blocked_ips": len(self.rate_limiter.blocked_ips),
            "active_rate_limits": len(self.rate_limiter.requests)
        }
    
    def get_rate_limit_info(self, identifier: str, endpoint: str = "default") -> Dict[str, Any]:
        """Get rate limit information for identifier"""
        return self.rate_limiter.get_rate_limit_info(identifier, endpoint)
    
    def reset_rate_limit(self, identifier: str):
        """Reset rate limit for identifier"""
        self.rate_limiter.reset_rate_limit(identifier)
    
    def add_blocked_ip(self, ip_address: str):
        """Add IP to blocked list"""
        self.rate_limiter.add_blocked_ip(ip_address)
    
    def remove_blocked_ip(self, ip_address: str):
        """Remove IP from blocked list"""
        self.rate_limiter.remove_blocked_ip(ip_address)

# Global security middleware instance
security_middleware = SecurityMiddleware()

# Example usage
if __name__ == "__main__":
    # Initialize security middleware
    config = SecurityConfig(
        enable_rate_limiting=True,
        enable_request_validation=True,
        enable_security_headers=True,
        rate_limits={
            "classification": RateLimit(requests=100, window_seconds=3600, burst_limit=20)
        }
    )
    
    security = SecurityMiddleware(config)
    
    # Test request processing
    test_request = {
        "query": "I want to write a novel",
        "use_ai": True
    }
    
    # Process request
    is_allowed, response, headers = security.process_request(
        request_data=test_request,
        identifier="test_user",
        endpoint="classification",
        ip_address="192.168.1.1",
        origin="https://example.com"
    )
    
    print(f"Request allowed: {is_allowed}")
    if not is_allowed:
        print(f"Response: {response}")
    print(f"Headers: {headers}")
    
    # Get security stats
    stats = security.get_security_stats()
    print(f"Security stats: {stats}")
    
    print("Security middleware initialized successfully")



























