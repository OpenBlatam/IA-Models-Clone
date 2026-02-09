"""
Security Enhancements
=====================

Advanced security features for microservices.
"""

from aws.modules.security.rate_limiter import AdvancedRateLimiter, RateLimitConfig
from aws.modules.security.security_headers import SecurityHeadersMiddleware
from aws.modules.security.auth_middleware import AuthMiddleware
from aws.modules.security.input_validator import InputValidator

__all__ = [
    "AdvancedRateLimiter",
    "RateLimitConfig",
    "SecurityHeadersMiddleware",
    "AuthMiddleware",
    "InputValidator",
]















