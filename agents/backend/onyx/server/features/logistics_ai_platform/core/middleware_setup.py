"""
Middleware configuration

This module provides functions to configure all application middleware.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import settings
from middleware.rate_limiter import RateLimiter, RateLimitMiddleware
from middleware.rate_limiter_redis import RedisRateLimiter, RedisRateLimitMiddleware
from middleware.performance import PerformanceMiddleware
from middleware.security import SecurityHeadersMiddleware, RequestIDMiddleware
from middleware.logging import RequestLoggingMiddleware
from utils.cache import cache_service


def setup_middleware(app: FastAPI) -> None:
    """Configure all application middleware"""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(RequestLoggingMiddleware)
    
    try:
        if hasattr(settings, 'REDIS_URL') and settings.REDIS_URL:
            rate_limiter = RedisRateLimiter(
                max_requests=settings.RATE_LIMIT_REQUESTS,
                window_seconds=settings.RATE_LIMIT_WINDOW
            )
            app.add_middleware(RedisRateLimitMiddleware, rate_limiter=rate_limiter)
        else:
            raise ValueError("Redis URL not configured")
    except Exception:
        rate_limiter = RateLimiter(
            max_requests=settings.RATE_LIMIT_REQUESTS,
            window_seconds=settings.RATE_LIMIT_WINDOW
        )
        app.add_middleware(RateLimitMiddleware, rate_limiter=rate_limiter)
    
    app.add_middleware(PerformanceMiddleware)

