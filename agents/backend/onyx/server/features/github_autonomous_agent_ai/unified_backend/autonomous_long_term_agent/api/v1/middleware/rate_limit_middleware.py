"""
Rate Limiting Middleware
Decorator for rate limiting endpoints
"""

import functools
import logging
from typing import Callable

from ....core.rate_limiter import RateLimiter
from ....core.exceptions import RateLimitExceededError
from ....config import settings

logger = logging.getLogger(__name__)

# Global rate limiter
_rate_limiter = RateLimiter(
    max_requests=settings.rate_limit_max_requests,
    window_seconds=settings.rate_limit_window_seconds
)


def rate_limit(key: str = "default"):
    """
    Decorator for rate limiting endpoints
    
    Args:
        key: Rate limit key (default: "default")
    
    Usage:
        @rate_limit("start_agent")
        async def start_agent(...):
            ...
    
    Raises:
        RateLimitExceededError: When rate limit is exceeded
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            allowed, remaining = await _rate_limiter.is_allowed(key)
            if not allowed:
                raise RateLimitExceededError(
                    key=key,
                    remaining=remaining
                )
            return await func(*args, **kwargs)
        return wrapper
    return decorator




