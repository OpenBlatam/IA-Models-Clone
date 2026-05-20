"""
Rate Limiting Utilities
========================
"""

from typing import Optional
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request

from ...config.app_config import get_config

config = get_config()

# Initialize rate limiter
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["100/hour", "10/minute"] if not config.debug else None
)


def get_rate_limiter() -> Limiter:
    """Obtener instancia del rate limiter."""
    return limiter

