"""
Rate Limiting Configuration
Configures rate limiting middleware using slowapi
"""

import logging
from datetime import datetime, timezone
from fastapi import FastAPI, Request
from fastapi.responses import ORJSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from ..utils.config import get_settings, get_rate_limit_config
from .models import ErrorResponse

logger = logging.getLogger(__name__)

def setup_rate_limiting(app: FastAPI) -> None:
    """Configure rate limiting for the application"""
    settings = get_settings()
    rate_limit_config = get_rate_limit_config()
    
    try:
        storage_uri = settings.redis_url if hasattr(settings, 'redis_url') and settings.redis_url else "memory://"
        
        limiter = Limiter(
            key_func=get_remote_address,
            default_limits=[f"{rate_limit_config['requests']}/{rate_limit_config['window']}second"],
            storage_uri=storage_uri
        )
        
        app.state.limiter = limiter
        
        @app.exception_handler(RateLimitExceeded)
        async def rate_limit_handler(request: Request, exc: RateLimitExceeded) -> ORJSONResponse:
            """Handle rate limit exceeded"""
            remote_address = get_remote_address(request)
            logger.warning(
                "Rate limit exceeded",
                extra={
                    "remote_address": remote_address,
                    "path": request.url.path,
                    "method": request.method
                }
            )
            
            error_response = ErrorResponse(
                error="Rate limit exceeded",
                status_code=429,
                timestamp=datetime.now(timezone.utc),
                details={"retry_after": exc.retry_after if hasattr(exc, 'retry_after') else None}
            )
            
            return ORJSONResponse(
                status_code=429,
                content=error_response.model_dump()
            )
        
        logger.info("Rate limiting configured", extra={
            "requests_per_window": rate_limit_config['requests'],
            "window_seconds": rate_limit_config['window'],
            "storage": "redis" if storage_uri != "memory://" else "memory"
        })
    except Exception as e:
        logger.error(f"Failed to setup rate limiting: {e}", exc_info=True)
        raise

