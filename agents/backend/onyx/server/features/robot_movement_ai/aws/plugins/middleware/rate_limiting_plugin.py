"""
Rate Limiting Plugin
====================
"""

import logging
from typing import Dict, Any
from fastapi import FastAPI
from aws.core.interfaces import MiddlewarePlugin

logger = logging.getLogger(__name__)


class RateLimitingMiddlewarePlugin(MiddlewarePlugin):
    """Rate limiting middleware plugin."""
    
    def get_name(self) -> str:
        return "rate_limiting"
    
    def is_enabled(self, config: Dict[str, Any]) -> bool:
        middleware_config = config.get("middleware", {})
        return middleware_config.get("enable_rate_limiting", True)
    
    def setup(self, app: FastAPI, config: Dict[str, Any]) -> FastAPI:
        """Setup rate limiting."""
        try:
            from slowapi import Limiter, _rate_limit_exceeded_handler
            from slowapi.util import get_remote_address
            from slowapi.errors import RateLimitExceeded
            from slowapi.middleware import SlowAPIMiddleware
            
            middleware_config = config.get("middleware", {})
            redis_url = middleware_config.get("redis_url")
            
            if redis_url:
                limiter = Limiter(
                    key_func=get_remote_address,
                    storage_uri=redis_url
                )
            else:
                limiter = Limiter(key_func=get_remote_address)
            
            app.state.limiter = limiter
            app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
            app.add_middleware(SlowAPIMiddleware)
            
            logger.info("Rate limiting enabled")
            
        except ImportError:
            logger.warning("SlowAPI not installed. Rate limiting disabled.")
        except Exception as e:
            logger.error(f"Failed to setup rate limiting: {e}")
        
        return app















