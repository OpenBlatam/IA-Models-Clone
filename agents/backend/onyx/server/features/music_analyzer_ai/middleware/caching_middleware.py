"""
Caching Middleware - Cache responses
"""

from typing import Any, Callable, Dict
import hashlib
import json
import logging

from .middleware import BaseMiddleware

logger = logging.getLogger(__name__)


class CachingMiddleware(BaseMiddleware):
    """
    Middleware that caches responses
    """
    
    def __init__(self, cache: Dict[str, Any] = None, ttl: int = 3600):
        super().__init__("CachingMiddleware")
        self.cache = cache or {}
        self.ttl = ttl
        self._access_times: Dict[str, float] = {}
    
    def _get_cache_key(self, request: Any) -> str:
        """Generate cache key from request"""
        if isinstance(request, dict):
            request_str = json.dumps(request, sort_keys=True)
        else:
            request_str = str(request)
        return hashlib.md5(request_str.encode()).hexdigest()
    
    def process(self, request: Any, next_handler: Callable) -> Any:
        """Process with caching"""
        cache_key = self._get_cache_key(request)
        
        # Check cache
        if cache_key in self.cache:
            logger.debug(f"Cache hit for key: {cache_key}")
            return self.cache[cache_key]
        
        # Process request
        response = next_handler(request)
        
        # Cache response
        self.cache[cache_key] = response
        self._access_times[cache_key] = self._get_current_time()
        
        logger.debug(f"Cached response for key: {cache_key}")
        return response
    
    def _get_current_time(self) -> float:
        """Get current time (can be overridden for testing)"""
        import time
        return time.time()
    
    def clear_cache(self):
        """Clear cache"""
        self.cache.clear()
        self._access_times.clear()
        logger.info("Cache cleared")








