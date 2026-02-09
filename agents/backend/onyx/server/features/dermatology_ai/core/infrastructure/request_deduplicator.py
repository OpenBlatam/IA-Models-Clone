"""
Request Deduplication
Prevents duplicate processing of identical requests
"""

import hashlib
import json
import time
from typing import Dict, Any, Optional, Callable
from functools import wraps
import asyncio
import logging

logger = logging.getLogger(__name__)


class RequestDeduplicator:
    """Deduplicate requests based on content hash"""
    
    def __init__(self, ttl: float = 60.0):
        """
        Initialize deduplicator
        
        Args:
            ttl: Time to live for deduplication cache (seconds)
        """
        self.ttl = ttl
        self.cache: Dict[str, tuple[Any, float]] = {}
        self._lock = asyncio.Lock()
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from function arguments"""
        key_data = {
            "args": args,
            "kwargs": kwargs
        }
        try:
            key_str = json.dumps(key_data, sort_keys=True, default=str)
        except (TypeError, ValueError):
            key_str = f"{args}{kwargs}"
        
        return hashlib.md5(key_str.encode()).hexdigest()
    
    async def deduplicate(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute function with deduplication
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result (cached if duplicate)
        """
        cache_key = self._generate_key(*args, **kwargs)
        
        async with self._lock:
            # Check cache
            if cache_key in self.cache:
                result, timestamp = self.cache[cache_key]
                
                # Check if still valid
                if time.time() - timestamp < self.ttl:
                    logger.debug(f"Deduplicated request: {cache_key}")
                    return result
                else:
                    # Expired, remove
                    del self.cache[cache_key]
        
        # Execute function
        if asyncio.iscoroutinefunction(func):
            result = await func(*args, **kwargs)
        else:
            result = func(*args, **kwargs)
        
        # Cache result
        async with self._lock:
            self.cache[cache_key] = (result, time.time())
        
        return result
    
    async def clear_expired(self):
        """Clear expired cache entries"""
        current_time = time.time()
        async with self._lock:
            expired_keys = [
                key for key, (_, timestamp) in self.cache.items()
                if current_time - timestamp >= self.ttl
            ]
            for key in expired_keys:
                del self.cache[key]
        
        if expired_keys:
            logger.debug(f"Cleared {len(expired_keys)} expired cache entries")
    
    def clear_all(self):
        """Clear all cache entries"""
        self.cache.clear()
        logger.debug("Cleared all deduplication cache")


def deduplicate_requests(ttl: float = 60.0):
    """
    Decorator for request deduplication
    
    Args:
        ttl: Time to live for deduplication (seconds)
    """
    deduplicator = RequestDeduplicator(ttl=ttl)
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await deduplicator.deduplicate(func, *args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            import asyncio
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(
                deduplicator.deduplicate(func, *args, **kwargs)
            )
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator















