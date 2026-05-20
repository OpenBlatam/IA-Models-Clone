"""Endpoint-level caching decorator."""
from functools import wraps
from typing import Callable, Any, Optional
from .cache import AsyncCache
import orjson
import hashlib


def cache_endpoint(
    ttl: int = 60,
    key_prefix: str = "endpoint",
    vary_by: Optional[list] = None
):
    """
    Cache decorator for FastAPI endpoints.
    
    Usage:
        @router.get("/expensive-operation")
        @cache_endpoint(ttl=300, vary_by=["query_param"])
        async def expensive_op(query_param: str):
            # Expensive operation
            return result
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get cache from app state (set in startup)
            request = args[0] if args else None
            if not request or not hasattr(request.app.state, "cache"):
                # No cache available, just execute
                return await func(*args, **kwargs)
            
            cache: AsyncCache = request.app.state.cache
            
            # Generate cache key
            cache_key_parts = [key_prefix, func.__name__]
            
            # Add varying parameters
            if vary_by:
                for param in vary_by:
                    if param in kwargs:
                        cache_key_parts.append(f"{param}:{kwargs[param]}")
            
            # Add query params if request available
            if request and hasattr(request, "query_params"):
                query_str = str(sorted(request.query_params.items()))
                cache_key_parts.append(hashlib.md5(query_str.encode()).hexdigest()[:8])
            
            cache_key = ":".join(cache_key_parts)
            
            # Check cache
            cached = await cache.get(cache_key)
            if cached:
                try:
                    return orjson.loads(cached)
                except Exception:
                    pass
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Store in cache
            try:
                cached_data = orjson.dumps(result).decode("utf-8")
                await cache.set(cache_key, cached_data, ttl)
            except Exception:
                pass
            
            return result
        
        return wrapper
    return decorator


