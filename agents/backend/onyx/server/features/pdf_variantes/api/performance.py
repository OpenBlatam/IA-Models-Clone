"""
PDF Variantes API - Performance Optimizations
High-performance optimizations for fast response times
"""

import functools
import time
from typing import Any, Callable, Dict, Optional
from fastapi import Request, Response
from fastapi.responses import JSONResponse
import orjson
import gzip
import asyncio


# Use orjson for faster JSON serialization
def json_response(
    content: Any,
    status_code: int = 200,
    headers: Optional[Dict[str, str]] = None,
    compress: bool = False
) -> Response:
    """Fast JSON response using orjson"""
    try:
        import orjson
    except ImportError:
        # Fallback to standard json if orjson not available
        import json
        body = json.dumps(content).encode('utf-8')
    else:
        # Serialize with orjson (much faster than default json)
        body = orjson.dumps(content, option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_NON_STR_KEYS)
    
    headers = headers or {}
    
    # Compress if requested and body is large enough
    if compress and len(body) > 1024:  # 1KB threshold
        body = gzip.compress(body)
        headers["Content-Encoding"] = "gzip"
        headers["Content-Length"] = str(len(body))
    
    headers["Content-Type"] = "application/json"
    
    return Response(
        content=body,
        status_code=status_code,
        headers=headers,
        media_type="application/json"
    )


def cache_response(
    max_age: int = 300,
    vary: Optional[str] = None
):
    """Decorator to cache function responses"""
    def decorator(func: Callable):
        cache = {}
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key
            cache_key = f"{func.__name__}:{args}:{kwargs}"
            
            # Check cache
            if cache_key in cache:
                cached_data, cached_time = cache[cache_key]
                if time.time() - cached_time < max_age:
                    return cached_data
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            cache[cache_key] = (result, time.time())
            
            return result
        
        return wrapper
    return decorator


def async_cache(ttl: int = 300):
    """Async function result caching"""
    cache: Dict[str, tuple[Any, float]] = {}
    
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            
            # Check cache
            if key in cache:
                value, timestamp = cache[key]
                if time.time() - timestamp < ttl:
                    return value
            
            # Execute and cache
            result = await func(*args, **kwargs)
            cache[key] = (result, time.time())
            
            return result
        
        return wrapper
    return decorator


def batch_requests(batch_size: int = 10, wait_ms: int = 50):
    """Batch multiple requests together for efficiency"""
    pending_requests = []
    last_batch_time = time.time()
    
    async def process_batch():
        if not pending_requests:
            return
        
        # Execute all pending requests
        results = await asyncio.gather(*[req["func"](*req["args"], **req["kwargs"]) for req in pending_requests])
        
        # Resolve all futures
        for req, result in zip(pending_requests, results):
            if not req["future"].done():
                req["future"].set_result(result)
        
        pending_requests.clear()
    
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            future = asyncio.Future()
            
            pending_requests.append({
                "func": func,
                "args": args,
                "kwargs": kwargs,
                "future": future
            })
            
            # Process batch if full or timeout
            current_time = time.time()
            if len(pending_requests) >= batch_size or (current_time - last_batch_time) * 1000 >= wait_ms:
                await process_batch()
                last_batch_time = current_time
            
            return await future
        
        return wrapper
    return decorator


class FastMiddleware:
    """Optimized middleware for performance"""
    
    @staticmethod
    async def compress_response(request: Request, call_next):
        """Compress responses larger than 1KB"""
        response = await call_next(request)
        
        # Only compress if not already compressed
        if "Content-Encoding" not in response.headers and response.status_code == 200:
            body = b""
            async for chunk in response.body_iterator:
                body += chunk
            
            if len(body) > 1024:
                compressed = gzip.compress(body)
                if len(compressed) < len(body):
                    response.body = compressed
                    response.headers["Content-Encoding"] = "gzip"
                    response.headers["Content-Length"] = str(len(compressed))
        
        return response
    
    @staticmethod
    async def optimize_headers(request: Request, call_next):
        """Add performance headers"""
        response = await call_next(request)
        
        # Performance headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Connection"] = "keep-alive"
        
        # Enable HTTP/2 Server Push hints
        if request.url.path.startswith("/api/v1/pdf/documents"):
            response.headers["Link"] = "</api/v1/pdf/documents>; rel=prefetch"
        
        return response


def optimize_json_serialization():
    """Configure orjson for optimal performance"""
    try:
        import orjson
        # Configure orjson defaults
        # Note: orjson options are set when calling dumps(), not globally
        pass
    except ImportError:
        # orjson is optional - will fallback to standard json
        pass


def eager_load(relationships: list):
    """Eager load relationships to avoid N+1 queries"""
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # This would integrate with ORM eager loading
            # For now, it's a placeholder
            return await func(*args, **kwargs)
        return wrapper
    return decorator


def connection_pool_optimizer():
    """Optimize database connection pooling"""
    # This would configure connection pool settings
    # For async database connections
    pass


def response_streaming(threshold: int = 1024 * 1024):
    """Stream large responses instead of loading everything in memory"""
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            
            # If result is large, stream it
            if isinstance(result, (list, dict)):
                import sys
                size = sys.getsizeof(str(result))
                if size > threshold:
                    # Convert to streaming response
                    from fastapi.responses import StreamingResponse
                    import io
                    
                    async def generate():
                        chunks = orjson.dumps(result, option=orjson.OPT_SERIALIZE_NUMPY)
                        yield chunks
                    
                    return StreamingResponse(generate(), media_type="application/json")
            
            return result
        
        return wrapper
    return decorator


# Performance monitoring
class PerformanceMonitor:
    """Monitor and log slow requests"""
    
    SLOW_REQUEST_THRESHOLD = 1.0  # seconds
    
    @staticmethod
    async def monitor(request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        duration = time.time() - start_time
        
        if duration > PerformanceMonitor.SLOW_REQUEST_THRESHOLD:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Slow request: {request.method} {request.url.path} "
                f"took {duration:.3f}s"
            )
        
        response.headers["X-Response-Time"] = f"{duration:.3f}s"
        return response

