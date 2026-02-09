"""
API Layer Optimizations

Optimizations for:
- Request/Response serialization
- Response compression
- Connection pooling
- Query optimization
- Caching strategies
- Rate limiting optimization
"""

import logging
import gzip
import json
from typing import Optional, Dict, Any, List
import orjson
import ujson
from functools import wraps
import time
from fastapi import Request, Response
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class FastJSONSerializer:
    """Ultra-fast JSON serialization."""
    
    @staticmethod
    def serialize(data: Any) -> bytes:
        """Serialize to JSON bytes using fastest method."""
        try:
            # Try orjson first (fastest)
            return orjson.dumps(data, option=orjson.OPT_SERIALIZE_NUMPY)
        except Exception:
            try:
                # Fallback to ujson
                return ujson.dumps(data).encode('utf-8')
            except Exception:
                # Fallback to standard json
                return json.dumps(data).encode('utf-8')
    
    @staticmethod
    def deserialize(data: bytes) -> Any:
        """Deserialize JSON bytes."""
        try:
            return orjson.loads(data)
        except Exception:
            try:
                return ujson.loads(data)
            except Exception:
                return json.loads(data)


class ResponseCompressor:
    """Smart response compression."""
    
    @staticmethod
    def compress(data: bytes, min_size: int = 500) -> tuple[bytes, bool]:
        """
        Compress data if beneficial.
        
        Args:
            data: Data to compress
            min_size: Minimum size to compress
            
        Returns:
            (compressed_data, was_compressed)
        """
        if len(data) < min_size:
            return data, False
        
        try:
            compressed = gzip.compress(data, compresslevel=6)
            # Only use if compression saves space
            if len(compressed) < len(data) * 0.9:
                return compressed, True
        except Exception as e:
            logger.debug(f"Compression failed: {e}")
        
        return data, False
    
    @staticmethod
    def compress_response(response: Response, min_size: int = 500) -> Response:
        """Compress response if beneficial."""
        if response.body and len(response.body) >= min_size:
            compressed, was_compressed = ResponseCompressor.compress(response.body, min_size)
            if was_compressed:
                response.body = compressed
                response.headers["Content-Encoding"] = "gzip"
        
        return response


class QueryOptimizer:
    """Database query optimization."""
    
    @staticmethod
    def optimize_query_filters(filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize query filters.
        
        Args:
            filters: Filter dictionary
            
        Returns:
            Optimized filters
        """
        optimized = {}
        
        # Remove None values
        optimized = {k: v for k, v in filters.items() if v is not None}
        
        # Sort filters by selectivity (most selective first)
        # This is a simplified version - in production, use actual statistics
        selectivity_order = ['id', 'user_id', 'created_at', 'genre', 'mood']
        
        sorted_filters = {}
        for key in selectivity_order:
            if key in optimized:
                sorted_filters[key] = optimized.pop(key)
        
        # Add remaining filters
        sorted_filters.update(optimized)
        
        return sorted_filters
    
    @staticmethod
    def build_efficient_query(base_query, filters: Dict[str, Any], limit: int = 100):
        """
        Build efficient database query.
        
        Args:
            base_query: Base SQLAlchemy query
            filters: Filter dictionary
            limit: Result limit
            
        Returns:
            Optimized query
        """
        # Apply filters in optimal order
        optimized_filters = QueryOptimizer.optimize_query_filters(filters)
        
        for key, value in optimized_filters.items():
            if hasattr(base_query.column_descriptions[0]['entity'], key):
                base_query = base_query.filter(getattr(base_query.column_descriptions[0]['entity'], key) == value)
        
        # Apply limit
        if limit:
            base_query = base_query.limit(limit)
        
        return base_query


class CacheStrategy:
    """Smart caching strategies."""
    
    @staticmethod
    def get_cache_key(request: Request, include_query: bool = True) -> str:
        """
        Generate cache key from request.
        
        Args:
            request: FastAPI request
            include_query: Include query parameters
            
        Returns:
            Cache key
        """
        import hashlib
        
        key_parts = [
            request.method,
            request.url.path,
        ]
        
        if include_query and request.query_params:
            key_parts.append(str(sorted(request.query_params.items())))
        
        # Include user if authenticated
        if hasattr(request.state, 'user_id'):
            key_parts.append(f"user:{request.state.user_id}")
        
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    @staticmethod
    def get_cache_ttl(endpoint: str) -> int:
        """
        Get cache TTL for endpoint.
        
        Args:
            endpoint: Endpoint path
            
        Returns:
            TTL in seconds
        """
        # Static content - long TTL
        if '/static/' in endpoint or '/assets/' in endpoint:
            return 86400  # 24 hours
        
        # Dynamic content - short TTL
        if '/generate' in endpoint or '/create' in endpoint:
            return 0  # No cache
        
        # List endpoints - medium TTL
        if '/list' in endpoint or '/search' in endpoint:
            return 300  # 5 minutes
        
        # Detail endpoints - longer TTL
        if '/get' in endpoint or '/detail' in endpoint:
            return 3600  # 1 hour
        
        # Default
        return 60  # 1 minute


class RateLimitOptimizer:
    """Optimized rate limiting."""
    
    def __init__(self):
        """Initialize rate limit optimizer."""
        self.request_counts: Dict[str, List[float]] = {}
        self.window_size = 60  # 1 minute window
    
    def check_rate_limit(
        self,
        identifier: str,
        max_requests: int = 100,
        window_seconds: int = 60
    ) -> tuple[bool, Optional[int]]:
        """
        Check rate limit efficiently.
        
        Args:
            identifier: Client identifier
            max_requests: Maximum requests
            window_seconds: Time window
            
        Returns:
            (is_allowed, retry_after_seconds)
        """
        now = time.time()
        
        # Clean old entries
        if identifier in self.request_counts:
            self.request_counts[identifier] = [
                t for t in self.request_counts[identifier]
                if now - t < window_seconds
            ]
        else:
            self.request_counts[identifier] = []
        
        # Check limit
        if len(self.request_counts[identifier]) >= max_requests:
            oldest_request = min(self.request_counts[identifier])
            retry_after = int(window_seconds - (now - oldest_request)) + 1
            return False, retry_after
        
        # Record request
        self.request_counts[identifier].append(now)
        return True, None


def optimize_response(func):
    """Decorator to optimize API responses."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        
        # Execute function
        result = await func(*args, **kwargs)
        
        # Optimize response
        if isinstance(result, dict):
            # Serialize with fast JSON
            response_data = FastJSONSerializer.serialize(result)
            
            # Compress if beneficial
            compressed, was_compressed = ResponseCompressor.compress(response_data)
            
            # Create response
            response = Response(
                content=compressed,
                media_type="application/json"
            )
            
            if was_compressed:
                response.headers["Content-Encoding"] = "gzip"
            
            # Add performance headers
            elapsed = time.time() - start_time
            response.headers["X-Response-Time"] = f"{elapsed:.3f}s"
            
            return response
        
        return result
    
    return wrapper


class ConnectionPoolOptimizer:
    """Database connection pool optimization."""
    
    @staticmethod
    def get_optimal_pool_size(
        expected_concurrent_requests: int = 100,
        avg_query_time_ms: float = 50.0
    ) -> Dict[str, int]:
        """
        Calculate optimal connection pool size.
        
        Args:
            expected_concurrent_requests: Expected concurrent requests
            avg_query_time_ms: Average query time in milliseconds
            
        Returns:
            Optimal pool configuration
        """
        # Formula: pool_size = (concurrent_requests * query_time) / target_response_time
        target_response_time_ms = 100.0
        
        pool_size = int(
            (expected_concurrent_requests * avg_query_time_ms) / target_response_time_ms
        )
        
        # Clamp to reasonable values
        pool_size = max(5, min(pool_size, 50))
        
        return {
            'pool_size': pool_size,
            'max_overflow': pool_size // 2,
            'pool_timeout': 30,
            'pool_recycle': 3600
        }


class BatchRequestProcessor:
    """Optimized batch request processing."""
    
    @staticmethod
    async def process_batch_requests(
        requests: List[Dict[str, Any]],
        processor_func,
        max_concurrent: int = 10
    ) -> List[Any]:
        """
        Process batch requests efficiently.
        
        Args:
            requests: List of requests
            processor_func: Function to process each request
            max_concurrent: Maximum concurrent processing
            
        Returns:
            List of results
        """
        import asyncio
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_one(request: Dict[str, Any], index: int):
            async with semaphore:
                try:
                    if asyncio.iscoroutinefunction(processor_func):
                        result = await processor_func(**request)
                    else:
                        result = processor_func(**request)
                    return {'index': index, 'result': result, 'error': None}
                except Exception as e:
                    logger.error(f"Batch request {index} failed: {e}")
                    return {'index': index, 'result': None, 'error': str(e)}
        
        # Process all requests
        tasks = [process_one(req, i) for i, req in enumerate(requests)]
        results = await asyncio.gather(*tasks)
        
        # Sort by index and extract results
        results.sort(key=lambda x: x['index'])
        return [r['result'] for r in results]


class ResponseCache:
    """In-memory response cache."""
    
    def __init__(self, max_size: int = 1000, ttl: int = 300):
        """
        Initialize response cache.
        
        Args:
            max_size: Maximum cache size
            ttl: Time to live in seconds
        """
        from collections import OrderedDict
        
        self.cache: OrderedDict = OrderedDict()
        self.max_size = max_size
        self.ttl = ttl
        self.timestamps: Dict[str, float] = {}
    
    def get(self, key: str) -> Optional[bytes]:
        """Get cached response."""
        if key not in self.cache:
            return None
        
        # Check TTL
        if time.time() - self.timestamps[key] > self.ttl:
            del self.cache[key]
            del self.timestamps[key]
            return None
        
        # Move to end (LRU)
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def set(self, key: str, value: bytes) -> None:
        """Set cached response."""
        # Remove if exists
        if key in self.cache:
            del self.cache[key]
        
        # Remove oldest if full
        if len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            del self.timestamps[oldest_key]
        
        # Add new entry
        self.cache[key] = value
        self.timestamps[key] = time.time()
    
    def clear(self) -> None:
        """Clear cache."""
        self.cache.clear()
        self.timestamps.clear()








