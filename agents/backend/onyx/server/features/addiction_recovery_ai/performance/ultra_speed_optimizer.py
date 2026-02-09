"""
Ultra-Speed Performance Optimizer
Maximum performance optimizations for ultra-fast responses
"""

import asyncio
import logging
import time
import hashlib
from typing import Any, Dict, List, Optional, Callable, Coroutine
from functools import lru_cache
from collections import defaultdict
import json

logger = logging.getLogger(__name__)

# Try to import advanced compression
try:
    import brotli
    BROTLI_AVAILABLE = True
except ImportError:
    BROTLI_AVAILABLE = False
    logger.warning("Brotli not available. Install brotli for better compression.")

# Try to import Redis for ultra-fast caching
try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not available. Using in-memory cache.")


class UltraSpeedOptimizer:
    """
    Ultra-speed performance optimizer
    
    Features:
    - Response streaming
    - Brotli compression (better than gzip)
    - Request coalescing
    - Smart prefetching
    - Early response optimization
    - Ultra-fast Redis caching
    - Response pre-computation
    """
    
    def __init__(self, redis_url: Optional[str] = None):
        self.redis_client: Optional[aioredis.Redis] = None
        self._in_memory_cache: Dict[str, tuple] = {}
        self._request_queue: Dict[str, List[asyncio.Future]] = defaultdict(list)
        self._prefetch_cache: Dict[str, Any] = {}
        self._response_cache: Dict[str, bytes] = {}
        self._coalesce_window = 0.01  # 10ms window for request coalescing
        
        # Initialize Redis if available
        if REDIS_AVAILABLE and redis_url:
            asyncio.create_task(self._init_redis(redis_url))
    
    async def _init_redis(self, redis_url: str):
        """Initialize Redis connection"""
        try:
            self.redis_client = await aioredis.from_url(
                redis_url,
                encoding="utf-8",
                decode_responses=False  # Keep as bytes for performance
            )
            logger.info("Redis connected for ultra-fast caching")
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}")
            self.redis_client = None
    
    async def get_cached(self, key: str, ttl: int = 300) -> Optional[bytes]:
        """Get cached value (Redis or in-memory)"""
        # Try Redis first
        if self.redis_client:
            try:
                value = await self.redis_client.get(key)
                if value:
                    return value
            except Exception as e:
                logger.warning(f"Redis get failed: {e}")
        
        # Fallback to in-memory
        if key in self._in_memory_cache:
            value, cached_time = self._in_memory_cache[key]
            if time.time() - cached_time < ttl:
                return value
            else:
                del self._in_memory_cache[key]
        
        return None
    
    async def set_cached(self, key: str, value: bytes, ttl: int = 300):
        """Set cached value (Redis or in-memory)"""
        # Try Redis first
        if self.redis_client:
            try:
                await self.redis_client.setex(key, ttl, value)
                return
            except Exception as e:
                logger.warning(f"Redis set failed: {e}")
        
        # Fallback to in-memory
        self._in_memory_cache[key] = (value, time.time())
    
    def compress_brotli(self, data: bytes) -> bytes:
        """Compress with Brotli (better compression than gzip)"""
        if BROTLI_AVAILABLE:
            return brotli.compress(data, quality=6)
        else:
            # Fallback to gzip
            import gzip
            return gzip.compress(data, compresslevel=6)
    
    def get_compression_algorithm(self, accept_encoding: str) -> Optional[str]:
        """Determine best compression algorithm"""
        if BROTLI_AVAILABLE and "br" in accept_encoding:
            return "br"
        elif "gzip" in accept_encoding:
            return "gzip"
        return None
    
    async def coalesce_requests(
        self,
        key: str,
        request_func: Callable[[], Coroutine]
    ) -> Any:
        """
        Coalesce duplicate requests within a time window
        Multiple identical requests return the same result
        """
        # Check if there's already a pending request
        if key in self._request_queue and self._request_queue[key]:
            # Wait for existing request
            future = asyncio.Future()
            self._request_queue[key].append(future)
            
            # Wait for result
            result = await future
            return result
        
        # Create new request
        future = asyncio.Future()
        self._request_queue[key].append(future)
        
        try:
            # Execute request
            result = await request_func()
            
            # Resolve all waiting futures
            for f in self._request_queue[key]:
                if not f.done():
                    f.set_result(result)
            
            return result
        except Exception as e:
            # Propagate error to all waiting futures
            for f in self._request_queue[key]:
                if not f.done():
                    f.set_exception(e)
            raise
        finally:
            # Clear queue after short delay
            await asyncio.sleep(self._coalesce_window)
            if key in self._request_queue:
                del self._request_queue[key]
    
    async def prefetch_likely_data(self, user_id: str, endpoint: str):
        """Prefetch data that's likely to be requested next"""
        # Common prefetch patterns
        prefetch_patterns = {
            "/recovery/profile": f"/recovery/stats/{user_id}",
            "/recovery/stats": f"/recovery/progress/{user_id}",
            "/recovery/progress": f"/recovery/timeline/{user_id}",
        }
        
        if endpoint in prefetch_patterns:
            next_endpoint = prefetch_patterns[endpoint]
            # Prefetch in background (don't wait)
            asyncio.create_task(self._prefetch_data(next_endpoint))
    
    async def _prefetch_data(self, endpoint: str):
        """Prefetch data for endpoint"""
        # This would call the actual endpoint and cache result
        # Placeholder implementation
        pass
    
    def generate_cache_key(self, method: str, path: str, query: str, body: Optional[bytes] = None) -> str:
        """Generate cache key for request"""
        key_data = f"{method}:{path}:{query}"
        if body:
            key_data += hashlib.md5(body).hexdigest()
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def get_or_compute(
        self,
        key: str,
        compute_func: Callable[[], Coroutine],
        ttl: int = 300
    ) -> Any:
        """Get from cache or compute and cache"""
        # Try cache first
        cached = await self.get_cached(key, ttl)
        if cached:
            try:
                from performance.serialization_optimizer import get_serializer
                serializer = get_serializer()
                return serializer.deserialize_json(cached)
            except Exception:
                pass
        
        # Compute
        result = await compute_func()
        
        # Cache result
        try:
            from performance.serialization_optimizer import get_serializer
            serializer = get_serializer()
            serialized = serializer.serialize_json(result)
            await self.set_cached(key, serialized, ttl)
        except Exception as e:
            logger.warning(f"Failed to cache result: {e}")
        
        return result
    
    def should_stream_response(self, data_size: int, threshold: int = 100000) -> bool:
        """Determine if response should be streamed"""
        return data_size > threshold
    
    async def stream_response(
        self,
        data_generator: Callable[[], Coroutine],
        chunk_size: int = 8192
    ):
        """Stream response in chunks"""
        async for chunk in data_generator():
            yield chunk


class ResponsePreComputer:
    """Pre-compute common responses"""
    
    def __init__(self):
        self._precomputed: Dict[str, bytes] = {}
    
    def precompute_response(self, endpoint: str, data: Any):
        """Pre-compute response for endpoint"""
        try:
            from performance.serialization_optimizer import get_serializer
            serializer = get_serializer()
            self._precomputed[endpoint] = serializer.serialize_json(data)
        except Exception as e:
            logger.warning(f"Failed to precompute response: {e}")
    
    def get_precomputed(self, endpoint: str) -> Optional[bytes]:
        """Get precomputed response"""
        return self._precomputed.get(endpoint)


class EarlyResponseOptimizer:
    """Optimize for early response patterns"""
    
    def __init__(self):
        self._early_responses: Dict[str, bytes] = {}
    
    def register_early_response(self, pattern: str, response: bytes):
        """Register early response for pattern"""
        self._early_responses[pattern] = response
    
    def get_early_response(self, path: str) -> Optional[bytes]:
        """Get early response if pattern matches"""
        for pattern, response in self._early_responses.items():
            if path.startswith(pattern):
                return response
        return None


# Global optimizer instance
_ultra_optimizer: Optional[UltraSpeedOptimizer] = None
_precomputer: Optional[ResponsePreComputer] = None
_early_optimizer: Optional[EarlyResponseOptimizer] = None


def get_ultra_optimizer(redis_url: Optional[str] = None) -> UltraSpeedOptimizer:
    """Get global ultra-speed optimizer"""
    global _ultra_optimizer
    if _ultra_optimizer is None:
        _ultra_optimizer = UltraSpeedOptimizer(redis_url)
    return _ultra_optimizer


def get_precomputer() -> ResponsePreComputer:
    """Get global response pre-computer"""
    global _precomputer
    if _precomputer is None:
        _precomputer = ResponsePreComputer()
    return _precomputer


def get_early_optimizer() -> EarlyResponseOptimizer:
    """Get global early response optimizer"""
    global _early_optimizer
    if _early_optimizer is None:
        _early_optimizer = EarlyResponseOptimizer()
    return _early_optimizer















