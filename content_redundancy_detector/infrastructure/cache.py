"""
Cache Service - Redis-based caching for stateless microservices
Implements circuit breaker pattern for resilience
"""

import logging
from typing import Optional, Any, Dict
import json
import hashlib

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("Redis not available - using in-memory cache")

from ..infrastructure.service_registry import Service
from ..core.logging_config import get_logger

logger = get_logger(__name__)


class CircuitBreaker:
    """Simple circuit breaker implementation"""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failures = 0
        self.state = "closed"  # closed, open, half_open
        self.last_failure_time = None
    
    def record_success(self):
        """Record successful operation"""
        self.failures = 0
        self.state = "closed"
    
    def record_failure(self):
        """Record failed operation"""
        self.failures += 1
        self.last_failure_time = __import__("time").time()
        
        if self.failures >= self.failure_threshold:
            self.state = "open"
            logger.warning(f"Circuit breaker opened after {self.failures} failures")
    
    def can_attempt(self) -> bool:
        """Check if operation can be attempted"""
        if self.state == "closed":
            return True
        
        if self.state == "open":
            if self.last_failure_time:
                elapsed = __import__("time").time() - self.last_failure_time
                if elapsed >= self.timeout:
                    self.state = "half_open"
                    return True
            return False
        
        return True  # half_open


class CacheService(Service):
    """
    Redis-based cache service with fallback to in-memory cache
    Stateless design for microservices
    """
    
    def __init__(self, settings):
        self.settings = settings
        self.redis_client: Optional[Any] = None
        self.memory_cache: Dict[str, Any] = {}
        self.circuit_breaker = CircuitBreaker()
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize cache service"""
        if not REDIS_AVAILABLE:
            logger.warning("Using in-memory cache (Redis not available)")
            self._initialized = True
            return
        
        try:
            self.redis_client = await redis.from_url(
                self.settings.redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            # Test connection
            await self.redis_client.ping()
            logger.info("✅ Cache service initialized (Redis)")
            self._initialized = True
            self.circuit_breaker.record_success()
        except Exception as e:
            logger.warning(f"Redis connection failed: {e} - Using in-memory cache")
            self._initialized = True
    
    async def shutdown(self) -> None:
        """Shutdown cache service"""
        if self.redis_client:
            try:
                await self.redis_client.close()
                logger.info("Cache service shutdown")
            except Exception as e:
                logger.error(f"Error closing Redis connection: {e}")
    
    def is_healthy(self) -> bool:
        """Check if cache service is healthy"""
        return self._initialized
    
    def _make_key(self, key: str) -> str:
        """Create cache key with prefix"""
        prefix = "crd:"
        return f"{prefix}{key}"
    
    def _hash_content(self, content: str) -> str:
        """Create hash for content"""
        return hashlib.sha256(content.encode()).hexdigest()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self.circuit_breaker.can_attempt():
            return None
        
        cache_key = self._make_key(key)
        
        try:
            if self.redis_client:
                value = await self.redis_client.get(cache_key)
                if value:
                    self.circuit_breaker.record_success()
                    return json.loads(value)
            else:
                # Fallback to memory
                return self.memory_cache.get(cache_key)
        
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
            self.circuit_breaker.record_failure()
            # Fallback to memory
            return self.memory_cache.get(cache_key)
        
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        if not self.circuit_breaker.can_attempt():
            return False
        
        cache_key = self._make_key(key)
        ttl = ttl or self.settings.cache_ttl
        
        try:
            if self.redis_client:
                await self.redis_client.setex(
                    cache_key,
                    ttl,
                    json.dumps(value, default=str)
                )
                self.circuit_breaker.record_success()
            else:
                # Fallback to memory (with size limit)
                if len(self.memory_cache) >= self.settings.max_cache_size:
                    # Remove oldest entry (simple FIFO)
                    oldest_key = next(iter(self.memory_cache))
                    del self.memory_cache[oldest_key]
                self.memory_cache[cache_key] = value
            
            return True
        
        except Exception as e:
            logger.warning(f"Cache set error: {e}")
            self.circuit_breaker.record_failure()
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        cache_key = self._make_key(key)
        
        try:
            if self.redis_client:
                await self.redis_client.delete(cache_key)
            else:
                self.memory_cache.pop(cache_key, None)
            return True
        except Exception as e:
            logger.warning(f"Cache delete error: {e}")
            return False
    
    async def clear(self) -> bool:
        """Clear all cache"""
        try:
            if self.redis_client:
                # Clear only our keys
                pattern = self._make_key("*")
                keys = await self.redis_client.keys(pattern)
                if keys:
                    await self.redis_client.delete(*keys)
            else:
                self.memory_cache.clear()
            return True
        except Exception as e:
            logger.warning(f"Cache clear error: {e}")
            return False






