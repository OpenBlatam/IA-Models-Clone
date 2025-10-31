"""
Cache Manager Module

High-performance caching with:
- Redis integration with fallback to in-memory cache
- TTL support and cache invalidation
- Performance monitoring and metrics
- Async operations with connection pooling
- Cache warming and preloading strategies
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Union
import json
import time
import asyncio
import hashlib
import structlog
from abc import ABC, abstractmethod
from dataclasses import dataclass
from contextlib import asynccontextmanager

logger = structlog.get_logger("cache")

# =============================================================================
# CACHE CONFIGURATION
# =============================================================================

@dataclass
class CacheConfig:
    """Cache configuration settings."""
    # Redis settings
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    redis_ssl: bool = False
    
    # Connection pool settings
    max_connections: int = 10
    retry_on_timeout: bool = True
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0
    
    # Cache settings
    default_ttl: int = 3600  # 1 hour
    max_memory: str = "100mb"
    compression_threshold: int = 1024  # 1KB
    
    # Fallback settings
    enable_fallback: bool = True
    fallback_max_size: int = 1000
    fallback_cleanup_interval: int = 300  # 5 minutes

# =============================================================================
# CACHE INTERFACES
# =============================================================================

class CacheBackend(ABC):
    """Abstract base class for cache backends."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """Clear all cache entries."""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close cache connection."""
        pass

# =============================================================================
# REDIS CACHE BACKEND
# =============================================================================

class RedisCacheBackend(CacheBackend):
    """Redis cache backend implementation."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.redis = None
        self._connected = False
    
    async def initialize(self) -> None:
        """Initialize Redis connection."""
        try:
            import redis.asyncio as redis
            
            # Create Redis connection pool
            self.redis = redis.ConnectionPool(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                password=self.config.redis_password,
                ssl=self.config.redis_ssl,
                max_connections=self.config.max_connections,
                retry_on_timeout=self.config.retry_on_timeout,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                decode_responses=True
            )
            
            # Test connection
            client = redis.Redis(connection_pool=self.redis)
            await client.ping()
            await client.close()
            
            self._connected = True
            logger.info("Redis cache backend initialized successfully")
            
        except ImportError:
            logger.warning("Redis not available, falling back to in-memory cache")
            raise
        except Exception as e:
            logger.error("Failed to initialize Redis cache backend", error=str(e))
            raise
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache."""
        if not self._connected:
            return None
        
        try:
            client = redis.Redis(connection_pool=self.redis)
            value = await client.get(key)
            await client.close()
            
            if value is None:
                return None
            
            # Deserialize JSON value
            return json.loads(value)
            
        except Exception as e:
            logger.warning("Redis get error", key=key, error=str(e))
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Redis cache."""
        if not self._connected:
            return False
        
        try:
            client = redis.Redis(connection_pool=self.redis)
            
            # Serialize value to JSON
            serialized_value = json.dumps(value, default=str)
            
            # Set with TTL
            ttl = ttl or self.config.default_ttl
            result = await client.setex(key, ttl, serialized_value)
            await client.close()
            
            return bool(result)
            
        except Exception as e:
            logger.warning("Redis set error", key=key, error=str(e))
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from Redis cache."""
        if not self._connected:
            return False
        
        try:
            client = redis.Redis(connection_pool=self.redis)
            result = await client.delete(key)
            await client.close()
            
            return bool(result)
            
        except Exception as e:
            logger.warning("Redis delete error", key=key, error=str(e))
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis cache."""
        if not self._connected:
            return False
        
        try:
            client = redis.Redis(connection_pool=self.redis)
            result = await client.exists(key)
            await client.close()
            
            return bool(result)
            
        except Exception as e:
            logger.warning("Redis exists error", key=key, error=str(e))
            return False
    
    async def clear(self) -> bool:
        """Clear all Redis cache entries."""
        if not self._connected:
            return False
        
        try:
            client = redis.Redis(connection_pool=self.redis)
            result = await client.flushdb()
            await client.close()
            
            return bool(result)
            
        except Exception as e:
            logger.warning("Redis clear error", error=str(e))
            return False
    
    async def close(self) -> None:
        """Close Redis connection."""
        if self.redis:
            await self.redis.disconnect()
            self._connected = False
            logger.info("Redis cache backend closed")

# =============================================================================
# IN-MEMORY CACHE BACKEND
# =============================================================================

class InMemoryCacheBackend(CacheBackend):
    """In-memory cache backend implementation."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache: Dict[str, Dict[str, Any]] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
    
    async def initialize(self) -> None:
        """Initialize in-memory cache."""
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("In-memory cache backend initialized")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from in-memory cache."""
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        
        # Check TTL
        if time.time() > entry['expires_at']:
            del self.cache[key]
            return None
        
        return entry['value']
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in in-memory cache."""
        ttl = ttl or self.config.default_ttl
        expires_at = time.time() + ttl
        
        self.cache[key] = {
            'value': value,
            'expires_at': expires_at,
            'created_at': time.time()
        }
        
        # Check size limit
        if len(self.cache) > self.config.fallback_max_size:
            await self._evict_oldest()
        
        return True
    
    async def delete(self, key: str) -> bool:
        """Delete key from in-memory cache."""
        if key in self.cache:
            del self.cache[key]
            return True
        return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in in-memory cache."""
        if key not in self.cache:
            return False
        
        # Check TTL
        if time.time() > self.cache[key]['expires_at']:
            del self.cache[key]
            return False
        
        return True
    
    async def clear(self) -> bool:
        """Clear all in-memory cache entries."""
        self.cache.clear()
        return True
    
    async def close(self) -> None:
        """Close in-memory cache."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        self.cache.clear()
        logger.info("In-memory cache backend closed")
    
    async def _cleanup_loop(self) -> None:
        """Background cleanup loop for expired entries."""
        while True:
            try:
                await asyncio.sleep(self.config.fallback_cleanup_interval)
                await self._cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("Cache cleanup error", error=str(e))
    
    async def _cleanup_expired(self) -> None:
        """Remove expired entries from cache."""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self.cache.items()
            if current_time > entry['expires_at']
        ]
        
        for key in expired_keys:
            del self.cache[key]
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    async def _evict_oldest(self) -> None:
        """Evict oldest entries when cache is full."""
        if not self.cache:
            return
        
        # Sort by creation time and remove oldest 10%
        sorted_entries = sorted(
            self.cache.items(),
            key=lambda x: x[1]['created_at']
        )
        
        evict_count = max(1, len(sorted_entries) // 10)
        for key, _ in sorted_entries[:evict_count]:
            del self.cache[key]
        
        logger.debug(f"Evicted {evict_count} oldest cache entries")

# =============================================================================
# CACHE MANAGER
# =============================================================================

class CacheManager:
    """High-level cache manager with fallback support."""
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self.primary_backend: Optional[CacheBackend] = None
        self.fallback_backend: Optional[CacheBackend] = None
        self._initialized = False
        self._stats = {
            'hits': 0,
            'misses': 0,
            'errors': 0,
            'fallback_uses': 0
        }
    
    async def initialize(self) -> None:
        """Initialize cache manager with primary and fallback backends."""
        try:
            # Initialize primary backend (Redis)
            self.primary_backend = RedisCacheBackend(self.config)
            await self.primary_backend.initialize()
            logger.info("Primary cache backend (Redis) initialized")
            
        except Exception as e:
            logger.warning("Failed to initialize primary cache backend", error=str(e))
            self.primary_backend = None
        
        # Initialize fallback backend (in-memory)
        if self.config.enable_fallback:
            self.fallback_backend = InMemoryCacheBackend(self.config)
            await self.fallback_backend.initialize()
            logger.info("Fallback cache backend (in-memory) initialized")
        
        self._initialized = True
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache with fallback support."""
        if not self._initialized:
            return None
        
        # Try primary backend first
        if self.primary_backend:
            try:
                value = await self.primary_backend.get(key)
                if value is not None:
                    self._stats['hits'] += 1
                    return value
                else:
                    self._stats['misses'] += 1
            except Exception as e:
                logger.warning("Primary cache get error", key=key, error=str(e))
                self._stats['errors'] += 1
        
        # Try fallback backend
        if self.fallback_backend:
            try:
                value = await self.fallback_backend.get(key)
                if value is not None:
                    self._stats['hits'] += 1
                    self._stats['fallback_uses'] += 1
                    return value
                else:
                    self._stats['misses'] += 1
            except Exception as e:
                logger.warning("Fallback cache get error", key=key, error=str(e))
                self._stats['errors'] += 1
        
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with fallback support."""
        if not self._initialized:
            return False
        
        success = False
        
        # Set in primary backend
        if self.primary_backend:
            try:
                if await self.primary_backend.set(key, value, ttl):
                    success = True
            except Exception as e:
                logger.warning("Primary cache set error", key=key, error=str(e))
                self._stats['errors'] += 1
        
        # Set in fallback backend
        if self.fallback_backend:
            try:
                if await self.fallback_backend.set(key, value, ttl):
                    success = True
            except Exception as e:
                logger.warning("Fallback cache set error", key=key, error=str(e))
                self._stats['errors'] += 1
        
        return success
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache with fallback support."""
        if not self._initialized:
            return False
        
        success = False
        
        # Delete from primary backend
        if self.primary_backend:
            try:
                if await self.primary_backend.delete(key):
                    success = True
            except Exception as e:
                logger.warning("Primary cache delete error", key=key, error=str(e))
                self._stats['errors'] += 1
        
        # Delete from fallback backend
        if self.fallback_backend:
            try:
                if await self.fallback_backend.delete(key):
                    success = True
            except Exception as e:
                logger.warning("Fallback cache delete error", key=key, error=str(e))
                self._stats['errors'] += 1
        
        return success
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        if not self._initialized:
            return False
        
        # Check primary backend first
        if self.primary_backend:
            try:
                if await self.primary_backend.exists(key):
                    return True
            except Exception as e:
                logger.warning("Primary cache exists error", key=key, error=str(e))
                self._stats['errors'] += 1
        
        # Check fallback backend
        if self.fallback_backend:
            try:
                if await self.fallback_backend.exists(key):
                    return True
            except Exception as e:
                logger.warning("Fallback cache exists error", key=key, error=str(e))
                self._stats['errors'] += 1
        
        return False
    
    async def clear(self) -> bool:
        """Clear all cache entries."""
        if not self._initialized:
            return False
        
        success = True
        
        # Clear primary backend
        if self.primary_backend:
            try:
                if not await self.primary_backend.clear():
                    success = False
            except Exception as e:
                logger.warning("Primary cache clear error", error=str(e))
                self._stats['errors'] += 1
                success = False
        
        # Clear fallback backend
        if self.fallback_backend:
            try:
                if not await self.fallback_backend.clear():
                    success = False
            except Exception as e:
                logger.warning("Fallback cache clear error", error=str(e))
                self._stats['errors'] += 1
                success = False
        
        return success
    
    async def close(self) -> None:
        """Close cache manager and all backends."""
        if self.primary_backend:
            await self.primary_backend.close()
        
        if self.fallback_backend:
            await self.fallback_backend.close()
        
        self._initialized = False
        logger.info("Cache manager closed")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._stats['hits'] + self._stats['misses']
        hit_rate = (self._stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            **self._stats,
            'total_requests': total_requests,
            'hit_rate_percent': round(hit_rate, 2),
            'initialized': self._initialized,
            'primary_available': self.primary_backend is not None,
            'fallback_available': self.fallback_backend is not None
        }
    
    def generate_key(self, prefix: str, *args: Any) -> str:
        """Generate cache key from prefix and arguments."""
        key_data = f"{prefix}:{':'.join(str(arg) for arg in args)}"
        return hashlib.md5(key_data.encode()).hexdigest()

# =============================================================================
# CACHE DECORATORS
# =============================================================================

def cached(ttl: int = 3600, key_prefix: str = "default"):
    """Decorator to cache function results."""
    
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"{key_prefix}:{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            cache_manager = getattr(async_wrapper, '_cache_manager', None)
            if cache_manager:
                cached_result = await cache_manager.get(cache_key)
                if cached_result is not None:
                    return cached_result
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            if cache_manager:
                await cache_manager.set(cache_key, result, ttl)
            
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"{key_prefix}:{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            cache_manager = getattr(sync_wrapper, '_cache_manager', None)
            if cache_manager:
                # For sync functions, we need to run async cache operations
                import asyncio
                loop = asyncio.get_event_loop()
                cached_result = loop.run_until_complete(cache_manager.get(cache_key))
                if cached_result is not None:
                    return cached_result
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Cache result
            if cache_manager:
                import asyncio
                loop = asyncio.get_event_loop()
                loop.run_until_complete(cache_manager.set(cache_key, result, ttl))
            
            return result
        
        # Return appropriate wrapper
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'CacheConfig',
    'CacheBackend',
    'RedisCacheBackend',
    'InMemoryCacheBackend',
    'CacheManager',
    'cached'
]






























