"""
Optimized Caching System for Video-OpusClip

High-performance caching with Redis, memory caching, and intelligent cache management
for improved response times and reduced API calls.
"""

import asyncio
import hashlib
import json
import time
import pickle
from typing import Any, Optional, Dict, List, Union, Callable
from functools import wraps
from dataclasses import dataclass
import structlog

# Optional Redis import
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    redis = None
    REDIS_AVAILABLE = False

# Optional memory cache import
try:
    from cachetools import TTLCache, LRUCache
    CACHETOOLS_AVAILABLE = True
except ImportError:
    TTLCache = None
    LRUCache = None
    CACHETOOLS_AVAILABLE = False

logger = structlog.get_logger()

# =============================================================================
# CACHE CONFIGURATION
# =============================================================================

@dataclass
class CacheConfig:
    """Configuration for caching system."""
    
    # Redis Configuration
    redis_url: str = "redis://localhost:6379"
    redis_ttl: int = 3600  # 1 hour
    redis_max_connections: int = 20
    redis_retry_on_timeout: bool = True
    
    # Memory Cache Configuration
    memory_cache_size: int = 1000
    memory_cache_ttl: int = 1800  # 30 minutes
    enable_memory_cache: bool = True
    
    # Cache Strategy
    enable_cache: bool = True
    cache_key_prefix: str = "video_opusclip:"
    enable_compression: bool = True
    compression_threshold: int = 1024  # 1KB
    
    # Cache Invalidation
    enable_auto_invalidation: bool = True
    invalidation_patterns: List[str] = None
    
    def __post_init__(self):
        if self.invalidation_patterns is None:
            self.invalidation_patterns = ["video:*", "analysis:*", "viral:*"]

# =============================================================================
# CACHE KEY GENERATION
# =============================================================================

class CacheKeyGenerator:
    """Generate consistent cache keys for different data types."""
    
    @staticmethod
    def generate_key(prefix: str, *args, **kwargs) -> str:
        """Generate a cache key from prefix and arguments."""
        # Create a string representation of arguments
        key_parts = [prefix]
        
        # Add positional arguments
        for arg in args:
            key_parts.append(str(arg))
        
        # Add keyword arguments (sorted for consistency)
        for key, value in sorted(kwargs.items()):
            key_parts.append(f"{key}:{value}")
        
        # Join and hash
        key_string = ":".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    @staticmethod
    def video_analysis_key(youtube_url: str, language: str, platform: str) -> str:
        """Generate cache key for video analysis."""
        return CacheKeyGenerator.generate_key(
            "video_analysis",
            youtube_url=youtube_url,
            language=language,
            platform=platform
        )
    
    @staticmethod
    def viral_analysis_key(content_hash: str, audience_profile: Dict) -> str:
        """Generate cache key for viral analysis."""
        audience_str = json.dumps(audience_profile, sort_keys=True)
        return CacheKeyGenerator.generate_key(
            "viral_analysis",
            content_hash=content_hash,
            audience=audience_str
        )
    
    @staticmethod
    def langchain_response_key(prompt: str, model: str, temperature: float) -> str:
        """Generate cache key for LangChain responses."""
        return CacheKeyGenerator.generate_key(
            "langchain_response",
            prompt_hash=hashlib.md5(prompt.encode()).hexdigest(),
            model=model,
            temperature=temperature
        )

# =============================================================================
# MEMORY CACHE
# =============================================================================

class MemoryCache:
    """High-performance in-memory cache with TTL and LRU eviction."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache = None
        self._initialize_cache()
    
    def _initialize_cache(self):
        """Initialize memory cache."""
        if CACHETOOLS_AVAILABLE and self.config.enable_memory_cache:
            self.cache = TTLCache(
                maxsize=self.config.memory_cache_size,
                ttl=self.config.memory_cache_ttl
            )
            logger.info("Memory cache initialized", size=self.config.memory_cache_size)
        else:
            self.cache = {}
            logger.warning("Using simple dict cache (cachetools not available)")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from memory cache."""
        if not self.config.enable_cache:
            return None
        
        try:
            value = self.cache.get(key)
            if value is not None:
                logger.debug("Memory cache hit", key=key)
                return value
            logger.debug("Memory cache miss", key=key)
            return None
        except Exception as e:
            logger.error("Memory cache get error", key=key, error=str(e))
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in memory cache."""
        if not self.config.enable_cache:
            return False
        
        try:
            self.cache[key] = value
            logger.debug("Memory cache set", key=key)
            return True
        except Exception as e:
            logger.error("Memory cache set error", key=key, error=str(e))
            return False
    
    def delete(self, key: str) -> bool:
        """Delete value from memory cache."""
        try:
            if key in self.cache:
                del self.cache[key]
                logger.debug("Memory cache delete", key=key)
                return True
            return False
        except Exception as e:
            logger.error("Memory cache delete error", key=key, error=str(e))
            return False
    
    def clear(self) -> bool:
        """Clear all memory cache."""
        try:
            self.cache.clear()
            logger.info("Memory cache cleared")
            return True
        except Exception as e:
            logger.error("Memory cache clear error", error=str(e))
            return False
    
    def size(self) -> int:
        """Get current cache size."""
        return len(self.cache)

# =============================================================================
# REDIS CACHE
# =============================================================================

class RedisCache:
    """Redis-based distributed cache with async support."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.redis_client = None
        self._initialize_redis()
    
    def _initialize_redis(self):
        """Initialize Redis connection."""
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available, skipping Redis cache initialization")
            return
        
        try:
            self.redis_client = redis.from_url(
                self.config.redis_url,
                max_connections=self.config.redis_max_connections,
                retry_on_timeout=self.config.redis_retry_on_timeout,
                decode_responses=False  # Keep as bytes for pickle
            )
            logger.info("Redis cache initialized", url=self.config.redis_url)
        except Exception as e:
            logger.error("Redis initialization failed", error=str(e))
            self.redis_client = None
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache."""
        if not self.config.enable_cache or not self.redis_client:
            return None
        
        try:
            full_key = f"{self.config.cache_key_prefix}{key}"
            value = await self.redis_client.get(full_key)
            
            if value is not None:
                # Decompress if needed
                if self.config.enable_compression and len(value) > self.config.compression_threshold:
                    import gzip
                    value = gzip.decompress(value)
                
                # Unpickle
                result = pickle.loads(value)
                logger.debug("Redis cache hit", key=key)
                return result
            
            logger.debug("Redis cache miss", key=key)
            return None
            
        except Exception as e:
            logger.error("Redis cache get error", key=key, error=str(e))
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Redis cache."""
        if not self.config.enable_cache or not self.redis_client:
            return False
        
        try:
            full_key = f"{self.config.cache_key_prefix}{key}"
            ttl = ttl or self.config.redis_ttl
            
            # Pickle the value
            pickled_value = pickle.dumps(value)
            
            # Compress if needed
            if self.config.enable_compression and len(pickled_value) > self.config.compression_threshold:
                import gzip
                pickled_value = gzip.compress(pickled_value)
            
            # Set with TTL
            await self.redis_client.setex(full_key, ttl, pickled_value)
            logger.debug("Redis cache set", key=key, ttl=ttl)
            return True
            
        except Exception as e:
            logger.error("Redis cache set error", key=key, error=str(e))
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from Redis cache."""
        if not self.redis_client:
            return False
        
        try:
            full_key = f"{self.config.cache_key_prefix}{key}"
            result = await self.redis_client.delete(full_key)
            if result:
                logger.debug("Redis cache delete", key=key)
            return bool(result)
        except Exception as e:
            logger.error("Redis cache delete error", key=key, error=str(e))
            return False
    
    async def clear_pattern(self, pattern: str) -> int:
        """Clear cache entries matching pattern."""
        if not self.redis_client:
            return 0
        
        try:
            full_pattern = f"{self.config.cache_key_prefix}{pattern}"
            keys = await self.redis_client.keys(full_pattern)
            if keys:
                deleted = await self.redis_client.delete(*keys)
                logger.info("Redis cache pattern clear", pattern=pattern, deleted=deleted)
                return deleted
            return 0
        except Exception as e:
            logger.error("Redis cache pattern clear error", pattern=pattern, error=str(e))
            return 0
    
    async def health_check(self) -> bool:
        """Check Redis connection health."""
        if not self.redis_client:
            return False
        
        try:
            await self.redis_client.ping()
            return True
        except Exception:
            return False

# =============================================================================
# HYBRID CACHE
# =============================================================================

class HybridCache:
    """Hybrid cache system using both memory and Redis for optimal performance."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.memory_cache = MemoryCache(config)
        self.redis_cache = RedisCache(config)
        self.key_generator = CacheKeyGenerator()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache (memory first, then Redis)."""
        # Try memory cache first
        value = self.memory_cache.get(key)
        if value is not None:
            return value
        
        # Try Redis cache
        value = await self.redis_cache.get(key)
        if value is not None:
            # Store in memory cache for faster future access
            self.memory_cache.set(key, value)
            return value
        
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in both memory and Redis cache."""
        # Set in memory cache
        memory_success = self.memory_cache.set(key, value, ttl)
        
        # Set in Redis cache
        redis_success = await self.redis_cache.set(key, value, ttl)
        
        return memory_success or redis_success
    
    async def delete(self, key: str) -> bool:
        """Delete value from both caches."""
        memory_success = self.memory_cache.delete(key)
        redis_success = await self.redis_cache.delete(key)
        return memory_success or redis_success
    
    async def clear_pattern(self, pattern: str) -> int:
        """Clear cache entries matching pattern."""
        # Clear memory cache (simple clear for now)
        self.memory_cache.clear()
        
        # Clear Redis cache pattern
        return await self.redis_cache.clear_pattern(pattern)
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of both cache systems."""
        return {
            "memory": self.memory_cache.size() >= 0,  # Simple check
            "redis": await self.redis_cache.health_check()
        }

# =============================================================================
# CACHE DECORATORS
# =============================================================================

def cached(ttl: Optional[int] = None, key_prefix: str = "func"):
    """Decorator for caching function results."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Get cache instance
            from .optimized_config import get_config
            config = get_config()
            cache = HybridCache(CacheConfig())
            
            # Generate cache key
            cache_key = CacheKeyGenerator.generate_key(
                key_prefix,
                func.__name__,
                *args,
                **kwargs
            )
            
            # Try to get from cache
            cached_result = await cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            await cache.set(cache_key, result, ttl)
            
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Get cache instance
            from .optimized_config import get_config
            config = get_config()
            cache = HybridCache(CacheConfig())
            
            # Generate cache key
            cache_key = CacheKeyGenerator.generate_key(
                key_prefix,
                func.__name__,
                *args,
                **kwargs
            )
            
            # Try to get from cache (sync version)
            cached_result = cache.memory_cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Cache result
            cache.memory_cache.set(cache_key, result, ttl)
            
            return result
        
        # Return appropriate wrapper
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

# =============================================================================
# CACHE MANAGER
# =============================================================================

class CacheManager:
    """Global cache manager for the application."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache = HybridCache(config)
        self.key_generator = CacheKeyGenerator()
    
    async def get_video_analysis(self, youtube_url: str, language: str, platform: str) -> Optional[Dict]:
        """Get cached video analysis."""
        key = self.key_generator.video_analysis_key(youtube_url, language, platform)
        return await self.cache.get(key)
    
    async def set_video_analysis(self, youtube_url: str, language: str, platform: str, analysis: Dict) -> bool:
        """Cache video analysis."""
        key = self.key_generator.video_analysis_key(youtube_url, language, platform)
        return await self.cache.set(key, analysis, self.config.redis_ttl)
    
    async def get_viral_analysis(self, content_hash: str, audience_profile: Dict) -> Optional[Dict]:
        """Get cached viral analysis."""
        key = self.key_generator.viral_analysis_key(content_hash, audience_profile)
        return await self.cache.get(key)
    
    async def set_viral_analysis(self, content_hash: str, audience_profile: Dict, analysis: Dict) -> bool:
        """Cache viral analysis."""
        key = self.key_generator.viral_analysis_key(content_hash, audience_profile)
        return await self.cache.set(key, analysis, self.config.redis_ttl)
    
    async def get_langchain_response(self, prompt: str, model: str, temperature: float) -> Optional[str]:
        """Get cached LangChain response."""
        key = self.key_generator.langchain_response_key(prompt, model, temperature)
        return await self.cache.get(key)
    
    async def set_langchain_response(self, prompt: str, model: str, temperature: float, response: str) -> bool:
        """Cache LangChain response."""
        key = self.key_generator.langchain_response_key(prompt, model, temperature)
        return await self.cache.set(key, response, self.config.redis_ttl)
    
    async def invalidate_video_cache(self, youtube_url: str) -> int:
        """Invalidate all cache entries for a video."""
        pattern = f"video_analysis:*{hashlib.md5(youtube_url.encode()).hexdigest()}*"
        return await self.cache.clear_pattern(pattern)
    
    async def health_check(self) -> Dict[str, bool]:
        """Check cache health."""
        return await self.cache.health_check()

# =============================================================================
# GLOBAL CACHE INSTANCE
# =============================================================================

# Global cache manager instance
cache_manager = None

def get_cache_manager() -> CacheManager:
    """Get the global cache manager instance."""
    global cache_manager
    if cache_manager is None:
        from .optimized_config import get_config
        config = get_config()
        cache_config = CacheConfig(
            redis_url=config.env.REDIS_URL,
            enable_cache=config.env.ENABLE_CACHING,
            memory_cache_size=config.env.CACHE_SIZE
        )
        cache_manager = CacheManager(cache_config)
    return cache_manager 