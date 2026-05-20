"""
Cache infrastructure for the ads feature.

This module consolidates cache functionality from:
- scalable_api_patterns.py (cache management)
- performance_optimizer.py (cache optimization)

Provides unified cache management with strategy pattern for different backends.
"""

import hashlib
try:
    import orjson as _fastjson  # type: ignore
except Exception:
    _fastjson = None
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from enum import Enum

try:
    import aioredis  # type: ignore
except Exception:  # pragma: no cover - optional in tests
    aioredis = None  # type: ignore[assignment]

try:
    from onyx.utils.logger import setup_logger  # type: ignore
except Exception:  # pragma: no cover - fallback minimal logger for tests
    import logging as _logging

    def setup_logger(name: str | None = None):  # type: ignore[override]
        logger = _logging.getLogger(name or __name__)
        if not _logging.getLogger().handlers:
            _logging.basicConfig(level=_logging.INFO)
        return logger
from ..config import get_optimized_settings

logger = setup_logger()

class CacheType(Enum):
    """Cache backend types."""
    MEMORY = "memory"
    REDIS = "redis"
    HYBRID = "hybrid"

@dataclass
class CacheConfig:
    """Cache configuration settings."""
    default_ttl: int = 300  # 5 minutes
    max_size: int = 1000
    cache_type: CacheType = CacheType.REDIS
    redis_url: Optional[str] = None
    redis_max_connections: int = 50
    compression_enabled: bool = True
    encryption_enabled: bool = False
    stats_enabled: bool = True

class CacheStrategy(ABC):
    """Abstract cache strategy interface."""
    
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
        """Delete value from cache."""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        pass
    
    @abstractmethod
    async def expire(self, key: str, ttl: int) -> bool:
        """Set expiration for key."""
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        pass

class MemoryCacheStrategy(CacheStrategy):
    """In-memory cache strategy."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_times: Dict[str, float] = {}
        self._stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "evictions": 0
        }
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from parameters."""
        data = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(data.encode()).hexdigest()
    
    def _cleanup_expired(self):
        """Remove expired entries from cache."""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.get('expires_at', 0) < current_time
        ]
        
        for key in expired_keys:
            del self._cache[key]
            del self._access_times[key]
            self._stats["evictions"] += 1
        
        # If still over max size, remove least recently used
        if len(self._cache) > self.config.max_size:
            sorted_keys = sorted(
                self._access_times.items(),
                key=lambda x: x[1]
            )
            keys_to_remove = len(self._cache) - self.config.max_size
            
            for key, _ in sorted_keys[:keys_to_remove]:
                del self._cache[key]
                del self._access_times[key]
                self._stats["evictions"] += 1
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from memory cache."""
        try:
            self._cleanup_expired()
            
            if key in self._cache:
                entry = self._cache[key]
                self._access_times[key] = time.time()
                self._stats["hits"] += 1
                return entry['value']
            
            self._stats["misses"] += 1
            return None
            
        except Exception as e:
            logger.error(f"Error getting from memory cache: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in memory cache."""
        try:
            self._cleanup_expired()
            
            ttl = ttl or self.config.default_ttl
            expires_at = time.time() + ttl
            
            self._cache[key] = {
                'value': value,
                'expires_at': expires_at,
                'created_at': time.time()
            }
            self._access_times[key] = time.time()
            self._stats["sets"] += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting memory cache: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from memory cache."""
        try:
            if key in self._cache:
                del self._cache[key]
                del self._access_times[key]
                self._stats["deletes"] += 1
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error deleting from memory cache: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in memory cache."""
        try:
            self._cleanup_expired()
            return key in self._cache
        except Exception as e:
            logger.error(f"Error checking existence in memory cache: {e}")
            return False
    
    async def expire(self, key: str, ttl: int) -> bool:
        """Set expiration for key in memory cache."""
        try:
            if key in self._cache:
                self._cache[key]['expires_at'] = time.time() + ttl
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error setting expiration in memory cache: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get memory cache statistics."""
        self._cleanup_expired()
        return {
            "type": "memory",
            "size": len(self._cache),
            "max_size": self.config.max_size,
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "sets": self._stats["sets"],
            "deletes": self._stats["deletes"],
            "evictions": self._stats["evictions"],
            "hit_rate": self._stats["hits"] / max(1, self._stats["hits"] + self._stats["misses"])
        }

class RedisCacheStrategy(CacheStrategy):
    """Redis cache strategy."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self._redis_client = None
        self._stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "errors": 0
        }
    
    @property
    async def redis_client(self):
        """Get Redis client."""
        if self._redis_client is None:
            redis_url = self.config.redis_url
            if not redis_url:
                settings = get_optimized_settings()
                redis_url = settings.redis_url
            
            self._redis_client = await aioredis.from_url(
                redis_url,
                encoding="utf-8",
                decode_responses=True,
                max_connections=self.config.redis_max_connections
            )
        return self._redis_client
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from parameters."""
        data = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(data.encode()).hexdigest()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache."""
        try:
            redis_client = await self.redis_client
            value = await redis_client.get(key)
            
            if value:
                self._stats["hits"] += 1
                try:
                    if _fastjson is not None:
                        return _fastjson.loads(value)
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value
            else:
                self._stats["misses"] += 1
                return None
                
        except Exception as e:
            logger.error(f"Error getting from Redis cache: {e}")
            self._stats["errors"] += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Redis cache."""
        try:
            redis_client = await self.redis_client
            ttl = ttl or self.config.default_ttl
            
            if isinstance(value, (dict, list)):
                if _fastjson is not None:
                    serialized_value = _fastjson.dumps(value).decode()
                else:
                    serialized_value = json.dumps(value)
            else:
                serialized_value = str(value)
            
            await redis_client.setex(key, ttl, serialized_value)
            self._stats["sets"] += 1
            return True
            
        except Exception as e:
            logger.error(f"Error setting Redis cache: {e}")
            self._stats["errors"] += 1
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from Redis cache."""
        try:
            redis_client = await self.redis_client
            result = await redis_client.delete(key)
            self._stats["deletes"] += 1
            return result > 0
            
        except Exception as e:
            logger.error(f"Error deleting from Redis cache: {e}")
            self._stats["errors"] += 1
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis cache."""
        try:
            redis_client = await self.redis_client
            return await redis_client.exists(key) > 0
        except Exception as e:
            logger.error(f"Error checking existence in Redis cache: {e}")
            self._stats["errors"] += 1
            return False
    
    async def expire(self, key: str, ttl: int) -> bool:
        """Set expiration for key in Redis cache."""
        try:
            redis_client = await self.redis_client
            return await redis_client.expire(key, ttl)
        except Exception as e:
            logger.error(f"Error setting expiration in Redis cache: {e}")
            self._stats["errors"] += 1
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get Redis cache statistics."""
        try:
            redis_client = await self.redis_client
            info = await redis_client.info()
            
            return {
                "type": "redis",
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": info.get("used_memory_human", "0B"),
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "sets": self._stats["sets"],
                "deletes": self._stats["deletes"],
                "errors": self._stats["errors"],
                "hit_rate": self._stats["hits"] / max(1, self._stats["hits"] + self._stats["misses"])
            }
        except Exception as e:
            logger.error(f"Error getting Redis stats: {e}")
            return {
                "type": "redis",
                "error": str(e),
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "sets": self._stats["sets"],
                "deletes": self._stats["deletes"],
                "errors": self._stats["errors"]
            }
    
    async def close(self):
        """Close Redis connection."""
        if self._redis_client:
            await self._redis_client.close()

class CacheManager:
    """Manages cache operations with strategy pattern."""
    
    def __init__(self, config: Optional[CacheConfig] = None):
        if config is None:
            settings = get_optimized_settings()
            config = CacheConfig(
                default_ttl=settings.cache_ttl,
                max_size=settings.cache_max_size,
                cache_type=CacheType.REDIS,
                redis_url=settings.redis_url
            )
        
        self.config = config
        self.strategy = self._create_strategy()
    
    def _create_strategy(self) -> CacheStrategy:
        """Create cache strategy based on configuration."""
        if self.config.cache_type == CacheType.MEMORY:
            return MemoryCacheStrategy(self.config)
        elif self.config.cache_type == CacheType.REDIS:
            return RedisCacheStrategy(self.config)
        else:
            # Default to Redis
            return RedisCacheStrategy(self.config)
    
    def generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from parameters."""
        try:
            if _fastjson is not None:
                blob = _fastjson.dumps([args, dict(sorted(kwargs.items()))])
                return hashlib.md5(blob).hexdigest()
            import json as _json
            blob = _json.dumps([args, dict(sorted(kwargs.items()))], separators=(",", ":"))
            return hashlib.md5(blob.encode()).hexdigest()
        except Exception:
            data = str(args) + str(sorted(kwargs.items()))
            return hashlib.md5(data.encode()).hexdigest()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache with JSON deserialization fallback."""
        value = await self.strategy.get(key)
        if isinstance(value, str):
            try:
                return json.loads(value)
            except Exception:
                return value
        return value
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with small-object fast path and string serialization fallback."""
        try:
            # Fast path for small primitives
            if isinstance(value, (str, int, float, bool)):
                return await self.strategy.set(key, value, ttl)
            # Attempt JSON serialization for complex types
            import json
            serialized = json.dumps(value, default=str)
            return await self.strategy.set(key, serialized, ttl)
        except Exception:
            # Fallback to direct set if strategy can handle it
            return await self.strategy.set(key, value, ttl)
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        return await self.strategy.delete(key)
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        return await self.strategy.exists(key)
    
    async def expire(self, key: str, ttl: int) -> bool:
        """Set expiration for key."""
        return await self.strategy.expire(key, ttl)
    
    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cache by pattern (Redis only)."""
        if isinstance(self.strategy, RedisCacheStrategy):
            try:
                redis_client = await self.strategy.redis_client
                keys = await redis_client.keys(pattern)
                if keys:
                    return await redis_client.delete(*keys)
                return 0
            except Exception as e:
                logger.error(f"Error invalidating cache pattern: {e}")
                return 0
        else:
            logger.warning("Pattern invalidation only supported for Redis cache")
            return 0
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return await self.strategy.get_stats()
    
    async def close(self):
        """Close cache connections."""
        if hasattr(self.strategy, 'close'):
            await self.strategy.close()

class CacheService:
    """High-level cache service with business logic."""
    
    def __init__(self, cache_manager: Optional[CacheManager] = None):
        self.cache_manager = cache_manager or CacheManager()
    
    async def get_cached_ads(self, user_id: int, ads_id: int) -> Optional[Dict[str, Any]]:
        """Get cached ads data."""
        cache_key = f"ads:{user_id}:{ads_id}"
        return await self.cache_manager.get(cache_key)
    
    async def cache_ads(self, user_id: int, ads_id: int, ads_data: Dict[str, Any], ttl: int = 3600) -> bool:
        """Cache ads data."""
        cache_key = f"ads:{user_id}:{ads_id}"
        return await self.cache_manager.set(cache_key, ads_data, ttl)
    
    async def invalidate_user_cache(self, user_id: int) -> int:
        """Invalidate all cache for a user."""
        pattern = f"ads:{user_id}:*"
        return await self.cache_manager.invalidate_pattern(pattern)
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return await self.cache_manager.get_stats()
    
    async def close(self):
        """Close cache service."""
        await self.cache_manager.close()

# Alias for backward compatibility
CacheStrategy = CacheStrategy
RedisCacheStrategy = RedisCacheStrategy
MemoryCacheStrategy = MemoryCacheStrategy
