"""
Enhanced Cache Service for AI Document Processor
===============================================

High-performance caching system with Redis support, memory optimization,
and intelligent cache invalidation strategies.
"""

import asyncio
import hashlib
import json
import logging
import pickle
import time
from typing import Any, Dict, Optional, Union, List
from dataclasses import dataclass
from datetime import datetime, timedelta
import weakref
import gc

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    value: Any
    created_at: datetime
    expires_at: Optional[datetime] = None
    access_count: int = 0
    last_accessed: datetime = None
    size_bytes: int = 0

class EnhancedCacheService:
    """Enhanced caching service with multiple backends and optimization strategies"""
    
    def __init__(self, 
                 max_memory_mb: int = 512,
                 default_ttl: int = 3600,
                 redis_url: Optional[str] = None,
                 enable_compression: bool = True):
        """
        Initialize enhanced cache service
        
        Args:
            max_memory_mb: Maximum memory usage in MB
            default_ttl: Default time-to-live in seconds
            redis_url: Redis connection URL (optional)
            enable_compression: Enable data compression
        """
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.default_ttl = default_ttl
        self.enable_compression = enable_compression
        
        # In-memory cache with LRU eviction
        self._memory_cache: Dict[str, CacheEntry] = {}
        self._access_order: List[str] = []
        self._current_memory_usage = 0
        
        # Redis connection (optional)
        self.redis_client: Optional[redis.Redis] = None
        if redis_url and REDIS_AVAILABLE:
            try:
                self.redis_client = redis.from_url(redis_url, decode_responses=False)
                logger.info("Redis cache initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Redis: {e}")
                self.redis_client = None
        
        # Cache statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'compressions': 0,
            'redis_hits': 0,
            'redis_misses': 0
        }
        
        # Start cleanup task
        self._cleanup_task = None
        self._start_cleanup_task()
    
    def _start_cleanup_task(self):
        """Start background cleanup task"""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
    
    async def _periodic_cleanup(self):
        """Periodic cleanup of expired entries and memory optimization"""
        while True:
            try:
                await asyncio.sleep(300)  # Cleanup every 5 minutes
                await self._cleanup_expired()
                await self._optimize_memory()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")
    
    async def _cleanup_expired(self):
        """Remove expired entries from cache"""
        now = datetime.now()
        expired_keys = []
        
        for key, entry in self._memory_cache.items():
            if entry.expires_at and entry.expires_at < now:
                expired_keys.append(key)
        
        for key in expired_keys:
            await self._remove_from_memory(key)
            logger.debug(f"Removed expired cache entry: {key}")
    
    async def _optimize_memory(self):
        """Optimize memory usage by evicting least recently used items"""
        if self._current_memory_usage <= self.max_memory_bytes:
            return
        
        # Calculate how much memory to free
        excess_memory = self._current_memory_usage - self.max_memory_bytes
        target_reduction = excess_memory + (self.max_memory_bytes * 0.1)  # Free 10% extra
        
        # Sort by access time and frequency
        sorted_keys = sorted(
            self._memory_cache.keys(),
            key=lambda k: (
                self._memory_cache[k].last_accessed or datetime.min,
                -self._memory_cache[k].access_count
            )
        )
        
        freed_memory = 0
        for key in sorted_keys:
            if freed_memory >= target_reduction:
                break
            
            entry = self._memory_cache[key]
            freed_memory += entry.size_bytes
            await self._remove_from_memory(key)
            self.stats['evictions'] += 1
        
        logger.info(f"Memory optimization: freed {freed_memory} bytes")
    
    async def _remove_from_memory(self, key: str):
        """Remove entry from memory cache"""
        if key in self._memory_cache:
            entry = self._memory_cache[key]
            self._current_memory_usage -= entry.size_bytes
            del self._memory_cache[key]
            
            if key in self._access_order:
                self._access_order.remove(key)
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of cached value"""
        try:
            if self.enable_compression:
                # Estimate compressed size
                serialized = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
                return len(serialized)
            else:
                return len(str(value).encode('utf-8'))
        except Exception:
            return 1024  # Default estimate
    
    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        key_data = {
            'prefix': prefix,
            'args': args,
            'kwargs': sorted(kwargs.items()) if kwargs else {}
        }
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        # Try memory cache first
        if key in self._memory_cache:
            entry = self._memory_cache[key]
            
            # Check expiration
            if entry.expires_at and entry.expires_at < datetime.now():
                await self._remove_from_memory(key)
                self.stats['misses'] += 1
                return None
            
            # Update access statistics
            entry.access_count += 1
            entry.last_accessed = datetime.now()
            
            # Move to end of access order (most recent)
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            
            self.stats['hits'] += 1
            return entry.value
        
        # Try Redis cache
        if self.redis_client:
            try:
                redis_value = await self.redis_client.get(f"cache:{key}")
                if redis_value:
                    value = pickle.loads(redis_value)
                    self.stats['redis_hits'] += 1
                    
                    # Store in memory cache for faster access
                    await self.set(key, value, ttl=self.default_ttl)
                    return value
                else:
                    self.stats['redis_misses'] += 1
            except Exception as e:
                logger.warning(f"Redis get error: {e}")
        
        self.stats['misses'] += 1
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        try:
            ttl = ttl or self.default_ttl
            expires_at = datetime.now() + timedelta(seconds=ttl)
            size_bytes = self._calculate_size(value)
            
            # Create cache entry
            entry = CacheEntry(
                value=value,
                created_at=datetime.now(),
                expires_at=expires_at,
                last_accessed=datetime.now(),
                size_bytes=size_bytes
            )
            
            # Store in memory cache
            await self._remove_from_memory(key)  # Remove existing if any
            self._memory_cache[key] = entry
            self._access_order.append(key)
            self._current_memory_usage += size_bytes
            
            # Store in Redis if available
            if self.redis_client:
                try:
                    serialized = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
                    await self.redis_client.setex(
                        f"cache:{key}", 
                        ttl, 
                        serialized
                    )
                except Exception as e:
                    logger.warning(f"Redis set error: {e}")
            
            # Trigger memory optimization if needed
            if self._current_memory_usage > self.max_memory_bytes:
                asyncio.create_task(self._optimize_memory())
            
            return True
            
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        try:
            # Remove from memory
            await self._remove_from_memory(key)
            
            # Remove from Redis
            if self.redis_client:
                await self.redis_client.delete(f"cache:{key}")
            
            return True
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False
    
    async def clear(self):
        """Clear all cache entries"""
        self._memory_cache.clear()
        self._access_order.clear()
        self._current_memory_usage = 0
        
        if self.redis_client:
            try:
                keys = await self.redis_client.keys("cache:*")
                if keys:
                    await self.redis_client.delete(*keys)
            except Exception as e:
                logger.warning(f"Redis clear error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = (self.stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            **self.stats,
            'hit_rate_percent': round(hit_rate, 2),
            'memory_usage_mb': round(self._current_memory_usage / 1024 / 1024, 2),
            'memory_usage_percent': round((self._current_memory_usage / self.max_memory_bytes) * 100, 2),
            'entries_count': len(self._memory_cache),
            'redis_available': self.redis_client is not None
        }
    
    async def close(self):
        """Close cache service and cleanup resources"""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        if self.redis_client:
            await self.redis_client.close()
        
        # Clear memory cache
        self._memory_cache.clear()
        self._access_order.clear()
        self._current_memory_usage = 0

# Global cache instance
_cache_service: Optional[EnhancedCacheService] = None

async def get_cache_service() -> EnhancedCacheService:
    """Get global cache service instance"""
    global _cache_service
    if _cache_service is None:
        _cache_service = EnhancedCacheService()
    return _cache_service

async def close_cache_service():
    """Close global cache service"""
    global _cache_service
    if _cache_service:
        await _cache_service.close()
        _cache_service = None

















