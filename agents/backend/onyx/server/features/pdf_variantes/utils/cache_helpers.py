"""
PDF Variantes Cache System
Advanced caching and performance optimization
"""

import asyncio
import json
import logging
import pickle
import hashlib
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
import redis.asyncio as redis
from pathlib import Path

from ..utils.config import Settings

logger = logging.getLogger(__name__)

class CacheManager:
    """Advanced cache manager with multiple backends"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        
        # Cache backends
        self.redis_client: Optional[redis.Redis] = None
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        
        # Cache configuration
        self.default_ttl = settings.CACHE_TTL_SECONDS
        self.max_memory_size = settings.CACHE_MAX_SIZE_MB * 1024 * 1024  # Convert to bytes
        self.enable_cache = settings.ENABLE_CACHE
        
        # Cache statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "memory_usage": 0
        }
    
    async def initialize(self):
        """Initialize cache manager"""
        try:
            if not self.enable_cache:
                logger.info("Caching disabled")
                return
            
            # Initialize Redis if available
            if self.settings.REDIS_URL:
                try:
                    self.redis_client = redis.from_url(
                        self.settings.REDIS_URL,
                        password=self.settings.REDIS_PASSWORD,
                        db=self.settings.REDIS_DB,
                        decode_responses=False  # We'll handle encoding ourselves
                    )
                    
                    # Test connection
                    await self.redis_client.ping()
                    logger.info("Redis cache initialized successfully")
                    
                except Exception as e:
                    logger.warning(f"Failed to initialize Redis cache: {e}")
                    self.redis_client = None
            
            logger.info("Cache Manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Cache Manager: {e}")
            raise
    
    async def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache"""
        try:
            if not self.enable_cache:
                return default
            
            # Try Redis first
            if self.redis_client:
                try:
                    value = await self.redis_client.get(key)
                    if value is not None:
                        self.stats["hits"] += 1
                        return self._deserialize(value)
                except Exception as e:
                    logger.warning(f"Redis get error: {e}")
            
            # Try memory cache
            if key in self.memory_cache:
                cache_entry = self.memory_cache[key]
                
                # Check expiration
                if cache_entry["expires_at"] > datetime.utcnow():
                    self.stats["hits"] += 1
                    return cache_entry["value"]
                else:
                    # Expired, remove from cache
                    del self.memory_cache[key]
            
            self.stats["misses"] += 1
            return default
            
        except Exception as e:
            logger.error(f"Error getting cache key {key}: {e}")
            return default
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        try:
            if not self.enable_cache:
                return True
            
            ttl = ttl or self.default_ttl
            expires_at = datetime.utcnow() + timedelta(seconds=ttl)
            
            # Set in Redis
            if self.redis_client:
                try:
                    serialized_value = self._serialize(value)
                    await self.redis_client.setex(key, ttl, serialized_value)
                except Exception as e:
                    logger.warning(f"Redis set error: {e}")
            
            # Set in memory cache
            cache_entry = {
                "value": value,
                "expires_at": expires_at,
                "created_at": datetime.utcnow()
            }
            
            self.memory_cache[key] = cache_entry
            self.stats["sets"] += 1
            
            # Cleanup expired entries periodically
            await self._cleanup_expired_entries()
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting cache key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        try:
            if not self.enable_cache:
                return True
            
            # Delete from Redis
            if self.redis_client:
                try:
                    await self.redis_client.delete(key)
                except Exception as e:
                    logger.warning(f"Redis delete error: {e}")
            
            # Delete from memory cache
            if key in self.memory_cache:
                del self.memory_cache[key]
                self.stats["deletes"] += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting cache key {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        try:
            if not self.enable_cache:
                return False
            
            # Check Redis
            if self.redis_client:
                try:
                    return await self.redis_client.exists(key) > 0
                except Exception as e:
                    logger.warning(f"Redis exists error: {e}")
            
            # Check memory cache
            if key in self.memory_cache:
                cache_entry = self.memory_cache[key]
                if cache_entry["expires_at"] > datetime.utcnow():
                    return True
                else:
                    del self.memory_cache[key]
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking cache key {key}: {e}")
            return False
    
    async def clear(self) -> bool:
        """Clear all cache"""
        try:
            if not self.enable_cache:
                return True
            
            # Clear Redis
            if self.redis_client:
                try:
                    await self.redis_client.flushdb()
                except Exception as e:
                    logger.warning(f"Redis clear error: {e}")
            
            # Clear memory cache
            self.memory_cache.clear()
            
            # Reset statistics
            self.stats = {
                "hits": 0,
                "misses": 0,
                "sets": 0,
                "deletes": 0,
                "memory_usage": 0
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            stats = self.stats.copy()
            
            # Add memory usage
            stats["memory_usage"] = self._calculate_memory_usage()
            stats["memory_entries"] = len(self.memory_cache)
            
            # Add Redis stats if available
            if self.redis_client:
                try:
                    redis_info = await self.redis_client.info("memory")
                    stats["redis_memory_usage"] = redis_info.get("used_memory", 0)
                    stats["redis_keys"] = await self.redis_client.dbsize()
                except Exception as e:
                    logger.warning(f"Redis stats error: {e}")
                    stats["redis_memory_usage"] = 0
                    stats["redis_keys"] = 0
            
            # Calculate hit rate
            total_requests = stats["hits"] + stats["misses"]
            stats["hit_rate"] = stats["hits"] / total_requests if total_requests > 0 else 0
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return self.stats.copy()
    
    def _serialize(self, value: Any) -> bytes:
        """Serialize value for storage"""
        try:
            return pickle.dumps(value)
        except Exception as e:
            logger.error(f"Error serializing value: {e}")
            return pickle.dumps(str(value))
    
    def _deserialize(self, value: bytes) -> Any:
        """Deserialize value from storage"""
        try:
            return pickle.loads(value)
        except Exception as e:
            logger.error(f"Error deserializing value: {e}")
            return None
    
    def _calculate_memory_usage(self) -> int:
        """Calculate memory usage of cache"""
        try:
            total_size = 0
            for key, cache_entry in self.memory_cache.items():
                total_size += len(key.encode('utf-8'))
                total_size += len(str(cache_entry).encode('utf-8'))
            return total_size
        except Exception as e:
            logger.error(f"Error calculating memory usage: {e}")
            return 0
    
    async def _cleanup_expired_entries(self):
        """Cleanup expired entries from memory cache"""
        try:
            current_time = datetime.utcnow()
            expired_keys = []
            
            for key, cache_entry in self.memory_cache.items():
                if cache_entry["expires_at"] <= current_time:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.memory_cache[key]
            
            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
            
        except Exception as e:
            logger.error(f"Error cleaning up expired entries: {e}")
    
    async def cleanup(self):
        """Cleanup cache manager"""
        try:
            if self.redis_client:
                await self.redis_client.close()
            
            self.memory_cache.clear()
            
            logger.info("Cache Manager cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Error cleaning up Cache Manager: {e}")

class PerformanceOptimizer:
    """Performance optimization utilities"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.cache_manager = CacheManager(settings)
        
        # Performance metrics
        self.metrics = {
            "request_count": 0,
            "total_response_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "slow_queries": 0
        }
    
    async def initialize(self):
        """Initialize performance optimizer"""
        try:
            await self.cache_manager.initialize()
            logger.info("Performance Optimizer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Performance Optimizer: {e}")
            raise
    
    async def optimize_query(self, query_key: str, query_func, ttl: int = 3600) -> Any:
        """Optimize query with caching"""
        try:
            # Check cache first
            cached_result = await self.cache_manager.get(query_key)
            if cached_result is not None:
                self.metrics["cache_hits"] += 1
                return cached_result
            
            # Execute query
            start_time = datetime.utcnow()
            result = await query_func()
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Track slow queries
            if execution_time > 1.0:  # Queries taking more than 1 second
                self.metrics["slow_queries"] += 1
                logger.warning(f"Slow query detected: {query_key} took {execution_time:.2f}s")
            
            # Cache result
            await self.cache_manager.set(query_key, result, ttl)
            self.metrics["cache_misses"] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Error optimizing query {query_key}: {e}")
            raise
    
    async def batch_process(self, items: List[Any], process_func, batch_size: int = 10) -> List[Any]:
        """Process items in batches for better performance"""
        try:
            results = []
            
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                batch_results = await asyncio.gather(*[process_func(item) for item in batch])
                results.extend(batch_results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            raise
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        try:
            metrics = self.metrics.copy()
            
            # Add cache stats
            cache_stats = await self.cache_manager.get_stats()
            metrics.update(cache_stats)
            
            # Calculate average response time
            if metrics["request_count"] > 0:
                metrics["avg_response_time"] = metrics["total_response_time"] / metrics["request_count"]
            else:
                metrics["avg_response_time"] = 0.0
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return self.metrics.copy()
    
    async def cleanup(self):
        """Cleanup performance optimizer"""
        try:
            await self.cache_manager.cleanup()
            logger.info("Performance Optimizer cleaned up successfully")
        except Exception as e:
            logger.error(f"Error cleaning up Performance Optimizer: {e}")

class CacheService:
    """Cache service for the application"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.cache_manager = CacheManager(settings)
        self.performance_optimizer = PerformanceOptimizer(settings)
    
    async def initialize(self):
        """Initialize cache service"""
        try:
            await self.cache_manager.initialize()
            await self.performance_optimizer.initialize()
            logger.info("Cache Service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Cache Service: {e}")
            raise
    
    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get document from cache"""
        try:
            cache_key = f"document:{document_id}"
            return await self.cache_manager.get(cache_key)
        except Exception as e:
            logger.error(f"Error getting document from cache: {e}")
            return None
    
    async def set_document(self, document_id: str, document_data: Dict[str, Any], ttl: int = 3600) -> bool:
        """Set document in cache"""
        try:
            cache_key = f"document:{document_id}"
            return await self.cache_manager.set(cache_key, document_data, ttl)
        except Exception as e:
            logger.error(f"Error setting document in cache: {e}")
            return False
    
    async def get_variant(self, variant_id: str) -> Optional[Dict[str, Any]]:
        """Get variant from cache"""
        try:
            cache_key = f"variant:{variant_id}"
            return await self.cache_manager.get(cache_key)
        except Exception as e:
            logger.error(f"Error getting variant from cache: {e}")
            return None
    
    async def set_variant(self, variant_id: str, variant_data: Dict[str, Any], ttl: int = 3600) -> bool:
        """Set variant in cache"""
        try:
            cache_key = f"variant:{variant_id}"
            return await self.cache_manager.set(cache_key, variant_data, ttl)
        except Exception as e:
            logger.error(f"Error setting variant in cache: {e}")
            return False
    
    async def get_user_permissions(self, user_id: str, document_id: str) -> Optional[List[str]]:
        """Get user permissions from cache"""
        try:
            cache_key = f"permissions:{user_id}:{document_id}"
            return await self.cache_manager.get(cache_key)
        except Exception as e:
            logger.error(f"Error getting user permissions from cache: {e}")
            return None
    
    async def set_user_permissions(self, user_id: str, document_id: str, permissions: List[str], ttl: int = 3600) -> bool:
        """Set user permissions in cache"""
        try:
            cache_key = f"permissions:{user_id}:{document_id}"
            return await self.cache_manager.set(cache_key, permissions, ttl)
        except Exception as e:
            logger.error(f"Error setting user permissions in cache: {e}")
            return False
    
    async def invalidate_document_cache(self, document_id: str) -> bool:
        """Invalidate all cache entries for a document"""
        try:
            # List of cache keys to invalidate
            keys_to_invalidate = [
                f"document:{document_id}",
                f"variants:{document_id}",
                f"topics:{document_id}",
                f"brainstorm:{document_id}"
            ]
            
            # Delete all keys
            for key in keys_to_invalidate:
                await self.cache_manager.delete(key)
            
            return True
            
        except Exception as e:
            logger.error(f"Error invalidating document cache: {e}")
            return False
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            return await self.cache_manager.get_stats()
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {}
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        try:
            return await self.performance_optimizer.get_performance_metrics()
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {}
    
    async def cleanup(self):
        """Cleanup cache service"""
        try:
            await self.cache_manager.cleanup()
            await self.performance_optimizer.cleanup()
            logger.info("Cache Service cleaned up successfully")
        except Exception as e:
            logger.error(f"Error cleaning up Cache Service: {e}")
