"""
Redis Caching Implementation for Email Sequence System

This module provides comprehensive Redis caching functionality
for performance optimization and data persistence.
"""

import json
import logging
import asyncio
from typing import Any, Optional, Dict, List, Union
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

import redis.asyncio as redis
from redis.asyncio import Redis

from .config import get_settings
from .exceptions import CacheError

logger = logging.getLogger(__name__)
settings = get_settings()


class CacheManager:
    """Redis cache manager with async operations"""
    
    def __init__(self, redis_url: str = None):
        """
        Initialize cache manager.
        
        Args:
            redis_url: Redis connection URL
        """
        self.redis_url = redis_url or settings.redis_url
        self.redis_client: Optional[Redis] = None
        self.is_connected = False
    
    async def connect(self) -> None:
        """Connect to Redis"""
        try:
            self.redis_client = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
                max_connections=settings.redis_max_connections,
                retry_on_timeout=True,
                socket_keepalive=True,
                socket_keepalive_options={}
            )
            
            # Test connection
            await self.redis_client.ping()
            self.is_connected = True
            
            logger.info("Connected to Redis successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.is_connected = False
            raise CacheError(f"Failed to connect to Redis: {e}")
    
    async def disconnect(self) -> None:
        """Disconnect from Redis"""
        if self.redis_client:
            await self.redis_client.close()
            self.is_connected = False
            logger.info("Disconnected from Redis")
    
    async def health_check(self) -> bool:
        """Check Redis health"""
        try:
            if not self.redis_client or not self.is_connected:
                return False
            
            await self.redis_client.ping()
            return True
            
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        try:
            if not self.is_connected:
                return None
            
            value = await self.redis_client.get(key)
            if value is None:
                return None
            
            # Try to deserialize JSON
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value
                
        except Exception as e:
            logger.error(f"Error getting cache key {key}: {e}")
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        serialize: bool = True
    ) -> bool:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            serialize: Whether to serialize as JSON
            
        Returns:
            True if successful
        """
        try:
            if not self.is_connected:
                return False
            
            # Serialize value if needed
            if serialize and not isinstance(value, (str, int, float, bool)):
                value = json.dumps(value, default=str)
            
            # Set TTL if not provided
            if ttl is None:
                ttl = settings.cache_ttl_seconds
            
            await self.redis_client.setex(key, ttl, value)
            return True
            
        except Exception as e:
            logger.error(f"Error setting cache key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """
        Delete key from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if successful
        """
        try:
            if not self.is_connected:
                return False
            
            result = await self.redis_client.delete(key)
            return result > 0
            
        except Exception as e:
            logger.error(f"Error deleting cache key {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if key exists
        """
        try:
            if not self.is_connected:
                return False
            
            result = await self.redis_client.exists(key)
            return result > 0
            
        except Exception as e:
            logger.error(f"Error checking cache key {key}: {e}")
            return False
    
    async def expire(self, key: str, ttl: int) -> bool:
        """
        Set expiration for key.
        
        Args:
            key: Cache key
            ttl: Time to live in seconds
            
        Returns:
            True if successful
        """
        try:
            if not self.is_connected:
                return False
            
            result = await self.redis_client.expire(key, ttl)
            return result
            
        except Exception as e:
            logger.error(f"Error setting expiration for cache key {key}: {e}")
            return False
    
    async def ttl(self, key: str) -> int:
        """
        Get TTL for key.
        
        Args:
            key: Cache key
            
        Returns:
            TTL in seconds, -1 if no expiration, -2 if key doesn't exist
        """
        try:
            if not self.is_connected:
                return -2
            
            return await self.redis_client.ttl(key)
            
        except Exception as e:
            logger.error(f"Error getting TTL for cache key {key}: {e}")
            return -2
    
    async def keys(self, pattern: str = "*") -> List[str]:
        """
        Get keys matching pattern.
        
        Args:
            pattern: Key pattern
            
        Returns:
            List of matching keys
        """
        try:
            if not self.is_connected:
                return []
            
            return await self.redis_client.keys(pattern)
            
        except Exception as e:
            logger.error(f"Error getting keys with pattern {pattern}: {e}")
            return []
    
    async def flush_all(self) -> bool:
        """
        Flush all keys from cache.
        
        Returns:
            True if successful
        """
        try:
            if not self.is_connected:
                return False
            
            await self.redis_client.flushdb()
            return True
            
        except Exception as e:
            logger.error(f"Error flushing cache: {e}")
            return False
    
    async def increment(self, key: str, amount: int = 1) -> Optional[int]:
        """
        Increment numeric value.
        
        Args:
            key: Cache key
            amount: Amount to increment
            
        Returns:
            New value or None
        """
        try:
            if not self.is_connected:
                return None
            
            return await self.redis_client.incrby(key, amount)
            
        except Exception as e:
            logger.error(f"Error incrementing cache key {key}: {e}")
            return None
    
    async def decrement(self, key: str, amount: int = 1) -> Optional[int]:
        """
        Decrement numeric value.
        
        Args:
            key: Cache key
            amount: Amount to decrement
            
        Returns:
            New value or None
        """
        try:
            if not self.is_connected:
                return None
            
            return await self.redis_client.decrby(key, amount)
            
        except Exception as e:
            logger.error(f"Error decrementing cache key {key}: {e}")
            return None
    
    async def list_push(self, key: str, *values: Any) -> Optional[int]:
        """
        Push values to list.
        
        Args:
            key: Cache key
            *values: Values to push
            
        Returns:
            New list length or None
        """
        try:
            if not self.is_connected:
                return None
            
            # Serialize values
            serialized_values = []
            for value in values:
                if not isinstance(value, (str, int, float, bool)):
                    value = json.dumps(value, default=str)
                serialized_values.append(value)
            
            return await self.redis_client.lpush(key, *serialized_values)
            
        except Exception as e:
            logger.error(f"Error pushing to list {key}: {e}")
            return None
    
    async def list_pop(self, key: str) -> Optional[Any]:
        """
        Pop value from list.
        
        Args:
            key: Cache key
            
        Returns:
            Popped value or None
        """
        try:
            if not self.is_connected:
                return None
            
            value = await self.redis_client.rpop(key)
            if value is None:
                return None
            
            # Try to deserialize JSON
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value
                
        except Exception as e:
            logger.error(f"Error popping from list {key}: {e}")
            return None
    
    async def list_length(self, key: str) -> int:
        """
        Get list length.
        
        Args:
            key: Cache key
            
        Returns:
            List length
        """
        try:
            if not self.is_connected:
                return 0
            
            return await self.redis_client.llen(key)
            
        except Exception as e:
            logger.error(f"Error getting list length {key}: {e}")
            return 0
    
    async def hash_set(self, key: str, field: str, value: Any) -> bool:
        """
        Set hash field.
        
        Args:
            key: Cache key
            field: Hash field
            value: Value to set
            
        Returns:
            True if successful
        """
        try:
            if not self.is_connected:
                return False
            
            # Serialize value if needed
            if not isinstance(value, (str, int, float, bool)):
                value = json.dumps(value, default=str)
            
            await self.redis_client.hset(key, field, value)
            return True
            
        except Exception as e:
            logger.error(f"Error setting hash field {key}.{field}: {e}")
            return False
    
    async def hash_get(self, key: str, field: str) -> Optional[Any]:
        """
        Get hash field.
        
        Args:
            key: Cache key
            field: Hash field
            
        Returns:
            Field value or None
        """
        try:
            if not self.is_connected:
                return None
            
            value = await self.redis_client.hget(key, field)
            if value is None:
                return None
            
            # Try to deserialize JSON
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value
                
        except Exception as e:
            logger.error(f"Error getting hash field {key}.{field}: {e}")
            return None
    
    async def hash_get_all(self, key: str) -> Dict[str, Any]:
        """
        Get all hash fields.
        
        Args:
            key: Cache key
            
        Returns:
            Dictionary of field-value pairs
        """
        try:
            if not self.is_connected:
                return {}
            
            data = await self.redis_client.hgetall(key)
            result = {}
            
            for field, value in data.items():
                # Try to deserialize JSON
                try:
                    result[field] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    result[field] = value
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting all hash fields {key}: {e}")
            return {}
    
    async def set_add(self, key: str, *values: Any) -> Optional[int]:
        """
        Add values to set.
        
        Args:
            key: Cache key
            *values: Values to add
            
        Returns:
            Number of new values added or None
        """
        try:
            if not self.is_connected:
                return None
            
            # Serialize values
            serialized_values = []
            for value in values:
                if not isinstance(value, (str, int, float, bool)):
                    value = json.dumps(value, default=str)
                serialized_values.append(value)
            
            return await self.redis_client.sadd(key, *serialized_values)
            
        except Exception as e:
            logger.error(f"Error adding to set {key}: {e}")
            return None
    
    async def set_members(self, key: str) -> List[Any]:
        """
        Get all set members.
        
        Args:
            key: Cache key
            
        Returns:
            List of set members
        """
        try:
            if not self.is_connected:
                return []
            
            members = await self.redis_client.smembers(key)
            result = []
            
            for member in members:
                # Try to deserialize JSON
                try:
                    result.append(json.loads(member))
                except (json.JSONDecodeError, TypeError):
                    result.append(member)
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting set members {key}: {e}")
            return []
    
    async def set_is_member(self, key: str, value: Any) -> bool:
        """
        Check if value is in set.
        
        Args:
            key: Cache key
            value: Value to check
            
        Returns:
            True if value is in set
        """
        try:
            if not self.is_connected:
                return False
            
            # Serialize value if needed
            if not isinstance(value, (str, int, float, bool)):
                value = json.dumps(value, default=str)
            
            return await self.redis_client.sismember(key, value)
            
        except Exception as e:
            logger.error(f"Error checking set membership {key}: {e}")
            return False


# Global cache manager instance
cache_manager = CacheManager()


# Cache decorators and utilities
def cache_key(*args, **kwargs) -> str:
    """Generate cache key from arguments"""
    key_parts = []
    
    for arg in args:
        if isinstance(arg, (str, int, float, bool)):
            key_parts.append(str(arg))
        else:
            key_parts.append(str(hash(str(arg))))
    
    for k, v in sorted(kwargs.items()):
        key_parts.append(f"{k}:{v}")
    
    return ":".join(key_parts)


async def cached(
    ttl: int = None,
    key_prefix: str = "",
    serialize: bool = True
):
    """
    Cache decorator for async functions.
    
    Args:
        ttl: Time to live in seconds
        key_prefix: Key prefix
        serialize: Whether to serialize as JSON
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Generate cache key
            key = f"{key_prefix}:{func.__name__}:{cache_key(*args, **kwargs)}"
            
            # Try to get from cache
            cached_result = await cache_manager.get(key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {key}")
                return cached_result
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            await cache_manager.set(key, result, ttl, serialize)
            logger.debug(f"Cached result for {key}")
            
            return result
        
        return wrapper
    return decorator


# Cache utilities for specific use cases
class SequenceCache:
    """Cache utilities for email sequences"""
    
    @staticmethod
    async def get_sequence(sequence_id: str) -> Optional[Dict[str, Any]]:
        """Get cached sequence"""
        return await cache_manager.get(f"sequence:{sequence_id}")
    
    @staticmethod
    async def set_sequence(sequence_id: str, sequence_data: Dict[str, Any], ttl: int = None) -> bool:
        """Cache sequence"""
        if ttl is None:
            ttl = settings.cache_sequence_ttl
        return await cache_manager.set(f"sequence:{sequence_id}", sequence_data, ttl)
    
    @staticmethod
    async def delete_sequence(sequence_id: str) -> bool:
        """Delete cached sequence"""
        return await cache_manager.delete(f"sequence:{sequence_id}")
    
    @staticmethod
    async def get_sequence_analytics(sequence_id: str, date_range: str) -> Optional[Dict[str, Any]]:
        """Get cached sequence analytics"""
        return await cache_manager.get(f"analytics:sequence:{sequence_id}:{date_range}")
    
    @staticmethod
    async def set_sequence_analytics(
        sequence_id: str,
        date_range: str,
        analytics_data: Dict[str, Any],
        ttl: int = None
    ) -> bool:
        """Cache sequence analytics"""
        if ttl is None:
            ttl = settings.cache_analytics_ttl
        return await cache_manager.set(
            f"analytics:sequence:{sequence_id}:{date_range}",
            analytics_data,
            ttl
        )


class SubscriberCache:
    """Cache utilities for subscribers"""
    
    @staticmethod
    async def get_subscriber(subscriber_id: str) -> Optional[Dict[str, Any]]:
        """Get cached subscriber"""
        return await cache_manager.get(f"subscriber:{subscriber_id}")
    
    @staticmethod
    async def set_subscriber(subscriber_id: str, subscriber_data: Dict[str, Any], ttl: int = None) -> bool:
        """Cache subscriber"""
        if ttl is None:
            ttl = settings.cache_ttl_seconds
        return await cache_manager.set(f"subscriber:{subscriber_id}", subscriber_data, ttl)
    
    @staticmethod
    async def delete_subscriber(subscriber_id: str) -> bool:
        """Delete cached subscriber"""
        return await cache_manager.delete(f"subscriber:{subscriber_id}")


class TemplateCache:
    """Cache utilities for email templates"""
    
    @staticmethod
    async def get_template(template_id: str) -> Optional[Dict[str, Any]]:
        """Get cached template"""
        return await cache_manager.get(f"template:{template_id}")
    
    @staticmethod
    async def set_template(template_id: str, template_data: Dict[str, Any], ttl: int = None) -> bool:
        """Cache template"""
        if ttl is None:
            ttl = settings.cache_ttl_seconds
        return await cache_manager.set(f"template:{template_id}", template_data, ttl)
    
    @staticmethod
    async def delete_template(template_id: str) -> bool:
        """Delete cached template"""
        return await cache_manager.delete(f"template:{template_id}")


# Initialize cache manager
async def init_cache() -> None:
    """Initialize cache manager"""
    await cache_manager.connect()


async def close_cache() -> None:
    """Close cache manager"""
    await cache_manager.disconnect()


async def check_cache_health() -> bool:
    """Check cache health"""
    return await cache_manager.health_check()






























