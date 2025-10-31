"""
Caching Layer
=============

Caching utilities and decorators for improved performance.
"""

import asyncio
import json
import hashlib
import logging
from typing import Any, Callable, Optional, Union, Dict
from functools import wraps
from datetime import datetime, timedelta
import redis.asyncio as redis
from ..config import config

logger = logging.getLogger(__name__)

class CacheManager:
    """Centralized cache management."""
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.memory_cache: Dict[str, Any] = {}
        self.cache_ttl = config.cache_ttl
        
    async def initialize(self):
        """Initialize cache connections."""
        try:
            if config.cache_type == "redis":
                self.redis_client = redis.from_url(
                    config.cache_url,
                    encoding="utf-8",
                    decode_responses=True
                )
                # Test connection
                await self.redis_client.ping()
                logger.info("Redis cache initialized successfully")
            else:
                logger.info("Using in-memory cache")
        except Exception as e:
            logger.error(f"Failed to initialize cache: {str(e)}")
            # Fallback to memory cache
            self.redis_client = None
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            if self.redis_client:
                value = await self.redis_client.get(key)
                if value:
                    return json.loads(value)
            else:
                # Check memory cache
                if key in self.memory_cache:
                    cached_item = self.memory_cache[key]
                    if cached_item["expires_at"] > datetime.now():
                        return cached_item["value"]
                    else:
                        del self.memory_cache[key]
            return None
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {str(e)}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        try:
            ttl = ttl or self.cache_ttl
            
            if self.redis_client:
                await self.redis_client.setex(
                    key, 
                    ttl, 
                    json.dumps(value, default=str)
                )
            else:
                # Store in memory cache
                self.memory_cache[key] = {
                    "value": value,
                    "expires_at": datetime.now() + timedelta(seconds=ttl)
                }
            return True
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {str(e)}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        try:
            if self.redis_client:
                await self.redis_client.delete(key)
            else:
                self.memory_cache.pop(key, None)
            return True
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {str(e)}")
            return False
    
    async def clear(self) -> bool:
        """Clear all cache."""
        try:
            if self.redis_client:
                await self.redis_client.flushdb()
            else:
                self.memory_cache.clear()
            return True
        except Exception as e:
            logger.error(f"Cache clear error: {str(e)}")
            return False
    
    def generate_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        # Create a hash of the arguments
        key_data = {
            "args": args,
            "kwargs": sorted(kwargs.items())
        }
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        key_hash = hashlib.md5(key_string.encode()).hexdigest()
        return f"{prefix}:{key_hash}"

# Global cache manager instance
cache_manager = CacheManager()

def cached(prefix: str, ttl: Optional[int] = None, key_func: Optional[Callable] = None):
    """Decorator for caching function results."""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = cache_manager.generate_key(prefix, *args, **kwargs)
            
            # Try to get from cache
            cached_result = await cache_manager.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for key: {cache_key}")
                return cached_result
            
            # Execute function and cache result
            logger.debug(f"Cache miss for key: {cache_key}")
            result = await func(*args, **kwargs)
            
            # Cache the result
            await cache_manager.set(cache_key, result, ttl)
            
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For sync functions, we'll run them in an async context
            return asyncio.run(async_wrapper(*args, **kwargs))
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

class CacheService:
    """Service for cache operations."""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
    
    async def cache_agent_list(self, business_area: Optional[str], is_active: Optional[bool], data: Any):
        """Cache agent list results."""
        key = self.cache_manager.generate_key(
            "agents:list",
            business_area=business_area,
            is_active=is_active
        )
        await self.cache_manager.set(key, data, ttl=300)  # 5 minutes
    
    async def get_cached_agent_list(self, business_area: Optional[str], is_active: Optional[bool]) -> Optional[Any]:
        """Get cached agent list."""
        key = self.cache_manager.generate_key(
            "agents:list",
            business_area=business_area,
            is_active=is_active
        )
        return await self.cache_manager.get(key)
    
    async def cache_workflow_list(self, business_area: Optional[str], status: Optional[str], data: Any):
        """Cache workflow list results."""
        key = self.cache_manager.generate_key(
            "workflows:list",
            business_area=business_area,
            status=status
        )
        await self.cache_manager.set(key, data, ttl=300)  # 5 minutes
    
    async def get_cached_workflow_list(self, business_area: Optional[str], status: Optional[str]) -> Optional[Any]:
        """Get cached workflow list."""
        key = self.cache_manager.generate_key(
            "workflows:list",
            business_area=business_area,
            status=status
        )
        return await self.cache_manager.get(key)
    
    async def cache_document_list(self, business_area: Optional[str], document_type: Optional[str], data: Any):
        """Cache document list results."""
        key = self.cache_manager.generate_key(
            "documents:list",
            business_area=business_area,
            document_type=document_type
        )
        await self.cache_manager.set(key, data, ttl=600)  # 10 minutes
    
    async def get_cached_document_list(self, business_area: Optional[str], document_type: Optional[str]) -> Optional[Any]:
        """Get cached document list."""
        key = self.cache_manager.generate_key(
            "documents:list",
            business_area=business_area,
            document_type=document_type
        )
        return await self.cache_manager.get(key)
    
    async def invalidate_agent_cache(self, agent_id: Optional[str] = None):
        """Invalidate agent-related cache."""
        if agent_id:
            # Invalidate specific agent cache
            key = self.cache_manager.generate_key("agents:detail", agent_id=agent_id)
            await self.cache_manager.delete(key)
        
        # Invalidate agent list cache
        await self.cache_manager.delete("agents:list")
    
    async def invalidate_workflow_cache(self, workflow_id: Optional[str] = None):
        """Invalidate workflow-related cache."""
        if workflow_id:
            # Invalidate specific workflow cache
            key = self.cache_manager.generate_key("workflows:detail", workflow_id=workflow_id)
            await self.cache_manager.delete(key)
        
        # Invalidate workflow list cache
        await self.cache_manager.delete("workflows:list")
    
    async def invalidate_document_cache(self, document_id: Optional[str] = None):
        """Invalidate document-related cache."""
        if document_id:
            # Invalidate specific document cache
            key = self.cache_manager.generate_key("documents:detail", document_id=document_id)
            await self.cache_manager.delete(key)
        
        # Invalidate document list cache
        await self.cache_manager.delete("documents:list")

# Global cache service instance
cache_service = CacheService(cache_manager)
