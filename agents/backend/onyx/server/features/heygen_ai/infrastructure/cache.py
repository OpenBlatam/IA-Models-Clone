from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import json
import pickle
from typing import Any, Optional, Dict, List, Union
from datetime import timedelta
import redis.asyncio as redis
import structlog
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Cache Manager

Manages Redis connections and caching operations.
"""


logger = structlog.get_logger()


class CacheManager:
    """
    Manages Redis cache connections and operations.
    
    Features:
    - Connection pooling
    - JSON and pickle serialization
    - TTL management
    - Health monitoring
    - Key pattern management
    """
    
    def __init__(self, config: Dict[str, Any]):
        
    """__init__ function."""
self.config = config
        self.redis_client: Optional[redis.Redis] = None
        self._is_initialized = False
    
    async def initialize(self) -> None:
        """Initialize Redis connection."""
        if self._is_initialized:
            return
        
        logger.info("Initializing cache connection", url=self._sanitize_url(self.config["url"]))
        
        # Create Redis connection with connection pooling
        self.redis_client = redis.from_url(
            self.config["url"],
            max_connections=self.config.get("max_connections", 20),
            decode_responses=self.config.get("decode_responses", True),
            health_check_interval=30,
            socket_keepalive=True,
            socket_keepalive_options={},
            retry_on_timeout=True,
            socket_connect_timeout=5,
            socket_timeout=5,
        )
        
        # Test connection
        await self._test_connection()
        
        self._is_initialized = True
        logger.info("Cache initialized successfully")
    
    async def shutdown(self) -> None:
        """Shutdown cache connections."""
        if not self._is_initialized:
            return
        
        logger.info("Shutting down cache connections")
        
        if self.redis_client:
            await self.redis_client.close()
            self.redis_client = None
        
        self._is_initialized = False
        logger.info("Cache shutdown completed")
    
    async def get(self, key: str, default: Any = None) -> Optional[Dict[str, Any]]:
        """Get value from cache."""
        if not self._is_initialized:
            return default
        
        try:
            value = await self.redis_client.get(key)
            if value is None:
                return default
            
            # Try to deserialize as JSON first, then pickle
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                try:
                    return pickle.loads(value.encode() if isinstance(value, str) else value)
                except Exception:
                    return value
        
        except Exception as e:
            logger.warning("Cache get failed", key=key, error=str(e))
            return default
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[Union[int, timedelta]] = None,
        serializer: str = "json"
    ) -> bool:
        """Set value in cache."""
        if not self._is_initialized:
            return False
        
        try:
            # Serialize value
            if serializer == "json":
                try:
                    serialized_value = json.dumps(value)
                except TypeError:
                    # Fall back to pickle for non-JSON serializable objects
                    serialized_value = pickle.dumps(value)
                    serializer = "pickle"
            else:
                serialized_value = pickle.dumps(value)
            
            # Set with TTL
            if ttl:
                if isinstance(ttl, timedelta):
                    ttl = int(ttl.total_seconds())
                await self.redis_client.setex(key, ttl, serialized_value)
            else:
                await self.redis_client.set(key, serialized_value)
            
            return True
        
        except Exception as e:
            logger.warning("Cache set failed", key=key, error=str(e))
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if not self._is_initialized:
            return False
        
        try:
            result = await self.redis_client.delete(key)
            return result > 0
        
        except Exception as e:
            logger.warning("Cache delete failed", key=key, error=str(e))
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        if not self._is_initialized:
            return False
        
        try:
            result = await self.redis_client.exists(key)
            return result > 0
        
        except Exception as e:
            logger.warning("Cache exists check failed", key=key, error=str(e))
            return False
    
    async def expire(self, key: str, ttl: Union[int, timedelta]) -> bool:
        """Set expiration for key."""
        if not self._is_initialized:
            return False
        
        try:
            if isinstance(ttl, timedelta):
                ttl = int(ttl.total_seconds())
            result = await self.redis_client.expire(key, ttl)
            return result
        
        except Exception as e:
            logger.warning("Cache expire failed", key=key, error=str(e))
            return False
    
    async def get_ttl(self, key: str) -> Optional[int]:
        """Get TTL for key."""
        if not self._is_initialized:
            return None
        
        try:
            result = await self.redis_client.ttl(key)
            return result if result >= 0 else None
        
        except Exception as e:
            logger.warning("Cache TTL check failed", key=key, error=str(e))
            return None
    
    async def increment(self, key: str, amount: int = 1) -> Optional[int]:
        """Increment counter."""
        if not self._is_initialized:
            return None
        
        try:
            result = await self.redis_client.incrby(key, amount)
            return result
        
        except Exception as e:
            logger.warning("Cache increment failed", key=key, error=str(e))
            return None
    
    async def get_keys(self, pattern: str) -> List[str]:
        """Get keys matching pattern."""
        if not self._is_initialized:
            return []
        
        try:
            keys = await self.redis_client.keys(pattern)
            return [key.decode() if isinstance(key, bytes) else key for key in keys]
        
        except Exception as e:
            logger.warning("Cache keys search failed", pattern=pattern, error=str(e))
            return []
    
    async def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern."""
        if not self._is_initialized:
            return 0
        
        try:
            keys = await self.get_keys(pattern)
            if keys:
                result = await self.redis_client.delete(*keys)
                return result
            return 0
        
        except Exception as e:
            logger.warning("Cache pattern delete failed", pattern=pattern, error=str(e))
            return 0
    
    async def flush_all(self) -> bool:
        """Flush all cache data."""
        if not self._is_initialized:
            return False
        
        try:
            await self.redis_client.flushall()
            return True
        
        except Exception as e:
            logger.warning("Cache flush failed", error=str(e))
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform cache health check."""
        if not self._is_initialized:
            return {"status": "not_initialized", "healthy": False}
        
        try:
            # Test ping
            await self.redis_client.ping()
            
            # Get info
            info = await self.redis_client.info()
            
            return {
                "status": "healthy",
                "healthy": True,
                "redis_version": info.get("redis_version"),
                "connected_clients": info.get("connected_clients"),
                "used_memory": info.get("used_memory"),
                "used_memory_human": info.get("used_memory_human"),
                "keyspace_hits": info.get("keyspace_hits"),
                "keyspace_misses": info.get("keyspace_misses"),
            }
        
        except Exception as e:
            logger.error("Cache health check failed", error=str(e))
            return {
                "status": "unhealthy",
                "healthy": False,
                "error": str(e)
            }
    
    def get_client(self) -> redis.Redis:
        """Get Redis client for direct access."""
        if not self._is_initialized:
            raise RuntimeError("Cache manager not initialized")
        return self.redis_client
    
    async def _test_connection(self) -> None:
        """Test cache connection."""
        try:
            await self.redis_client.ping()
            logger.info("Cache connection test successful")
        except Exception as e:
            logger.error("Cache connection test failed", error=str(e))
            raise
    
    def _sanitize_url(self, url: str) -> str:
        """Sanitize Redis URL for logging."""
        try:
            if "@" in url:
                protocol_part, rest = url.split("://", 1)
                if "@" in rest:
                    creds_part, host_part = rest.split("@", 1)
                    return f"{protocol_part}://***@{host_part}"
            return url
        except Exception:
            return "***SANITIZED***"


# Cache key patterns for consistent naming
class CacheKeys:
    """Standard cache key patterns."""
    
    @staticmethod
    def user(user_id: str) -> str:
        return f"user:{user_id}"
    
    @staticmethod
    def user_by_email(email: str) -> str:
        return f"user_email:{email}"
    
    @staticmethod
    def user_by_username(username: str) -> str:
        return f"user_username:{username}"
    
    @staticmethod
    def video(video_id: str) -> str:
        return f"video:{video_id}"
    
    @staticmethod
    def user_videos(user_id: str, page: int = 0) -> str:
        return f"user_videos:{user_id}:page:{page}"
    
    @staticmethod
    def video_processing(video_id: str) -> str:
        return f"video_processing:{video_id}"
    
    @staticmethod
    async def api_rate_limit(ip: str) -> str:
        return f"rate_limit:{ip}"
    
    @staticmethod
    def session(session_id: str) -> str:
        return f"session:{session_id}"


# Convenience functions for dependency injection
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Get global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        raise RuntimeError("Cache manager not initialized")
    return _cache_manager


def set_cache_manager(manager: CacheManager) -> None:
    """Set global cache manager instance."""
    global _cache_manager
    _cache_manager = manager


async def get_cache_client() -> redis.Redis:
    """FastAPI dependency for cache client."""
    cache_manager = get_cache_manager()
    return cache_manager.get_client() 