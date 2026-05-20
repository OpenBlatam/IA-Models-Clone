"""Redis distributed cache"""
from typing import Any, Optional
import json
import logging

logger = logging.getLogger(__name__)


class RedisCache:
    """Redis distributed cache"""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        ttl: int = 3600
    ):
        """
        Initialize Redis cache
        
        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Optional password
            ttl: Default TTL in seconds
        """
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.ttl = ttl
        self.client = None
        self._connect()
    
    def _connect(self):
        """Connect to Redis"""
        try:
            import redis
            self.client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=True
            )
            # Test connection
            self.client.ping()
            logger.info(f"Connected to Redis at {self.host}:{self.port}")
        except ImportError:
            logger.warning("redis not installed, Redis cache disabled")
            self.client = None
        except Exception as e:
            logger.error(f"Error connecting to Redis: {e}")
            self.client = None
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        if not self.client:
            return None
        
        try:
            value = self.client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Error getting from Redis cache: {e}")
            return None
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ):
        """
        Set value in cache
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional TTL in seconds
        """
        if not self.client:
            return
        
        try:
            ttl = ttl or self.ttl
            serialized = json.dumps(value)
            self.client.setex(key, ttl, serialized)
        except Exception as e:
            logger.error(f"Error setting Redis cache: {e}")
    
    def delete(self, key: str):
        """
        Delete key from cache
        
        Args:
            key: Cache key
        """
        if not self.client:
            return
        
        try:
            self.client.delete(key)
        except Exception as e:
            logger.error(f"Error deleting from Redis cache: {e}")
    
    def clear(self):
        """Clear all cache"""
        if not self.client:
            return
        
        try:
            self.client.flushdb()
        except Exception as e:
            logger.error(f"Error clearing Redis cache: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            Cache statistics
        """
        if not self.client:
            return {
                "enabled": False,
                "host": self.host,
                "port": self.port
            }
        
        try:
            info = self.client.info()
            return {
                "enabled": True,
                "host": self.host,
                "port": self.port,
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": info.get("used_memory_human", "0B"),
                "keyspace": info.get("db0", {})
            }
        except Exception as e:
            logger.error(f"Error getting Redis stats: {e}")
            return {
                "enabled": True,
                "error": str(e)
            }


# Global Redis cache
_redis_cache: Optional[RedisCache] = None


def get_redis_cache() -> Optional[RedisCache]:
    """Get global Redis cache"""
    global _redis_cache
    if _redis_cache is None:
        import os
        host = os.getenv("REDIS_HOST", "localhost")
        port = int(os.getenv("REDIS_PORT", 6379))
        password = os.getenv("REDIS_PASSWORD")
        _redis_cache = RedisCache(host=host, port=port, password=password)
    return _redis_cache

