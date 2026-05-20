"""
Redis Connection Management
Centralized Redis connection handling
"""

import logging
from typing import Optional
from redis.asyncio import Redis

from ...utils.config import get_settings

logger = logging.getLogger(__name__)

class RedisConnectionManager:
    """Manages Redis connections"""
    
    def __init__(self, redis_url: Optional[str] = None, decode_responses: bool = True):
        """Initialize Redis connection manager"""
        settings = get_settings()
        self.redis_url = redis_url or settings.redis_url
        self.decode_responses = decode_responses
        self.client: Optional[Redis] = None
    
    def initialize(self) -> bool:
        """Initialize Redis connection"""
        try:
            if not self.redis_url:
                redis_host = getattr(get_settings(), 'redis_host', 'localhost')
                redis_port = getattr(get_settings(), 'redis_port', 6379)
                redis_db = getattr(get_settings(), 'redis_db', 0)
                self.redis_url = f"redis://{redis_host}:{redis_port}/{redis_db}"
            
            self.client = Redis.from_url(
                self.redis_url,
                decode_responses=self.decode_responses,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True
            )
            logger.info("Redis connection established")
            return True
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            self.client = None
            return False
    
    async def close(self):
        """Close Redis connection"""
        if self.client:
            try:
                await self.client.aclose()
                logger.info("Redis connection closed")
            except Exception as e:
                logger.error(f"Error closing Redis connection: {e}")
            finally:
                self.client = None
    
    def get_client(self) -> Optional[Redis]:
        """Get Redis client"""
        if self.client is None:
            self.initialize()
        return self.client

_redis_manager: Optional[RedisConnectionManager] = None

def get_redis_manager(redis_url: Optional[str] = None, decode_responses: bool = True) -> RedisConnectionManager:
    """Get the global Redis connection manager"""
    global _redis_manager
    if _redis_manager is None:
        _redis_manager = RedisConnectionManager(redis_url, decode_responses)
        _redis_manager.initialize()
    return _redis_manager







