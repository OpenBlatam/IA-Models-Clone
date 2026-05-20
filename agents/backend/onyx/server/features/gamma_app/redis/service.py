"""
Redis Service Implementation
"""

from typing import Any, Optional
import logging
from datetime import timedelta
import redis.asyncio as redis

from .base import RedisBase, RedisConnection

logger = logging.getLogger(__name__)


class RedisService(RedisBase):
    """Redis service implementation"""
    
    def __init__(self, connection_string: str, config_service=None):
        """Initialize Redis service"""
        self.connection_string = connection_string
        self.config_service = config_service
        self._client: Optional[redis.Redis] = None
        self._connected = False
    
    async def connect(self) -> bool:
        """Connect to Redis"""
        try:
            self._client = await redis.from_url(
                self.connection_string,
                decode_responses=False
            )
            await self._client.ping()
            self._connected = True
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to Redis: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from Redis"""
        try:
            if self._client:
                await self._client.close()
                self._client = None
            self._connected = False
            return True
            
        except Exception as e:
            logger.error(f"Error disconnecting from Redis: {e}")
            return False
    
    async def get(self, key: str) -> Optional[bytes]:
        """Get value by key"""
        try:
            if not self._connected:
                await self.connect()
            
            return await self._client.get(key)
            
        except Exception as e:
            logger.error(f"Error getting key: {e}")
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[timedelta] = None
    ) -> bool:
        """Set value by key"""
        try:
            if not self._connected:
                await self.connect()
            
            if ttl:
                await self._client.setex(key, int(ttl.total_seconds()), value)
            else:
                await self._client.set(key, value)
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting key: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key"""
        try:
            if not self._connected:
                await self.connect()
            
            await self._client.delete(key)
            return True
            
        except Exception as e:
            logger.error(f"Error deleting key: {e}")
            return False
    
    async def lpush(self, key: str, value: Any) -> int:
        """Push to list"""
        try:
            if not self._connected:
                await self.connect()
            
            return await self._client.lpush(key, value)
            
        except Exception as e:
            logger.error(f"Error pushing to list: {e}")
            return 0
    
    async def rpop(self, key: str) -> Optional[bytes]:
        """Pop from list"""
        try:
            if not self._connected:
                await self.connect()
            
            return await self._client.rpop(key)
            
        except Exception as e:
            logger.error(f"Error popping from list: {e}")
            return None
    
    async def setex(self, key: str, seconds: int, value: Any) -> bool:
        """Set with expiration"""
        return await self.set(key, value, timedelta(seconds=seconds))
    
    async def exists(self, key: str) -> int:
        """Check if key exists"""
        try:
            if not self._connected:
                await self.connect()
            
            return await self._client.exists(key)
            
        except Exception as e:
            logger.error(f"Error checking key existence: {e}")
            return 0

