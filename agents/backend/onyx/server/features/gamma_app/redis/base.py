"""
Redis Base Classes and Interfaces
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, List
from datetime import timedelta


class RedisConnection:
    """Redis connection definition"""
    
    def __init__(self, url: str, pool_size: int = 10):
        self.url = url
        self.pool_size = pool_size
        self.connected = False


class RedisBase(ABC):
    """Base interface for Redis"""
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to Redis"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from Redis"""
        pass
    
    @abstractmethod
    async def get(self, key: str) -> Optional[bytes]:
        """Get value by key"""
        pass
    
    @abstractmethod
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[timedelta] = None
    ) -> bool:
        """Set value by key"""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete key"""
        pass
    
    @abstractmethod
    async def lpush(self, key: str, value: Any) -> int:
        """Push to list"""
        pass
    
    @abstractmethod
    async def rpop(self, key: str) -> Optional[bytes]:
        """Pop from list"""
        pass

