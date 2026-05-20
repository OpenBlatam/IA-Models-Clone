"""
Key-Value Store Service Implementation
"""

from typing import Any, Optional
import logging
from datetime import timedelta

from .base import KVStoreBase, KVStore

logger = logging.getLogger(__name__)


class KeyValueStoreService(KVStoreBase):
    """Key-value store service implementation"""
    
    def __init__(self, redis_client=None):
        """Initialize key-value store service"""
        self.redis_client = redis_client
        self._store: dict = {}
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value by key"""
        try:
            if self.redis_client:
                value = await self.redis_client.get(key)
                return value.decode() if value else None
            
            return self._store.get(key)
            
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
            if self.redis_client:
                if ttl:
                    await self.redis_client.setex(
                        key,
                        int(ttl.total_seconds()),
                        str(value)
                    )
                else:
                    await self.redis_client.set(key, str(value))
            else:
                self._store[key] = value
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting key: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key"""
        try:
            if self.redis_client:
                await self.redis_client.delete(key)
            else:
                if key in self._store:
                    del self._store[key]
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting key: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        try:
            if self.redis_client:
                return await self.redis_client.exists(key) > 0
            else:
                return key in self._store
                
        except Exception as e:
            logger.error(f"Error checking key existence: {e}")
            return False

