"""
Cache Service Adapter

Adapter that wraps CacheManager to implement ICacheService interface.
"""

import logging
from typing import Any, Optional, Dict

from ...domain.interfaces.cache import ICacheService

logger = logging.getLogger(__name__)


class CacheServiceAdapter(ICacheService):
    """
    Adapter that wraps existing CacheManager to implement ICacheService.
    
    This allows the existing CacheManager to be used with the new architecture
    without modifying the original implementation.
    """
    
    def __init__(self, cache_manager):
        """
        Initialize adapter with CacheManager instance.
        
        Args:
            cache_manager: Instance of CacheManager
        """
        self.cache_manager = cache_manager
    
    async def get(
        self,
        namespace: str,
        key: str
    ) -> Optional[Any]:
        """Get value from cache"""
        try:
            # CacheManager.get is synchronous
            return self.cache_manager.get(namespace, key)
        except Exception as e:
            logger.warning(f"Cache get failed for {namespace}:{key}: {e}")
            return None
    
    async def set(
        self,
        namespace: str,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """Set value in cache"""
        try:
            # CacheManager.set is synchronous
            self.cache_manager.set(namespace, key, value, ttl=ttl)
            return True
        except Exception as e:
            logger.warning(f"Cache set failed for {namespace}:{key}: {e}")
            return False
    
    async def delete(
        self,
        namespace: str,
        key: str
    ) -> bool:
        """Delete value from cache"""
        try:
            self.cache_manager.delete(namespace, key)
            return True
        except Exception as e:
            logger.warning(f"Cache delete failed for {namespace}:{key}: {e}")
            return False
    
    async def clear_namespace(self, namespace: str) -> int:
        """Clear all keys in a namespace"""
        try:
            self.cache_manager.clear(prefix=namespace)
            return 1  # CacheManager doesn't return count, so we return 1
        except Exception as e:
            logger.warning(f"Cache clear failed for namespace {namespace}: {e}")
            return 0
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            return self.cache_manager.get_stats()
        except Exception as e:
            logger.warning(f"Cache stats failed: {e}")
            return {}

