"""
Cache Submodule
Aggregates cache components.
"""

from typing import Optional, Any
import logging
from .manager import CacheManager as BaseCacheManager
from .storage import store_item, set_item
from .cleanup import cleanup_expired

logger = logging.getLogger(__name__)


class CacheManager(BaseCacheManager):
    """
    Complete cache manager combining all functionality.
    """
    
    def set(self, prefix: str, identifier: str, data: Any, ttl: Optional[int] = None) -> None:
        """Almacena un valor en el cache"""
        if not self.cache_enabled:
            return
        
        key = self._generate_key(prefix, identifier)
        ttl = ttl or self.cache_ttl
        
        set_item(
            self.memory_cache,
            key,
            data,
            ttl,
            self.max_memory_items
        )
        
        logger.debug(f"Cache set: {prefix}:{identifier} (expires in {ttl}s)")
    
    def _cleanup_expired(self) -> None:
        """Elimina items expirados del cache"""
        cleanup_expired(self.memory_cache)


# Instancia global del cache
cache_manager = CacheManager()

__all__ = [
    "CacheManager",
    "store_item",
    "set_item",
    "cleanup_expired",
    "cache_manager",
]

