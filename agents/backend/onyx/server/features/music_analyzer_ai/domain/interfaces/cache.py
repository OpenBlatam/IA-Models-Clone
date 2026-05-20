"""
Cache Service Interface

Defines contract for caching functionality.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Dict


class ICacheService(ABC):
    """Interface for cache service"""
    
    @abstractmethod
    async def get(
        self,
        namespace: str,
        key: str
    ) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            namespace: Cache namespace (e.g., 'spotify', 'analysis')
            key: Cache key
        
        Returns:
            Cached value or None if not found
        """
        pass
    
    @abstractmethod
    async def set(
        self,
        namespace: str,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set value in cache.
        
        Args:
            namespace: Cache namespace
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (None for default)
        
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    async def delete(
        self,
        namespace: str,
        key: str
    ) -> bool:
        """
        Delete value from cache.
        
        Args:
            namespace: Cache namespace
            key: Cache key
        
        Returns:
            True if deleted, False if not found
        """
        pass
    
    @abstractmethod
    async def clear_namespace(self, namespace: str) -> int:
        """
        Clear all keys in a namespace.
        
        Args:
            namespace: Cache namespace to clear
        
        Returns:
            Number of keys cleared
        """
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache stats (hits, misses, size, etc.)
        """
        pass

