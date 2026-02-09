"""
Cache Manager

Centralized cache manager with multiple cache instances.
"""

import threading
from typing import Dict, Any, Optional

from .lru_cache import LRUCache


class CacheManager:
    """Centralized cache manager with multiple cache instances"""
    
    _instance: Optional['CacheManager'] = None
    _lock = threading.Lock()
    
    def __init__(self):
        self._caches: Dict[str, LRUCache] = {}
        self._default_cache = LRUCache(max_size=1000, default_ttl=3600.0)
    
    @classmethod
    def get_instance(cls) -> 'CacheManager':
        """Get singleton instance"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def get_cache(self, name: str = "default", max_size: int = 1000, default_ttl: float = 3600.0) -> LRUCache:
        """Get or create a named cache"""
        if name not in self._caches:
            self._caches[name] = LRUCache(max_size=max_size, default_ttl=default_ttl)
        return self._caches[name]
    
    def get_default_cache(self) -> LRUCache:
        """Get default cache"""
        return self._default_cache
    
    def clear_all(self) -> Dict[str, int]:
        """Clear all caches"""
        results = {}
        for name, cache in self._caches.items():
            results[name] = cache.clear()
        results["default"] = self._default_cache.clear()
        return results
    
    def cleanup_all(self) -> Dict[str, int]:
        """Cleanup expired entries in all caches"""
        results = {}
        for name, cache in self._caches.items():
            results[name] = cache.cleanup_expired()
        results["default"] = self._default_cache.cleanup_expired()
        return results
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all caches"""
        stats = {}
        for name, cache in self._caches.items():
            stats[name] = cache.get_stats()
        stats["default"] = self._default_cache.get_stats()
        return stats


def get_cache_manager() -> CacheManager:
    """Get the global cache manager instance"""
    return CacheManager.get_instance()




