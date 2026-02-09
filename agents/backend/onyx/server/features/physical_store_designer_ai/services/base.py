"""
Base classes for services - Legacy compatibility
Note: New services should use core.service_base instead
"""

from typing import Any, Dict, Optional
from ..core.service_base import BaseService as CoreBaseService, TimestampedService as CoreTimestampedService

# Re-export for backward compatibility
BaseService = CoreBaseService
TimestampedService = CoreTimestampedService


class StorageMixin:
    """Mixin for services that need storage"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._storage: Dict[str, Any] = {}
    
    def store(self, key: str, value: Any) -> None:
        """Store a value"""
        self._storage[key] = value
    
    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve a value"""
        return self._storage.get(key)
    
    def delete(self, key: str) -> bool:
        """Delete a value"""
        if key in self._storage:
            del self._storage[key]
            return True
        return False
    
    def list_keys(self) -> list[str]:
        """List all stored keys"""
        return list(self._storage.keys())


class CacheMixin:
    """Mixin for services that need caching with per-key TTL"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache: Dict[str, tuple[Any, float, float]] = {}  # key -> (value, timestamp, ttl)
        self._default_ttl: float = 3600  # 1 hour default
    
    def get_cached(self, key: str) -> Optional[Any]:
        """Get cached value if not expired"""
        if key in self._cache:
            value, timestamp, ttl = self._cache[key]
            import time
            if time.time() - timestamp < ttl:
                return value
            else:
                del self._cache[key]
        return None
    
    def set_cached(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Cache a value with optional per-key TTL"""
        import time
        cache_ttl = ttl if ttl is not None else self._default_ttl
        self._cache[key] = (value, time.time(), cache_ttl)
    
    def clear_cache(self) -> None:
        """Clear all cached values"""
        self._cache.clear()
    
    def clear_expired(self) -> int:
        """Clear expired cache entries, returns number of cleared entries"""
        import time
        current_time = time.time()
        expired_keys = [
            key for key, (_, timestamp, ttl) in self._cache.items()
            if current_time - timestamp >= ttl
        ]
        for key in expired_keys:
            del self._cache[key]
        return len(expired_keys)

