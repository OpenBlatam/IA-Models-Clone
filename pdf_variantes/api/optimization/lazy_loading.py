"""
Lazy Loading Optimization
Load data only when needed
"""

from typing import Any, Optional, Callable
from functools import cached_property, lru_cache
import asyncio


class LazyLoader:
    """Lazy load data on demand"""
    
    def __init__(self, loader_func: Callable):
        self.loader_func = loader_func
        self._value: Optional[Any] = None
        self._loaded = False
        self._lock = asyncio.Lock()
    
    async def get(self) -> Any:
        """Get value, loading if needed"""
        if self._loaded:
            return self._value
        
        async with self._lock:
            if not self._loaded:
                self._value = await self.loader_func()
                self._loaded = True
        
        return self._value
    
    def reset(self):
        """Reset loader to force reload"""
        self._loaded = False
        self._value = None


class AsyncLazyProperty:
    """Lazy property for async operations"""
    
    def __init__(self, func: Callable):
        self.func = func
        self.name = func.__name__
        self._cache = {}
    
    def __get__(self, instance, owner):
        if instance is None:
            return self
        
        if instance not in self._cache:
            self._cache[instance] = LazyLoader(
                lambda: self.func(instance)
            )
        
        return self._cache[instance]


def async_lazy_property(func: Callable):
    """Decorator for async lazy property"""
    return AsyncLazyProperty(func)


class PrefetchOptimizer:
    """Prefetch related data to avoid N+1 queries"""
    
    def __init__(self):
        self.prefetch_cache: Dict[str, Any] = {}
    
    async def prefetch_related(
        self,
        items: List[Any],
        relation_name: str,
        fetch_func: Callable
    ):
        """Prefetch related data for items"""
        # Collect all IDs
        ids = [item.id for item in items]
        
        # Fetch all at once
        related_data = await fetch_func(ids)
        
        # Cache by ID
        for item in related_data:
            self.prefetch_cache[f"{relation_name}_{item.id}"] = item
        
        return related_data
    
    def get_prefetched(self, relation_name: str, item_id: str) -> Optional[Any]:
        """Get prefetched data"""
        return self.prefetch_cache.get(f"{relation_name}_{item_id}")






