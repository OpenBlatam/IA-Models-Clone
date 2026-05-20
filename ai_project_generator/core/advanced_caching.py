"""
Advanced Caching - Caching Avanzado
==================================

Estrategias de caching avanzadas:
- Multi-layer caching
- Cache invalidation strategies
- Cache warming
- Distributed caching
- Cache analytics
"""

import hashlib
from typing import Optional, Dict, Any, List, Callable, Tuple
from functools import wraps
from enum import Enum
from datetime import datetime, timedelta

from .shared_utils import get_logger
from .json_utils import json_dumps_str

logger = get_logger(__name__)


class CacheStrategy(str, Enum):
    """Estrategias de cache"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    TTL = "ttl"  # Time To Live
    WRITE_THROUGH = "write_through"
    WRITE_BACK = "write_back"


class CacheEntry:
    """Entrada de cache"""
    
    def __init__(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        created_at: Optional[datetime] = None
    ) -> None:
        self.key = key
        self.value = value
        self.ttl = ttl
        self.created_at = created_at or datetime.now()
        self.access_count = 0
        self.last_accessed = self.created_at
    
    def is_expired(self) -> bool:
        """Verifica si está expirado"""
        if self.ttl is None:
            return False
        
        elapsed = (datetime.now() - self.created_at).total_seconds()
        return elapsed > self.ttl
    
    def access(self) -> None:
        """Registra acceso"""
        self.access_count += 1
        self.last_accessed = datetime.now()


class AdvancedCache:
    """
    Cache avanzado con múltiples estrategias.
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        strategy: CacheStrategy = CacheStrategy.LRU,
        default_ttl: Optional[int] = None
    ) -> None:
        self.max_size = max_size
        self.strategy = strategy
        self.default_ttl = default_ttl
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order: List[str] = []
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "sets": 0
        }
    
    def _generate_key(self, *args: Any, **kwargs: Any) -> str:
        """Genera clave de cache"""
        key_data = {
            "args": args,
            "kwargs": kwargs
        }
        key_string = json_dumps_str(key_data)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Obtiene valor del cache"""
        if key not in self.cache:
            self.stats["misses"] += 1
            return None
        
        entry = self.cache[key]
        
        if entry.is_expired():
            del self.cache[key]
            if key in self.access_order:
                self.access_order.remove(key)
            self.stats["misses"] += 1
            return None
        
        entry.access()
        self.stats["hits"] += 1
        
        # Actualizar orden de acceso para LRU
        if self.strategy == CacheStrategy.LRU:
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
        
        return entry.value
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> None:
        """Establece valor en cache"""
        ttl = ttl or self.default_ttl
        
        # Verificar si necesitamos evictar
        if len(self.cache) >= self.max_size and key not in self.cache:
            self._evict()
        
        entry = CacheEntry(key, value, ttl)
        self.cache[key] = entry
        self.stats["sets"] += 1
        
        # Actualizar orden
        if self.strategy == CacheStrategy.LRU:
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
    
    def _evict(self) -> None:
        """Evicta entrada según estrategia"""
        if not self.cache:
            return
        
        if self.strategy == CacheStrategy.LRU:
            # Remover least recently used
            if self.access_order:
                key_to_remove = self.access_order.pop(0)
                if key_to_remove in self.cache:
                    del self.cache[key_to_remove]
                    self.stats["evictions"] += 1
        elif self.strategy == CacheStrategy.LFU:
            # Remover least frequently used
            least_used = min(
                self.cache.items(),
                key=lambda x: x[1].access_count
            )
            del self.cache[least_used[0]]
            if least_used[0] in self.access_order:
                self.access_order.remove(least_used[0])
            self.stats["evictions"] += 1
        elif self.strategy == CacheStrategy.FIFO:
            # Remover first in
            if self.access_order:
                key_to_remove = self.access_order.pop(0)
                if key_to_remove in self.cache:
                    del self.cache[key_to_remove]
                    self.stats["evictions"] += 1
        else:
            # Default: remover primera entrada
            first_key = next(iter(self.cache))
            del self.cache[first_key]
            if first_key in self.access_order:
                self.access_order.remove(first_key)
            self.stats["evictions"] += 1
    
    def delete(self, key: str) -> bool:
        """Elimina entrada del cache"""
        if key in self.cache:
            del self.cache[key]
            if key in self.access_order:
                self.access_order.remove(key)
            return True
        return False
    
    def clear(self) -> None:
        """Limpia todo el cache"""
        self.cache.clear()
        self.access_order.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del cache"""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = (self.stats["hits"] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            **self.stats,
            "size": len(self.cache),
            "max_size": self.max_size,
            "hit_rate": f"{hit_rate:.2f}%",
            "strategy": self.strategy.value
        }
    
    def cache_decorator(
        self,
        ttl: Optional[int] = None,
        key_prefix: Optional[str] = None
    ):
        """Decorator para caching automático"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                cache_key = self._generate_key(key_prefix or func.__name__, *args, **kwargs)
                cached = self.get(cache_key)
                if cached is not None:
                    return cached
                
                result = await func(*args, **kwargs)
                self.set(cache_key, result, ttl)
                return result
            
            @wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                cache_key = self._generate_key(key_prefix or func.__name__, *args, **kwargs)
                cached = self.get(cache_key)
                if cached is not None:
                    return cached
                
                result = func(*args, **kwargs)
                self.set(cache_key, result, ttl)
                return result
            
            import asyncio
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            return sync_wrapper
        
        return decorator


def get_advanced_cache(
    max_size: int = 1000,
    strategy: CacheStrategy = CacheStrategy.LRU,
    default_ttl: Optional[int] = None
) -> AdvancedCache:
    """Obtiene cache avanzado"""
    return AdvancedCache(max_size=max_size, strategy=strategy, default_ttl=default_ttl)

