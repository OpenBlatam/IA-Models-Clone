"""
Cache - Sistema de caché
==========================

Sistema de caché para mejorar rendimiento.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, Callable
from datetime import datetime, timedelta
from collections import OrderedDict

logger = logging.getLogger(__name__)


class CacheEntry:
    """Entrada de caché"""
    
    def __init__(self, key: str, value: Any, ttl: Optional[float] = None):
        self.key = key
        self.value = value
        self.created_at = time.time()
        self.ttl = ttl
        self.access_count = 0
        self.last_accessed = time.time()
    
    def is_expired(self) -> bool:
        """Verificar si la entrada expiró"""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    def access(self):
        """Registrar acceso"""
        self.access_count += 1
        self.last_accessed = time.time()


class Cache:
    """Sistema de caché"""
    
    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: Optional[float] = None,
        eviction_policy: str = "lru"  # lru, lfu, fifo
    ):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.eviction_policy = eviction_policy
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order = OrderedDict()
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Obtener valor del caché"""
        async with self._lock:
            if key not in self._cache:
                return None
            
            entry = self._cache[key]
            
            # Verificar expiración
            if entry.is_expired():
                del self._cache[key]
                if key in self._access_order:
                    del self._access_order[key]
                return None
            
            # Registrar acceso
            entry.access()
            self._access_order.move_to_end(key)
            
            return entry.value
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None
    ):
        """Establecer valor en caché"""
        async with self._lock:
            # Verificar tamaño
            if len(self._cache) >= self.max_size and key not in self._cache:
                await self._evict()
            
            ttl = ttl or self.default_ttl
            entry = CacheEntry(key, value, ttl)
            self._cache[key] = entry
            self._access_order[key] = True
            self._access_order.move_to_end(key)
    
    async def delete(self, key: str) -> bool:
        """Eliminar entrada del caché"""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                if key in self._access_order:
                    del self._access_order[key]
                return True
            return False
    
    async def clear(self):
        """Limpiar todo el caché"""
        async with self._lock:
            self._cache.clear()
            self._access_order.clear()
    
    async def _evict(self):
        """Eliminar entrada según política"""
        if not self._cache:
            return
        
        if self.eviction_policy == "lru":
            # Least Recently Used
            key = next(iter(self._access_order))
        elif self.eviction_policy == "lfu":
            # Least Frequently Used
            key = min(
                self._cache.keys(),
                key=lambda k: self._cache[k].access_count
            )
        else:  # fifo
            # First In First Out
            key = next(iter(self._access_order))
        
        del self._cache[key]
        if key in self._access_order:
            del self._access_order[key]
    
    async def get_or_set(
        self,
        key: str,
        callable: Callable,
        ttl: Optional[float] = None
    ) -> Any:
        """Obtener del caché o calcular y guardar"""
        value = await self.get(key)
        if value is not None:
            return value
        
        # Calcular valor
        if asyncio.iscoroutinefunction(callable):
            value = await callable()
        else:
            value = callable()
        
        await self.set(key, value, ttl)
        return value
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del caché"""
        total_size = len(self._cache)
        expired_count = sum(1 for entry in self._cache.values() if entry.is_expired())
        
        return {
            "size": total_size,
            "max_size": self.max_size,
            "expired_entries": expired_count,
            "eviction_policy": self.eviction_policy,
            "hit_rate": 0.0  # Se calcularía con más tracking
        }


class CommandCache:
    """Caché específico para resultados de comandos"""
    
    def __init__(self, max_size: int = 500, ttl: float = 3600):
        self.cache = Cache(max_size=max_size, default_ttl=ttl, eviction_policy="lru")
    
    async def get_result(self, command: str) -> Optional[str]:
        """Obtener resultado de comando desde caché"""
        cache_key = self._get_cache_key(command)
        return await self.cache.get(cache_key)
    
    async def set_result(self, command: str, result: str, ttl: Optional[float] = None):
        """Guardar resultado de comando en caché"""
        cache_key = self._get_cache_key(command)
        await self.cache.set(cache_key, result, ttl)
    
    def _get_cache_key(self, command: str) -> str:
        """Generar clave de caché para comando"""
        import hashlib
        return f"cmd_{hashlib.md5(command.encode()).hexdigest()}"



