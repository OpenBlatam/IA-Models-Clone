"""
Cache - Sistema de caché
==========================

Sistema de caché para mejorar rendimiento.
"""

import asyncio
import logging
import time
import random
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
        eviction_policy: str = "lru"  # lru, lfu, fifo, ttl, random
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
    
    async def _evict(self) -> None:
        """
        Eliminar entrada según política de evicción.
        
        Políticas soportadas:
        - lru: Least Recently Used
        - lfu: Least Frequently Used
        - fifo: First In First Out
        - ttl: Time To Live (elimina entradas más cercanas a expirar)
        - random: Eliminación aleatoria
        """
        if not self._cache:
            return
        
        # Limpiar entradas expiradas primero
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.is_expired()
        ]
        for key in expired_keys:
            del self._cache[key]
            if key in self._access_order:
                del self._access_order[key]
        
        # Si aún necesitamos espacio, aplicar política de evicción
        if len(self._cache) >= self.max_size:
            if self.eviction_policy == "lru":
                if self._access_order:
                    key = next(iter(self._access_order))
                    del self._cache[key]
                    del self._access_order[key]
            elif self.eviction_policy == "lfu":
                key = min(
                    self._cache.keys(),
                    key=lambda k: self._cache[k].access_count
                )
                del self._cache[key]
                if key in self._access_order:
                    del self._access_order[key]
            elif self.eviction_policy == "ttl":
                # Eliminar entrada más cercana a expirar
                key = min(
                    self._cache.keys(),
                    key=lambda k: (
                        self._cache[k].ttl or float('inf'),
                        self._cache[k].created_at
                    )
                )
                del self._cache[key]
                if key in self._access_order:
                    del self._access_order[key]
            elif self.eviction_policy == "random":
                key = random.choice(list(self._cache.keys()))
                del self._cache[key]
                if key in self._access_order:
                    del self._access_order[key]
            else:  # fifo (default)
                if self._access_order:
                    key = next(iter(self._access_order))
                    del self._cache[key]
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
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas detalladas del caché.
        
        Returns:
            Diccionario con estadísticas del caché
        """
        async with self._lock:
            total_size = len(self._cache)
            expired_count = sum(1 for entry in self._cache.values() if entry.is_expired())
            
            total_accesses = sum(entry.access_count for entry in self._cache.values())
            avg_accesses = total_accesses / total_size if total_size > 0 else 0
            
            # Calcular hit rate basado en accesos
            hit_rate = 0.0
            if total_accesses > 0 and total_size > 0:
                entries_with_accesses = sum(1 for entry in self._cache.values() if entry.access_count > 0)
                hit_rate = entries_with_accesses / total_size
            
            # Encontrar entrada más y menos accedida
            most_accessed = None
            least_accessed = None
            if self._cache:
                sorted_entries = sorted(
                    self._cache.items(),
                    key=lambda x: x[1].access_count,
                    reverse=True
                )
                most_accessed = {
                    "key": sorted_entries[0][0][:50],
                    "access_count": sorted_entries[0][1].access_count
                }
                least_accessed = {
                    "key": sorted_entries[-1][0][:50],
                    "access_count": sorted_entries[-1][1].access_count
                }
            
            return {
                "size": total_size,
                "max_size": self.max_size,
                "usage_percent": round((total_size / self.max_size * 100) if self.max_size > 0 else 0, 2),
                "expired_entries": expired_count,
                "eviction_policy": self.eviction_policy,
                "default_ttl": self.default_ttl,
                "total_accesses": total_accesses,
                "avg_accesses_per_entry": round(avg_accesses, 2),
                "hit_rate": round(hit_rate, 4),
                "most_accessed": most_accessed,
                "least_accessed": least_accessed
            }


class CommandCache:
    """
    Caché específico para resultados de comandos.
    
    Optimizado para cachear resultados de ejecución de comandos,
    reduciendo la necesidad de re-ejecutar comandos idénticos.
    """
    
    def __init__(self, max_size: int = 500, ttl: float = 3600):
        """
        Inicializar caché de comandos.
        
        Args:
            max_size: Tamaño máximo del caché
            ttl: Tiempo de vida por defecto en segundos
        """
        self.cache = Cache(max_size=max_size, default_ttl=ttl, eviction_policy="lru")
        self._hits = 0
        self._misses = 0
    
    async def get_result(self, command: str) -> Optional[str]:
        """
        Obtener resultado de comando desde caché.
        
        Args:
            command: Comando a buscar
            
        Returns:
            Resultado cacheado o None si no existe
        """
        cache_key = self._get_cache_key(command)
        result = await self.cache.get(cache_key)
        if result is not None:
            self._hits += 1
        else:
            self._misses += 1
        return result
    
    async def set_result(self, command: str, result: str, ttl: Optional[float] = None) -> None:
        """
        Guardar resultado de comando en caché.
        
        Args:
            command: Comando ejecutado
            result: Resultado a cachear
            ttl: Tiempo de vida en segundos (opcional)
        """
        cache_key = self._get_cache_key(command)
        await self.cache.set(cache_key, result, ttl)
    
    def _get_cache_key(self, command: str) -> str:
        """
        Generar clave de caché para comando.
        
        Args:
            command: Comando a hashear
            
        Returns:
            Clave única para el comando
        """
        import hashlib
        return f"cmd_{hashlib.md5(command.encode()).hexdigest()}"
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas del caché de comandos.
        
        Returns:
            Diccionario con estadísticas
        """
        cache_stats = await self.cache.get_stats()
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0.0
        
        return {
            **cache_stats,
            "command_cache_hits": self._hits,
            "command_cache_misses": self._misses,
            "command_cache_hit_rate": round(hit_rate, 4),
            "total_command_requests": total_requests
        }
    
    async def clear(self) -> None:
        """Limpiar todo el caché de comandos"""
        await self.cache.clear()
        self._hits = 0
        self._misses = 0


