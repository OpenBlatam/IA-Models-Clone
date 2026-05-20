"""
Advanced Cache Utilities - Utilidades avanzadas de caché
==========================================================

Sistema de caché avanzado con múltiples estrategias, invalidación inteligente,
y estadísticas detalladas.
"""

import logging
import time
import hashlib
import json
from typing import Any, Dict, Optional, Callable, List, Tuple
from datetime import datetime, timedelta
from collections import OrderedDict
from threading import Lock
from enum import Enum

logger = logging.getLogger(__name__)


class CacheStrategy(str, Enum):
    """Estrategias de caché."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    TTL = "ttl"  # Time To Live


class CacheEntry:
    """Entrada de caché."""
    
    def __init__(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
        created_at: Optional[float] = None
    ):
        """
        Inicializar entrada de caché.
        
        Args:
            key: Clave de la entrada
            value: Valor a cachear
            ttl: Time to live en segundos (None = sin expiración)
            created_at: Timestamp de creación (None = ahora)
        """
        self.key = key
        self.value = value
        self.ttl = ttl
        self.created_at = created_at or time.time()
        self.access_count = 0
        self.last_accessed = self.created_at
    
    def is_expired(self) -> bool:
        """Verificar si la entrada está expirada."""
        if self.ttl is None:
            return False
        return (time.time() - self.created_at) > self.ttl
    
    def access(self) -> None:
        """Registrar acceso a la entrada."""
        self.access_count += 1
        self.last_accessed = time.time()
    
    def age(self) -> float:
        """Obtener edad de la entrada en segundos."""
        return time.time() - self.created_at


class AdvancedCache:
    """
    Caché avanzado con múltiples estrategias.
    
    Soporta diferentes estrategias de evicción y invalidación inteligente.
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: Optional[float] = None,
        strategy: CacheStrategy = CacheStrategy.LRU
    ):
        """
        Inicializar caché avanzado.
        
        Args:
            max_size: Tamaño máximo del caché
            default_ttl: TTL por defecto en segundos (None = sin expiración)
            strategy: Estrategia de evicción
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.strategy = strategy
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: OrderedDict = OrderedDict()
        self._lock = Lock()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "evictions": 0,
            "invalidations": 0,
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Obtener valor del caché.
        
        Args:
            key: Clave
            default: Valor por defecto si no existe
        
        Returns:
            Valor cacheado o default
        """
        with self._lock:
            entry = self._cache.get(key)
            
            if entry is None:
                self._stats["misses"] += 1
                return default
            
            if entry.is_expired():
                del self._cache[key]
                if key in self._access_order:
                    del self._access_order[key]
                self._stats["misses"] += 1
                return default
            
            entry.access()
            self._stats["hits"] += 1
            
            # Actualizar orden de acceso
            if self.strategy == CacheStrategy.LRU:
                if key in self._access_order:
                    del self._access_order[key]
                self._access_order[key] = time.time()
            
            return entry.value
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None
    ) -> None:
        """
        Establecer valor en caché.
        
        Args:
            key: Clave
            value: Valor
            ttl: TTL en segundos (None = usar default_ttl)
        """
        with self._lock:
            # Limpiar expirados
            self._cleanup_expired()
            
            # Verificar si necesitamos evicción
            if len(self._cache) >= self.max_size and key not in self._cache:
                self._evict()
            
            # Crear entrada
            entry = CacheEntry(key, value, ttl or self.default_ttl)
            self._cache[key] = entry
            
            # Actualizar orden
            if self.strategy == CacheStrategy.LRU:
                if key in self._access_order:
                    del self._access_order[key]
                self._access_order[key] = time.time()
            
            self._stats["sets"] += 1
    
    def delete(self, key: str) -> bool:
        """
        Eliminar entrada del caché.
        
        Args:
            key: Clave
        
        Returns:
            True si se eliminó, False si no existía
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                if key in self._access_order:
                    del self._access_order[key]
                self._stats["invalidations"] += 1
                return True
            return False
    
    def clear(self) -> None:
        """Limpiar todo el caché."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._stats["invalidations"] += len(self._cache)
    
    def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidar entradas que coincidan con patrón.
        
        Args:
            pattern: Patrón de clave (substring)
        
        Returns:
            Número de entradas invalidadas
        """
        import re
        compiled = re.compile(pattern)
        
        with self._lock:
            to_delete = [key for key in self._cache.keys() if compiled.search(key)]
            for key in to_delete:
                del self._cache[key]
                if key in self._access_order:
                    del self._access_order[key]
            
            self._stats["invalidations"] += len(to_delete)
            return len(to_delete)
    
    def _cleanup_expired(self) -> None:
        """Limpiar entradas expiradas."""
        expired = [
            key for key, entry in self._cache.items()
            if entry.is_expired()
        ]
        for key in expired:
            del self._cache[key]
            if key in self._access_order:
                del self._access_order[key]
    
    def _evict(self) -> None:
        """Evictar entrada según estrategia."""
        if not self._cache:
            return
        
        if self.strategy == CacheStrategy.LRU:
            # Eliminar menos recientemente usado
            if self._access_order:
                key = next(iter(self._access_order))
                del self._cache[key]
                del self._access_order[key]
        elif self.strategy == CacheStrategy.LFU:
            # Eliminar menos frecuentemente usado
            key = min(self._cache.items(), key=lambda x: x[1].access_count)[0]
            del self._cache[key]
            if key in self._access_order:
                del self._access_order[key]
        elif self.strategy == CacheStrategy.FIFO:
            # Eliminar más antiguo
            key = min(self._cache.items(), key=lambda x: x[1].created_at)[0]
            del self._cache[key]
            if key in self._access_order:
                del self._access_order[key]
        
        self._stats["evictions"] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas del caché.
        
        Returns:
            Diccionario con estadísticas
        """
        with self._lock:
            total_requests = self._stats["hits"] + self._stats["misses"]
            hit_rate = (
                self._stats["hits"] / total_requests
                if total_requests > 0 else 0.0
            )
            
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "strategy": self.strategy.value,
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "hit_rate": hit_rate,
                "sets": self._stats["sets"],
                "evictions": self._stats["evictions"],
                "invalidations": self._stats["invalidations"],
            }
    
    def reset_stats(self) -> None:
        """Resetear estadísticas."""
        with self._lock:
            self._stats = {
                "hits": 0,
                "misses": 0,
                "sets": 0,
                "evictions": 0,
                "invalidations": 0,
            }


def make_cache_key(*args, **kwargs) -> str:
    """
    Crear clave de caché a partir de argumentos.
    
    Args:
        *args: Argumentos posicionales
        **kwargs: Argumentos nombrados
    
    Returns:
        Clave de caché (hash)
    
    Example:
        key = make_cache_key("user", user_id=123, operation="read")
    """
    key_data = {
        "args": args,
        "kwargs": sorted(kwargs.items())
    }
    key_str = json.dumps(key_data, sort_keys=True, default=str)
    return hashlib.sha256(key_str.encode()).hexdigest()


__all__ = [
    "CacheStrategy",
    "CacheEntry",
    "AdvancedCache",
    "make_cache_key",
]

