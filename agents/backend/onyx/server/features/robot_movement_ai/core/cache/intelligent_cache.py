"""
Intelligent Cache
=================

Cache inteligente con TTL y múltiples estrategias de eviction.
"""

import time
import hashlib
import threading
import functools
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import OrderedDict


class EvictionPolicy(Enum):
    """Políticas de eviction."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    TTL = "ttl"  # Time To Live


@dataclass
class CacheEntry:
    """Entrada de cache."""
    key: str
    value: Any
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    ttl: Optional[timedelta] = None
    
    def is_expired(self) -> bool:
        """Verificar si está expirado."""
        if self.ttl is None:
            return False
        return datetime.now() - self.created_at > self.ttl
    
    def touch(self):
        """Actualizar último acceso."""
        self.last_accessed = datetime.now()
        self.access_count += 1


@dataclass
class CacheStats:
    """Estadísticas de cache."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0
    max_size: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Tasa de hits."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class IntelligentCache:
    """
    Cache inteligente con TTL y múltiples estrategias de eviction.
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: Optional[float] = None,
        eviction_policy: EvictionPolicy = EvictionPolicy.LRU
    ):
        """
        Inicializar cache.
        
        Args:
            max_size: Tamaño máximo
            ttl_seconds: TTL en segundos (opcional)
            eviction_policy: Política de eviction
        """
        self.max_size = max_size
        self.ttl = timedelta(seconds=ttl_seconds) if ttl_seconds else None
        self.eviction_policy = eviction_policy
        
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: OrderedDict = OrderedDict()  # Para LRU
        self._access_counts: Dict[str, int] = {}  # Para LFU
        self._lock = threading.Lock()
        self._stats = CacheStats(max_size=max_size)
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generar clave de cache."""
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Obtener valor del cache.
        
        Args:
            key: Clave
            
        Returns:
            Valor o None
        """
        with self._lock:
            entry = self._cache.get(key)
            
            if entry is None:
                self._stats.misses += 1
                return None
            
            # Verificar expiración
            if entry.is_expired():
                del self._cache[key]
                self._stats.misses += 1
                return None
            
            # Actualizar acceso
            entry.touch()
            self._stats.hits += 1
            
            # Actualizar orden de acceso (LRU)
            if self.eviction_policy == EvictionPolicy.LRU:
                self._access_order.move_to_end(key)
            
            # Actualizar conteo de acceso (LFU)
            if self.eviction_policy == EvictionPolicy.LFU:
                self._access_counts[key] = self._access_counts.get(key, 0) + 1
            
            return entry.value
    
    def put(
        self,
        key: str,
        value: Any,
        ttl: Optional[timedelta] = None
    ):
        """
        Agregar valor al cache.
        
        Args:
            key: Clave
            value: Valor
            ttl: TTL específico (opcional)
        """
        with self._lock:
            # Verificar si necesitamos evictar
            if len(self._cache) >= self.max_size and key not in self._cache:
                self._evict()
            
            # Crear entrada
            entry = CacheEntry(
                key=key,
                value=value,
                ttl=ttl or self.ttl
            )
            
            self._cache[key] = entry
            self._stats.size = len(self._cache)
            
            # Actualizar orden (LRU)
            if self.eviction_policy == EvictionPolicy.LRU:
                self._access_order[key] = True
            
            # Inicializar conteo (LFU)
            if self.eviction_policy == EvictionPolicy.LFU:
                self._access_counts[key] = 0
    
    def _evict(self):
        """Evictar entrada según política."""
        if not self._cache:
            return
        
        if self.eviction_policy == EvictionPolicy.LRU:
            # Evictar menos recientemente usado
            key = next(iter(self._access_order))
            del self._access_order[key]
        elif self.eviction_policy == EvictionPolicy.LFU:
            # Evictar menos frecuentemente usado
            key = min(self._access_counts.items(), key=lambda x: x[1])[0]
            del self._access_counts[key]
        elif self.eviction_policy == EvictionPolicy.FIFO:
            # Evictar más antiguo
            key = min(self._cache.items(), key=lambda x: x[1].created_at)[0]
        else:
            # TTL: evictar expirado o más antiguo
            expired = [k for k, v in self._cache.items() if v.is_expired()]
            if expired:
                key = expired[0]
            else:
                key = min(self._cache.items(), key=lambda x: x[1].created_at)[0]
        
        del self._cache[key]
        self._stats.evictions += 1
        self._stats.size = len(self._cache)
    
    def clear(self):
        """Limpiar cache."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._access_counts.clear()
            self._stats = CacheStats(max_size=self.max_size)
    
    def get_stats(self) -> CacheStats:
        """Obtener estadísticas."""
        with self._lock:
            # Limpiar expirados
            expired = [k for k, v in self._cache.items() if v.is_expired()]
            for key in expired:
                del self._cache[key]
            
            self._stats.size = len(self._cache)
            return self._stats
    
    def cached(
        self,
        ttl: Optional[float] = None,
        key_func: Optional[Callable] = None
    ):
        """
        Decorator para cachear resultados de función.
        
        Args:
            ttl: TTL en segundos (opcional)
            key_func: Función para generar clave (opcional)
            
        Returns:
            Decorator
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Generar clave
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    cache_key = self._generate_key(*args, **kwargs)
                
                # Intentar obtener del cache
                cached_value = self.get(cache_key)
                if cached_value is not None:
                    return cached_value
                
                # Ejecutar función
                result = func(*args, **kwargs)
                
                # Guardar en cache
                ttl_delta = timedelta(seconds=ttl) if ttl else None
                self.put(cache_key, result, ttl=ttl_delta)
                
                return result
            
            return wrapper
        return decorator


# Aliases para políticas
TTL = EvictionPolicy.TTL
LRU = EvictionPolicy.LRU
LFU = EvictionPolicy.LFU
FIFO = EvictionPolicy.FIFO

