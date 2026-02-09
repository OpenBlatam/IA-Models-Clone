"""
Distributed Cache System
========================

Sistema de cache distribuido.
"""

import logging
import json
import hashlib
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Estrategia de cache."""
    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"
    TTL = "ttl"


@dataclass
class CacheEntry:
    """Entrada de cache."""
    key: str
    value: Any
    created_at: float
    expires_at: Optional[float] = None
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class DistributedCache:
    """
    Cache distribuido.
    
    Gestiona cache con múltiples estrategias.
    """
    
    def __init__(
        self,
        name: str,
        max_size: int = 1000,
        default_ttl: float = 3600.0,
        strategy: CacheStrategy = CacheStrategy.LRU
    ):
        """
        Inicializar cache distribuido.
        
        Args:
            name: Nombre del cache
            max_size: Tamaño máximo
            default_ttl: TTL por defecto en segundos
            strategy: Estrategia de cache
        """
        self.name = name
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.strategy = strategy
        
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order: List[str] = []  # Para LRU/FIFO
        self.access_counts: Dict[str, int] = defaultdict(int)  # Para LFU
    
    def get(self, key: str) -> Optional[Any]:
        """
        Obtener valor del cache.
        
        Args:
            key: Clave
            
        Returns:
            Valor o None si no existe o expiró
        """
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        now = time.time()
        
        # Verificar expiración
        if entry.expires_at and now > entry.expires_at:
            del self.cache[key]
            if key in self.access_order:
                self.access_order.remove(key)
            if key in self.access_counts:
                del self.access_counts[key]
            return None
        
        # Actualizar acceso
        entry.last_accessed = now
        entry.access_count += 1
        self.access_counts[key] = entry.access_count
        
        # Actualizar orden para LRU
        if self.strategy == CacheStrategy.LRU:
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
        
        return entry.value
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None
    ) -> None:
        """
        Establecer valor en cache.
        
        Args:
            key: Clave
            value: Valor
            ttl: TTL en segundos (opcional)
        """
        now = time.time()
        expires_at = now + (ttl or self.default_ttl) if ttl or self.default_ttl else None
        
        # Verificar tamaño
        if len(self.cache) >= self.max_size and key not in self.cache:
            self._evict_entry()
        
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=now,
            expires_at=expires_at,
            last_accessed=now
        )
        
        self.cache[key] = entry
        
        # Agregar a orden
        if self.strategy in [CacheStrategy.LRU, CacheStrategy.FIFO]:
            if key not in self.access_order:
                self.access_order.append(key)
        
        self.access_counts[key] = 0
    
    def _evict_entry(self) -> None:
        """Eliminar entrada según estrategia."""
        if not self.cache:
            return
        
        if self.strategy == CacheStrategy.LRU:
            # Eliminar menos recientemente usado
            if self.access_order:
                key_to_remove = self.access_order.pop(0)
                if key_to_remove in self.cache:
                    del self.cache[key_to_remove]
        elif self.strategy == CacheStrategy.LFU:
            # Eliminar menos frecuentemente usado
            if self.access_counts:
                key_to_remove = min(self.access_counts.items(), key=lambda x: x[1])[0]
                del self.cache[key_to_remove]
                del self.access_counts[key_to_remove]
                if key_to_remove in self.access_order:
                    self.access_order.remove(key_to_remove)
        elif self.strategy == CacheStrategy.FIFO:
            # Eliminar primero en entrar
            if self.access_order:
                key_to_remove = self.access_order.pop(0)
                if key_to_remove in self.cache:
                    del self.cache[key_to_remove]
        elif self.strategy == CacheStrategy.TTL:
            # Eliminar expirado más antiguo
            expired_keys = [
                k for k, v in self.cache.items()
                if v.expires_at and time.time() > v.expires_at
            ]
            if expired_keys:
                key_to_remove = expired_keys[0]
                del self.cache[key_to_remove]
            else:
                # Si no hay expirados, eliminar más antiguo
                oldest_key = min(self.cache.items(), key=lambda x: x[1].created_at)[0]
                del self.cache[oldest_key]
    
    def delete(self, key: str) -> bool:
        """Eliminar entrada del cache."""
        if key in self.cache:
            del self.cache[key]
            if key in self.access_order:
                self.access_order.remove(key)
            if key in self.access_counts:
                del self.access_counts[key]
            return True
        return False
    
    def clear(self) -> None:
        """Limpiar cache."""
        self.cache.clear()
        self.access_order.clear()
        self.access_counts.clear()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas del cache."""
        now = time.time()
        expired = sum(
            1 for entry in self.cache.values()
            if entry.expires_at and now > entry.expires_at
        )
        
        return {
            "name": self.name,
            "size": len(self.cache),
            "max_size": self.max_size,
            "strategy": self.strategy.value,
            "expired_entries": expired,
            "hit_rate": 0.0  # Se calcularía con contadores de hits/misses
        }


# Instancia global de caches
_distributed_caches: Dict[str, DistributedCache] = {}


def create_distributed_cache(
    name: str,
    max_size: int = 1000,
    default_ttl: float = 3600.0,
    strategy: CacheStrategy = CacheStrategy.LRU
) -> DistributedCache:
    """
    Crear cache distribuido.
    
    Args:
        name: Nombre del cache
        max_size: Tamaño máximo
        default_ttl: TTL por defecto
        strategy: Estrategia
        
    Returns:
        Cache distribuido
    """
    cache = DistributedCache(name, max_size, default_ttl, strategy)
    _distributed_caches[name] = cache
    return cache


def get_distributed_cache(name: str) -> Optional[DistributedCache]:
    """Obtener cache distribuido por nombre."""
    return _distributed_caches.get(name)


# Importar defaultdict
from collections import defaultdict






