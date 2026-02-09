"""
MCP Advanced Cache - Estrategias avanzadas de cache
=====================================================
"""

import hashlib
import json
import logging
from typing import Any, Dict, Optional, Callable, List
from datetime import datetime, timedelta
from enum import Enum

from .cache import MCPCache

logger = logging.getLogger(__name__)


class CacheStrategy(str, Enum):
    """Estrategias de cache"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    TTL = "ttl"  # Time To Live
    ADAPTIVE = "adaptive"  # Adaptativa


class CacheEntry:
    """Entrada de cache con metadata"""
    
    def __init__(self, key: str, value: Any, ttl: Optional[int] = None):
        self.key = key
        self.value = value
        self.created_at = datetime.utcnow()
        self.last_accessed = datetime.utcnow()
        self.access_count = 0
        self.ttl = ttl
        self.expires_at = (
            datetime.utcnow() + timedelta(seconds=ttl)
            if ttl else None
        )
    
    def is_expired(self) -> bool:
        """Verifica si la entrada expiró"""
        if self.expires_at:
            return datetime.utcnow() > self.expires_at
        return False
    
    def access(self):
        """Registra acceso"""
        self.last_accessed = datetime.utcnow()
        self.access_count += 1


class AdvancedCache(MCPCache):
    """
    Cache avanzado con múltiples estrategias
    
    Extiende MCPCache con estrategias LRU, LFU, FIFO, etc.
    """
    
    def __init__(
        self,
        default_ttl: int = 300,
        max_size: int = 1000,
        strategy: CacheStrategy = CacheStrategy.LRU,
    ):
        """
        Args:
            default_ttl: TTL por defecto en segundos
            max_size: Tamaño máximo del cache
            strategy: Estrategia de cache
        """
        super().__init__(default_ttl)
        self.max_size = max_size
        self.strategy = strategy
        self._entries: Dict[str, CacheEntry] = {}
        self._access_order: List[str] = []  # Para LRU/FIFO
    
    def get(
        self,
        resource_id: str,
        operation: str,
        parameters: Dict[str, Any],
    ) -> Optional[Any]:
        """Obtiene valor del cache con estrategia avanzada"""
        key = self._make_key(resource_id, operation, parameters)
        entry = self._entries.get(key)
        
        if not entry:
            return None
        
        # Verificar expiración
        if entry.is_expired():
            del self._entries[key]
            if key in self._access_order:
                self._access_order.remove(key)
            return None
        
        # Registrar acceso
        entry.access()
        
        # Actualizar orden de acceso para LRU
        if self.strategy == CacheStrategy.LRU:
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
        
        return entry.value
    
    def set(
        self,
        resource_id: str,
        operation: str,
        parameters: Dict[str, Any],
        value: Any,
        ttl: Optional[int] = None,
    ):
        """Guarda valor en cache con estrategia avanzada"""
        key = self._make_key(resource_id, operation, parameters)
        ttl = ttl or self.default_ttl
        
        # Verificar tamaño máximo
        if len(self._entries) >= self.max_size:
            self._evict_entry()
        
        # Crear entrada
        entry = CacheEntry(key, value, ttl)
        self._entries[key] = entry
        
        # Actualizar orden
        if self.strategy in [CacheStrategy.LRU, CacheStrategy.FIFO]:
            if key not in self._access_order:
                self._access_order.append(key)
    
    def _evict_entry(self):
        """Elimina entrada según estrategia"""
        if not self._entries:
            return
        
        if self.strategy == CacheStrategy.LRU:
            # Eliminar menos recientemente usado
            if self._access_order:
                key = self._access_order.pop(0)
                del self._entries[key]
        
        elif self.strategy == CacheStrategy.LFU:
            # Eliminar menos frecuentemente usado
            least_used = min(
                self._entries.items(),
                key=lambda x: x[1].access_count
            )
            key = least_used[0]
            del self._entries[key]
            if key in self._access_order:
                self._access_order.remove(key)
        
        elif self.strategy == CacheStrategy.FIFO:
            # Eliminar más antiguo
            if self._access_order:
                key = self._access_order.pop(0)
                del self._entries[key]
        
        elif self.strategy == CacheStrategy.TTL:
            # Eliminar expirado más antiguo
            expired = [
                (k, e) for k, e in self._entries.items()
                if e.is_expired()
            ]
            if expired:
                key = min(expired, key=lambda x: x[1].expires_at)[0]
                del self._entries[key]
                if key in self._access_order:
                    self._access_order.remove(key)
            else:
                # Si no hay expirados, usar LRU
                if self._access_order:
                    key = self._access_order.pop(0)
                    del self._entries[key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del cache avanzado"""
        base_stats = super().get_stats()
        
        hit_rate = 0
        if self._entries:
            total_access = sum(e.access_count for e in self._entries.values())
            if total_access > 0:
                hits = sum(1 for e in self._entries.values() if e.access_count > 0)
                hit_rate = hits / len(self._entries) if self._entries else 0
        
        return {
            **base_stats,
            "strategy": self.strategy.value,
            "max_size": self.max_size,
            "current_size": len(self._entries),
            "hit_rate": hit_rate,
            "entries_by_access": {
                "most_accessed": sorted(
                    [(k, e.access_count) for k, e in self._entries.items()],
                    key=lambda x: x[1],
                    reverse=True
                )[:10],
            },
        }

