"""
Distributed Cache Service - Cache distribuido
==============================================

Sistema de cache distribuido con múltiples estrategias y TTL.
"""

import logging
import hashlib
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Entrada de cache"""
    key: str
    value: Any
    ttl: int  # segundos
    created_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)


@dataclass
class CacheStats:
    """Estadísticas de cache"""
    total_keys: int
    hit_count: int
    miss_count: int
    hit_rate: float
    memory_usage: int  # bytes estimado
    evicted_keys: int


class DistributedCacheService:
    """Servicio de cache distribuido"""
    
    def __init__(self, max_size: int = 10000):
        """Inicializar servicio"""
        self.cache: Dict[str, CacheEntry] = {}
        self.max_size = max_size
        self.hit_count = 0
        self.miss_count = 0
        self.evicted_keys = 0
        logger.info("DistributedCacheService initialized")
    
    def get(self, key: str) -> Optional[Any]:
        """Obtener valor del cache"""
        entry = self.cache.get(key)
        
        if not entry:
            self.miss_count += 1
            return None
        
        # Verificar TTL
        if datetime.now() - entry.created_at > timedelta(seconds=entry.ttl):
            del self.cache[key]
            self.miss_count += 1
            return None
        
        # Actualizar estadísticas de acceso
        entry.access_count += 1
        entry.last_accessed = datetime.now()
        self.hit_count += 1
        
        return entry.value
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: int = 3600  # 1 hora por defecto
    ) -> bool:
        """Establecer valor en cache"""
        # Verificar tamaño máximo
        if len(self.cache) >= self.max_size and key not in self.cache:
            self._evict_lru()
        
        entry = CacheEntry(
            key=key,
            value=value,
            ttl=ttl,
        )
        
        self.cache[key] = entry
        
        return True
    
    def delete(self, key: str) -> bool:
        """Eliminar del cache"""
        if key in self.cache:
            del self.cache[key]
            return True
        return False
    
    def clear(self) -> int:
        """Limpiar todo el cache"""
        count = len(self.cache)
        self.cache.clear()
        return count
    
    def _evict_lru(self):
        """Eliminar entrada menos recientemente usada (LRU)"""
        if not self.cache:
            return
        
        # Encontrar entrada con menor last_accessed
        lru_key = min(
            self.cache.keys(),
            key=lambda k: self.cache[k].last_accessed
        )
        
        del self.cache[lru_key]
        self.evicted_keys += 1
    
    def get_stats(self) -> CacheStats:
        """Obtener estadísticas de cache"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total_requests * 100) if total_requests > 0 else 0.0
        
        # Estimar uso de memoria (simplificado)
        memory_usage = sum(
            len(str(entry.value)) + len(entry.key)
            for entry in self.cache.values()
        )
        
        return CacheStats(
            total_keys=len(self.cache),
            hit_count=self.hit_count,
            miss_count=self.miss_count,
            hit_rate=round(hit_rate, 2),
            memory_usage=memory_usage,
            evicted_keys=self.evicted_keys,
        )
    
    def generate_cache_key(
        self,
        prefix: str,
        params: Dict[str, Any]
    ) -> str:
        """Generar clave de cache"""
        # Ordenar parámetros para consistencia
        sorted_params = json.dumps(params, sort_keys=True)
        param_hash = hashlib.md5(sorted_params.encode()).hexdigest()[:8]
        
        return f"{prefix}:{param_hash}"
    
    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidar claves que coincidan con patrón"""
        count = 0
        keys_to_delete = [
            key for key in self.cache.keys()
            if pattern in key
        ]
        
        for key in keys_to_delete:
            del self.cache[key]
            count += 1
        
        return count




