from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import hashlib
import json
import pickle
from typing import Any, Dict, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict
import logging
from typing import Any, List, Dict, Optional
"""
💾 Production Cache System
==========================

Sistema de cache de producción con métricas, TTL y limpieza automática.
"""



@dataclass
class CacheMetrics:
    """Métricas del cache."""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0
    
    def hit_rate(self) -> float:
        """Calcular tasa de aciertos."""
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0
    
    def total_operations(self) -> int:
        """Total de operaciones."""
        return self.hits + self.misses + self.sets + self.deletes


@dataclass
class CacheEntry:
    """Entrada del cache."""
    value: Any
    created_at: datetime
    expires_at: datetime
    access_count: int = 0
    last_accessed: datetime = None
    
    def is_expired(self) -> bool:
        """Verificar si ha expirado."""
        return datetime.now() > self.expires_at
    
    def access(self) -> None:
        """Registrar acceso."""
        self.access_count += 1
        self.last_accessed = datetime.now()


class ProductionCache:
    """
    Sistema de cache de producción con características empresariales.
    
    Características:
    - TTL configurable por entrada
    - Métricas detalladas
    - Limpieza automática
    - Serialización eficiente
    - Límites de memoria
    - Políticas de eviction
    """
    
    def __init__(
        self,
        default_ttl: int = 3600,
        max_size: int = 10000,
        cleanup_interval: int = 300,
        eviction_policy: str = "lru"
    ):
        
    """__init__ function."""
self.default_ttl = default_ttl
        self.max_size = max_size
        self.cleanup_interval = cleanup_interval
        self.eviction_policy = eviction_policy
        
        self.cache: Dict[str, CacheEntry] = {}
        self.metrics = CacheMetrics()
        self.logger = logging.getLogger("production_cache")
        
        # Iniciar limpieza automática
        self._cleanup_task = None
        self._start_cleanup_task()
    
    def _start_cleanup_task(self) -> None:
        """Iniciar tarea de limpieza automática."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def _cleanup_loop(self) -> None:
        """Loop de limpieza automática."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self.cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}")
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Obtener valor del cache.
        
        Args:
            key: Clave del cache
            
        Returns:
            Valor o None si no existe/expiró
        """
        if key not in self.cache:
            self.metrics.misses += 1
            return None
        
        entry = self.cache[key]
        
        # Verificar expiración
        if entry.is_expired():
            del self.cache[key]
            self.metrics.misses += 1
            self.metrics.deletes += 1
            return None
        
        # Registrar acceso
        entry.access()
        self.metrics.hits += 1
        
        return entry.value
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None
    ) -> bool:
        """
        Establecer valor en cache.
        
        Args:
            key: Clave del cache
            value: Valor a cachear
            ttl: Time to live en segundos
            
        Returns:
            True si se estableció correctamente
        """
        try:
            # Verificar límite de tamaño
            if len(self.cache) >= self.max_size and key not in self.cache:
                await self._evict_entries()
            
            # Calcular expiración
            ttl = ttl or self.default_ttl
            expires_at = datetime.now() + timedelta(seconds=ttl)
            
            # Crear entrada
            entry = CacheEntry(
                value=value,
                created_at=datetime.now(),
                expires_at=expires_at
            )
            
            self.cache[key] = entry
            self.metrics.sets += 1
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting cache key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """
        Eliminar entrada del cache.
        
        Args:
            key: Clave a eliminar
            
        Returns:
            True si se eliminó
        """
        if key in self.cache:
            del self.cache[key]
            self.metrics.deletes += 1
            return True
        return False
    
    async def exists(self, key: str) -> bool:
        """Verificar si existe una clave."""
        if key not in self.cache:
            return False
        
        entry = self.cache[key]
        if entry.is_expired():
            del self.cache[key]
            self.metrics.deletes += 1
            return False
        
        return True
    
    async def clear(self) -> None:
        """Limpiar todo el cache."""
        cleared_count = len(self.cache)
        self.cache.clear()
        self.metrics.deletes += cleared_count
        self.logger.info(f"Cache cleared: {cleared_count} entries removed")
    
    async def cleanup_expired(self) -> int:
        """
        Limpiar entradas expiradas.
        
        Returns:
            Número de entradas eliminadas
        """
        now = datetime.now()
        expired_keys = [
            key for key, entry in self.cache.items()
            if entry.expires_at < now
        ]
        
        for key in expired_keys:
            del self.cache[key]
        
        removed_count = len(expired_keys)
        self.metrics.deletes += removed_count
        
        if removed_count > 0:
            self.logger.info(f"Cleanup: {removed_count} expired entries removed")
        
        return removed_count
    
    async def _evict_entries(self) -> None:
        """Evict entradas según política."""
        if self.eviction_policy == "lru":
            await self._evict_lru()
        elif self.eviction_policy == "lfu":
            await self._evict_lfu()
        else:
            await self._evict_oldest()
    
    async def _evict_lru(self) -> None:
        """Evict Least Recently Used."""
        if not self.cache:
            return
        
        # Encontrar entrada menos recientemente usada
        lru_key = min(
            self.cache.keys(),
            key=lambda k: self.cache[k].last_accessed or self.cache[k].created_at
        )
        
        del self.cache[lru_key]
        self.metrics.evictions += 1
        self.logger.debug(f"LRU eviction: {lru_key}")
    
    async def _evict_lfu(self) -> None:
        """Evict Least Frequently Used."""
        if not self.cache:
            return
        
        # Encontrar entrada menos frecuentemente usada
        lfu_key = min(
            self.cache.keys(),
            key=lambda k: self.cache[k].access_count
        )
        
        del self.cache[lfu_key]
        self.metrics.evictions += 1
        self.logger.debug(f"LFU eviction: {lfu_key}")
    
    async def _evict_oldest(self) -> None:
        """Evict entrada más antigua."""
        if not self.cache:
            return
        
        oldest_key = min(
            self.cache.keys(),
            key=lambda k: self.cache[k].created_at
        )
        
        del self.cache[oldest_key]
        self.metrics.evictions += 1
        self.logger.debug(f"Oldest eviction: {oldest_key}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del cache."""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hit_rate": self.metrics.hit_rate(),
            "metrics": {
                "hits": self.metrics.hits,
                "misses": self.metrics.misses,
                "sets": self.metrics.sets,
                "deletes": self.metrics.deletes,
                "evictions": self.metrics.evictions,
                "total_operations": self.metrics.total_operations()
            },
            "config": {
                "default_ttl": self.default_ttl,
                "cleanup_interval": self.cleanup_interval,
                "eviction_policy": self.eviction_policy
            }
        }
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Información detallada del cache."""
        now = datetime.now()
        
        entries_info = []
        for key, entry in list(self.cache.items())[:10]:  # Solo primeras 10
            entries_info.append({
                "key": key,
                "created_at": entry.created_at.isoformat(),
                "expires_at": entry.expires_at.isoformat(),
                "access_count": entry.access_count,
                "last_accessed": entry.last_accessed.isoformat() if entry.last_accessed else None,
                "is_expired": entry.is_expired(),
                "ttl_remaining": (entry.expires_at - now).total_seconds()
            })
        
        return {
            "total_entries": len(self.cache),
            "sample_entries": entries_info,
            "stats": self.get_stats()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check del cache."""
        await self.cleanup_expired()
        
        stats = self.get_stats()
        
        # Determinar estado de salud
        health_status = "healthy"
        if stats["hit_rate"] < 50:
            health_status = "degraded"
        if len(self.cache) >= self.max_size * 0.9:
            health_status = "degraded"
        
        return {
            "status": health_status,
            "stats": stats,
            "issues": self._check_issues()
        }
    
    def _check_issues(self) -> List[str]:
        """Verificar problemas potenciales."""
        issues = []
        
        if self.metrics.hit_rate() < 50:
            issues.append("Low cache hit rate")
        
        if len(self.cache) >= self.max_size * 0.9:
            issues.append("Cache near capacity")
        
        if self.metrics.evictions > self.metrics.sets * 0.1:
            issues.append("High eviction rate")
        
        return issues
    
    async def close(self) -> None:
        """Cerrar cache y limpiar recursos."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        await self.clear()
        self.logger.info("Cache closed")


def generate_cache_key(text: str, analyzers: List[str] = None, **kwargs) -> str:
    """
    Generar clave de cache consistente.
    
    Args:
        text: Texto a analizar
        analyzers: Lista de analizadores
        **kwargs: Parámetros adicionales
        
    Returns:
        Clave de cache MD5
    """
    analyzers = sorted(analyzers or [])
    
    # Crear contenido para hash
    content_parts = [text]
    if analyzers:
        content_parts.append(":".join(analyzers))
    
    # Agregar kwargs ordenados
    if kwargs:
        sorted_kwargs = sorted(kwargs.items())
        content_parts.append(json.dumps(sorted_kwargs, sort_keys=True))
    
    content = "|".join(content_parts)
    
    # Generar hash MD5
    return hashlib.md5(content.encode('utf-8')).hexdigest()


# Cache global para uso en producción
_global_cache: Optional[ProductionCache] = None


async def get_global_cache() -> ProductionCache:
    """Obtener instancia global del cache."""
    global _global_cache
    if _global_cache is None:
        _global_cache = ProductionCache()
    return _global_cache


async def close_global_cache() -> None:
    """Cerrar cache global."""
    global _global_cache
    if _global_cache:
        await _global_cache.close()
        _global_cache = None 