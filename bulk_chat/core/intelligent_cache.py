"""
Intelligent Cache - Caché Inteligente
=====================================

Sistema de caché avanzado con invalidación inteligente, prefetching y análisis de patrones de acceso.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, OrderedDict
import hashlib
import json

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Estrategia de caché."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    TTL = "ttl"  # Time To Live
    ADAPTIVE = "adaptive"  # Adaptativo basado en patrones


@dataclass
class CacheEntry:
    """Entrada de caché."""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    hit_count: int = 0
    miss_count: int = 0
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheStats:
    """Estadísticas de caché."""
    total_entries: int = 0
    total_hits: int = 0
    total_misses: int = 0
    hit_rate: float = 0.0
    memory_usage: int = 0
    evictions: int = 0


class IntelligentCache:
    """Caché inteligente."""
    
    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: Optional[int] = None,
        strategy: CacheStrategy = CacheStrategy.ADAPTIVE,
    ):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.strategy = strategy
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.access_patterns: Dict[str, List[datetime]] = defaultdict(list)
        self.prefetch_predictions: Dict[str, float] = {}
        self._lock = asyncio.Lock()
        self.stats = CacheStats()
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generar clave de caché."""
        key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True)
        return hashlib.sha256(key_data.encode()).hexdigest()
    
    async def get(
        self,
        key: str,
        default: Any = None,
        update_access: bool = True,
    ) -> Optional[Any]:
        """Obtener valor del caché."""
        async with self._lock:
            entry = self.cache.get(key)
            
            if entry is None:
                self.stats.total_misses += 1
                return default
            
            # Verificar expiración
            if entry.expires_at and datetime.now() > entry.expires_at:
                del self.cache[key]
                self.stats.total_misses += 1
                return default
            
            # Actualizar acceso
            if update_access:
                entry.last_accessed = datetime.now()
                entry.access_count += 1
                entry.hit_count += 1
                self.stats.total_hits += 1
                
                # Mover al final (LRU)
                if self.strategy == CacheStrategy.LRU:
                    self.cache.move_to_end(key)
                
                # Registrar patrón de acceso
                self.access_patterns[key].append(datetime.now())
                
                # Limitar historial de patrones
                if len(self.access_patterns[key]) > 100:
                    self.access_patterns[key].pop(0)
            
            self._update_hit_rate()
            return entry.value
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Guardar valor en caché."""
        expires_at = None
        if ttl or self.default_ttl:
            ttl_seconds = ttl or self.default_ttl
            expires_at = datetime.now() + timedelta(seconds=ttl_seconds)
        
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            expires_at=expires_at,
            metadata=metadata or {},
        )
        
        async with self._lock:
            # Si existe, actualizar
            if key in self.cache:
                old_entry = self.cache[key]
                entry.access_count = old_entry.access_count
                entry.hit_count = old_entry.hit_count
            
            # Si está lleno, evictar
            if len(self.cache) >= self.max_size and key not in self.cache:
                await self._evict_entry()
            
            self.cache[key] = entry
            
            # Actualizar estadísticas
            self.stats.total_entries = len(self.cache)
    
    async def _evict_entry(self):
        """Evictar entrada según estrategia."""
        if not self.cache:
            return
        
        evict_key = None
        
        if self.strategy == CacheStrategy.LRU:
            # Eliminar el menos recientemente usado (primero)
            evict_key = next(iter(self.cache))
        
        elif self.strategy == CacheStrategy.LFU:
            # Eliminar el menos frecuentemente usado
            evict_key = min(
                self.cache.keys(),
                key=lambda k: self.cache[k].access_count,
            )
        
        elif self.strategy == CacheStrategy.FIFO:
            # Eliminar el primero (más antiguo)
            evict_key = next(iter(self.cache))
        
        elif self.strategy == CacheStrategy.ADAPTIVE:
            # Usar combinación de frecuencia y tiempo
            now = datetime.now()
            scores = {}
            
            for k, entry in self.cache.items():
                # Penalizar entradas antiguas sin acceso
                age_penalty = (now - entry.last_accessed).total_seconds() / 3600
                frequency_score = entry.access_count
                score = frequency_score - age_penalty
                scores[k] = score
            
            evict_key = min(scores.keys(), key=lambda k: scores[k])
        
        if evict_key:
            del self.cache[evict_key]
            self.stats.evictions += 1
            logger.debug(f"Evicted cache entry: {evict_key}")
    
    async def invalidate(self, key: str):
        """Invalidar entrada."""
        async with self._lock:
            if key in self.cache:
                del self.cache[key]
                self.stats.total_entries = len(self.cache)
    
    async def invalidate_pattern(self, pattern: str):
        """Invalidar entradas que coincidan con patrón."""
        async with self._lock:
            keys_to_remove = [k for k in self.cache.keys() if pattern in k]
            for key in keys_to_remove:
                del self.cache[key]
            self.stats.total_entries = len(self.cache)
            return len(keys_to_remove)
    
    async def prefetch(self, key: str, fetch_func: Callable):
        """Pre-cargar entrada."""
        if key in self.cache:
            return
        
        try:
            value = await fetch_func()
            await self.set(key, value)
            logger.debug(f"Prefetched cache entry: {key}")
        except Exception as e:
            logger.error(f"Prefetch failed for {key}: {e}")
    
    def _update_hit_rate(self):
        """Actualizar tasa de aciertos."""
        total = self.stats.total_hits + self.stats.total_misses
        if total > 0:
            self.stats.hit_rate = self.stats.total_hits / total
    
    def analyze_access_patterns(self) -> Dict[str, Any]:
        """Analizar patrones de acceso."""
        now = datetime.now()
        patterns = {}
        
        for key, accesses in self.access_patterns.items():
            if not accesses:
                continue
            
            # Calcular frecuencia de acceso en última hora
            recent_accesses = [
                a for a in accesses
                if (now - a).total_seconds() < 3600
            ]
            
            patterns[key] = {
                "total_accesses": len(accesses),
                "recent_accesses": len(recent_accesses),
                "access_rate": len(recent_accesses) / 3600 if recent_accesses else 0.0,
                "last_access": accesses[-1].isoformat() if accesses else None,
            }
        
        return patterns
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de caché."""
        self._update_hit_rate()
        
        return {
            "total_entries": self.stats.total_entries,
            "max_size": self.max_size,
            "total_hits": self.stats.total_hits,
            "total_misses": self.stats.total_misses,
            "hit_rate": self.stats.hit_rate,
            "evictions": self.stats.evictions,
            "strategy": self.strategy.value,
            "usage_percentage": (self.stats.total_entries / self.max_size * 100) if self.max_size > 0 else 0.0,
        }
    
    async def clear(self):
        """Limpiar caché."""
        async with self._lock:
            self.cache.clear()
            self.access_patterns.clear()
            self.prefetch_predictions.clear()
            self.stats = CacheStats()

