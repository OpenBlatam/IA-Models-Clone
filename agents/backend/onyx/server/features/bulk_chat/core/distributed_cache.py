"""
Distributed Cache Manager
==========================

Sistema avanzado de cache distribuido con replicación, consistencia,
invalidación inteligente y múltiples estrategias de cache.
"""

import asyncio
import hashlib
import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Estrategias de cache."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    TTL = "ttl"  # Time To Live
    ADAPTIVE = "adaptive"  # Adaptativo basado en uso


class ConsistencyLevel(Enum):
    """Niveles de consistencia."""
    STRONG = "strong"  # Consistencia fuerte
    EVENTUAL = "eventual"  # Consistencia eventual
    WEAK = "weak"  # Consistencia débil


@dataclass
class CacheEntry:
    """Entrada de cache."""
    key: str
    value: Any
    created_at: datetime
    expires_at: Optional[datetime] = None
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    size: int = 0
    version: int = 1
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Verifica si la entrada ha expirado."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    def touch(self):
        """Actualiza el tiempo de acceso."""
        self.last_accessed = datetime.now()
        self.access_count += 1


@dataclass
class CacheStats:
    """Estadísticas de cache."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    sets: int = 0
    gets: int = 0
    invalidations: int = 0
    total_size: int = 0
    entry_count: int = 0

    @property
    def hit_rate(self) -> float:
        """Calcula la tasa de aciertos."""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total * 100


class DistributedCache:
    """
    Sistema de cache distribuido con replicación y consistencia.
    """

    def __init__(
        self,
        strategy: CacheStrategy = CacheStrategy.LRU,
        max_size: int = 10000,
        default_ttl: Optional[int] = None,
        consistency_level: ConsistencyLevel = ConsistencyLevel.EVENTUAL,
        replication_factor: int = 2,
        enable_compression: bool = False,
    ):
        self.strategy = strategy
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.consistency_level = consistency_level
        self.replication_factor = replication_factor
        self.enable_compression = enable_compression

        self.cache: Dict[str, CacheEntry] = {}
        self.stats = CacheStats()
        self.node_id = str(uuid4())
        self.replicas: Dict[str, "DistributedCache"] = {}
        self.invalidation_queue: deque = deque(maxlen=1000)
        self.tag_index: Dict[str, Set[str]] = defaultdict(set)

        # Estrategias específicas
        self.access_order: deque = deque()  # Para LRU/FIFO
        self.frequency: Dict[str, int] = defaultdict(int)  # Para LFU

        logger.info(
            f"DistributedCache initialized: strategy={strategy.value}, "
            f"max_size={max_size}, consistency={consistency_level.value}"
        )

    async def get(
        self,
        key: str,
        default: Any = None,
        update_access: bool = True,
    ) -> Any:
        """
        Obtiene un valor del cache.

        Args:
            key: Clave del cache
            default: Valor por defecto si no existe
            update_access: Si actualizar estadísticas de acceso

        Returns:
            Valor cacheado o default
        """
        self.stats.gets += 1

        entry = self.cache.get(key)
        if entry is None:
            # Intentar obtener de réplicas
            if self.consistency_level == ConsistencyLevel.WEAK:
                entry = await self._get_from_replicas(key)
                if entry:
                    await self.set(key, entry.value, ttl=self.default_ttl)
                    self.stats.hits += 1
                    return entry.value

            self.stats.misses += 1
            return default

        if entry.is_expired():
            await self.delete(key)
            self.stats.misses += 1
            return default

        if update_access:
            entry.touch()
            if self.strategy == CacheStrategy.LRU:
                self._update_lru_order(key)

        self.stats.hits += 1
        return entry.value

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Establece un valor en el cache.

        Args:
            key: Clave del cache
            value: Valor a cachear
            ttl: Tiempo de vida en segundos
            tags: Tags para invalidation por tags
            metadata: Metadatos adicionales

        Returns:
            True si se estableció correctamente
        """
        try:
            # Calcular tamaño
            size = self._calculate_size(value)

            # Verificar si necesitamos hacer eviction
            if len(self.cache) >= self.max_size and key not in self.cache:
                await self._evict()

            # Crear entrada
            expires_at = None
            if ttl is None:
                ttl = self.default_ttl
            if ttl is not None:
                expires_at = datetime.now() + timedelta(seconds=ttl)

            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.now(),
                expires_at=expires_at,
                size=size,
                tags=set(tags or []),
                metadata=metadata or {},
            )
            entry.touch()

            # Actualizar índices
            if tags:
                for tag in tags:
                    self.tag_index[tag].add(key)

            # Actualizar estrategias
            if self.strategy == CacheStrategy.LRU or self.strategy == CacheStrategy.FIFO:
                self.access_order.append(key)
                if len(self.access_order) > self.max_size * 2:
                    self.access_order.popleft()

            if self.strategy == CacheStrategy.LFU:
                self.frequency[key] = 1

            self.cache[key] = entry
            self.stats.sets += 1
            self.stats.total_size += size
            self.stats.entry_count = len(self.cache)

            # Replicar si es necesario
            if self.consistency_level != ConsistencyLevel.WEAK:
                await self._replicate_set(key, value, ttl, tags, metadata)

            return True

        except Exception as e:
            logger.error(f"Error setting cache key {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Elimina una entrada del cache."""
        if key not in self.cache:
            return False

        entry = self.cache.pop(key)
        self.stats.total_size -= entry.size
        self.stats.entry_count = len(self.cache)

        # Limpiar índices
        for tag in entry.tags:
            self.tag_index[tag].discard(key)
            if not self.tag_index[tag]:
                del self.tag_index[tag]

        # Limpiar estrategias
        if self.strategy == CacheStrategy.LRU or self.strategy == CacheStrategy.FIFO:
            try:
                self.access_order.remove(key)
            except ValueError:
                pass

        if self.strategy == CacheStrategy.LFU:
            self.frequency.pop(key, None)

        # Replicar eliminación
        await self._replicate_delete(key)

        return True

    async def invalidate(self, tags: Optional[List[str]] = None, pattern: Optional[str] = None) -> int:
        """
        Invalida entradas del cache.

        Args:
            tags: Lista de tags para invalidar
            pattern: Patrón de clave para invalidar

        Returns:
            Número de entradas invalidadas
        """
        invalidated = 0

        if tags:
            keys_to_invalidate = set()
            for tag in tags:
                keys_to_invalidate.update(self.tag_index.get(tag, set()))
            for key in keys_to_invalidate:
                if await self.delete(key):
                    invalidated += 1

        if pattern:
            # Invalidación por patrón
            keys_to_delete = [k for k in self.cache.keys() if pattern in k]
            for key in keys_to_delete:
                if await self.delete(key):
                    invalidated += 1

        self.stats.invalidations += invalidated
        return invalidated

    async def clear(self):
        """Limpia todo el cache."""
        self.cache.clear()
        self.tag_index.clear()
        self.access_order.clear()
        self.frequency.clear()
        self.stats = CacheStats()

        # Replicar limpieza
        for replica in self.replicas.values():
            await replica.clear()

    async def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del cache."""
        return {
            "node_id": self.node_id,
            "strategy": self.strategy.value,
            "max_size": self.max_size,
            "entry_count": self.stats.entry_count,
            "total_size": self.stats.total_size,
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "hit_rate": self.stats.hit_rate,
            "evictions": self.stats.evictions,
            "invalidations": self.stats.invalidations,
            "replicas": len(self.replicas),
            "consistency_level": self.consistency_level.value,
        }

    async def add_replica(self, replica_id: str, replica: "DistributedCache"):
        """Agrega una réplica al cache."""
        self.replicas[replica_id] = replica
        logger.info(f"Added replica {replica_id}")

    async def remove_replica(self, replica_id: str):
        """Elimina una réplica."""
        self.replicas.pop(replica_id, None)
        logger.info(f"Removed replica {replica_id}")

    async def _evict(self):
        """Ejecuta la estrategia de eviction."""
        if not self.cache:
            return

        key_to_evict = None

        if self.strategy == CacheStrategy.LRU:
            # Eliminar el menos recientemente usado
            while self.access_order:
                candidate = self.access_order.popleft()
                if candidate in self.cache:
                    key_to_evict = candidate
                    break

        elif self.strategy == CacheStrategy.LFU:
            # Eliminar el menos frecuentemente usado
            if self.frequency:
                key_to_evict = min(self.frequency.items(), key=lambda x: x[1])[0]

        elif self.strategy == CacheStrategy.FIFO:
            # Eliminar el primero
            if self.access_order:
                key_to_evict = self.access_order.popleft()

        elif self.strategy == CacheStrategy.TTL:
            # Eliminar el más antiguo expirado o el más antiguo
            oldest_key = min(
                self.cache.items(),
                key=lambda x: x[1].created_at
            )[0]
            key_to_evict = oldest_key

        elif self.strategy == CacheStrategy.ADAPTIVE:
            # Combinación de LRU y LFU
            scores = {}
            for key, entry in self.cache.items():
                age = (datetime.now() - entry.last_accessed).total_seconds()
                frequency = self.frequency.get(key, 1)
                scores[key] = age / frequency
            key_to_evict = max(scores.items(), key=lambda x: x[1])[0]

        if key_to_evict:
            await self.delete(key_to_evict)
            self.stats.evictions += 1

    async def _get_from_replicas(self, key: str) -> Optional[CacheEntry]:
        """Obtiene un valor de las réplicas."""
        for replica in self.replicas.values():
            try:
                entry = await replica.get(key, update_access=False)
                if entry:
                    return entry
            except Exception as e:
                logger.warning(f"Error getting from replica: {e}")
        return None

    async def _replicate_set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int],
        tags: Optional[List[str]],
        metadata: Optional[Dict[str, Any]],
    ):
        """Replica un set a las réplicas."""
        if not self.replicas:
            return

        tasks = []
        for replica in list(self.replicas.values())[:self.replication_factor]:
            tasks.append(replica.set(key, value, ttl, tags, metadata))

        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.warning(f"Error replicating set: {e}")

    async def _replicate_delete(self, key: str):
        """Replica una eliminación a las réplicas."""
        if not self.replicas:
            return

        tasks = [replica.delete(key) for replica in self.replicas.values()]
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.warning(f"Error replicating delete: {e}")

    def _update_lru_order(self, key: str):
        """Actualiza el orden LRU."""
        try:
            self.access_order.remove(key)
        except ValueError:
            pass
        self.access_order.append(key)

    def _calculate_size(self, value: Any) -> int:
        """Calcula el tamaño aproximado de un valor."""
        try:
            if isinstance(value, (str, bytes)):
                return len(value)
            return len(json.dumps(value, default=str))
        except Exception:
            return 1024  # Tamaño por defecto


