"""
Advanced Caching System for Ultra-High Performance
Sistema de Caché Avanzado para ultra-alto rendimiento
"""

import asyncio
import logging
import time
import json
import threading
import hashlib
import pickle
import gzip
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.routing import APIRouter
from collections import defaultdict, deque, OrderedDict
import numpy as np
import weakref
import heapq
from functools import wraps, lru_cache
import redis
import aioredis

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Estrategias de caché"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    FIFO = "fifo"  # First In First Out
    LIFO = "lifo"  # Last In First Out
    RANDOM = "random"
    ADAPTIVE = "adaptive"


class CacheLevel(Enum):
    """Niveles de caché"""
    L1 = "l1"  # In-memory cache
    L2 = "l2"  # Redis cache
    L3 = "l3"  # Disk cache
    L4 = "l4"  # Distributed cache


class CacheOperation(Enum):
    """Operaciones de caché"""
    GET = "get"
    SET = "set"
    DELETE = "delete"
    CLEAR = "clear"
    EXISTS = "exists"
    EXPIRE = "expire"
    INCREMENT = "increment"
    DECREMENT = "decrement"


@dataclass
class CacheEntry:
    """Entrada de caché"""
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int
    size: int
    ttl: Optional[float]
    tags: List[str]
    metadata: Dict[str, Any]


@dataclass
class CacheStats:
    """Estadísticas de caché"""
    hits: int
    misses: int
    sets: int
    deletes: int
    evictions: int
    total_size: int
    entry_count: int
    hit_rate: float
    miss_rate: float


@dataclass
class CacheConfig:
    """Configuración de caché"""
    max_size: int
    max_memory_mb: int
    default_ttl: float
    strategy: CacheStrategy
    compression: bool
    serialization: str  # json, pickle, msgpack
    persistence: bool
    backup_interval: float


class L1Cache:
    """Caché L1 - In-memory"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache: OrderedDict = OrderedDict()
        self.stats = CacheStats(0, 0, 0, 0, 0, 0, 0, 0.0, 0.0)
        self._lock = threading.RLock()
        self._access_times: Dict[str, float] = {}
        self._access_counts: Dict[str, int] = defaultdict(int)
    
    async def get(self, key: str) -> Optional[Any]:
        """Obtener valor del caché"""
        async with self._lock:
            if key in self.cache:
                # Actualizar estadísticas
                self.stats.hits += 1
                self._access_times[key] = time.time()
                self._access_counts[key] += 1
                
                # Mover al final (LRU)
                if self.config.strategy == CacheStrategy.LRU:
                    self.cache.move_to_end(key)
                
                return self.cache[key].value
            else:
                self.stats.misses += 1
                return None
    
    async def set(self, key: str, value: Any, ttl: Optional[float] = None, 
                 tags: List[str] = None, metadata: Dict[str, Any] = None) -> bool:
        """Establecer valor en caché"""
        async with self._lock:
            try:
                # Serializar valor
                serialized_value = await self._serialize_value(value)
                value_size = len(serialized_value)
                
                # Verificar límites
                if not await self._check_limits(key, value_size):
                    return False
                
                # Crear entrada de caché
                entry = CacheEntry(
                    key=key,
                    value=serialized_value,
                    created_at=time.time(),
                    last_accessed=time.time(),
                    access_count=0,
                    size=value_size,
                    ttl=ttl or self.config.default_ttl,
                    tags=tags or [],
                    metadata=metadata or {}
                )
                
                # Agregar al caché
                self.cache[key] = entry
                self._access_times[key] = time.time()
                self._access_counts[key] = 0
                
                # Actualizar estadísticas
                self.stats.sets += 1
                self.stats.total_size += value_size
                self.stats.entry_count += 1
                
                return True
                
            except Exception as e:
                logger.error(f"Error setting cache value for key {key}: {e}")
                return False
    
    async def delete(self, key: str) -> bool:
        """Eliminar valor del caché"""
        async with self._lock:
            if key in self.cache:
                entry = self.cache.pop(key)
                self.stats.total_size -= entry.size
                self.stats.entry_count -= 1
                self.stats.deletes += 1
                
                # Limpiar metadatos
                self._access_times.pop(key, None)
                self._access_counts.pop(key, None)
                
                return True
            return False
    
    async def clear(self):
        """Limpiar caché"""
        async with self._lock:
            self.cache.clear()
            self._access_times.clear()
            self._access_counts.clear()
            self.stats = CacheStats(0, 0, 0, 0, 0, 0, 0, 0.0, 0.0)
    
    async def exists(self, key: str) -> bool:
        """Verificar si existe clave"""
        async with self._lock:
            return key in self.cache
    
    async def expire(self, key: str, ttl: float) -> bool:
        """Establecer TTL para clave"""
        async with self._lock:
            if key in self.cache:
                self.cache[key].ttl = ttl
                return True
            return False
    
    async def get_stats(self) -> CacheStats:
        """Obtener estadísticas"""
        async with self._lock:
            total_requests = self.stats.hits + self.stats.misses
            if total_requests > 0:
                self.stats.hit_rate = self.stats.hits / total_requests
                self.stats.miss_rate = self.stats.misses / total_requests
            
            return self.stats
    
    async def _serialize_value(self, value: Any) -> bytes:
        """Serializar valor"""
        if self.config.serialization == "json":
            return json.dumps(value).encode('utf-8')
        elif self.config.serialization == "pickle":
            return pickle.dumps(value)
        else:
            return str(value).encode('utf-8')
    
    async def _deserialize_value(self, data: bytes) -> Any:
        """Deserializar valor"""
        if self.config.serialization == "json":
            return json.loads(data.decode('utf-8'))
        elif self.config.serialization == "pickle":
            return pickle.loads(data)
        else:
            return data.decode('utf-8')
    
    async def _check_limits(self, key: str, value_size: int) -> bool:
        """Verificar límites de caché"""
        # Verificar límite de tamaño
        if value_size > self.config.max_memory_mb * 1024 * 1024:
            return False
        
        # Verificar límite de entradas
        if len(self.cache) >= self.config.max_size:
            await self._evict_entries()
        
        # Verificar límite de memoria total
        if self.stats.total_size + value_size > self.config.max_memory_mb * 1024 * 1024:
            await self._evict_entries()
        
        return True
    
    async def _evict_entries(self):
        """Eliminar entradas según estrategia"""
        if not self.cache:
            return
        
        # Calcular número de entradas a eliminar (10% del caché)
        entries_to_evict = max(1, len(self.cache) // 10)
        
        if self.config.strategy == CacheStrategy.LRU:
            # Eliminar entradas menos recientemente usadas
            for _ in range(entries_to_evict):
                if self.cache:
                    key, entry = self.cache.popitem(last=False)
                    self.stats.total_size -= entry.size
                    self.stats.entry_count -= 1
                    self.stats.evictions += 1
                    self._access_times.pop(key, None)
                    self._access_counts.pop(key, None)
        
        elif self.config.strategy == CacheStrategy.LFU:
            # Eliminar entradas menos frecuentemente usadas
            sorted_keys = sorted(self._access_counts.keys(), key=lambda k: self._access_counts[k])
            for key in sorted_keys[:entries_to_evict]:
                if key in self.cache:
                    entry = self.cache.pop(key)
                    self.stats.total_size -= entry.size
                    self.stats.entry_count -= 1
                    self.stats.evictions += 1
                    self._access_times.pop(key, None)
                    self._access_counts.pop(key, None)
        
        elif self.config.strategy == CacheStrategy.TTL:
            # Eliminar entradas expiradas
            current_time = time.time()
            expired_keys = []
            for key, entry in self.cache.items():
                if entry.ttl and current_time - entry.created_at > entry.ttl:
                    expired_keys.append(key)
            
            for key in expired_keys[:entries_to_evict]:
                entry = self.cache.pop(key)
                self.stats.total_size -= entry.size
                self.stats.entry_count -= 1
                self.stats.evictions += 1
                self._access_times.pop(key, None)
                self._access_counts.pop(key, None)


class L2Cache:
    """Caché L2 - Redis"""
    
    def __init__(self, config: CacheConfig, redis_url: str = "redis://localhost:6379"):
        self.config = config
        self.redis_url = redis_url
        self.redis: Optional[aioredis.Redis] = None
        self.stats = CacheStats(0, 0, 0, 0, 0, 0, 0, 0.0, 0.0)
        self._lock = threading.RLock()
    
    async def connect(self):
        """Conectar a Redis"""
        try:
            self.redis = await aioredis.from_url(self.redis_url)
            await self.redis.ping()
            logger.info("Connected to Redis cache")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis = None
    
    async def disconnect(self):
        """Desconectar de Redis"""
        if self.redis:
            await self.redis.close()
            self.redis = None
    
    async def get(self, key: str) -> Optional[Any]:
        """Obtener valor del caché"""
        if not self.redis:
            return None
        
        try:
            async with self._lock:
                data = await self.redis.get(key)
                if data:
                    self.stats.hits += 1
                    return await self._deserialize_value(data)
                else:
                    self.stats.misses += 1
                    return None
        except Exception as e:
            logger.error(f"Error getting cache value for key {key}: {e}")
            self.stats.misses += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[float] = None, 
                 tags: List[str] = None, metadata: Dict[str, Any] = None) -> bool:
        """Establecer valor en caché"""
        if not self.redis:
            return False
        
        try:
            async with self._lock:
                serialized_value = await self._serialize_value(value)
                ttl_seconds = int(ttl or self.config.default_ttl)
                
                await self.redis.setex(key, ttl_seconds, serialized_value)
                
                # Almacenar metadatos
                if tags or metadata:
                    meta_key = f"{key}:meta"
                    meta_data = {
                        "tags": tags or [],
                        "metadata": metadata or {},
                        "created_at": time.time()
                    }
                    await self.redis.setex(meta_key, ttl_seconds, json.dumps(meta_data))
                
                self.stats.sets += 1
                return True
                
        except Exception as e:
            logger.error(f"Error setting cache value for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Eliminar valor del caché"""
        if not self.redis:
            return False
        
        try:
            async with self._lock:
                result = await self.redis.delete(key)
                await self.redis.delete(f"{key}:meta")
                self.stats.deletes += 1
                return result > 0
        except Exception as e:
            logger.error(f"Error deleting cache value for key {key}: {e}")
            return False
    
    async def clear(self):
        """Limpiar caché"""
        if not self.redis:
            return
        
        try:
            async with self._lock:
                await self.redis.flushdb()
                self.stats = CacheStats(0, 0, 0, 0, 0, 0, 0, 0.0, 0.0)
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
    
    async def exists(self, key: str) -> bool:
        """Verificar si existe clave"""
        if not self.redis:
            return False
        
        try:
            return await self.redis.exists(key) > 0
        except Exception as e:
            logger.error(f"Error checking cache existence for key {key}: {e}")
            return False
    
    async def expire(self, key: str, ttl: float) -> bool:
        """Establecer TTL para clave"""
        if not self.redis:
            return False
        
        try:
            return await self.redis.expire(key, int(ttl))
        except Exception as e:
            logger.error(f"Error setting cache expiration for key {key}: {e}")
            return False
    
    async def get_stats(self) -> CacheStats:
        """Obtener estadísticas"""
        if not self.redis:
            return self.stats
        
        try:
            info = await self.redis.info('memory')
            self.stats.total_size = info.get('used_memory', 0)
            self.stats.entry_count = await self.redis.dbsize()
            
            total_requests = self.stats.hits + self.stats.misses
            if total_requests > 0:
                self.stats.hit_rate = self.stats.hits / total_requests
                self.stats.miss_rate = self.stats.misses / total_requests
            
            return self.stats
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return self.stats
    
    async def _serialize_value(self, value: Any) -> bytes:
        """Serializar valor"""
        if self.config.serialization == "json":
            data = json.dumps(value).encode('utf-8')
        elif self.config.serialization == "pickle":
            data = pickle.dumps(value)
        else:
            data = str(value).encode('utf-8')
        
        if self.config.compression:
            data = gzip.compress(data)
        
        return data
    
    async def _deserialize_value(self, data: bytes) -> Any:
        """Deserializar valor"""
        if self.config.compression:
            data = gzip.decompress(data)
        
        if self.config.serialization == "json":
            return json.loads(data.decode('utf-8'))
        elif self.config.serialization == "pickle":
            return pickle.loads(data)
        else:
            return data.decode('utf-8')


class L3Cache:
    """Caché L3 - Disk"""
    
    def __init__(self, config: CacheConfig, cache_dir: str = "/tmp/cache"):
        self.config = config
        self.cache_dir = cache_dir
        self.stats = CacheStats(0, 0, 0, 0, 0, 0, 0, 0.0, 0.0)
        self._lock = threading.RLock()
        self._ensure_cache_dir()
    
    def _ensure_cache_dir(self):
        """Asegurar que el directorio de caché existe"""
        import os
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _get_cache_path(self, key: str) -> str:
        """Obtener ruta del archivo de caché"""
        # Crear hash de la clave para el nombre del archivo
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return f"{self.cache_dir}/{key_hash}.cache"
    
    async def get(self, key: str) -> Optional[Any]:
        """Obtener valor del caché"""
        try:
            async with self._lock:
                cache_path = self._get_cache_path(key)
                
                if not os.path.exists(cache_path):
                    self.stats.misses += 1
                    return None
                
                # Verificar TTL
                if await self._is_expired(cache_path):
                    await self.delete(key)
                    self.stats.misses += 1
                    return None
                
                # Leer archivo
                with open(cache_path, 'rb') as f:
                    data = f.read()
                
                self.stats.hits += 1
                return await self._deserialize_value(data)
                
        except Exception as e:
            logger.error(f"Error getting cache value for key {key}: {e}")
            self.stats.misses += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[float] = None, 
                 tags: List[str] = None, metadata: Dict[str, Any] = None) -> bool:
        """Establecer valor en caché"""
        try:
            async with self._lock:
                cache_path = self._get_cache_path(key)
                serialized_value = await self._serialize_value(value)
                
                # Escribir archivo
                with open(cache_path, 'wb') as f:
                    f.write(serialized_value)
                
                # Establecer tiempo de modificación para TTL
                if ttl:
                    import os
                    os.utime(cache_path, (time.time(), time.time() + ttl))
                
                self.stats.sets += 1
                return True
                
        except Exception as e:
            logger.error(f"Error setting cache value for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Eliminar valor del caché"""
        try:
            async with self._lock:
                cache_path = self._get_cache_path(key)
                
                if os.path.exists(cache_path):
                    os.remove(cache_path)
                    self.stats.deletes += 1
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Error deleting cache value for key {key}: {e}")
            return False
    
    async def clear(self):
        """Limpiar caché"""
        try:
            async with self._lock:
                import os
                import glob
                
                cache_files = glob.glob(f"{self.cache_dir}/*.cache")
                for cache_file in cache_files:
                    os.remove(cache_file)
                
                self.stats = CacheStats(0, 0, 0, 0, 0, 0, 0, 0.0, 0.0)
                
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
    
    async def exists(self, key: str) -> bool:
        """Verificar si existe clave"""
        try:
            cache_path = self._get_cache_path(key)
            return os.path.exists(cache_path) and not await self._is_expired(cache_path)
        except Exception as e:
            logger.error(f"Error checking cache existence for key {key}: {e}")
            return False
    
    async def expire(self, key: str, ttl: float) -> bool:
        """Establecer TTL para clave"""
        try:
            cache_path = self._get_cache_path(key)
            if os.path.exists(cache_path):
                import os
                os.utime(cache_path, (time.time(), time.time() + ttl))
                return True
            return False
        except Exception as e:
            logger.error(f"Error setting cache expiration for key {key}: {e}")
            return False
    
    async def _is_expired(self, cache_path: str) -> bool:
        """Verificar si el archivo de caché ha expirado"""
        try:
            import os
            stat = os.stat(cache_path)
            return time.time() > stat.st_mtime
        except:
            return True
    
    async def _serialize_value(self, value: Any) -> bytes:
        """Serializar valor"""
        if self.config.serialization == "json":
            data = json.dumps(value).encode('utf-8')
        elif self.config.serialization == "pickle":
            data = pickle.dumps(value)
        else:
            data = str(value).encode('utf-8')
        
        if self.config.compression:
            data = gzip.compress(data)
        
        return data
    
    async def _deserialize_value(self, data: bytes) -> Any:
        """Deserializar valor"""
        if self.config.compression:
            data = gzip.decompress(data)
        
        if self.config.serialization == "json":
            return json.loads(data.decode('utf-8'))
        elif self.config.serialization == "pickle":
            return pickle.loads(data)
        else:
            return data.decode('utf-8')


class AdvancedCachingSystem:
    """Sistema de caché avanzado multi-nivel"""
    
    def __init__(self):
        self.l1_cache: Optional[L1Cache] = None
        self.l2_cache: Optional[L2Cache] = None
        self.l3_cache: Optional[L3Cache] = None
        self.config: Optional[CacheConfig] = None
        self.is_running = False
        self._cleanup_task = None
        self._backup_task = None
        self._lock = threading.RLock()
    
    async def initialize(self, config: CacheConfig, redis_url: str = None, cache_dir: str = None):
        """Inicializar sistema de caché"""
        try:
            self.config = config
            
            # Inicializar L1 cache (siempre)
            self.l1_cache = L1Cache(config)
            
            # Inicializar L2 cache (Redis) si está disponible
            if redis_url:
                self.l2_cache = L2Cache(config, redis_url)
                await self.l2_cache.connect()
            
            # Inicializar L3 cache (Disk) si está habilitado
            if config.persistence and cache_dir:
                self.l3_cache = L3Cache(config, cache_dir)
            
            # Iniciar tareas de mantenimiento
            self.is_running = True
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            if config.backup_interval > 0:
                self._backup_task = asyncio.create_task(self._backup_loop())
            
            logger.info("Advanced caching system initialized")
            
        except Exception as e:
            logger.error(f"Error initializing caching system: {e}")
            raise
    
    async def shutdown(self):
        """Cerrar sistema de caché"""
        try:
            self.is_running = False
            
            # Detener tareas
            if self._cleanup_task:
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass
            
            if self._backup_task:
                self._backup_task.cancel()
                try:
                    await self._backup_task
                except asyncio.CancelledError:
                    pass
            
            # Desconectar L2 cache
            if self.l2_cache:
                await self.l2_cache.disconnect()
            
            logger.info("Advanced caching system shutdown")
            
        except Exception as e:
            logger.error(f"Error shutting down caching system: {e}")
    
    async def get(self, key: str) -> Optional[Any]:
        """Obtener valor del caché (multi-nivel)"""
        try:
            # Intentar L1 cache primero
            if self.l1_cache:
                value = await self.l1_cache.get(key)
                if value is not None:
                    return value
            
            # Intentar L2 cache
            if self.l2_cache:
                value = await self.l2_cache.get(key)
                if value is not None:
                    # Promover a L1 cache
                    if self.l1_cache:
                        await self.l1_cache.set(key, value)
                    return value
            
            # Intentar L3 cache
            if self.l3_cache:
                value = await self.l3_cache.get(key)
                if value is not None:
                    # Promover a L1 y L2 cache
                    if self.l1_cache:
                        await self.l1_cache.set(key, value)
                    if self.l2_cache:
                        await self.l2_cache.set(key, value)
                    return value
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting cache value for key {key}: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[float] = None, 
                 tags: List[str] = None, metadata: Dict[str, Any] = None) -> bool:
        """Establecer valor en caché (multi-nivel)"""
        try:
            success = True
            
            # Establecer en L1 cache
            if self.l1_cache:
                success &= await self.l1_cache.set(key, value, ttl, tags, metadata)
            
            # Establecer en L2 cache
            if self.l2_cache:
                success &= await self.l2_cache.set(key, value, ttl, tags, metadata)
            
            # Establecer en L3 cache si está habilitado
            if self.l3_cache and self.config.persistence:
                success &= await self.l3_cache.set(key, value, ttl, tags, metadata)
            
            return success
            
        except Exception as e:
            logger.error(f"Error setting cache value for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Eliminar valor del caché (multi-nivel)"""
        try:
            success = True
            
            # Eliminar de L1 cache
            if self.l1_cache:
                success &= await self.l1_cache.delete(key)
            
            # Eliminar de L2 cache
            if self.l2_cache:
                success &= await self.l2_cache.delete(key)
            
            # Eliminar de L3 cache
            if self.l3_cache:
                success &= await self.l3_cache.delete(key)
            
            return success
            
        except Exception as e:
            logger.error(f"Error deleting cache value for key {key}: {e}")
            return False
    
    async def clear(self):
        """Limpiar todos los niveles de caché"""
        try:
            if self.l1_cache:
                await self.l1_cache.clear()
            
            if self.l2_cache:
                await self.l2_cache.clear()
            
            if self.l3_cache:
                await self.l3_cache.clear()
                
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
    
    async def exists(self, key: str) -> bool:
        """Verificar si existe clave en cualquier nivel"""
        try:
            if self.l1_cache and await self.l1_cache.exists(key):
                return True
            
            if self.l2_cache and await self.l2_cache.exists(key):
                return True
            
            if self.l3_cache and await self.l3_cache.exists(key):
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking cache existence for key {key}: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de todos los niveles"""
        try:
            stats = {
                "l1_cache": await self.l1_cache.get_stats() if self.l1_cache else None,
                "l2_cache": await self.l2_cache.get_stats() if self.l2_cache else None,
                "l3_cache": await self.l3_cache.get_stats() if self.l3_cache else None,
                "config": asdict(self.config) if self.config else None
            }
            
            # Calcular estadísticas agregadas
            total_hits = 0
            total_misses = 0
            total_sets = 0
            total_deletes = 0
            
            for level_stats in [stats["l1_cache"], stats["l2_cache"], stats["l3_cache"]]:
                if level_stats:
                    total_hits += level_stats.hits
                    total_misses += level_stats.misses
                    total_sets += level_stats.sets
                    total_deletes += level_stats.deletes
            
            total_requests = total_hits + total_misses
            stats["aggregated"] = {
                "total_hits": total_hits,
                "total_misses": total_misses,
                "total_sets": total_sets,
                "total_deletes": total_deletes,
                "hit_rate": total_hits / total_requests if total_requests > 0 else 0,
                "miss_rate": total_misses / total_requests if total_requests > 0 else 0
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {}
    
    async def _cleanup_loop(self):
        """Loop de limpieza"""
        while self.is_running:
            try:
                # Limpiar entradas expiradas
                await self._cleanup_expired_entries()
                
                # Optimizar caché
                await self._optimize_cache()
                
                await asyncio.sleep(300)  # Limpiar cada 5 minutos
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(300)
    
    async def _backup_loop(self):
        """Loop de backup"""
        while self.is_running:
            try:
                # Realizar backup del caché
                await self._backup_cache()
                
                await asyncio.sleep(self.config.backup_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in backup loop: {e}")
                await asyncio.sleep(self.config.backup_interval)
    
    async def _cleanup_expired_entries(self):
        """Limpiar entradas expiradas"""
        # Implementar limpieza de entradas expiradas
        pass
    
    async def _optimize_cache(self):
        """Optimizar caché"""
        # Implementar optimización de caché
        pass
    
    async def _backup_cache(self):
        """Realizar backup del caché"""
        # Implementar backup del caché
        pass


# Instancia global del sistema de caché avanzado
advanced_caching_system = AdvancedCachingSystem()


# Router para endpoints del sistema de caché avanzado
advanced_caching_router = APIRouter()


@advanced_caching_router.post("/cache/initialize")
async def initialize_cache_endpoint(config_data: dict):
    """Inicializar sistema de caché"""
    try:
        config = CacheConfig(
            max_size=config_data.get("max_size", 10000),
            max_memory_mb=config_data.get("max_memory_mb", 100),
            default_ttl=config_data.get("default_ttl", 3600),
            strategy=CacheStrategy(config_data.get("strategy", "lru")),
            compression=config_data.get("compression", True),
            serialization=config_data.get("serialization", "json"),
            persistence=config_data.get("persistence", False),
            backup_interval=config_data.get("backup_interval", 0)
        )
        
        redis_url = config_data.get("redis_url")
        cache_dir = config_data.get("cache_dir")
        
        await advanced_caching_system.initialize(config, redis_url, cache_dir)
        
        return {
            "message": "Advanced caching system initialized successfully",
            "config": asdict(config)
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid cache strategy: {e}")
    except Exception as e:
        logger.error(f"Error initializing cache: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to initialize cache: {str(e)}")


@advanced_caching_router.get("/cache/{key}")
async def get_cache_value_endpoint(key: str):
    """Obtener valor del caché"""
    try:
        value = await advanced_caching_system.get(key)
        
        if value is not None:
            return {
                "key": key,
                "value": value,
                "found": True
            }
        else:
            return {
                "key": key,
                "value": None,
                "found": False
            }
            
    except Exception as e:
        logger.error(f"Error getting cache value: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get cache value: {str(e)}")


@advanced_caching_router.post("/cache/{key}")
async def set_cache_value_endpoint(key: str, cache_data: dict):
    """Establecer valor en caché"""
    try:
        value = cache_data["value"]
        ttl = cache_data.get("ttl")
        tags = cache_data.get("tags", [])
        metadata = cache_data.get("metadata", {})
        
        success = await advanced_caching_system.set(key, value, ttl, tags, metadata)
        
        return {
            "message": "Cache value set successfully" if success else "Failed to set cache value",
            "key": key,
            "success": success
        }
        
    except Exception as e:
        logger.error(f"Error setting cache value: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to set cache value: {str(e)}")


@advanced_caching_router.delete("/cache/{key}")
async def delete_cache_value_endpoint(key: str):
    """Eliminar valor del caché"""
    try:
        success = await advanced_caching_system.delete(key)
        
        return {
            "message": "Cache value deleted successfully" if success else "Cache value not found",
            "key": key,
            "success": success
        }
        
    except Exception as e:
        logger.error(f"Error deleting cache value: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete cache value: {str(e)}")


@advanced_caching_router.post("/cache/clear")
async def clear_cache_endpoint():
    """Limpiar caché"""
    try:
        await advanced_caching_system.clear()
        
        return {
            "message": "Cache cleared successfully"
        }
        
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")


@advanced_caching_router.get("/cache/stats")
async def get_cache_stats_endpoint():
    """Obtener estadísticas del caché"""
    try:
        stats = await advanced_caching_system.get_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get cache stats: {str(e)}")


# Decoradores de caché
def cache_result(ttl: float = 3600, tags: List[str] = None, metadata: Dict[str, Any] = None):
    """Decorador para cachear resultado de función"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generar clave de caché
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Intentar obtener del caché
            cached_result = await advanced_caching_system.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Ejecutar función
            result = await func(*args, **kwargs)
            
            # Cachear resultado
            await advanced_caching_system.set(cache_key, result, ttl, tags, metadata)
            
            return result
        return wrapper
    return decorator


def cache_invalidate(pattern: str = None, tags: List[str] = None):
    """Decorador para invalidar caché"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            
            # Invalidar caché basado en patrón o tags
            if pattern:
                # Implementar invalidación por patrón
                pass
            elif tags:
                # Implementar invalidación por tags
                pass
            
            return result
        return wrapper
    return decorator


# Funciones de utilidad para integración
async def initialize_advanced_caching(config: CacheConfig, redis_url: str = None, cache_dir: str = None):
    """Inicializar sistema de caché avanzado"""
    await advanced_caching_system.initialize(config, redis_url, cache_dir)


async def shutdown_advanced_caching():
    """Cerrar sistema de caché avanzado"""
    await advanced_caching_system.shutdown()


async def get_cached_value(key: str) -> Optional[Any]:
    """Obtener valor del caché"""
    return await advanced_caching_system.get(key)


async def set_cached_value(key: str, value: Any, ttl: Optional[float] = None, 
                          tags: List[str] = None, metadata: Dict[str, Any] = None) -> bool:
    """Establecer valor en caché"""
    return await advanced_caching_system.set(key, value, ttl, tags, metadata)


async def get_advanced_caching_stats() -> Dict[str, Any]:
    """Obtener estadísticas del sistema de caché avanzado"""
    return await advanced_caching_system.get_stats()


logger.info("Advanced caching system module loaded successfully")

