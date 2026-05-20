from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import asyncio
import time
import logging
from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, Union
from dataclasses import dataclass, field
from cachetools import TTLCache, LRUCache
from .config import CacheConfig, CacheBackend
    import orjson
    import json as orjson
    import redis.asyncio as aioredis
from typing import Any, List, Dict, Optional
"""
Gestor de cache ultra-optimizado con múltiples backends.
"""



try:
    ORJSON_AVAILABLE = True
except ImportError:
    ORJSON_AVAILABLE = False

try:
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class CacheStats:
    """Estadísticas del cache."""
    hits: int = 0
    misses: int = 0
    errors: int = 0
    total_requests: int = 0
    avg_get_time_ms: float = 0.0
    avg_set_time_ms: float = 0.0
    
    @property
    def hit_rate(self) -> float:
        """Calcular tasa de aciertos."""
        if self.total_requests == 0:
            return 0.0
        return (self.hits / self.total_requests) * 100
    
    def record_hit(self, get_time_ms: float = 0.0):
        """Registrar acierto."""
        self.hits += 1
        self.total_requests += 1
        self._update_avg_get_time(get_time_ms)
    
    def record_miss(self, get_time_ms: float = 0.0):
        """Registrar fallo."""
        self.misses += 1
        self.total_requests += 1
        self._update_avg_get_time(get_time_ms)
    
    def record_error(self) -> Any:
        """Registrar error."""
        self.errors += 1
    
    def record_set(self, set_time_ms: float):
        """Registrar operación de escritura."""
        # Actualizar tiempo promedio de set
        if self.total_requests > 0:
            self.avg_set_time_ms = (
                (self.avg_set_time_ms * (self.total_requests - 1) + set_time_ms) / 
                self.total_requests
            )
        else:
            self.avg_set_time_ms = set_time_ms
    
    def _update_avg_get_time(self, get_time_ms: float):
        """Actualizar tiempo promedio de get."""
        if self.total_requests > 1:
            self.avg_get_time_ms = (
                (self.avg_get_time_ms * (self.total_requests - 1) + get_time_ms) / 
                self.total_requests
            )
        else:
            self.avg_get_time_ms = get_time_ms

class CacheBackendInterface(ABC):
    """Interfaz para backends de cache."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Obtener valor del cache."""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Guardar valor en cache."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Eliminar valor del cache."""
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """Limpiar todo el cache."""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Verificar si existe una key."""
        pass

class MemoryCacheBackend(CacheBackendInterface):
    """Backend de cache en memoria usando TTLCache."""
    
    def __init__(self, config: CacheConfig):
        
    """__init__ function."""
self.config = config
        self.cache = TTLCache(
            maxsize=config.memory_size,
            ttl=config.ttl_seconds
        )
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Obtener valor del cache en memoria."""
        async with self._lock:
            return self.cache.get(key)
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Guardar valor en cache en memoria."""
        try:
            async with self._lock:
                # TTLCache no soporta TTL por key, usamos el global
                self.cache[key] = value
                return True
        except Exception as e:
            logger.error(f"Error setting memory cache: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Eliminar valor del cache."""
        try:
            async with self._lock:
                if key in self.cache:
                    del self.cache[key]
                    return True
                return False
        except Exception:
            return False
    
    async def clear(self) -> bool:
        """Limpiar cache."""
        try:
            async with self._lock:
                self.cache.clear()
                return True
        except Exception:
            return False
    
    async def exists(self, key: str) -> bool:
        """Verificar existencia."""
        async with self._lock:
            return key in self.cache

class RedisCacheBackend(CacheBackendInterface):
    """Backend de cache usando Redis."""
    
    def __init__(self, config: CacheConfig):
        
    """__init__ function."""
self.config = config
        self.client: Optional[aioredis.Redis] = None
        self._initialized = False
    
    async def _ensure_connected(self) -> Any:
        """Asegurar conexión con Redis."""
        if not self._initialized:
            try:
                self.client = aioredis.from_url(
                    self.config.redis_url,
                    socket_timeout=self.config.redis_timeout,
                    socket_connect_timeout=self.config.redis_timeout
                )
                await self.client.ping()
                self._initialized = True
                logger.info("Redis cache backend initialized")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                self.client = None
                raise
    
    def _make_key(self, key: str) -> str:
        """Crear key con prefijo."""
        return f"{self.config.redis_prefix}:{key}"
    
    async def get(self, key: str) -> Optional[Any]:
        """Obtener valor de Redis."""
        try:
            await self._ensure_connected()
            if not self.client:
                return None
            
            redis_key = self._make_key(key)
            value = await self.client.get(redis_key)
            
            if value is None:
                return None
            
            # Deserializar
            if ORJSON_AVAILABLE:
                return orjson.loads(value)
            else:
                return orjson.loads(value.decode())
        except Exception as e:
            logger.warning(f"Redis get error: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Guardar valor en Redis."""
        try:
            await self._ensure_connected()
            if not self.client:
                return False
            
            redis_key = self._make_key(key)
            ttl = ttl or self.config.ttl_seconds
            
            # Serializar
            if ORJSON_AVAILABLE:
                data = orjson.dumps(value)
            else:
                data = orjson.dumps(value).encode()
            
            await self.client.setex(redis_key, ttl, data)
            return True
        except Exception as e:
            logger.warning(f"Redis set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Eliminar de Redis."""
        try:
            await self._ensure_connected()
            if not self.client:
                return False
            
            redis_key = self._make_key(key)
            result = await self.client.delete(redis_key)
            return result > 0
        except Exception as e:
            logger.warning(f"Redis delete error: {e}")
            return False
    
    async def clear(self) -> bool:
        """Limpiar cache de Redis."""
        try:
            await self._ensure_connected()
            if not self.client:
                return False
            
            pattern = f"{self.config.redis_prefix}:*"
            keys = await self.client.keys(pattern)
            if keys:
                await self.client.delete(*keys)
            return True
        except Exception as e:
            logger.warning(f"Redis clear error: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Verificar existencia en Redis."""
        try:
            await self._ensure_connected()
            if not self.client:
                return False
            
            redis_key = self._make_key(key)
            result = await self.client.exists(redis_key)
            return result > 0
        except Exception:
            return False

class HybridCacheBackend(CacheBackendInterface):
    """Backend híbrido: memoria + Redis."""
    
    def __init__(self, config: CacheConfig):
        
    """__init__ function."""
self.config = config
        self.memory_backend = MemoryCacheBackend(config)
        self.redis_backend = RedisCacheBackend(config) if REDIS_AVAILABLE else None
    
    async def get(self, key: str) -> Optional[Any]:
        """Obtener con fallback memoria -> Redis."""
        # Intentar memoria primero
        value = await self.memory_backend.get(key)
        if value is not None:
            return value
        
        # Fallback a Redis
        if self.redis_backend:
            value = await self.redis_backend.get(key)
            if value is not None:
                # Guardar en memoria para próxima vez
                await self.memory_backend.set(key, value)
                return value
        
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Guardar en ambos backends."""
        memory_success = await self.memory_backend.set(key, value, ttl)
        redis_success = True
        
        if self.redis_backend:
            redis_success = await self.redis_backend.set(key, value, ttl)
        
        return memory_success or redis_success
    
    async def delete(self, key: str) -> bool:
        """Eliminar de ambos backends."""
        memory_success = await self.memory_backend.delete(key)
        redis_success = True
        
        if self.redis_backend:
            redis_success = await self.redis_backend.delete(key)
        
        return memory_success or redis_success
    
    async def clear(self) -> bool:
        """Limpiar ambos backends."""
        memory_success = await self.memory_backend.clear()
        redis_success = True
        
        if self.redis_backend:
            redis_success = await self.redis_backend.clear()
        
        return memory_success and redis_success
    
    async def exists(self, key: str) -> bool:
        """Verificar en ambos backends."""
        if await self.memory_backend.exists(key):
            return True
        
        if self.redis_backend:
            return await self.redis_backend.exists(key)
        
        return False

class CacheManager:
    """Gestor principal de cache."""
    
    def __init__(self, config: CacheConfig):
        
    """__init__ function."""
self.config = config
        self.stats = CacheStats()
        self.backend = self._create_backend()
    
    def _create_backend(self) -> CacheBackendInterface:
        """Crear backend apropiado según configuración."""
        if self.config.backend == CacheBackend.MEMORY:
            return MemoryCacheBackend(self.config)
        elif self.config.backend == CacheBackend.REDIS:
            if not REDIS_AVAILABLE:
                logger.warning("Redis not available, falling back to memory")
                return MemoryCacheBackend(self.config)
            return RedisCacheBackend(self.config)
        elif self.config.backend == CacheBackend.HYBRID:
            return HybridCacheBackend(self.config)
        else:
            return MemoryCacheBackend(self.config)
    
    async def get(self, key: str) -> Optional[Any]:
        """Obtener valor del cache."""
        start_time = time.time()
        try:
            value = await self.backend.get(key)
            get_time_ms = (time.time() - start_time) * 1000
            
            if value is not None:
                self.stats.record_hit(get_time_ms)
                logger.debug(f"Cache hit: {key}")
            else:
                self.stats.record_miss(get_time_ms)
                logger.debug(f"Cache miss: {key}")
            
            return value
        except Exception as e:
            self.stats.record_error()
            logger.error(f"Cache get error for {key}: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Guardar valor en cache."""
        start_time = time.time()
        try:
            success = await self.backend.set(key, value, ttl)
            set_time_ms = (time.time() - start_time) * 1000
            self.stats.record_set(set_time_ms)
            
            if success:
                logger.debug(f"Cache set: {key}")
            else:
                logger.warning(f"Cache set failed: {key}")
            
            return success
        except Exception as e:
            self.stats.record_error()
            logger.error(f"Cache set error for {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Eliminar del cache."""
        try:
            return await self.backend.delete(key)
        except Exception as e:
            self.stats.record_error()
            logger.error(f"Cache delete error for {key}: {e}")
            return False
    
    async def clear(self) -> bool:
        """Limpiar cache."""
        try:
            success = await self.backend.clear()
            if success:
                logger.info("Cache cleared")
            return success
        except Exception as e:
            self.stats.record_error()
            logger.error(f"Cache clear error: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Verificar existencia."""
        try:
            return await self.backend.exists(key)
        except Exception as e:
            logger.error(f"Cache exists error for {key}: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del cache."""
        return {
            'hits': self.stats.hits,
            'misses': self.stats.misses,
            'errors': self.stats.errors,
            'total_requests': self.stats.total_requests,
            'hit_rate': self.stats.hit_rate,
            'avg_get_time_ms': self.stats.avg_get_time_ms,
            'avg_set_time_ms': self.stats.avg_set_time_ms,
            'backend_type': self.config.backend.value
        } 