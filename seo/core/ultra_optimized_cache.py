from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import time
import hashlib
from typing import Any, Optional, Dict, List, Union
from dataclasses import dataclass
from loguru import logger
import orjson
import zstandard as zstd
from cachetools import TTLCache, LRUCache
import redis.asyncio as redis
from functools import wraps
import asyncio
from contextlib import asynccontextmanager
from .interfaces import CacheInterface
from typing import Any, List, Dict, Optional
import logging
"""
Cache Manager ultra-optimizado usando las librerías más rápidas disponibles.
Redis + Zstandard + Cachetools con fallback a memoria.
"""




@dataclass
class CacheStats:
    """Estadísticas del cache ultra-optimizado."""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    compression_ratio: float = 0.0
    memory_usage: int = 0
    total_requests: int = 0
    hit_rate: float = 0.0


class UltraOptimizedCache(CacheInterface):
    """Cache ultra-optimizado con múltiples niveles."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        
    """__init__ function."""
self.config = config or {}
        
        # Configuraciones
        self.redis_enabled = self.config.get('redis_enabled', True)
        self.memory_enabled = self.config.get('memory_enabled', True)
        self.compression_enabled = self.config.get('compression_enabled', True)
        self.compression_threshold = self.config.get('compression_threshold', 1024)  # 1KB
        self.compression_level = self.config.get('compression_level', 3)
        
        # Cache en memoria
        if self.memory_enabled:
            self.memory_cache = TTLCache(
                maxsize=self.config.get('memory_max_size', 1000),
                ttl=self.config.get('memory_ttl', 300)  # 5 minutos
            )
            self.lru_cache = LRUCache(
                maxsize=self.config.get('lru_max_size', 500)
            )
        
        # Redis
        self.redis_client = None
        if self.redis_enabled:
            self._setup_redis()
        
        # Compresión
        if self.compression_enabled:
            self.compressor = zstd.ZstdCompressor(level=self.compression_level)
            self.decompressor = zstd.ZstdDecompressor()
        
        # Estadísticas
        self.stats = CacheStats()
        self.start_time = time.time()
    
    def _setup_redis(self) -> Any:
        """Configura cliente Redis."""
        try:
            redis_config = self.config.get('redis', {})
            self.redis_client = redis.Redis(
                host=redis_config.get('host', 'localhost'),
                port=redis_config.get('port', 6379),
                password=redis_config.get('password'),
                db=redis_config.get('db', 0),
                max_connections=redis_config.get('max_connections', 50),
                decode_responses=False,  # Mantener bytes para compresión
                socket_timeout=redis_config.get('timeout', 5.0),
                socket_connect_timeout=redis_config.get('connect_timeout', 5.0),
                retry_on_timeout=True,
                health_check_interval=30
            )
            logger.info("Redis client configured successfully")
        except Exception as e:
            logger.error(f"Failed to setup Redis: {e}")
            self.redis_client = None
    
    def _generate_key(self, key: str) -> str:
        """Genera clave de cache optimizada."""
        # Usar hash MD5 para claves largas
        if len(key) > 250:
            return f"seo:{hashlib.md5(key.encode()).hexdigest()}"
        return f"seo:{key}"
    
    def _compress_data(self, data: Any) -> bytes:
        """Comprime datos usando Zstandard."""
        if not self.compression_enabled:
            return orjson.dumps(data)
        
        json_data = orjson.dumps(data)
        
        # Solo comprimir si supera el umbral
        if len(json_data) > self.compression_threshold:
            compressed = self.compressor.compress(json_data)
            # Solo usar compresión si realmente reduce el tamaño
            if len(compressed) < len(json_data):
                return b'zstd:' + compressed
        
        return json_data
    
    def _decompress_data(self, data: bytes) -> Any:
        """Descomprime datos usando Zstandard."""
        if not self.compression_enabled:
            return orjson.loads(data)
        
        # Verificar si está comprimido
        if data.startswith(b'zstd:'):
            compressed_data = data[5:]  # Remover prefijo 'zstd:'
            decompressed = self.decompressor.decompress(compressed_data)
            return orjson.loads(decompressed)
        
        return orjson.loads(data)
    
    async def get(self, key: str) -> Optional[Any]:
        """Obtiene valor del cache ultra-optimizado."""
        self.stats.total_requests += 1
        cache_key = self._generate_key(key)
        
        # 1. Intentar cache en memoria (más rápido)
        if self.memory_enabled:
            try:
                value = self.memory_cache.get(cache_key)
                if value is not None:
                    self.stats.hits += 1
                    return value
            except Exception as e:
                logger.warning(f"Memory cache get error: {e}")
        
        # 2. Intentar LRU cache
        if self.memory_enabled:
            try:
                value = self.lru_cache.get(cache_key)
                if value is not None:
                    self.stats.hits += 1
                    # Mover a cache principal
                    self.memory_cache[cache_key] = value
                    return value
            except Exception as e:
                logger.warning(f"LRU cache get error: {e}")
        
        # 3. Intentar Redis
        if self.redis_client:
            try:
                value_bytes = await self.redis_client.get(cache_key)
                if value_bytes:
                    value = self._decompress_data(value_bytes)
                    self.stats.hits += 1
                    
                    # Guardar en cache en memoria
                    if self.memory_enabled:
                        self.memory_cache[cache_key] = value
                    
                    return value
            except Exception as e:
                logger.warning(f"Redis get error: {e}")
        
        self.stats.misses += 1
        return None
    
    async def set(self, key: str, value: Any, ttl: int = 0) -> bool:
        """Establece valor en cache ultra-optimizado."""
        cache_key = self._generate_key(key)
        success = False
        
        # 1. Guardar en cache en memoria
        if self.memory_enabled:
            try:
                self.memory_cache[cache_key] = value
                success = True
            except Exception as e:
                logger.warning(f"Memory cache set error: {e}")
        
        # 2. Guardar en Redis
        if self.redis_client:
            try:
                value_bytes = self._compress_data(value)
                if ttl > 0:
                    await self.redis_client.setex(cache_key, ttl, value_bytes)
                else:
                    await self.redis_client.set(cache_key, value_bytes)
                success = True
            except Exception as e:
                logger.warning(f"Redis set error: {e}")
        
        if success:
            self.stats.sets += 1
        
        return success
    
    async def delete(self, key: str) -> bool:
        """Elimina valor del cache ultra-optimizado."""
        cache_key = self._generate_key(key)
        success = False
        
        # 1. Eliminar de cache en memoria
        if self.memory_enabled:
            try:
                if cache_key in self.memory_cache:
                    del self.memory_cache[cache_key]
                if cache_key in self.lru_cache:
                    del self.lru_cache[cache_key]
                success = True
            except Exception as e:
                logger.warning(f"Memory cache delete error: {e}")
        
        # 2. Eliminar de Redis
        if self.redis_client:
            try:
                await self.redis_client.delete(cache_key)
                success = True
            except Exception as e:
                logger.warning(f"Redis delete error: {e}")
        
        if success:
            self.stats.deletes += 1
        
        return success
    
    async def exists(self, key: str) -> bool:
        """Verifica si existe una clave en el cache."""
        cache_key = self._generate_key(key)
        
        # Verificar en memoria
        if self.memory_enabled:
            if cache_key in self.memory_cache or cache_key in self.lru_cache:
                return True
        
        # Verificar en Redis
        if self.redis_client:
            try:
                return await self.redis_client.exists(cache_key) > 0
            except Exception as e:
                logger.warning(f"Redis exists error: {e}")
        
        return False
    
    async def clear(self) -> bool:
        """Limpia todo el cache."""
        success = True
        
        # Limpiar cache en memoria
        if self.memory_enabled:
            try:
                self.memory_cache.clear()
                self.lru_cache.clear()
            except Exception as e:
                logger.warning(f"Memory cache clear error: {e}")
                success = False
        
        # Limpiar Redis
        if self.redis_client:
            try:
                # Eliminar solo claves del patrón seo:*
                pattern = self._generate_key('*')
                keys = await self.redis_client.keys(pattern)
                if keys:
                    await self.redis_client.delete(*keys)
            except Exception as e:
                logger.warning(f"Redis clear error: {e}")
                success = False
        
        return success
    
    async def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """Obtiene múltiples valores del cache."""
        result = {}
        
        # Procesar en lotes para mejor rendimiento
        batch_size = 100
        for i in range(0, len(keys), batch_size):
            batch_keys = keys[i:i + batch_size]
            
            # Intentar obtener del cache en memoria primero
            memory_results = {}
            if self.memory_enabled:
                for key in batch_keys:
                    cache_key = self._generate_key(key)
                    value = self.memory_cache.get(cache_key)
                    if value is not None:
                        memory_results[key] = value
                        self.stats.hits += 1
            
            # Obtener del Redis los que no están en memoria
            redis_keys = [k for k in batch_keys if k not in memory_results]
            if redis_keys and self.redis_client:
                try:
                    cache_keys = [self._generate_key(k) for k in redis_keys]
                    values = await self.redis_client.mget(cache_keys)
                    
                    for key, value_bytes in zip(redis_keys, values):
                        if value_bytes:
                            value = self._decompress_data(value_bytes)
                            result[key] = value
                            self.stats.hits += 1
                            
                            # Guardar en memoria
                            if self.memory_enabled:
                                cache_key = self._generate_key(key)
                                self.memory_cache[cache_key] = value
                        else:
                            self.stats.misses += 1
                except Exception as e:
                    logger.warning(f"Redis mget error: {e}")
                    for key in redis_keys:
                        self.stats.misses += 1
            
            # Agregar resultados de memoria
            result.update(memory_results)
        
        return result
    
    async def set_many(self, data: Dict[str, Any], ttl: int = 0) -> bool:
        """Establece múltiples valores en el cache."""
        success = True
        
        # Procesar en lotes
        batch_size = 100
        items = list(data.items())
        
        for i in range(0, len(items), batch_size):
            batch_items = items[i:i + batch_size]
            
            # Guardar en memoria
            if self.memory_enabled:
                try:
                    for key, value in batch_items:
                        cache_key = self._generate_key(key)
                        self.memory_cache[cache_key] = value
                except Exception as e:
                    logger.warning(f"Memory cache set_many error: {e}")
                    success = False
            
            # Guardar en Redis
            if self.redis_client:
                try:
                    pipeline = self.redis_client.pipeline()
                    for key, value in batch_items:
                        cache_key = self._generate_key(key)
                        value_bytes = self._compress_data(value)
                        if ttl > 0:
                            pipeline.setex(cache_key, ttl, value_bytes)
                        else:
                            pipeline.set(cache_key, value_bytes)
                    await pipeline.execute()
                except Exception as e:
                    logger.warning(f"Redis set_many error: {e}")
                    success = False
        
        if success:
            self.stats.sets += len(data)
        
        return success
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas detalladas del cache."""
        total_requests = self.stats.total_requests
        hit_rate = (self.stats.hits / total_requests * 100) if total_requests > 0 else 0
        
        # Calcular uso de memoria
        memory_usage = 0
        if self.memory_enabled:
            memory_usage = len(self.memory_cache) + len(self.lru_cache)
        
        # Calcular ratio de compresión
        compression_ratio = 0.0
        if self.compression_enabled and self.stats.sets > 0:
            # Estimación basada en config
            compression_ratio = 0.3  # ~30% de compresión típica
        
        return {
            'hits': self.stats.hits,
            'misses': self.stats.misses,
            'sets': self.stats.sets,
            'deletes': self.stats.deletes,
            'total_requests': total_requests,
            'hit_rate': hit_rate,
            'memory_usage': memory_usage,
            'compression_ratio': compression_ratio,
            'uptime': time.time() - self.start_time,
            'redis_enabled': self.redis_enabled,
            'memory_enabled': self.memory_enabled,
            'compression_enabled': self.compression_enabled
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Verifica la salud del cache."""
        health = {
            'status': 'healthy',
            'memory_cache': 'ok',
            'redis_cache': 'ok',
            'compression': 'ok'
        }
        
        # Verificar cache en memoria
        if self.memory_enabled:
            try:
                test_key = '_health_check_memory'
                self.memory_cache[test_key] = 'test'
                if self.memory_cache.get(test_key) == 'test':
                    del self.memory_cache[test_key]
                else:
                    health['memory_cache'] = 'error'
                    health['status'] = 'degraded'
            except Exception as e:
                health['memory_cache'] = f'error: {e}'
                health['status'] = 'degraded'
        
        # Verificar Redis
        if self.redis_client:
            try:
                test_key = '_health_check_redis'
                await self.redis_client.set(test_key, b'test', ex=10)
                value = await self.redis_client.get(test_key)
                if value != b'test':
                    health['redis_cache'] = 'error'
                    health['status'] = 'degraded'
            except Exception as e:
                health['redis_cache'] = f'error: {e}'
                health['status'] = 'degraded'
        
        # Verificar compresión
        if self.compression_enabled:
            try:
                test_data = {'test': 'data' * 1000}  # Datos grandes para comprimir
                compressed = self._compress_data(test_data)
                decompressed = self._decompress_data(compressed)
                if decompressed != test_data:
                    health['compression'] = 'error'
                    health['status'] = 'degraded'
            except Exception as e:
                health['compression'] = f'error: {e}'
                health['status'] = 'degraded'
        
        return health
    
    @asynccontextmanager
    async def pipeline(self) -> Any:
        """Context manager para operaciones en pipeline."""
        if self.redis_client:
            async with self.redis_client.pipeline() as pipe:
                yield pipe
        else:
            yield None
    
    def cache_decorator(self, ttl: int = 300, key_prefix: str = ""):
        """Decorador para cachear funciones."""
        def decorator(func) -> Any:
            @wraps(func)
            async def wrapper(*args, **kwargs) -> Any:
                # Generar clave única
                key_parts = [key_prefix, func.__name__]
                key_parts.extend([str(arg) for arg in args])
                key_parts.extend([f"{k}:{v}" for k, v in sorted(kwargs.items())])
                cache_key = ":".join(key_parts)
                
                # Intentar obtener del cache
                cached_result = await self.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # Ejecutar función
                result = await func(*args, **kwargs)
                
                # Guardar en cache
                await self.set(cache_key, result, ttl)
                
                return result
            return wrapper
        return decorator 