from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import time
import asyncio
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from loguru import logger
import orjson
import zstandard as zstd
from cachetools import LRUCache, TTLCache
import aioredis
import diskcache
import hashlib
from contextlib import asynccontextmanager
from .interfaces import CacheInterface
            import json
                        import json
                import json
from typing import Any, List, Dict, Optional
import logging
"""
Ultra-Optimized Cache Manager v2.0
Multi-level caching with memory, Redis, and disk using the fastest libraries
"""




@dataclass
class CacheStats:
    """Estadísticas de cache ultra-optimizado."""
    memory_hits: int = 0
    memory_misses: int = 0
    redis_hits: int = 0
    redis_misses: int = 0
    disk_hits: int = 0
    disk_misses: int = 0
    compression_savings: float = 0.0
    total_operations: int = 0
    average_response_time: float = 0.0


class UltraOptimizedCacheV2(CacheInterface):
    """Cache ultra-optimizado con múltiples niveles."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        
    """__init__ function."""
self.config = config or {}
        
        # Configuraciones de cache
        self.memory_size = self.config.get('memory_size', 1000)
        self.memory_ttl = self.config.get('memory_ttl', 300)  # 5 minutos
        self.redis_url = self.config.get('redis_url', 'redis://localhost:6379/0')
        self.redis_ttl = self.config.get('redis_ttl', 3600)  # 1 hora
        self.disk_path = self.config.get('disk_path', '/tmp/seo-cache')
        self.disk_size = self.config.get('disk_size', 100 * 1024 * 1024)  # 100MB
        self.disk_ttl = self.config.get('disk_ttl', 86400)  # 24 horas
        
        # Configuraciones de compresión
        self.enable_compression = self.config.get('enable_compression', True)
        self.compression_threshold = self.config.get('compression_threshold', 1024)  # 1KB
        self.compression_level = self.config.get('compression_level', 3)
        
        # Configuraciones de serialización
        self.use_orjson = self.config.get('use_orjson', True)
        self.enable_stats = self.config.get('enable_stats', True)
        
        # Inicializar niveles de cache
        self._init_memory_cache()
        self._init_redis_client()
        self._init_disk_cache()
        
        # Compresor Zstandard
        if self.enable_compression:
            self.compressor = zstd.ZstdCompressor(level=self.compression_level)
            self.decompressor = zstd.ZstdDecompressor()
        
        # Estadísticas
        self.stats = CacheStats()
        self.operation_times = []
        
        logger.info("Ultra-Optimized Cache v2.0 initialized")
    
    def _init_memory_cache(self) -> Any:
        """Inicializar cache en memoria."""
        self.memory_cache = TTLCache(
            maxsize=self.memory_size,
            ttl=self.memory_ttl
        )
        logger.info(f"Memory cache initialized: size={self.memory_size}, ttl={self.memory_ttl}s")
    
    def _init_redis_client(self) -> Any:
        """Inicializar cliente Redis."""
        self.redis_client = None
        self.redis_available = False
        
        if self.redis_url:
            try:
                # Redis se inicializará de forma lazy
                logger.info(f"Redis client configured: {self.redis_url}")
            except Exception as e:
                logger.warning(f"Redis configuration failed: {e}")
    
    async def _get_redis_client(self) -> Optional[Dict[str, Any]]:
        """Obtener cliente Redis (lazy initialization)."""
        if self.redis_client is None and self.redis_url:
            try:
                self.redis_client = aioredis.from_url(
                    self.redis_url,
                    encoding="utf-8",
                    decode_responses=False,
                    max_connections=20,
                    retry_on_timeout=True,
                    socket_keepalive=True,
                    socket_keepalive_options={}
                )
                self.redis_available = True
                logger.info("Redis client connected")
            except Exception as e:
                logger.error(f"Redis connection failed: {e}")
                self.redis_available = False
        
        return self.redis_client if self.redis_available else None
    
    def _init_disk_cache(self) -> Any:
        """Inicializar cache en disco."""
        try:
            self.disk_cache = diskcache.Cache(
                directory=self.disk_path,
                size_limit=self.disk_size,
                timeout=1,
                disk_min_file_size=1024,  # 1KB
                disk_pickle_protocol=4
            )
            logger.info(f"Disk cache initialized: path={self.disk_path}, size={self.disk_size}")
        except Exception as e:
            logger.error(f"Disk cache initialization failed: {e}")
            self.disk_cache = None
    
    async def get(self, key: str) -> Optional[Any]:
        """Obtener valor del cache (multi-nivel)."""
        start_time = time.perf_counter()
        
        try:
            # 1. Cache en memoria (más rápido)
            if key in self.memory_cache:
                self.stats.memory_hits += 1
                value = self.memory_cache[key]
                self._record_operation_time(time.perf_counter() - start_time)
                return value
            
            self.stats.memory_misses += 1
            
            # 2. Redis (medio)
            redis_client = await self._get_redis_client()
            if redis_client:
                try:
                    redis_value = await redis_client.get(key)
                    if redis_value is not None:
                        self.stats.redis_hits += 1
                        value = self._deserialize(redis_value)
                        
                        # Promover a memoria
                        self.memory_cache[key] = value
                        
                        self._record_operation_time(time.perf_counter() - start_time)
                        return value
                    
                    self.stats.redis_misses += 1
                except Exception as e:
                    logger.warning(f"Redis get failed: {e}")
                    self.stats.redis_misses += 1
            
            # 3. Cache en disco (más lento)
            if self.disk_cache and key in self.disk_cache:
                try:
                    self.stats.disk_hits += 1
                    value = self.disk_cache[key]
                    
                    # Promover a memoria y Redis
                    self.memory_cache[key] = value
                    if redis_client:
                        try:
                            await redis_client.setex(
                                key, 
                                self.redis_ttl, 
                                self._serialize(value)
                            )
                        except Exception:
                            pass
                    
                    self._record_operation_time(time.perf_counter() - start_time)
                    return value
                
                except Exception as e:
                    logger.warning(f"Disk cache get failed: {e}")
                    self.stats.disk_misses += 1
            else:
                self.stats.disk_misses += 1
            
            self._record_operation_time(time.perf_counter() - start_time)
            return None
            
        except Exception as e:
            logger.error(f"Cache get failed: {e}")
            self._record_operation_time(time.perf_counter() - start_time)
            return None
    
    async def set(self, key: str, value: Any, ttl: int = 0) -> bool:
        """Establecer valor en cache (multi-nivel)."""
        start_time = time.perf_counter()
        
        try:
            # Determinar TTL
            if ttl == 0:
                ttl = self.redis_ttl
            
            # 1. Cache en memoria
            self.memory_cache[key] = value
            
            # 2. Redis (async)
            redis_client = await self._get_redis_client()
            if redis_client:
                try:
                    serialized_value = self._serialize(value)
                    await redis_client.setex(key, ttl, serialized_value)
                except Exception as e:
                    logger.warning(f"Redis set failed: {e}")
            
            # 3. Cache en disco (para persistencia)
            if self.disk_cache:
                try:
                    self.disk_cache.set(key, value, expire=ttl)
                except Exception as e:
                    logger.warning(f"Disk cache set failed: {e}")
            
            self._record_operation_time(time.perf_counter() - start_time)
            return True
            
        except Exception as e:
            logger.error(f"Cache set failed: {e}")
            self._record_operation_time(time.perf_counter() - start_time)
            return False
    
    async def delete(self, key: str) -> bool:
        """Eliminar valor del cache (multi-nivel)."""
        start_time = time.perf_counter()
        
        try:
            # 1. Cache en memoria
            if key in self.memory_cache:
                del self.memory_cache[key]
            
            # 2. Redis
            redis_client = await self._get_redis_client()
            if redis_client:
                try:
                    await redis_client.delete(key)
                except Exception as e:
                    logger.warning(f"Redis delete failed: {e}")
            
            # 3. Cache en disco
            if self.disk_cache and key in self.disk_cache:
                try:
                    del self.disk_cache[key]
                except Exception as e:
                    logger.warning(f"Disk cache delete failed: {e}")
            
            self._record_operation_time(time.perf_counter() - start_time)
            return True
            
        except Exception as e:
            logger.error(f"Cache delete failed: {e}")
            self._record_operation_time(time.perf_counter() - start_time)
            return False
    
    async def exists(self, key: str) -> bool:
        """Verificar si existe clave en cache."""
        # Verificar memoria primero (más rápido)
        if key in self.memory_cache:
            return True
        
        # Verificar Redis
        redis_client = await self._get_redis_client()
        if redis_client:
            try:
                return await redis_client.exists(key) > 0
            except Exception:
                pass
        
        # Verificar disco
        if self.disk_cache:
            return key in self.disk_cache
        
        return False
    
    async def clear(self, pattern: str = None) -> bool:
        """Limpiar cache."""
        try:
            # Limpiar memoria
            self.memory_cache.clear()
            
            # Limpiar Redis
            redis_client = await self._get_redis_client()
            if redis_client and pattern:
                try:
                    keys = await redis_client.keys(pattern)
                    if keys:
                        await redis_client.delete(*keys)
                except Exception as e:
                    logger.warning(f"Redis clear failed: {e}")
            elif redis_client:
                try:
                    await redis_client.flushdb()
                except Exception as e:
                    logger.warning(f"Redis flush failed: {e}")
            
            # Limpiar disco
            if self.disk_cache:
                try:
                    self.disk_cache.clear()
                except Exception as e:
                    logger.warning(f"Disk cache clear failed: {e}")
            
            logger.info("Cache cleared successfully")
            return True
            
        except Exception as e:
            logger.error(f"Cache clear failed: {e}")
            return False
    
    async def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """Obtener múltiples valores del cache."""
        results = {}
        
        # Procesar en paralelo
        tasks = [self.get(key) for key in keys]
        values = await asyncio.gather(*tasks, return_exceptions=True)
        
        for key, value in zip(keys, values):
            if not isinstance(value, Exception) and value is not None:
                results[key] = value
        
        return results
    
    async def set_many(self, data: Dict[str, Any], ttl: int = 0) -> bool:
        """Establecer múltiples valores en cache."""
        try:
            # Procesar en paralelo
            tasks = [self.set(key, value, ttl) for key, value in data.items()]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Verificar si todos fueron exitosos
            return all(not isinstance(result, Exception) and result for result in results)
            
        except Exception as e:
            logger.error(f"Cache set_many failed: {e}")
            return False
    
    async def pipeline(self) -> Any:
        """Obtener pipeline de Redis para operaciones en lote."""
        redis_client = await self._get_redis_client()
        if redis_client:
            return redis_client.pipeline()
        return None
    
    def _serialize(self, data: Any) -> bytes:
        """Serializar datos con compresión opcional."""
        if self.use_orjson:
            json_data = orjson.dumps(data)
        else:
            json_data = json.dumps(data).encode('utf-8')
        
        # Comprimir si está habilitado y el tamaño supera el umbral
        if (self.enable_compression and 
            len(json_data) > self.compression_threshold):
            try:
                compressed_data = self.compressor.compress(json_data)
                compression_ratio = (len(json_data) - len(compressed_data)) / len(json_data)
                self.stats.compression_savings += compression_ratio
                return compressed_data
            except Exception as e:
                logger.warning(f"Compression failed: {e}")
        
        return json_data
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserializar datos con descompresión automática."""
        try:
            # Intentar descomprimir primero
            if self.enable_compression:
                try:
                    decompressed_data = self.decompressor.decompress(data)
                    if self.use_orjson:
                        return orjson.loads(decompressed_data)
                    else:
                        return json.loads(decompressed_data.decode('utf-8'))
                except Exception:
                    # Si falla la descompresión, intentar como JSON normal
                    pass
            
            # Deserializar como JSON normal
            if self.use_orjson:
                return orjson.loads(data)
            else:
                return json.loads(data.decode('utf-8'))
                
        except Exception as e:
            logger.error(f"Deserialization failed: {e}")
            return None
    
    def _record_operation_time(self, duration: float):
        """Registrar tiempo de operación para estadísticas."""
        self.stats.total_operations += 1
        self.operation_times.append(duration)
        
        # Mantener solo los últimos 1000 tiempos
        if len(self.operation_times) > 1000:
            self.operation_times = self.operation_times[-1000:]
        
        # Calcular promedio
        self.stats.average_response_time = sum(self.operation_times) / len(self.operation_times)
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del cache."""
        total_hits = (self.stats.memory_hits + 
                     self.stats.redis_hits + 
                     self.stats.disk_hits)
        
        total_requests = (self.stats.memory_hits + self.stats.memory_misses +
                         self.stats.redis_hits + self.stats.redis_misses +
                         self.stats.disk_hits + self.stats.disk_misses)
        
        hit_ratio = total_hits / max(total_requests, 1)
        
        return {
            'cache_type': 'ultra_optimized_v2',
            'memory_cache_size': len(self.memory_cache),
            'memory_cache_maxsize': self.memory_size,
            'redis_available': self.redis_available,
            'disk_cache_size': len(self.disk_cache) if self.disk_cache else 0,
            'total_operations': self.stats.total_operations,
            'hit_ratio': hit_ratio,
            'memory_hit_ratio': self.stats.memory_hits / max(self.stats.memory_hits + self.stats.memory_misses, 1),
            'redis_hit_ratio': self.stats.redis_hits / max(self.stats.redis_hits + self.stats.redis_misses, 1),
            'disk_hit_ratio': self.stats.disk_hits / max(self.stats.disk_hits + self.stats.disk_misses, 1),
            'average_response_time': self.stats.average_response_time,
            'compression_savings': self.stats.compression_savings,
            'compression_enabled': self.enable_compression,
            'compression_threshold': self.compression_threshold
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check del cache."""
        health_status = {
            'status': 'healthy',
            'cache_type': 'ultra_optimized_v2',
            'memory_cache': 'ok',
            'redis_cache': 'unknown',
            'disk_cache': 'unknown'
        }
        
        # Verificar Redis
        redis_client = await self._get_redis_client()
        if redis_client:
            try:
                await redis_client.ping()
                health_status['redis_cache'] = 'ok'
            except Exception:
                health_status['redis_cache'] = 'error'
                health_status['status'] = 'degraded'
        
        # Verificar disco
        if self.disk_cache:
            try:
                # Test write/read
                test_key = '__health_check__'
                test_value = {'timestamp': time.time()}
                self.disk_cache[test_key] = test_value
                retrieved_value = self.disk_cache[test_key]
                del self.disk_cache[test_key]
                
                if retrieved_value == test_value:
                    health_status['disk_cache'] = 'ok'
                else:
                    health_status['disk_cache'] = 'error'
                    health_status['status'] = 'degraded'
            except Exception:
                health_status['disk_cache'] = 'error'
                health_status['status'] = 'degraded'
        
        return health_status
    
    async def close(self) -> Any:
        """Cerrar conexiones del cache."""
        try:
            # Cerrar Redis
            if self.redis_client:
                await self.redis_client.close()
                self.redis_client = None
            
            # Cerrar disco cache
            if self.disk_cache:
                self.disk_cache.close()
                self.disk_cache = None
            
            logger.info("Cache connections closed")
            
        except Exception as e:
            logger.error(f"Error closing cache: {e}")
    
    async def __aenter__(self) -> Any:
        """Context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> Any:
        """Context manager exit."""
        await self.close() 