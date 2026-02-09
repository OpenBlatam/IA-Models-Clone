"""
Routing Cache Optimizations
============================

Optimizaciones avanzadas de cache y almacenamiento.
Incluye: Distributed cache, Redis integration, Cache warming, etc.
"""

import logging
import hashlib
import pickle
from typing import Dict, Any, List, Optional, Tuple
from collections import OrderedDict
import time
import threading

logger = logging.getLogger(__name__)

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not available, distributed cache disabled")

try:
    import memcached
    MEMCACHED_AVAILABLE = True
except ImportError:
    MEMCACHED_AVAILABLE = False


class DistributedCache:
    """Cache distribuido usando Redis."""
    
    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379, redis_db: int = 0):
        """
        Inicializar cache distribuido.
        
        Args:
            redis_host: Host de Redis
            redis_port: Puerto de Redis
            redis_db: Base de datos de Redis
        """
        if not REDIS_AVAILABLE:
            raise ImportError("Redis not available")
        
        try:
            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                decode_responses=False
            )
            # Test connection
            self.redis_client.ping()
            logger.info(f"Redis connection established: {redis_host}:{redis_port}")
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}")
            self.redis_client = None
    
    def get(self, key: str) -> Optional[Any]:
        """
        Obtener valor del cache.
        
        Args:
            key: Clave del cache
        
        Returns:
            Valor o None
        """
        if not self.redis_client:
            return None
        
        try:
            value = self.redis_client.get(key)
            if value:
                return pickle.loads(value)
            return None
        except Exception as e:
            logger.debug(f"Redis get error: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: int = 3600):
        """
        Guardar valor en cache.
        
        Args:
            key: Clave del cache
            value: Valor a guardar
            ttl: Time to live en segundos
        """
        if not self.redis_client:
            return
        
        try:
            serialized = pickle.dumps(value)
            self.redis_client.setex(key, ttl, serialized)
        except Exception as e:
            logger.debug(f"Redis set error: {e}")
    
    def delete(self, key: str):
        """Eliminar clave del cache."""
        if not self.redis_client:
            return
        
        try:
            self.redis_client.delete(key)
        except Exception as e:
            logger.debug(f"Redis delete error: {e}")
    
    def clear_pattern(self, pattern: str):
        """Eliminar todas las claves que coincidan con el patrón."""
        if not self.redis_client:
            return
        
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                self.redis_client.delete(*keys)
        except Exception as e:
            logger.debug(f"Redis clear pattern error: {e}")


class CacheWarmer:
    """Precalentador de cache para mejor rendimiento."""
    
    def __init__(self, cache: Any):
        """
        Inicializar cache warmer.
        
        Args:
            cache: Instancia de cache a precalentar
        """
        self.cache = cache
        self.warming_queue: List[Tuple[str, Any]] = []
        self.lock = threading.Lock()
    
    def add_to_warmup(self, key: str, value: Any):
        """
        Agregar item a la cola de precalentamiento.
        
        Args:
            key: Clave del cache
            value: Valor a precalentar
        """
        with self.lock:
            self.warming_queue.append((key, value))
    
    def warm_cache(self, batch_size: int = 100):
        """
        Precalentar cache con items en cola.
        
        Args:
            batch_size: Tamaño del batch
        """
        with self.lock:
            batch = self.warming_queue[:batch_size]
            self.warming_queue = self.warming_queue[batch_size:]
        
        for key, value in batch:
            try:
                if hasattr(self.cache, 'set'):
                    self.cache.set(key, value)
                elif hasattr(self.cache, 'put'):
                    self.cache.put(key, value)
            except Exception as e:
                logger.debug(f"Cache warmup error for {key}: {e}")


class MultiLevelCache:
    """Cache multi-nivel (L1: memoria, L2: distribuido)."""
    
    def __init__(self, l1_max_size: int = 1000, l2_cache: Optional[DistributedCache] = None):
        """
        Inicializar cache multi-nivel.
        
        Args:
            l1_max_size: Tamaño máximo del cache L1 (memoria)
            l2_cache: Cache L2 (distribuido)
        """
        self.l1_cache: OrderedDict = OrderedDict()
        self.l1_max_size = l1_max_size
        self.l2_cache = l2_cache
        self.lock = threading.Lock()
        self.stats = {
            'l1_hits': 0,
            'l1_misses': 0,
            'l2_hits': 0,
            'l2_misses': 0
        }
    
    def get(self, key: str) -> Optional[Any]:
        """Obtener valor del cache multi-nivel."""
        with self.lock:
            # Intentar L1 (memoria)
            if key in self.l1_cache:
                # Mover al final (LRU)
                self.l1_cache.move_to_end(key)
                self.stats['l1_hits'] += 1
                return self.l1_cache[key]
            
            self.stats['l1_misses'] += 1
            
            # Intentar L2 (distribuido)
            if self.l2_cache:
                value = self.l2_cache.get(key)
                if value is not None:
                    # Guardar en L1 para acceso futuro
                    self._put_l1(key, value)
                    self.stats['l2_hits'] += 1
                    return value
                self.stats['l2_misses'] += 1
            
            return None
    
    def _put_l1(self, key: str, value: Any):
        """Guardar en cache L1."""
        if len(self.l1_cache) >= self.l1_max_size:
            # Evictar LRU
            self.l1_cache.popitem(last=False)
        
        self.l1_cache[key] = value
    
    def put(self, key: str, value: Any, ttl: int = 3600):
        """Guardar valor en ambos niveles."""
        with self.lock:
            # Guardar en L1
            self._put_l1(key, value)
            
            # Guardar en L2
            if self.l2_cache:
                self.l2_cache.set(key, value, ttl)
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del cache."""
        total_requests = sum(self.stats.values())
        l1_hit_rate = self.stats['l1_hits'] / max(total_requests, 1)
        l2_hit_rate = self.stats['l2_hits'] / max(total_requests, 1)
        
        return {
            'l1_size': len(self.l1_cache),
            'l1_max_size': self.l1_max_size,
            'l1_hits': self.stats['l1_hits'],
            'l1_misses': self.stats['l1_misses'],
            'l1_hit_rate': l1_hit_rate,
            'l2_hits': self.stats['l2_hits'],
            'l2_misses': self.stats['l2_misses'],
            'l2_hit_rate': l2_hit_rate,
            'total_requests': total_requests
        }


class CacheOptimizer:
    """Optimizador completo de cache."""
    
    def __init__(self, use_distributed: bool = False, redis_host: str = "localhost"):
        """
        Inicializar optimizador de cache.
        
        Args:
            use_distributed: Usar cache distribuido
            redis_host: Host de Redis
        """
        self.distributed_cache = None
        if use_distributed and REDIS_AVAILABLE:
            try:
                self.distributed_cache = DistributedCache(redis_host=redis_host)
            except Exception as e:
                logger.warning(f"Failed to initialize distributed cache: {e}")
        
        self.multi_level_cache = MultiLevelCache(l2_cache=self.distributed_cache)
        self.cache_warmer = CacheWarmer(self.multi_level_cache)
    
    def optimize_cache(self):
        """Optimizar configuración de cache."""
        logger.info("Cache optimizations configured")
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        stats = self.multi_level_cache.get_stats()
        stats['distributed_cache_enabled'] = self.distributed_cache is not None
        return stats

