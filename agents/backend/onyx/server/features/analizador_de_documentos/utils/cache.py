"""
Sistema de Caché para Analizador de Documentos
===============================================

Sistema de caché inteligente con múltiples backends (memoria, Redis, disco)
para optimizar el rendimiento.
"""

import os
import hashlib
import json
import pickle
import logging
from typing import Any, Optional, Dict, Callable
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
import time

logger = logging.getLogger(__name__)


class CacheBackend:
    """Interfaz para backends de caché"""
    
    def get(self, key: str) -> Optional[Any]:
        """Obtener valor del caché"""
        raise NotImplementedError
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Guardar valor en caché"""
        raise NotImplementedError
    
    def delete(self, key: str):
        """Eliminar valor del caché"""
        raise NotImplementedError
    
    def clear(self):
        """Limpiar todo el caché"""
        raise NotImplementedError
    
    def exists(self, key: str) -> bool:
        """Verificar si existe una clave"""
        raise NotImplementedError


class MemoryCache(CacheBackend):
    """Caché en memoria"""
    
    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, tuple] = {}
        self.max_size = max_size
        self.access_times: Dict[str, float] = {}
    
    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            value, expiry = self.cache[key]
            if expiry is None or time.time() < expiry:
                self.access_times[key] = time.time()
                return value
            else:
                del self.cache[key]
                if key in self.access_times:
                    del self.access_times[key]
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        # LRU eviction si está lleno
        if len(self.cache) >= self.max_size and key not in self.cache:
            # Eliminar el menos usado recientemente
            if self.access_times:
                lru_key = min(self.access_times, key=self.access_times.get)
                del self.cache[lru_key]
                del self.access_times[lru_key]
        
        expiry = None
        if ttl:
            expiry = time.time() + ttl
        
        self.cache[key] = (value, expiry)
        self.access_times[key] = time.time()
    
    def delete(self, key: str):
        if key in self.cache:
            del self.cache[key]
        if key in self.access_times:
            del self.access_times[key]
    
    def clear(self):
        self.cache.clear()
        self.access_times.clear()
    
    def exists(self, key: str) -> bool:
        return self.get(key) is not None


class DiskCache(CacheBackend):
    """Caché en disco"""
    
    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_file_size = 10 * 1024 * 1024  # 10MB
    
    def _get_path(self, key: str) -> Path:
        """Obtener ruta del archivo de caché"""
        # Usar hash para nombres de archivo seguros
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"
    
    def get(self, key: str) -> Optional[Any]:
        cache_file = self._get_path(key)
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
            
            # Verificar TTL
            if data.get("expiry") and time.time() > data["expiry"]:
                cache_file.unlink()
                return None
            
            return data.get("value")
        except Exception as e:
            logger.warning(f"Error leyendo caché {key}: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        cache_file = self._get_path(key)
        
        try:
            expiry = None
            if ttl:
                expiry = time.time() + ttl
            
            data = {
                "value": value,
                "expiry": expiry,
                "created": time.time()
            }
            
            with open(cache_file, "wb") as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.warning(f"Error guardando en caché {key}: {e}")
    
    def delete(self, key: str):
        cache_file = self._get_path(key)
        if cache_file.exists():
            cache_file.unlink()
    
    def clear(self):
        for cache_file in self.cache_dir.glob("*.cache"):
            cache_file.unlink()
    
    def exists(self, key: str) -> bool:
        return self._get_path(key).exists()


class RedisCache(CacheBackend):
    """Caché usando Redis (opcional)"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        try:
            import redis
            self.redis_client = redis.from_url(redis_url, decode_responses=False)
            self.available = True
        except ImportError:
            logger.warning("Redis no disponible, usando caché en memoria")
            self.available = False
    
    def get(self, key: str) -> Optional[Any]:
        if not self.available:
            return None
        
        try:
            data = self.redis_client.get(key)
            if data:
                return pickle.loads(data)
        except Exception as e:
            logger.warning(f"Error obteniendo de Redis {key}: {e}")
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        if not self.available:
            return
        
        try:
            data = pickle.dumps(value)
            if ttl:
                self.redis_client.setex(key, ttl, data)
            else:
                self.redis_client.set(key, data)
        except Exception as e:
            logger.warning(f"Error guardando en Redis {key}: {e}")
    
    def delete(self, key: str):
        if self.available:
            try:
                self.redis_client.delete(key)
            except Exception as e:
                logger.warning(f"Error eliminando de Redis {key}: {e}")
    
    def clear(self):
        if self.available:
            try:
                self.redis_client.flushdb()
            except Exception as e:
                logger.warning(f"Error limpiando Redis: {e}")
    
    def exists(self, key: str) -> bool:
        if not self.available:
            return False
        try:
            return self.redis_client.exists(key) > 0
        except:
            return False


class CacheManager:
    """Gestor de caché con múltiples backends"""
    
    def __init__(
        self,
        backend: str = "memory",
        **backend_kwargs
    ):
        """
        Inicializar gestor de caché
        
        Args:
            backend: Tipo de backend ('memory', 'disk', 'redis', 'auto')
            **backend_kwargs: Argumentos para el backend
        """
        self.backend_type = backend
        
        if backend == "memory":
            self.backend = MemoryCache(**backend_kwargs)
        elif backend == "disk":
            self.backend = DiskCache(**backend_kwargs)
        elif backend == "redis":
            self.backend = RedisCache(**backend_kwargs)
        elif backend == "auto":
            # Intentar Redis, luego disco, luego memoria
            try:
                self.backend = RedisCache(**backend_kwargs)
                if self.backend.available:
                    logger.info("Usando Redis como backend de caché")
                else:
                    raise
            except:
                try:
                    self.backend = DiskCache(**backend_kwargs)
                    logger.info("Usando disco como backend de caché")
                except:
                    self.backend = MemoryCache(**backend_kwargs)
                    logger.info("Usando memoria como backend de caché")
        else:
            raise ValueError(f"Backend no soportado: {backend}")
        
        logger.info(f"CacheManager inicializado con backend: {backend}")
    
    def get(self, key: str) -> Optional[Any]:
        """Obtener valor del caché"""
        return self.backend.get(key)
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Guardar valor en caché"""
        self.backend.set(key, value, ttl)
    
    def delete(self, key: str):
        """Eliminar valor del caché"""
        self.backend.delete(key)
    
    def clear(self):
        """Limpiar todo el caché"""
        self.backend.clear()
    
    def exists(self, key: str) -> bool:
        """Verificar si existe una clave"""
        return self.backend.exists(key)
    
    def generate_key(self, prefix: str, *args, **kwargs) -> str:
        """Generar clave de caché única"""
        key_parts = [prefix]
        
        # Agregar argumentos posicionales
        for arg in args:
            if isinstance(arg, (str, int, float, bool)):
                key_parts.append(str(arg))
            else:
                key_parts.append(hashlib.md5(str(arg).encode()).hexdigest()[:8])
        
        # Agregar argumentos keyword
        if kwargs:
            sorted_kwargs = sorted(kwargs.items())
            for k, v in sorted_kwargs:
                if isinstance(v, (str, int, float, bool)):
                    key_parts.append(f"{k}:{v}")
                else:
                    key_parts.append(f"{k}:{hashlib.md5(str(v).encode()).hexdigest()[:8]}")
        
        return ":".join(key_parts)


def cached(
    ttl: int = 3600,
    key_prefix: str = "cache",
    cache_manager: Optional[CacheManager] = None
):
    """
    Decorator para cachear resultados de funciones
    
    Args:
        ttl: Tiempo de vida del caché en segundos
        key_prefix: Prefijo para las claves
        cache_manager: Instancia de CacheManager (opcional)
    """
    if cache_manager is None:
        cache_manager = CacheManager()
    
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Generar clave de caché
            cache_key = cache_manager.generate_key(
                key_prefix,
                func.__name__,
                *args,
                **kwargs
            )
            
            # Intentar obtener del caché
            cached_result = cache_manager.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit para {cache_key}")
                return cached_result
            
            # Ejecutar función
            logger.debug(f"Cache miss para {cache_key}")
            result = await func(*args, **kwargs)
            
            # Guardar en caché
            cache_manager.set(cache_key, result, ttl=ttl)
            
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Generar clave de caché
            cache_key = cache_manager.generate_key(
                key_prefix,
                func.__name__,
                *args,
                **kwargs
            )
            
            # Intentar obtener del caché
            cached_result = cache_manager.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit para {cache_key}")
                return cached_result
            
            # Ejecutar función
            logger.debug(f"Cache miss para {cache_key}")
            result = func(*args, **kwargs)
            
            # Guardar en caché
            cache_manager.set(cache_key, result, ttl=ttl)
            
            return result
        
        # Retornar wrapper apropiado
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Instancia global de caché
_default_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Obtener instancia global de CacheManager"""
    global _default_cache_manager
    if _default_cache_manager is None:
        backend = os.getenv("CACHE_BACKEND", "auto")
        _default_cache_manager = CacheManager(backend=backend)
    return _default_cache_manager
















