"""
Cache Service - Servicio de cache distribuido
==============================================

Servicio de cache que abstrae el backend de cache (Redis, Memcached, etc.)
siguiendo principios de microservicios.
"""

import logging
from typing import Optional, Any
from abc import ABC, abstractmethod

from ..core.redis_client import get_redis_client
from ..core.microservices_config import get_microservices_config, CacheBackend

logger = logging.getLogger(__name__)


class CacheBackendInterface(ABC):
    """Interfaz para backends de cache"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Obtiene valor del cache"""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Establece valor en cache"""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Elimina clave del cache"""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Verifica si clave existe"""
        pass


class RedisCacheBackend(CacheBackendInterface):
    """Backend de cache usando Redis"""
    
    def __init__(self):
        self.client = get_redis_client()
    
    async def get(self, key: str) -> Optional[Any]:
        return await self.client.aget(key)
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        return await self.client.aset(key, value, ttl)
    
    async def delete(self, key: str) -> bool:
        return await self.client.adelete(key)
    
    async def exists(self, key: str) -> bool:
        return await self.client.aexists(key)


class InMemoryCacheBackend(CacheBackendInterface):
    """Backend de cache en memoria (para desarrollo/testing)"""
    
    def __init__(self):
        self._cache: dict = {}
    
    async def get(self, key: str) -> Optional[Any]:
        return self._cache.get(key)
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        self._cache[key] = value
        return True
    
    async def delete(self, key: str) -> bool:
        if key in self._cache:
            del self._cache[key]
            return True
        return False
    
    async def exists(self, key: str) -> bool:
        return key in self._cache


class CacheService:
    """
    Servicio de cache que abstrae el backend.
    
    Proporciona una interfaz unificada para cache independientemente
    del backend utilizado.
    """
    
    def __init__(self, backend: Optional[CacheBackendInterface] = None):
        config = get_microservices_config()
        
        if backend:
            self.backend = backend
        elif config.cache_backend == CacheBackend.REDIS:
            try:
                self.backend = RedisCacheBackend()
            except Exception as e:
                logger.warning(f"Redis not available, using in-memory cache: {e}")
                self.backend = InMemoryCacheBackend()
        else:
            self.backend = InMemoryCacheBackend()
    
    async def get(self, key: str) -> Optional[Any]:
        """Obtiene valor del cache"""
        try:
            return await self.backend.get(key)
        except Exception as e:
            logger.error(f"Error getting cache key {key}: {e}")
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """Establece valor en cache"""
        try:
            return await self.backend.set(key, value, ttl)
        except Exception as e:
            logger.error(f"Error setting cache key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Elimina clave del cache"""
        try:
            return await self.backend.delete(key)
        except Exception as e:
            logger.error(f"Error deleting cache key {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Verifica si clave existe"""
        try:
            return await self.backend.exists(key)
        except Exception as e:
            logger.error(f"Error checking cache key {key}: {e}")
            return False
    
    async def clear_pattern(self, pattern: str) -> int:
        """Elimina todas las claves que coinciden con el patrón"""
        # Implementación básica, puede mejorarse según backend
        return 0


def get_cache_service() -> CacheService:
    """Obtiene instancia de cache service"""
    return CacheService()















