"""
Cache Strategy - Estrategias de cache
=====================================

Implementa el patrón Strategy para diferentes backends de cache.
"""

from abc import ABC, abstractmethod
from typing import Optional, Any
import logging

logger = logging.getLogger(__name__)


class CacheStrategy(ABC):
    """Interfaz para estrategias de cache"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Obtiene valor"""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Establece valor"""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Elimina clave"""
        pass


class RedisCacheStrategy(CacheStrategy):
    """Estrategia de cache usando Redis"""
    
    def __init__(self, redis_client):
        self.redis_client = redis_client
    
    async def get(self, key: str) -> Optional[Any]:
        """Obtiene valor de Redis"""
        return await self.redis_client.aget(key)
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Establece valor en Redis"""
        return await self.redis_client.aset(key, value, ttl)
    
    async def delete(self, key: str) -> bool:
        """Elimina clave de Redis"""
        return await self.redis_client.adelete(key)


class MemoryCacheStrategy(CacheStrategy):
    """Estrategia de cache en memoria"""
    
    def __init__(self):
        self._cache: dict = {}
    
    async def get(self, key: str) -> Optional[Any]:
        """Obtiene valor de memoria"""
        return self._cache.get(key)
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Establece valor en memoria"""
        self._cache[key] = value
        return True
    
    async def delete(self, key: str) -> bool:
        """Elimina clave de memoria"""
        if key in self._cache:
            del self._cache[key]
            return True
        return False















