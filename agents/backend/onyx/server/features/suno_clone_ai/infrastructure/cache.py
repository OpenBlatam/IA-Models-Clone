"""
Cache Infrastructure
Abstracción para diferentes backends de caché
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class CacheType(Enum):
    """Tipos de caché soportados"""
    MEMORY = "memory"
    REDIS = "redis"
    DYNAMODB = "dynamodb"
    S3 = "s3"


class CacheManager(ABC):
    """Interfaz abstracta para gestión de caché"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Obtiene un valor del caché"""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Establece un valor en el caché"""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Elimina un valor del caché"""
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """Limpia todo el caché"""
        pass


class MemoryCacheManager(CacheManager):
    """Implementación de caché en memoria"""
    
    def __init__(self):
        self._cache: dict = {}
        self._ttl: dict = {}
        import asyncio
        self._loop = asyncio.get_event_loop()
    
    async def get(self, key: str) -> Optional[Any]:
        """Obtiene un valor del caché en memoria"""
        if key in self._cache:
            # Verificar TTL
            if key in self._ttl:
                import time
                if time.time() > self._ttl[key]:
                    del self._cache[key]
                    del self._ttl[key]
                    return None
            return self._cache[key]
        return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Establece un valor en el caché en memoria"""
        import time
        self._cache[key] = value
        if ttl > 0:
            self._ttl[key] = time.time() + ttl
        return True
    
    async def delete(self, key: str) -> bool:
        """Elimina un valor del caché en memoria"""
        if key in self._cache:
            del self._cache[key]
            if key in self._ttl:
                del self._ttl[key]
            return True
        return False
    
    async def clear(self) -> bool:
        """Limpia todo el caché en memoria"""
        self._cache.clear()
        self._ttl.clear()
        return True


class RedisCacheManager(CacheManager):
    """Implementación de caché con Redis"""
    
    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self._client = None
    
    async def connect(self):
        """Conecta a Redis"""
        try:
            import redis.asyncio as redis
            
            self._client = await redis.from_url(
                self.redis_url,
                decode_responses=True
            )
            logger.info("Connected to Redis")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    async def get(self, key: str) -> Optional[Any]:
        """Obtiene un valor de Redis"""
        if not self._client:
            await self.connect()
        
        import json
        value = await self._client.get(key)
        if value:
            try:
                return json.loads(value)
            except:
                return value
        return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Establece un valor en Redis"""
        if not self._client:
            await self.connect()
        
        import json
        serialized = json.dumps(value) if not isinstance(value, str) else value
        await self._client.setex(key, ttl, serialized)
        return True
    
    async def delete(self, key: str) -> bool:
        """Elimina un valor de Redis"""
        if not self._client:
            await self.connect()
        
        result = await self._client.delete(key)
        return result > 0
    
    async def clear(self) -> bool:
        """Limpia todo el caché de Redis"""
        if not self._client:
            await self.connect()
        
        await self._client.flushdb()
        return True


# Factory function
_cache_manager: Optional[CacheManager] = None


def get_cache() -> Optional[CacheManager]:
    """Obtiene la instancia global del gestor de caché"""
    return _cache_manager


def create_cache_manager(cache_type: CacheType, **kwargs) -> CacheManager:
    """Crea un gestor de caché según el tipo"""
    if cache_type == CacheType.MEMORY:
        return MemoryCacheManager()
    elif cache_type == CacheType.REDIS:
        redis_url = kwargs.get('redis_url', 'redis://localhost:6379/0')
        return RedisCacheManager(redis_url)
    else:
        raise ValueError(f"Unsupported cache type: {cache_type}")















