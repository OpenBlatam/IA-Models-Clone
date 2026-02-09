"""
Caching distribuido con Redis para Robot Movement AI v2.0
"""

import json
from typing import Any, Optional
from datetime import timedelta

try:
    import redis
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None
    aioredis = None


class RedisCache:
    """Cache distribuido usando Redis"""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        password: Optional[str] = None,
        db: int = 0,
        decode_responses: bool = True
    ):
        """
        Inicializar cache Redis
        
        Args:
            host: Host de Redis
            port: Puerto de Redis
            password: Contraseña (opcional)
            db: Base de datos de Redis
            decode_responses: Decodificar respuestas como strings
        """
        if not REDIS_AVAILABLE:
            raise ImportError("redis package is required. Install with: pip install redis")
        
        self.host = host
        self.port = port
        self.password = password
        self.db = db
        self.decode_responses = decode_responses
        self._client: Optional[redis.Redis] = None
        self._async_client: Optional[aioredis.Redis] = None
    
    def _get_client(self) -> redis.Redis:
        """Obtener cliente Redis síncrono"""
        if self._client is None:
            self._client = redis.Redis(
                host=self.host,
                port=self.port,
                password=self.password,
                db=self.db,
                decode_responses=self.decode_responses
            )
        return self._client
    
    async def _get_async_client(self) -> aioredis.Redis:
        """Obtener cliente Redis async"""
        if self._async_client is None:
            self._async_client = aioredis.Redis(
                host=self.host,
                port=self.port,
                password=self.password,
                db=self.db,
                decode_responses=self.decode_responses
            )
        return self._async_client
    
    def get(self, key: str) -> Optional[Any]:
        """Obtener valor del cache"""
        client = self._get_client()
        value = client.get(key)
        
        if value is None:
            return None
        
        # Intentar deserializar JSON
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value
    
    async def aget(self, key: str) -> Optional[Any]:
        """Obtener valor del cache (async)"""
        client = await self._get_async_client()
        value = await client.get(key)
        
        if value is None:
            return None
        
        # Intentar deserializar JSON
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        ttl_timedelta: Optional[timedelta] = None
    ):
        """Establecer valor en cache"""
        client = self._get_client()
        
        # Serializar si es necesario
        if not isinstance(value, (str, int, float)):
            value = json.dumps(value)
        
        # Calcular TTL
        if ttl_timedelta:
            ttl = int(ttl_timedelta.total_seconds())
        
        if ttl:
            client.setex(key, ttl, value)
        else:
            client.set(key, value)
    
    async def aset(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        ttl_timedelta: Optional[timedelta] = None
    ):
        """Establecer valor en cache (async)"""
        client = await self._get_async_client()
        
        # Serializar si es necesario
        if not isinstance(value, (str, int, float)):
            value = json.dumps(value)
        
        # Calcular TTL
        if ttl_timedelta:
            ttl = int(ttl_timedelta.total_seconds())
        
        if ttl:
            await client.setex(key, ttl, value)
        else:
            await client.set(key, value)
    
    def delete(self, key: str):
        """Eliminar clave del cache"""
        client = self._get_client()
        client.delete(key)
    
    async def adelete(self, key: str):
        """Eliminar clave del cache (async)"""
        client = await self._get_async_client()
        await client.delete(key)
    
    def exists(self, key: str) -> bool:
        """Verificar si existe clave"""
        client = self._get_client()
        return bool(client.exists(key))
    
    async def aexists(self, key: str) -> bool:
        """Verificar si existe clave (async)"""
        client = await self._get_async_client()
        return bool(await client.exists(key))
    
    def clear_pattern(self, pattern: str):
        """Eliminar todas las claves que coincidan con patrón"""
        client = self._get_client()
        keys = client.keys(pattern)
        if keys:
            client.delete(*keys)
    
    async def aclear_pattern(self, pattern: str):
        """Eliminar todas las claves que coincidan con patrón (async)"""
        client = await self._get_async_client()
        keys = await client.keys(pattern)
        if keys:
            await client.delete(*keys)
    
    def get_stats(self) -> dict:
        """Obtener estadísticas de Redis"""
        client = self._get_client()
        info = client.info()
        return {
            "connected_clients": info.get("connected_clients", 0),
            "used_memory": info.get("used_memory_human", "0B"),
            "total_keys": client.dbsize(),
        }
    
    async def close(self):
        """Cerrar conexión"""
        if self._async_client:
            await self._async_client.close()
        if self._client:
            self._client.close()


# Instancia global
_redis_cache: Optional[RedisCache] = None


def get_redis_cache(
    host: Optional[str] = None,
    port: Optional[int] = None,
    password: Optional[str] = None
) -> Optional[RedisCache]:
    """Obtener instancia global de Redis cache"""
    global _redis_cache
    
    if not REDIS_AVAILABLE:
        return None
    
    if _redis_cache is None:
        from core.architecture.config import get_config
        config = get_config()
        
        # Intentar obtener de variables de entorno o config
        redis_host = host or getattr(config, 'redis_host', 'localhost')
        redis_port = port or getattr(config, 'redis_port', 6379)
        redis_password = password or getattr(config, 'redis_password', None)
        
        try:
            _redis_cache = RedisCache(
                host=redis_host,
                port=redis_port,
                password=redis_password
            )
        except Exception as e:
            print(f"Warning: Could not connect to Redis: {e}")
            return None
    
    return _redis_cache




