"""
Response Cache - Cache de respuestas
====================================

Sistema de cache para respuestas HTTP.
"""

import hashlib
import json
import logging
from typing import Optional, Any, Callable, Dict
from datetime import datetime, timedelta
try:
    import aioredis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False
    aioredis = None

logger = logging.getLogger(__name__)


class ResponseCache:
    """Cache de respuestas HTTP."""
    
    def __init__(self, redis_client: Optional[Any] = None, default_ttl: int = 300):
        """
        Inicializar cache.
        
        Args:
            redis_client: Cliente Redis (opcional).
            default_ttl: TTL por defecto en segundos.
        """
        self.redis = redis_client
        self.default_ttl = default_ttl
        self._memory_cache: Dict[str, tuple] = {}  # Fallback en memoria
    
    def _make_key(self, method: str, path: str, query_params: str = "") -> str:
        """Crear clave de cache."""
        key_string = f"{method}:{path}:{query_params}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    async def get(self, method: str, path: str, query_params: str = "") -> Optional[Any]:
        """
        Obtener respuesta del cache.
        
        Args:
            method: Método HTTP.
            path: Path de la request.
            query_params: Parámetros de query.
        
        Returns:
            Respuesta cacheada o None.
        """
        key = self._make_key(method, path, query_params)
        
        # Intentar Redis primero
        if self.redis:
            try:
                cached = await self.redis.get(f"cache:{key}")
                if cached:
                    return json.loads(cached)
            except Exception as e:
                logger.warning(f"Redis cache get failed: {e}")
        
        # Fallback a memoria
        if key in self._memory_cache:
            data, expiry = self._memory_cache[key]
            if datetime.now() < expiry:
                return data
            else:
                del self._memory_cache[key]
        
        return None
    
    async def set(
        self,
        method: str,
        path: str,
        data: Any,
        ttl: Optional[int] = None,
        query_params: str = ""
    ) -> None:
        """
        Guardar respuesta en cache.
        
        Args:
            method: Método HTTP.
            path: Path de la request.
            data: Datos a cachear.
            ttl: TTL en segundos (None = default).
            query_params: Parámetros de query.
        """
        key = self._make_key(method, path, query_params)
        ttl = ttl or self.default_ttl
        
        # Guardar en Redis
        if self.redis:
            try:
                await self.redis.setex(
                    f"cache:{key}",
                    ttl,
                    json.dumps(data)
                )
            except Exception as e:
                logger.warning(f"Redis cache set failed: {e}")
        
        # Fallback a memoria
        expiry = datetime.now() + timedelta(seconds=ttl)
        self._memory_cache[key] = (data, expiry)
        
        # Limpiar memoria si es necesario
        if len(self._memory_cache) > 1000:
            now = datetime.now()
            self._memory_cache = {
                k: v for k, v in self._memory_cache.items()
                if v[1] > now
            }
    
    async def invalidate(self, pattern: str) -> None:
        """
        Invalidar cache por patrón.
        
        Args:
            pattern: Patrón de path a invalidar.
        """
        # Invalidar en Redis
        if self.redis:
            try:
                keys = await self.redis.keys(f"cache:*{pattern}*")
                if keys:
                    await self.redis.delete(*keys)
            except Exception as e:
                logger.warning(f"Redis cache invalidate failed: {e}")
        
        # Invalidar en memoria
        keys_to_delete = [k for k in self._memory_cache.keys() if pattern in k]
        for key in keys_to_delete:
            del self._memory_cache[key]


# Cache global
_response_cache: Optional[ResponseCache] = None


def get_response_cache() -> Optional[ResponseCache]:
    """Obtener cache de respuestas global."""
    global _response_cache
    
    if _response_cache is None:
        try:
            import os
            import aioredis
            
            redis_url = os.getenv("REDIS_URL")
            if redis_url:
                redis_client = aioredis.from_url(redis_url)
                _response_cache = ResponseCache(redis_client=redis_client)
                logger.info("Response cache initialized with Redis")
            else:
                _response_cache = ResponseCache()
                logger.info("Response cache initialized (memory only)")
        except Exception as e:
            logger.warning(f"Failed to initialize response cache: {e}")
            _response_cache = ResponseCache()
    
    return _response_cache

