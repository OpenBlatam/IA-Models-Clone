"""
Caching Optimizations - Optimizaciones de cache
===============================================

Cache inteligente con TTL, invalidation y estrategias avanzadas.
"""

import logging
import hashlib
import json
from typing import Optional, Any, Callable, Dict
from functools import wraps
from datetime import datetime, timedelta
import asyncio

logger = logging.getLogger(__name__)


class SmartCache:
    """
    Cache inteligente con:
    - TTL automático
    - Invalidation por patrón
    - Cache warming
    - Statistics
    """
    
    def __init__(self, backend, default_ttl: int = 3600):
        self.backend = backend
        self.default_ttl = default_ttl
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0
        }
        self._invalidation_patterns: Dict[str, list] = {}
    
    async def get(
        self,
        key: str,
        default: Optional[Any] = None
    ) -> Optional[Any]:
        """Obtiene valor con estadísticas"""
        value = await self.backend.get(key)
        if value is not None:
            self.stats["hits"] += 1
            return value
        else:
            self.stats["misses"] += 1
            return default
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        tags: Optional[list] = None
    ) -> bool:
        """Establece valor con tags para invalidation"""
        ttl = ttl or self.default_ttl
        success = await self.backend.set(key, value, ttl)
        if success:
            self.stats["sets"] += 1
            # Registrar tags para invalidation
            if tags:
                for tag in tags:
                    if tag not in self._invalidation_patterns:
                        self._invalidation_patterns[tag] = []
                    self._invalidation_patterns[tag].append(key)
        return success
    
    async def delete(self, key: str) -> bool:
        """Elimina clave"""
        success = await self.backend.delete(key)
        if success:
            self.stats["deletes"] += 1
        return success
    
    async def invalidate_by_tag(self, tag: str) -> int:
        """Invalida todas las claves con un tag"""
        count = 0
        if tag in self._invalidation_patterns:
            for key in self._invalidation_patterns[tag]:
                if await self.delete(key):
                    count += 1
            del self._invalidation_patterns[tag]
        return count
    
    async def warm_cache(self, keys: list, loader: Callable):
        """Pre-carga cache"""
        for key in keys:
            if await self.get(key) is None:
                value = await loader(key)
                if value:
                    await self.set(key, value)
    
    def get_stats(self) -> dict:
        """Obtiene estadísticas de cache"""
        total = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total if total > 0 else 0
        return {
            **self.stats,
            "hit_rate": hit_rate,
            "total_requests": total
        }


def CacheDecorator(
    ttl: int = 3600,
    key_prefix: str = "",
    tags: Optional[list] = None
):
    """
    Decorator para cachear resultados de funciones.
    
    Args:
        ttl: Tiempo de vida del cache (segundos)
        key_prefix: Prefijo para las claves
        tags: Tags para invalidation
    
    Example:
        ```python
        @CacheDecorator(ttl=3600, tags=["projects"])
        async def get_project(project_id: str):
            return await repository.get_by_id(project_id)
        ```
    """
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Generar clave de cache
            cache_key = _generate_cache_key(
                func.__name__,
                key_prefix,
                args,
                kwargs
            )
            
            # Intentar obtener de cache
            from ..infrastructure.cache import get_cache_service
            cache = get_cache_service()
            cached = await cache.get(cache_key)
            if cached is not None:
                return cached
            
            # Ejecutar función
            result = await func(*args, **kwargs)
            
            # Guardar en cache
            if result is not None:
                await cache.set(cache_key, result, ttl=ttl)
            
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Para funciones sync, convertir a async
            import asyncio
            return asyncio.run(async_wrapper(*args, **kwargs))
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def _generate_cache_key(
    func_name: str,
    prefix: str,
    args: tuple,
    kwargs: dict
) -> str:
    """Genera clave de cache única"""
    key_data = {
        "func": func_name,
        "args": str(args),
        "kwargs": json.dumps(kwargs, sort_keys=True)
    }
    key_string = json.dumps(key_data, sort_keys=True)
    key_hash = hashlib.md5(key_string.encode()).hexdigest()
    return f"{prefix}:{func_name}:{key_hash}" if prefix else f"{func_name}:{key_hash}"


def get_smart_cache(backend=None, default_ttl: int = 3600) -> SmartCache:
    """Obtiene instancia de SmartCache"""
    if backend is None:
        from ..infrastructure.cache import get_cache_service
        backend = get_cache_service()
    return SmartCache(backend, default_ttl)















