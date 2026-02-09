"""
Fast Cache
Sistema de caché ultra-rápido con múltiples niveles
"""

import logging
import time
import hashlib
from typing import Any, Optional, Dict
from functools import wraps
from collections import OrderedDict

logger = logging.getLogger(__name__)


class LRUCache:
    """LRU Cache en memoria ultra-rápido"""
    
    def __init__(self, maxsize: int = 1000):
        self.cache = OrderedDict()
        self.maxsize = maxsize
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Obtiene valor del cache"""
        if key in self.cache:
            # Mover al final (más reciente)
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None
    
    def set(self, key: str, value: Any):
        """Establece valor en cache"""
        if key in self.cache:
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.maxsize:
            # Remover el más antiguo
            self.cache.popitem(last=False)
        
        self.cache[key] = value
    
    def delete(self, key: str):
        """Elimina del cache"""
        self.cache.pop(key, None)
    
    def clear(self):
        """Limpia el cache"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    def stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas"""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            "size": len(self.cache),
            "maxsize": self.maxsize,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.2f}%"
        }


class FastCache:
    """Sistema de caché multi-nivel rápido"""
    
    def __init__(self):
        self._l1_cache = LRUCache(maxsize=1000)  # Cache L1 (memoria)
        self._l2_cache = None  # Cache L2 (Redis, opcional)
        self._ttl_cache: Dict[str, float] = {}  # TTL por key
    
    def set_l2_cache(self, redis_client):
        """Configura cache L2 (Redis)"""
        self._l2_cache = redis_client
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Genera clave de cache"""
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def get(self, key: str) -> Optional[Any]:
        """Obtiene valor del cache (L1 primero, luego L2)"""
        # Verificar TTL
        if key in self._ttl_cache:
            if time.time() > self._ttl_cache[key]:
                self._l1_cache.delete(key)
                del self._ttl_cache[key]
                return None
        
        # Intentar L1
        value = self._l1_cache.get(key)
        if value is not None:
            return value
        
        # Intentar L2 (Redis)
        if self._l2_cache:
            try:
                import json
                cached = await self._l2_cache.get(key)
                if cached:
                    value = json.loads(cached) if isinstance(cached, str) else cached
                    # Promover a L1
                    self._l1_cache.set(key, value)
                    return value
            except Exception as e:
                logger.warning(f"L2 cache error: {e}")
        
        return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600):
        """Establece valor en cache (L1 y L2)"""
        # L1
        self._l1_cache.set(key, value)
        if ttl > 0:
            self._ttl_cache[key] = time.time() + ttl
        
        # L2 (Redis)
        if self._l2_cache:
            try:
                import json
                serialized = json.dumps(value) if not isinstance(value, str) else value
                await self._l2_cache.set(key, serialized, ttl=ttl)
            except Exception as e:
                logger.warning(f"L2 cache error: {e}")
    
    def delete(self, key: str):
        """Elimina del cache"""
        self._l1_cache.delete(key)
        if key in self._ttl_cache:
            del self._ttl_cache[key]
        if self._l2_cache:
            try:
                self._l2_cache.delete(key)
            except Exception:
                pass
    
    def stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas"""
        stats = {
            "l1": self._l1_cache.stats()
        }
        if self._l2_cache:
            stats["l2"] = "enabled"
        return stats


def fast_cache(ttl: int = 3600, key_func=None):
    """Decorator para cache rápido"""
    cache = FastCache()
    
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Generar key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = cache._generate_key(*args, **kwargs)
            
            # Intentar cache
            cached = await cache.get(cache_key)
            if cached is not None:
                return cached
            
            # Ejecutar función
            result = await func(*args, **kwargs)
            
            # Guardar en cache
            await cache.set(cache_key, result, ttl=ttl)
            
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Para funciones sync, usar cache L1 solo
            cache_key = cache._generate_key(*args, **kwargs)
            cached = cache._l1_cache.get(cache_key)
            if cached is not None:
                return cached
            
            result = func(*args, **kwargs)
            cache._l1_cache.set(cache_key, result)
            return result
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Instancia global
_fast_cache: Optional[FastCache] = None


def get_fast_cache() -> FastCache:
    """Obtiene el cache rápido"""
    global _fast_cache
    if _fast_cache is None:
        _fast_cache = FastCache()
    return _fast_cache















