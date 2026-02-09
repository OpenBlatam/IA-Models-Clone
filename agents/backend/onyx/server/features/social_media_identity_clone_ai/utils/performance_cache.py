"""
Sistema de caché avanzado con TTL y estrategias de invalidación
"""

import logging
import time
import hashlib
import json
import asyncio
from typing import Any, Optional, Dict, Callable
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path

from ..config import get_settings

logger = logging.getLogger(__name__)


class AdvancedCache:
    """Caché avanzado con múltiples estrategias"""
    
    def __init__(self):
        self.settings = get_settings()
        self.cache_dir = Path(self.settings.storage_path) / "advanced_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Caché en memoria para acceso rápido
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        self.max_memory_items = 1000
    
    def _get_cache_key(self, prefix: str, *args, **kwargs) -> str:
        """Genera clave de caché única"""
        key_data = {
            "prefix": prefix,
            "args": str(args),
            "kwargs": str(sorted(kwargs.items()))
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Obtiene valor del caché"""
        # Intentar memoria primero
        if key in self.memory_cache:
            item = self.memory_cache[key]
            if item["expires_at"] > time.time():
                logger.debug(f"Cache hit (memory): {key}")
                return item["value"]
            else:
                del self.memory_cache[key]
        
        # Intentar disco
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    data = json.load(f)
                
                if data["expires_at"] > time.time():
                    value = data["value"]
                    # Cargar a memoria
                    self._add_to_memory(key, value, data["expires_at"])
                    logger.debug(f"Cache hit (disk): {key}")
                    return value
                else:
                    cache_file.unlink()
            except Exception as e:
                logger.error(f"Error leyendo caché {key}: {e}")
        
        return None
    
    def set(self, key: str, value: Any, ttl: int = 3600):
        """Guarda valor en caché"""
        expires_at = time.time() + ttl
        
        # Guardar en memoria
        self._add_to_memory(key, value, expires_at)
        
        # Guardar en disco
        try:
            cache_file = self.cache_dir / f"{key}.json"
            with open(cache_file, "w") as f:
                json.dump({
                    "value": value,
                    "expires_at": expires_at,
                    "created_at": time.time()
                }, f, default=str)
        except Exception as e:
            logger.error(f"Error guardando caché {key}: {e}")
    
    def _add_to_memory(self, key: str, value: Any, expires_at: float):
        """Agrega a caché en memoria con límite"""
        if len(self.memory_cache) >= self.max_memory_items:
            # Eliminar el más antiguo
            oldest_key = min(
                self.memory_cache.keys(),
                key=lambda k: self.memory_cache[k]["expires_at"]
            )
            del self.memory_cache[oldest_key]
        
        self.memory_cache[key] = {
            "value": value,
            "expires_at": expires_at
        }
    
    def delete(self, key: str):
        """Elimina del caché"""
        self.memory_cache.pop(key, None)
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            cache_file.unlink()
    
    def clear(self, prefix: Optional[str] = None):
        """Limpia caché"""
        if prefix:
            keys_to_delete = [
                k for k in self.memory_cache.keys()
                if k.startswith(prefix)
            ]
            for key in keys_to_delete:
                del self.memory_cache[key]
            
            for cache_file in self.cache_dir.glob(f"{prefix}*.json"):
                cache_file.unlink()
        else:
            self.memory_cache.clear()
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()


def cached(ttl: int = 3600, key_prefix: str = ""):
    """Decorator para cachear resultados de funciones"""
    def decorator(func: Callable) -> Callable:
        cache = AdvancedCache()
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache_key = cache._get_cache_key(key_prefix or func.__name__, *args, **kwargs)
            
            # Intentar obtener del caché
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                return cached_value
            
            # Ejecutar función
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Guardar en caché
            cache.set(cache_key, result, ttl=ttl)
            
            return result
        
        return wrapper
    return decorator


# Singleton global
_advanced_cache: Optional[AdvancedCache] = None


def get_advanced_cache() -> AdvancedCache:
    """Obtiene instancia singleton del caché avanzado"""
    global _advanced_cache
    if _advanced_cache is None:
        _advanced_cache = AdvancedCache()
    return _advanced_cache

