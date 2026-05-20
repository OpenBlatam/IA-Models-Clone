"""
Advanced Caching
================
Utilidades avanzadas para caching.
"""

from typing import Any, Optional, Callable, Dict
from datetime import datetime, timedelta
import hashlib
import json
from functools import wraps


class LRUCache:
    """Cache LRU (Least Recently Used)."""
    
    def __init__(self, maxsize: int = 128):
        """
        Inicializar cache LRU.
        
        Args:
            maxsize: Tamaño máximo del cache
        """
        self.maxsize = maxsize
        self.cache: Dict[str, tuple] = {}
        self.access_order = []
    
    def get(self, key: str) -> Optional[Any]:
        """Obtener valor del cache."""
        if key in self.cache:
            # Mover a final (más reciente)
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key][0]
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None):
        """Agregar valor al cache."""
        expires_at = None
        if ttl:
            expires_at = datetime.now() + timedelta(seconds=ttl)
        
        if key in self.cache:
            # Actualizar existente
            self.access_order.remove(key)
        elif len(self.cache) >= self.maxsize:
            # Remover menos reciente
            oldest = self.access_order.pop(0)
            del self.cache[oldest]
        
        self.cache[key] = (value, expires_at)
        self.access_order.append(key)
    
    def delete(self, key: str):
        """Eliminar del cache."""
        if key in self.cache:
            del self.cache[key]
            self.access_order.remove(key)
    
    def clear(self):
        """Limpiar cache."""
        self.cache.clear()
        self.access_order.clear()
    
    def _clean_expired(self):
        """Limpiar entradas expiradas."""
        now = datetime.now()
        expired_keys = [
            key for key, (_, expires_at) in self.cache.items()
            if expires_at and expires_at < now
        ]
        for key in expired_keys:
            self.delete(key)
    
    def get_stats(self) -> Dict:
        """Obtener estadísticas del cache."""
        self._clean_expired()
        return {
            "size": len(self.cache),
            "maxsize": self.maxsize,
            "usage_percent": (len(self.cache) / self.maxsize * 100) if self.maxsize > 0 else 0
        }


def cache_key_generator(*args, **kwargs) -> str:
    """
    Generar clave de cache desde argumentos.
    
    Args:
        *args: Argumentos posicionales
        **kwargs: Argumentos de palabra clave
        
    Returns:
        Clave de cache
    """
    key_parts = []
    
    for arg in args:
        if isinstance(arg, (str, int, float, bool)):
            key_parts.append(str(arg))
        else:
            key_parts.append(json.dumps(arg, sort_keys=True, default=str))
    
    for key, value in sorted(kwargs.items()):
        if isinstance(value, (str, int, float, bool)):
            key_parts.append(f"{key}:{value}")
        else:
            key_parts.append(f"{key}:{json.dumps(value, sort_keys=True, default=str)}")
    
    key_string = "|".join(key_parts)
    return hashlib.md5(key_string.encode()).hexdigest()


def cached_function(ttl: Optional[float] = None, maxsize: int = 128):
    """
    Decorador para cachear resultados de funciones.
    
    Args:
        ttl: Time to live en segundos
        maxsize: Tamaño máximo del cache
    """
    cache = LRUCache(maxsize=maxsize)
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            key = cache_key_generator(func.__name__, *args, **kwargs)
            
            # Intentar obtener del cache
            cached = cache.get(key)
            if cached is not None:
                return cached
            
            # Ejecutar función
            result = await func(*args, **kwargs)
            
            # Guardar en cache
            cache.set(key, result, ttl=ttl)
            
            return result
        
        wrapper.cache = cache  # Exponer cache para acceso directo
        return wrapper
    return decorator

