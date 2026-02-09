"""
Sistema de cache para resultados de validación y checks de pipelines
"""

import time
import hashlib
import json
from typing import Dict, Any, Optional, Callable, TypeVar, Generic
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import wraps
import threading

T = TypeVar('T')


@dataclass
class CacheEntry(Generic[T]):
    """Entrada de cache"""
    key: str
    value: T
    timestamp: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.utcnow)


class PipelineCache:
    """Cache para resultados de pipelines"""
    
    def __init__(
        self,
        default_ttl: float = 300.0,  # 5 minutos por defecto
        max_size: int = 100
    ):
        """
        Inicializa el cache.
        
        Args:
            default_ttl: Tiempo de vida por defecto en segundos
            max_size: Tamaño máximo del cache
        """
        self.default_ttl = default_ttl
        self.max_size = max_size
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Genera una clave única para los argumentos"""
        key_data = {
            "args": str(args),
            "kwargs": json.dumps(kwargs, sort_keys=True, default=str)
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Obtiene un valor del cache.
        
        Args:
            key: Clave del cache
            
        Returns:
            Valor cacheado o None si no existe o expiró
        """
        with self._lock:
            entry = self._cache.get(key)
            
            if entry is None:
                self._misses += 1
                return None
            
            # Verificar expiración
            if entry.expires_at and datetime.utcnow() > entry.expires_at:
                del self._cache[key]
                self._misses += 1
                return None
            
            # Actualizar estadísticas
            entry.access_count += 1
            entry.last_accessed = datetime.utcnow()
            self._hits += 1
            
            return entry.value
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None
    ) -> None:
        """
        Guarda un valor en el cache.
        
        Args:
            key: Clave del cache
            value: Valor a cachear
            ttl: Tiempo de vida en segundos (None para usar default)
        """
        with self._lock:
            # Limpiar si está lleno
            if len(self._cache) >= self.max_size:
                self._evict_oldest()
            
            expires_at = None
            if ttl is not None:
                expires_at = datetime.utcnow() + timedelta(seconds=ttl)
            elif self.default_ttl:
                expires_at = datetime.utcnow() + timedelta(seconds=self.default_ttl)
            
            self._cache[key] = CacheEntry(
                key=key,
                value=value,
                expires_at=expires_at
            )
    
    def _evict_oldest(self) -> None:
        """Elimina la entrada más antigua"""
        if not self._cache:
            return
        
        oldest_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].last_accessed
        )
        del self._cache[oldest_key]
    
    def clear(self) -> None:
        """Limpia todo el cache"""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
    
    def invalidate(self, key: str) -> bool:
        """
        Invalida una entrada específica.
        
        Args:
            key: Clave a invalidar
            
        Returns:
            True si se invalidó, False si no existía
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas del cache"""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0.0
            
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(hit_rate, 2),
                "total_requests": total_requests
            }
    
    def cached(
        self,
        ttl: Optional[float] = None,
        key_func: Optional[Callable] = None
    ):
        """
        Decorador para cachear resultados de funciones.
        
        Args:
            ttl: Tiempo de vida del cache
            key_func: Función personalizada para generar la clave
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generar clave
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    cache_key = self._generate_key(func.__name__, *args, **kwargs)
                
                # Intentar obtener del cache
                cached_value = self.get(cache_key)
                if cached_value is not None:
                    return cached_value
                
                # Ejecutar función y cachear resultado
                result = func(*args, **kwargs)
                self.set(cache_key, result, ttl=ttl)
                
                return result
            
            return wrapper
        return decorator


# Instancia global del cache
_global_cache: Optional[PipelineCache] = None


def get_cache(
    default_ttl: float = 300.0,
    max_size: int = 100
) -> PipelineCache:
    """
    Obtiene la instancia global del cache.
    
    Args:
        default_ttl: Tiempo de vida por defecto
        max_size: Tamaño máximo
        
    Returns:
        PipelineCache instance
    """
    global _global_cache
    
    if _global_cache is None:
        _global_cache = PipelineCache(default_ttl=default_ttl, max_size=max_size)
    
    return _global_cache


def cached_check(ttl: float = 60.0):
    """
    Decorador para cachear resultados de check_compatibility.
    
    Args:
        ttl: Tiempo de vida en segundos
    """
    cache = get_cache()
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = "check_compatibility"
            cached = cache.get(cache_key)
            
            if cached is not None:
                return cached
            
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl=ttl)
            
            return result
        
        return wrapper
    return decorator

