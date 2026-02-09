"""
Optimizaciones de performance para Robot Movement AI v2.0
Incluye caching avanzado, connection pooling, y optimizaciones de queries
"""

import functools
import time
import hashlib
import json
from typing import Any, Callable, Optional, Dict, Tuple
from datetime import datetime, timedelta
from collections import OrderedDict
from dataclasses import dataclass, field


@dataclass
class CacheEntry:
    """Entrada de cache con metadata"""
    value: Any
    timestamp: datetime
    ttl: timedelta
    access_count: int = 0
    last_access: datetime = field(default_factory=datetime.now)
    
    def is_expired(self) -> bool:
        """Verificar si la entrada ha expirado"""
        return datetime.now() - self.timestamp > self.ttl
    
    def touch(self):
        """Actualizar último acceso"""
        self.last_access = datetime.now()
        self.access_count += 1


class LRUCache:
    """Cache LRU (Least Recently Used) con TTL"""
    
    def __init__(self, max_size: int = 1000, default_ttl: timedelta = timedelta(minutes=5)):
        """
        Inicializar cache LRU
        
        Args:
            max_size: Tamaño máximo del cache
            default_ttl: TTL por defecto para entradas
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Obtener valor del cache"""
        if key not in self.cache:
            self.misses += 1
            return None
        
        entry = self.cache[key]
        
        # Verificar expiración
        if entry.is_expired():
            del self.cache[key]
            self.misses += 1
            return None
        
        # Mover al final (más reciente)
        self.cache.move_to_end(key)
        entry.touch()
        self.hits += 1
        
        return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[timedelta] = None):
        """Establecer valor en cache"""
        ttl = ttl or self.default_ttl
        
        # Si existe, actualizar
        if key in self.cache:
            self.cache[key].value = value
            self.cache[key].timestamp = datetime.now()
            self.cache[key].ttl = ttl
            self.cache.move_to_end(key)
            return
        
        # Si está lleno, eliminar el más antiguo
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)
        
        # Agregar nueva entrada
        self.cache[key] = CacheEntry(
            value=value,
            timestamp=datetime.now(),
            ttl=ttl
        )
    
    def delete(self, key: str):
        """Eliminar entrada del cache"""
        self.cache.pop(key, None)
    
    def clear(self):
        """Limpiar todo el cache"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    def cleanup_expired(self):
        """Limpiar entradas expiradas"""
        expired_keys = [
            key for key, entry in self.cache.items()
            if entry.is_expired()
        ]
        for key in expired_keys:
            del self.cache[key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del cache"""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.2f}%",
            "total_requests": total
        }


class PerformanceMonitor:
    """Monitor de performance para funciones"""
    
    def __init__(self):
        self.metrics: Dict[str, list] = {}
    
    def record(self, function_name: str, duration: float):
        """Registrar duración de ejecución"""
        if function_name not in self.metrics:
            self.metrics[function_name] = []
        self.metrics[function_name].append(duration)
        
        # Mantener solo últimos 1000 registros
        if len(self.metrics[function_name]) > 1000:
            self.metrics[function_name].pop(0)
    
    def get_stats(self, function_name: str) -> Optional[Dict[str, float]]:
        """Obtener estadísticas de una función"""
        if function_name not in self.metrics or not self.metrics[function_name]:
            return None
        
        durations = self.metrics[function_name]
        return {
            "count": len(durations),
            "min": min(durations),
            "max": max(durations),
            "avg": sum(durations) / len(durations),
            "p50": sorted(durations)[len(durations) // 2],
            "p95": sorted(durations)[int(len(durations) * 0.95)],
            "p99": sorted(durations)[int(len(durations) * 0.99)],
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Obtener estadísticas de todas las funciones"""
        return {
            name: self.get_stats(name)
            for name in self.metrics.keys()
        }


# Instancias globales
_performance_cache = LRUCache()
_performance_monitor = PerformanceMonitor()


def cached(
    ttl: Optional[timedelta] = None,
    key_func: Optional[Callable] = None,
    cache_instance: Optional[LRUCache] = None
):
    """
    Decorator para cachear resultados de funciones
    
    Args:
        ttl: Tiempo de vida del cache
        key_func: Función para generar clave de cache
        cache_instance: Instancia de cache a usar
    """
    cache = cache_instance or _performance_cache
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Generar clave
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = _generate_cache_key(func.__name__, args, kwargs)
            
            # Intentar obtener del cache
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                return cached_value
            
            # Ejecutar función
            result = await func(*args, **kwargs)
            
            # Guardar en cache
            cache.set(cache_key, result, ttl)
            
            return result
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Generar clave
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = _generate_cache_key(func.__name__, args, kwargs)
            
            # Intentar obtener del cache
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                return cached_value
            
            # Ejecutar función
            result = func(*args, **kwargs)
            
            # Guardar en cache
            cache.set(cache_key, result, ttl)
            
            return result
        
        # Retornar wrapper apropiado
        import inspect
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def timed(func: Callable = None, monitor: Optional[PerformanceMonitor] = None):
    """
    Decorator para medir tiempo de ejecución
    
    Args:
        func: Función a decorar
        monitor: Monitor de performance a usar
    """
    monitor = monitor or _performance_monitor
    
    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        async def async_wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = await f(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start
                monitor.record(f.__name__, duration)
        
        @functools.wraps(f)
        def sync_wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = f(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start
                monitor.record(f.__name__, duration)
        
        import inspect
        if inspect.iscoroutinefunction(f):
            return async_wrapper
        return sync_wrapper
    
    if func is None:
        return decorator
    return decorator(func)


def _generate_cache_key(func_name: str, args: tuple, kwargs: dict) -> str:
    """Generar clave de cache única"""
    # Serializar argumentos
    key_data = {
        "func": func_name,
        "args": str(args),
        "kwargs": json.dumps(kwargs, sort_keys=True, default=str)
    }
    key_str = json.dumps(key_data, sort_keys=True)
    
    # Generar hash
    return hashlib.md5(key_str.encode()).hexdigest()


def get_performance_cache() -> LRUCache:
    """Obtener instancia global de cache"""
    return _performance_cache


def get_performance_monitor() -> PerformanceMonitor:
    """Obtener instancia global de monitor"""
    return _performance_monitor




