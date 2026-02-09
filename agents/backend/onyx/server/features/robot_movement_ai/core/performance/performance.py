"""
Performance Utilities
=====================

Utilidades para optimización y profiling de performance.
"""

import time
import functools
from typing import Callable, Any, Optional, Dict
import logging
from contextlib import contextmanager

try:
    from ..system.metrics import record_timing, get_metrics_collector
except ImportError:
    from ...system.metrics import record_timing, get_metrics_collector

logger = logging.getLogger(__name__)


@contextmanager
def measure_time(metric_name: str, tags: Optional[Dict[str, str]] = None):
    """
    Context manager para medir tiempo de ejecución.
    
    Usage:
        with measure_time("trajectory_optimization"):
            # código a medir
            ...
    """
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        record_timing(metric_name, duration, tags)
        logger.debug(f"{metric_name} took {duration:.4f}s")


def timeit(metric_name: Optional[str] = None):
    """
    Decorador para medir tiempo de ejecución.
    
    Usage:
        @timeit("my_function")
        def my_function():
            ...
    """
    def decorator(func: Callable) -> Callable:
        name = metric_name or f"{func.__module__}.{func.__name__}"
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with measure_time(name):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


def timeit_async(metric_name: Optional[str] = None):
    """
    Decorador para medir tiempo de ejecución (async).
    
    Usage:
        @timeit_async("my_async_function")
        async def my_async_function():
            ...
    """
    def decorator(func: Callable) -> Callable:
        name = metric_name or f"{func.__module__}.{func.__name__}"
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                return await func(*args, **kwargs)
            finally:
                duration = time.time() - start_time
                record_timing(name, duration)
                logger.debug(f"{name} took {duration:.4f}s")
        
        return wrapper
    return decorator


class PerformanceProfiler:
    """
    Profiler de performance para análisis detallado.
    """
    
    def __init__(self):
        """Inicializar profiler."""
        self.sections: Dict[str, float] = {}
        self.start_times: Dict[str, float] = {}
        self.call_counts: Dict[str, int] = defaultdict(int)
    
    def start_section(self, name: str):
        """Iniciar sección de profiling."""
        self.start_times[name] = time.time()
    
    def end_section(self, name: str) -> float:
        """
        Finalizar sección de profiling.
        
        Returns:
            Duración en segundos
        """
        if name not in self.start_times:
            logger.warning(f"Section {name} was not started")
            return 0.0
        
        duration = time.time() - self.start_times[name]
        if name in self.sections:
            self.sections[name] += duration
        else:
            self.sections[name] = duration
        
        self.call_counts[name] += 1
        del self.start_times[name]
        return duration
    
    @contextmanager
    def section(self, name: str):
        """
        Context manager para sección de profiling.
        
        Usage:
            with profiler.section("optimization"):
                # código
        """
        self.start_section(name)
        try:
            yield
        finally:
            self.end_section(name)
    
    def get_report(self) -> Dict[str, Any]:
        """Obtener reporte de profiling."""
        total_time = sum(self.sections.values())
        
        return {
            "total_time": total_time,
            "sections": {
                name: {
                    "total_time": time,
                    "call_count": self.call_counts.get(name, 0),
                    "average_time": time / self.call_counts.get(name, 1),
                    "percentage": (time / total_time * 100) if total_time > 0 else 0
                }
                for name, time in self.sections.items()
            }
        }
    
    def reset(self):
        """Resetear profiler."""
        self.sections.clear()
        self.start_times.clear()
        self.call_counts.clear()


class CacheManager:
    """
    Gestor de caché con estadísticas y limpieza automática.
    """
    
    def __init__(self, max_size: int = 1000, ttl_seconds: Optional[float] = None):
        """
        Inicializar gestor de caché.
        
        Args:
            max_size: Tamaño máximo del caché
            ttl_seconds: Time-to-live en segundos (None = sin expiración)
        """
        self.cache: Dict[str, tuple] = {}  # {key: (value, timestamp)}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Obtener valor del caché."""
        if key not in self.cache:
            self.misses += 1
            return None
        
        value, timestamp = self.cache[key]
        
        # Verificar TTL
        if self.ttl_seconds and (time.time() - timestamp) > self.ttl_seconds:
            del self.cache[key]
            self.misses += 1
            return None
        
        self.hits += 1
        return value
    
    def set(self, key: str, value: Any):
        """Guardar valor en caché."""
        # Limpiar si está lleno
        if len(self.cache) >= self.max_size:
            # Eliminar más antiguo
            oldest_key = min(
                self.cache.keys(),
                key=lambda k: self.cache[k][1]
            )
            del self.cache[oldest_key]
        
        self.cache[key] = (value, time.time())
    
    def clear(self):
        """Limpiar caché."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas del caché."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0.0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "ttl_seconds": self.ttl_seconds
        }


def batch_process(items: list, batch_size: int = 10, processor: Optional[Callable] = None):
    """
    Procesar items en lotes.
    
    Args:
        items: Lista de items a procesar
        batch_size: Tamaño del lote
        processor: Función para procesar cada lote (opcional)
        
    Yields:
        Lotes de items
    """
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        if processor:
            yield processor(batch)
        else:
            yield batch


def parallel_map(func: Callable, items: list, max_workers: int = 4):
    """
    Aplicar función a items en paralelo.
    
    Args:
        func: Función a aplicar
        items: Lista de items
        max_workers: Número máximo de workers
        
    Returns:
        Lista de resultados
    """
    from concurrent.futures import ThreadPoolExecutor
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(func, items))






