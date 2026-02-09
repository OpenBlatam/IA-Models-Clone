"""
Performance Utilities
=====================

Utilidades para monitoreo y optimización de rendimiento.
"""

import time
import functools
from typing import Any, Callable, Optional, Dict
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


def measure_time(func: Optional[Callable] = None, *, log: bool = True, threshold_ms: float = 0.0):
    """
    Decorador para medir tiempo de ejecución.
    
    Args:
        func: Función a decorar (si se usa como decorador sin paréntesis)
        log: Si debe loguear el tiempo
        threshold_ms: Solo loguear si excede este tiempo en ms
    
    Returns:
        Decorador o función decorada
    """
    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def sync_wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = f(*args, **kwargs)
                return result
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000
                if log and elapsed_ms > threshold_ms:
                    logger.debug(f"{f.__name__} took {elapsed_ms:.2f}ms")
        
        @functools.wraps(f)
        async def async_wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = await f(*args, **kwargs)
                return result
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000
                if log and elapsed_ms > threshold_ms:
                    logger.debug(f"{f.__name__} took {elapsed_ms:.2f}ms")
        
        if hasattr(f, '__code__') and f.__code__.co_flags & 0x80:  # CO_COROUTINE
            return async_wrapper
        return sync_wrapper
    
    if func is None:
        return decorator
    return decorator(func)


@contextmanager
def timer(name: str = "operation", log: bool = True):
    """
    Context manager para medir tiempo.
    
    Args:
        name: Nombre de la operación
        log: Si debe loguear el tiempo
    
    Yields:
        None
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        if log:
            logger.debug(f"{name} took {elapsed_ms:.2f}ms")


class PerformanceMonitor:
    """Monitor de rendimiento para operaciones."""
    
    def __init__(self, name: str = "operation"):
        """
        Inicializar monitor.
        
        Args:
            name: Nombre de la operación
        """
        self.name = name
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.metrics: Dict[str, Any] = {}
    
    def start(self) -> None:
        """Iniciar medición."""
        self.start_time = time.perf_counter()
    
    def stop(self) -> float:
        """
        Detener medición.
        
        Returns:
            Tiempo transcurrido en segundos
        """
        if self.start_time is None:
            raise RuntimeError("Monitor not started")
        self.end_time = time.perf_counter()
        return self.end_time - self.start_time
    
    def add_metric(self, key: str, value: Any) -> None:
        """
        Agregar métrica.
        
        Args:
            key: Nombre de la métrica
            value: Valor
        """
        self.metrics[key] = value
    
    def get_report(self) -> Dict[str, Any]:
        """
        Obtener reporte de rendimiento.
        
        Returns:
            Diccionario con métricas
        """
        elapsed = None
        if self.start_time is not None and self.end_time is not None:
            elapsed = self.end_time - self.start_time
        
        return {
            "name": self.name,
            "elapsed_seconds": elapsed,
            "elapsed_ms": elapsed * 1000 if elapsed else None,
            "metrics": self.metrics.copy()
        }
    
    def __enter__(self):
        """Entrar al contexto."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Salir del contexto."""
        self.stop()


def cache_result(maxsize: int = 128, ttl: Optional[float] = None):
    """
    Decorador para cachear resultados.
    
    Args:
        maxsize: Tamaño máximo del cache
        ttl: Time to live en segundos (None = sin expiración)
    
    Returns:
        Decorador
    """
    def decorator(func: Callable) -> Callable:
        cache: Dict[Any, tuple] = {}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Crear clave del cache
            key = (args, tuple(sorted(kwargs.items())))
            
            # Verificar cache
            if key in cache:
                result, timestamp = cache[key]
                if ttl is None or (time.time() - timestamp) < ttl:
                    return result
                else:
                    del cache[key]
            
            # Ejecutar función
            result = func(*args, **kwargs)
            
            # Guardar en cache
            if len(cache) >= maxsize:
                # Eliminar el más antiguo
                oldest_key = min(cache.keys(), key=lambda k: cache[k][1])
                del cache[oldest_key]
            
            cache[key] = (result, time.time())
            return result
        
        return wrapper
    return decorator

