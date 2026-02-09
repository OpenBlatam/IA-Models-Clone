"""
Performance Utilities - Utilidades de rendimiento
==================================================

Utilidades para optimización y profiling de performance.
"""

import time
import functools
from typing import Callable, Any, Optional, Dict
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


def timeit(func: Optional[Callable] = None, log_level: str = "DEBUG"):
    """
    Decorador para medir tiempo de ejecución.
    
    Args:
        func: Función a decorar
        log_level: Nivel de log para el tiempo
    
    Example:
        @timeit
        def expensive_operation():
            ...
    """
    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        async def async_wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = await f(*args, **kwargs)
                return result
            finally:
                duration = time.perf_counter() - start
                log_func = getattr(logger, log_level.lower())
                log_func(f"{f.__name__} took {duration*1000:.2f}ms")
        
        @functools.wraps(f)
        def sync_wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = f(*args, **kwargs)
                return result
            finally:
                duration = time.perf_counter() - start
                log_func = getattr(logger, log_level.lower())
                log_func(f"{f.__name__} took {duration*1000:.2f}ms")
        
        if hasattr(f, '__code__') and f.__code__.co_flags & 0x80:
            return async_wrapper
        return sync_wrapper
    
    if func is None:
        return decorator
    return decorator(func)


@contextmanager
def timer(operation_name: str, log_level: str = "DEBUG"):
    """
    Context manager para medir tiempo de operación.
    
    Args:
        operation_name: Nombre de la operación
        log_level: Nivel de log
    
    Example:
        with timer("data_processing"):
            process_data()
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        log_func = getattr(logger, log_level.lower())
        log_func(f"{operation_name} took {duration*1000:.2f}ms")


class PerformanceMonitor:
    """Monitor de performance para operaciones."""
    
    def __init__(self):
        """Inicializar monitor."""
        self.operations: Dict[str, list] = {}
        self.counts: Dict[str, int] = {}
    
    def record(self, operation: str, duration_ms: float):
        """
        Registrar duración de operación.
        
        Args:
            operation: Nombre de la operación
            duration_ms: Duración en milisegundos
        """
        if operation not in self.operations:
            self.operations[operation] = []
            self.counts[operation] = 0
        
        self.operations[operation].append(duration_ms)
        self.counts[operation] += 1
    
    def get_stats(self, operation: str) -> Optional[Dict[str, float]]:
        """
        Obtener estadísticas de operación.
        
        Args:
            operation: Nombre de la operación
        
        Returns:
            Estadísticas o None si no existe
        """
        if operation not in self.operations:
            return None
        
        durations = self.operations[operation]
        if not durations:
            return None
        
        return {
            "count": self.counts[operation],
            "total_ms": sum(durations),
            "avg_ms": sum(durations) / len(durations),
            "min_ms": min(durations),
            "max_ms": max(durations),
            "p50_ms": sorted(durations)[len(durations) // 2],
            "p95_ms": sorted(durations)[int(len(durations) * 0.95)] if len(durations) > 1 else durations[0],
            "p99_ms": sorted(durations)[int(len(durations) * 0.99)] if len(durations) > 1 else durations[0]
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Obtener estadísticas de todas las operaciones."""
        return {
            op: self.get_stats(op)
            for op in self.operations.keys()
        }
    
    def reset(self):
        """Resetear todas las estadísticas."""
        self.operations.clear()
        self.counts.clear()


_global_monitor = PerformanceMonitor()


def get_performance_monitor() -> PerformanceMonitor:
    """Obtener monitor de performance global."""
    return _global_monitor

