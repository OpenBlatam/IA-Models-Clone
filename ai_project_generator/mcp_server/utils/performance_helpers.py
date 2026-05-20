"""
Performance Helpers - Utilidades para monitoreo de rendimiento
===============================================================

Funciones helper para medir y optimizar el rendimiento.
"""

import time
import functools
import asyncio
from typing import Callable, Any, Optional, Dict, List
from contextlib import contextmanager
from collections import defaultdict
import statistics

from .helpers import format_duration


class PerformanceMonitor:
    """
    Monitor de rendimiento para operaciones.
    
    Permite rastrear métricas de rendimiento y generar estadísticas.
    """
    
    def __init__(self):
        """Inicializar monitor de rendimiento"""
        self._metrics: Dict[str, List[float]] = defaultdict(list)
        self._counts: Dict[str, int] = defaultdict(int)
        self._errors: Dict[str, int] = defaultdict(int)
    
    def record(self, operation: str, duration: float, success: bool = True) -> None:
        """
        Registrar métrica de operación.
        
        Args:
            operation: Nombre de la operación
            duration: Duración en segundos
            success: Si la operación fue exitosa
        """
        self._metrics[operation].append(duration)
        self._counts[operation] += 1
        if not success:
            self._errors[operation] += 1
    
    def get_stats(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """
        Obtener estadísticas de rendimiento.
        
        Args:
            operation: Nombre de operación específica (opcional)
            
        Returns:
            Diccionario con estadísticas
        """
        if operation:
            if operation not in self._metrics:
                return {}
            
            durations = self._metrics[operation]
            return {
                "operation": operation,
                "count": self._counts[operation],
                "errors": self._errors[operation],
                "total_time": sum(durations),
                "avg_time": statistics.mean(durations) if durations else 0,
                "min_time": min(durations) if durations else 0,
                "max_time": max(durations) if durations else 0,
                "median_time": statistics.median(durations) if durations else 0,
                "p95_time": self._percentile(durations, 95) if durations else 0,
                "p99_time": self._percentile(durations, 99) if durations else 0,
            }
        
        # Estadísticas globales
        all_stats = {}
        for op in self._metrics.keys():
            all_stats[op] = self.get_stats(op)
        
        return all_stats
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calcular percentil"""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def reset(self) -> None:
        """Resetear todas las métricas"""
        self._metrics.clear()
        self._counts.clear()
        self._errors.clear()


# Monitor global
_global_monitor = PerformanceMonitor()


def get_performance_monitor() -> PerformanceMonitor:
    """
    Obtener monitor de rendimiento global.
    
    Returns:
        PerformanceMonitor instance
    """
    return _global_monitor


@contextmanager
def measure_time(operation: str, monitor: Optional[PerformanceMonitor] = None):
    """
    Context manager para medir tiempo de ejecución.
    
    Args:
        operation: Nombre de la operación
        monitor: Monitor de rendimiento (opcional, usa global)
    
    Yields:
        None
    """
    monitor = monitor or _global_monitor
    start_time = time.time()
    success = True
    
    try:
        yield
    except Exception:
        success = False
        raise
    finally:
        duration = time.time() - start_time
        monitor.record(operation, duration, success)


def timed_function(operation: Optional[str] = None, monitor: Optional[PerformanceMonitor] = None):
    """
    Decorator para medir tiempo de ejecución de funciones.
    
    Args:
        operation: Nombre de la operación (opcional, usa nombre de función)
        monitor: Monitor de rendimiento (opcional)
    
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        op_name = operation or f"{func.__module__}.{func.__name__}"
        perf_monitor = monitor or _global_monitor
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            with measure_time(op_name, perf_monitor):
                return func(*args, **kwargs)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            with measure_time(op_name, perf_monitor):
                return await func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def benchmark_function(iterations: int = 100):
    """
    Decorator para hacer benchmark de funciones.
    
    Args:
        iterations: Número de iteraciones
    
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            times = []
            result = None
            
            for _ in range(iterations):
                start = time.time()
                result = func(*args, **kwargs)
                times.append(time.time() - start)
            
            avg_time = statistics.mean(times)
            min_time = min(times)
            max_time = max(times)
            total_time = sum(times)
            
            print(f"Benchmark: {func.__name__}")
            print(f"  Iterations: {iterations}")
            print(f"  Total time: {format_duration(total_time)}")
            print(f"  Average: {format_duration(avg_time)}")
            print(f"  Min: {format_duration(min_time)}")
            print(f"  Max: {format_duration(max_time)}")
            
            return result
        
        return wrapper
    return decorator


class RateLimiter:
    """
    Rate limiter simple para operaciones.
    
    Permite limitar la frecuencia de ejecución de operaciones.
    """
    
    def __init__(self, max_calls: int, period: float):
        """
        Inicializar rate limiter.
        
        Args:
            max_calls: Número máximo de llamadas
            period: Período en segundos
        """
        self.max_calls = max_calls
        self.period = period
        self.calls: List[float] = []
    
    def acquire(self) -> bool:
        """
        Intentar adquirir permiso para ejecutar.
        
        Returns:
            True si se puede ejecutar, False si está limitado
        """
        now = time.time()
        
        # Limpiar llamadas antiguas
        self.calls = [call_time for call_time in self.calls if now - call_time < self.period]
        
        if len(self.calls) >= self.max_calls:
            return False
        
        self.calls.append(now)
        return True
    
    def wait_time(self) -> float:
        """
        Calcular tiempo de espera hasta la próxima ejecución permitida.
        
        Returns:
            Tiempo en segundos
        """
        if not self.calls:
            return 0.0
        
        now = time.time()
        oldest_call = min(self.calls)
        elapsed = now - oldest_call
        
        if elapsed >= self.period:
            return 0.0
        
        return self.period - elapsed


def cache_result(ttl: float = 60.0):
    """
    Decorator para cachear resultados de funciones.
    
    Args:
        ttl: Time to live en segundos
    
    Returns:
        Decorator function
    """
    cache: Dict[str, tuple] = {}
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generar clave de cache
            cache_key = str((args, tuple(sorted(kwargs.items()))))
            
            # Verificar cache
            if cache_key in cache:
                result, timestamp = cache[cache_key]
                if time.time() - timestamp < ttl:
                    return result
            
            # Ejecutar función y cachear
            result = func(*args, **kwargs)
            cache[cache_key] = (result, time.time())
            
            return result
        
        return wrapper
    return decorator

