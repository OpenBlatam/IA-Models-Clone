"""
Performance Utilities - Utilidades de rendimiento
==================================================

Utilidades para optimizar el rendimiento del agente.
"""

import asyncio
import functools
import time
from typing import Any, Callable, TypeVar, Optional
from datetime import datetime, timedelta

T = TypeVar('T')


def cached_async(ttl: float = 60.0, maxsize: int = 128):
    """
    Decorador para cachear resultados de funciones async con TTL.
    
    Args:
        ttl: Tiempo de vida del caché en segundos
        maxsize: Tamaño máximo del caché
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        cache: dict = {}
        cache_times: dict = {}
        
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Generar clave del caché
            key = str(args) + str(sorted(kwargs.items()))
            
            # Verificar si está en caché y no ha expirado
            if key in cache:
                cache_time = cache_times.get(key)
                if cache_time and (datetime.now() - cache_time).total_seconds() < ttl:
                    return cache[key]
                else:
                    # Limpiar entrada expirada
                    cache.pop(key, None)
                    cache_times.pop(key, None)
            
            # Ejecutar función y cachear resultado
            result = await func(*args, **kwargs)
            
            # Limpiar si el caché está lleno
            if len(cache) >= maxsize:
                # Eliminar la entrada más antigua
                oldest_key = min(cache_times.items(), key=lambda x: x[1])[0]
                cache.pop(oldest_key, None)
                cache_times.pop(oldest_key, None)
            
            cache[key] = result
            cache_times[key] = datetime.now()
            
            return result
        
        # Agregar método para limpiar caché
        wrapper.clear_cache = lambda: (cache.clear(), cache_times.clear())
        
        return wrapper
    return decorator


def rate_limit(calls: int = 10, period: float = 60.0):
    """
    Decorador para limitar la tasa de llamadas a una función.
    
    Args:
        calls: Número máximo de llamadas permitidas
        period: Período de tiempo en segundos
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        call_times: list[float] = []
        lock = asyncio.Lock()
        
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            async with lock:
                now = time.time()
                # Limpiar llamadas fuera del período
                call_times[:] = [t for t in call_times if now - t < period]
                
                if len(call_times) >= calls:
                    # Calcular tiempo de espera
                    oldest_call = min(call_times)
                    wait_time = period - (now - oldest_call)
                    if wait_time > 0:
                        await asyncio.sleep(wait_time)
                        now = time.time()
                        call_times[:] = [t for t in call_times if now - t < period]
                
                call_times.append(now)
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


def retry_async(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorador para reintentar funciones async con backoff exponencial.
    
    Nota: Para estrategias más avanzadas, usar retry_strategy.retry_with_strategy
    
    Args:
        max_attempts: Número máximo de intentos
        delay: Delay inicial en segundos
        backoff: Factor de backoff exponencial
        exceptions: Tupla de excepciones que deben causar reintento
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        raise
            
            if last_exception:
                raise last_exception
        
        return wrapper
    return decorator


class PerformanceMonitor:
    """Monitor de rendimiento para operaciones"""
    
    def __init__(self):
        self.metrics: dict[str, list[float]] = {}
        self.counts: dict[str, int] = {}
    
    def record(self, operation: str, duration: float) -> None:
        """Registrar duración de una operación"""
        if operation not in self.metrics:
            self.metrics[operation] = []
            self.counts[operation] = 0
        
        self.metrics[operation].append(duration)
        self.counts[operation] += 1
        
        # Mantener solo las últimas 1000 mediciones
        if len(self.metrics[operation]) > 1000:
            self.metrics[operation].pop(0)
    
    def get_stats(self, operation: str) -> Optional[dict[str, float]]:
        """Obtener estadísticas de una operación"""
        if operation not in self.metrics or not self.metrics[operation]:
            return None
        
        durations = self.metrics[operation]
        return {
            "count": self.counts[operation],
            "min": min(durations),
            "max": max(durations),
            "avg": sum(durations) / len(durations),
            "p50": sorted(durations)[len(durations) // 2],
            "p95": sorted(durations)[int(len(durations) * 0.95)],
            "p99": sorted(durations)[int(len(durations) * 0.99)]
        }
    
    def get_all_stats(self) -> dict[str, dict[str, float]]:
        """Obtener estadísticas de todas las operaciones"""
        return {
            op: self.get_stats(op)
            for op in self.metrics.keys()
            if self.get_stats(op) is not None
        }


def timed_async(monitor: Optional[PerformanceMonitor] = None):
    """
    Decorador para medir el tiempo de ejecución de funciones async.
    
    Args:
        monitor: Monitor de rendimiento opcional para registrar métricas
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start
                if monitor:
                    monitor.record(func.__name__, duration)
        
        return wrapper
    return decorator

