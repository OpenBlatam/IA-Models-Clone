"""
Utilidades de Desarrollo y Debugging.
"""

import time
import functools
import traceback
from typing import Any, Callable, Dict, Optional
from datetime import datetime
from contextlib import contextmanager

from config.logging_config import get_logger

logger = get_logger(__name__)


def timing_decorator(func: Callable) -> Callable:
    """
    Decorador para medir tiempo de ejecución.
    
    Args:
        func: Función a decorar
        
    Returns:
        Función decorada
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start
            logger.debug(f"{func.__name__} ejecutado en {duration:.4f}s")
            return result
        except Exception as e:
            duration = time.time() - start
            logger.error(f"{func.__name__} falló después de {duration:.4f}s: {e}")
            raise
    
    return wrapper


def async_timing_decorator(func: Callable) -> Callable:
    """
    Decorador para medir tiempo de ejecución de funciones async.
    
    Args:
        func: Función async a decorar
        
    Returns:
        Función decorada
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start
            logger.debug(f"{func.__name__} ejecutado en {duration:.4f}s")
            return result
        except Exception as e:
            duration = time.time() - start
            logger.error(f"{func.__name__} falló después de {duration:.4f}s: {e}")
            raise
    
    return wrapper


@contextmanager
def measure_time(operation_name: str):
    """
    Context manager para medir tiempo de ejecución.
    
    Args:
        operation_name: Nombre de la operación
        
    Example:
        with measure_time("database_query"):
            result = db.query(...)
    """
    start = time.time()
    try:
        yield
    finally:
        duration = time.time() - start
        logger.debug(f"{operation_name} completado en {duration:.4f}s")


def log_function_call(func: Callable) -> Callable:
    """
    Decorador para loggear llamadas a funciones.
    
    Args:
        func: Función a decorar
        
    Returns:
        Función decorada
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"Llamando {func.__name__} con args={args}, kwargs={kwargs}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} retornó: {result}")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} lanzó excepción: {e}")
            logger.error(traceback.format_exc())
            raise
    
    return wrapper


def print_memory_usage():
    """Imprimir uso de memoria."""
    try:
        import psutil
        import os
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        logger.info(f"Memoria usada: {mem_info.rss / 1024 / 1024:.2f} MB")
    except ImportError:
        logger.warning("psutil no disponible, no se puede medir memoria")


def debug_dict(d: Dict[str, Any], prefix: str = "") -> None:
    """
    Imprimir diccionario de forma legible.
    
    Args:
        d: Diccionario a imprimir
        prefix: Prefijo para cada línea
    """
    for key, value in d.items():
        if isinstance(value, dict):
            logger.debug(f"{prefix}{key}:")
            debug_dict(value, prefix + "  ")
        else:
            logger.debug(f"{prefix}{key}: {value}")


class PerformanceMonitor:
    """Monitor de rendimiento."""
    
    def __init__(self):
        """Inicializar monitor."""
        self.operations: Dict[str, list] = {}
    
    def record(self, operation: str, duration: float) -> None:
        """
        Registrar duración de operación.
        
        Args:
            operation: Nombre de la operación
            duration: Duración en segundos
        """
        if operation not in self.operations:
            self.operations[operation] = []
        self.operations[operation].append(duration)
    
    def get_stats(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """
        Obtener estadísticas.
        
        Args:
            operation: Operación específica (opcional)
            
        Returns:
            Estadísticas
        """
        if operation:
            if operation not in self.operations:
                return {}
            durations = self.operations[operation]
        else:
            # Estadísticas globales
            all_durations = []
            for durations in self.operations.values():
                all_durations.extend(durations)
            durations = all_durations
        
        if not durations:
            return {}
        
        durations.sort()
        return {
            "count": len(durations),
            "min": min(durations),
            "max": max(durations),
            "avg": sum(durations) / len(durations),
            "median": durations[len(durations) // 2],
            "p95": durations[int(len(durations) * 0.95)],
            "p99": durations[int(len(durations) * 0.99)]
        }
    
    def reset(self) -> None:
        """Resetear estadísticas."""
        self.operations.clear()


# Instancia global
performance_monitor = PerformanceMonitor()



