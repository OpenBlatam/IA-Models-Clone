"""
Performance Utilities
=====================

Utilidades para optimización de rendimiento.
"""

import logging
import time
import functools
from typing import Callable, Any, Dict
from datetime import datetime

logger = logging.getLogger(__name__)


def measure_time(func: Callable) -> Callable:
    """
    Decorador para medir tiempo de ejecución.
    
    Usage:
        @measure_time
        def my_function():
            ...
    """
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.debug(f"{func.__name__} took {elapsed:.3f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"{func.__name__} failed after {elapsed:.3f}s: {str(e)}")
            raise
    
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.debug(f"{func.__name__} took {elapsed:.3f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"{func.__name__} failed after {elapsed:.3f}s: {str(e)}")
            raise
    
    if hasattr(func, '__code__') and func.__code__.co_flags & 0x80:  # CO_COROUTINE
        return async_wrapper
    return sync_wrapper


class PerformanceMonitor:
    """Monitor de rendimiento."""
    
    def __init__(self):
        """Inicializar monitor."""
        self.metrics: Dict[str, list] = {}
        self._logger = logger
    
    def record_operation(
        self,
        operation_name: str,
        duration: float,
        success: bool = True
    ):
        """
        Registrar operación.
        
        Args:
            operation_name: Nombre de la operación
            duration: Duración en segundos
            success: Si fue exitosa
        """
        if operation_name not in self.metrics:
            self.metrics[operation_name] = []
        
        self.metrics[operation_name].append({
            "duration": duration,
            "success": success,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_statistics(self, operation_name: str) -> Dict[str, Any]:
        """
        Obtener estadísticas de una operación.
        
        Args:
            operation_name: Nombre de la operación
        
        Returns:
            Estadísticas
        """
        if operation_name not in self.metrics:
            return {
                "count": 0,
                "avg_duration": 0.0,
                "min_duration": 0.0,
                "max_duration": 0.0,
                "success_rate": 0.0
            }
        
        operations = self.metrics[operation_name]
        durations = [op["duration"] for op in operations]
        successes = [op["success"] for op in operations]
        
        return {
            "count": len(operations),
            "avg_duration": sum(durations) / len(durations) if durations else 0.0,
            "min_duration": min(durations) if durations else 0.0,
            "max_duration": max(durations) if durations else 0.0,
            "success_rate": sum(successes) / len(successes) if successes else 0.0
        }
    
    def get_all_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        Obtener estadísticas de todas las operaciones.
        
        Returns:
            Diccionario de estadísticas
        """
        return {
            name: self.get_statistics(name)
            for name in self.metrics.keys()
        }




