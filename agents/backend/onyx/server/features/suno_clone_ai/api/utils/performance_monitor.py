"""
Monitor de rendimiento para endpoints

Incluye decoradores y utilidades para monitorear el rendimiento de endpoints.
"""

import time
import logging
from functools import wraps
from typing import Callable, Any, Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Métricas de rendimiento
_performance_metrics: dict = {}


@contextmanager
def measure_time(operation_name: str):
    """
    Context manager para medir el tiempo de ejecución.
    
    Args:
        operation_name: Nombre de la operación a medir
        
    Example:
        with measure_time("database_query"):
            result = db.query(...)
    """
    start_time = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        _record_metric(operation_name, elapsed)
        logger.debug(f"{operation_name} took {elapsed:.3f}s")


def performance_monitor(operation_name: Optional[str] = None):
    """
    Decorador para monitorear el rendimiento de funciones.
    
    Args:
        operation_name: Nombre de la operación (opcional, usa nombre de función si no se proporciona)
        
    Example:
        @performance_monitor("generate_song")
        async def generate_song(...):
            ...
    """
    def decorator(func: Callable) -> Callable:
        op_name = operation_name or func.__name__
        
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                elapsed = time.time() - start_time
                _record_metric(op_name, elapsed)
                logger.debug(f"{op_name} took {elapsed:.3f}s")
        
        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                elapsed = time.time() - start_time
                _record_metric(op_name, elapsed)
                logger.debug(f"{op_name} took {elapsed:.3f}s")
        
        # Retornar wrapper apropiado
        import inspect
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def _record_metric(operation_name: str, elapsed_time: float) -> None:
    """Registra una métrica de rendimiento"""
    if operation_name not in _performance_metrics:
        _performance_metrics[operation_name] = {
            "count": 0,
            "total_time": 0.0,
            "min_time": float('inf'),
            "max_time": 0.0
        }
    
    metrics = _performance_metrics[operation_name]
    metrics["count"] += 1
    metrics["total_time"] += elapsed_time
    metrics["min_time"] = min(metrics["min_time"], elapsed_time)
    metrics["max_time"] = max(metrics["max_time"], elapsed_time)


def get_performance_stats(operation_name: Optional[str] = None) -> dict:
    """
    Obtiene estadísticas de rendimiento.
    
    Args:
        operation_name: Nombre de la operación (opcional, retorna todas si no se especifica)
        
    Returns:
        Diccionario con estadísticas de rendimiento
    """
    if operation_name:
        if operation_name not in _performance_metrics:
            return {}
        
        metrics = _performance_metrics[operation_name]
        count = metrics["count"]
        return {
            "operation": operation_name,
            "count": count,
            "total_time": metrics["total_time"],
            "avg_time": metrics["total_time"] / count if count > 0 else 0,
            "min_time": metrics["min_time"] if metrics["min_time"] != float('inf') else 0,
            "max_time": metrics["max_time"]
        }
    
    # Retornar todas las métricas
    result = {}
    for op_name, metrics in _performance_metrics.items():
        count = metrics["count"]
        result[op_name] = {
            "count": count,
            "total_time": metrics["total_time"],
            "avg_time": metrics["total_time"] / count if count > 0 else 0,
            "min_time": metrics["min_time"] if metrics["min_time"] != float('inf') else 0,
            "max_time": metrics["max_time"]
        }
    
    return result


def clear_performance_stats() -> None:
    """Limpia todas las métricas de rendimiento"""
    _performance_metrics.clear()

