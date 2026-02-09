"""
Helper functions for metrics tracking.
Eliminates repetitive metrics.increment() and metrics.timer() patterns.
"""

from typing import Optional, Dict, Any
from contextlib import contextmanager
from functools import wraps
import time
import logging

logger = logging.getLogger(__name__)


def track_metric(
    metric_name: str,
    tags: Optional[Dict[str, str]] = None,
    increment: bool = True
):
    """
    Decorador para tracking automático de métricas.
    
    Args:
        metric_name: Nombre de la métrica
        tags: Tags adicionales para la métrica
        increment: Si incrementar contador (default: True)
        
    Usage:
        @track_metric("profile_extraction", tags={"platform": "tiktok"})
        async def extract_profile(username: str):
            # código
    """
    def decorator(func):
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            from ..analytics.metrics import get_metrics_collector
            metrics = get_metrics_collector()
            
            if increment:
                metrics.increment(metric_name, tags=tags)
            
            with metrics.timer(f"{metric_name}_duration", tags=tags):
                return func(*args, **kwargs)
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            from ..analytics.metrics import get_metrics_collector
            metrics = get_metrics_collector()
            
            if increment:
                metrics.increment(metric_name, tags=tags)
            
            with metrics.timer(f"{metric_name}_duration", tags=tags):
                return await func(*args, **kwargs)
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


@contextmanager
def track_operation(
    operation_name: str,
    tags: Optional[Dict[str, str]] = None,
    increment: bool = True
):
    """
    Context manager para tracking de operaciones con métricas.
    
    Args:
        operation_name: Nombre de la operación
        tags: Tags adicionales
        increment: Si incrementar contador (default: True)
        
    Usage:
        with track_operation("extract_profile", tags={"platform": "tiktok"}):
            # código de la operación
    """
    from ..analytics.metrics import get_metrics_collector
    metrics = get_metrics_collector()
    
    if increment:
        metrics.increment(operation_name, tags=tags)
    
    with metrics.timer(f"{operation_name}_duration", tags=tags):
        yield


def increment_metric(
    metric_name: str,
    tags: Optional[Dict[str, str]] = None,
    value: float = 1.0
):
    """
    Incrementa una métrica.
    
    Args:
        metric_name: Nombre de la métrica
        tags: Tags adicionales
        value: Valor a incrementar (default: 1.0)
        
    Usage:
        increment_metric("profile_extraction_requests", tags={"platform": "tiktok"})
    """
    from ..analytics.metrics import get_metrics_collector
    metrics = get_metrics_collector()
    metrics.increment(metric_name, tags=tags, value=value)


def set_gauge(
    metric_name: str,
    value: float,
    tags: Optional[Dict[str, str]] = None
):
    """
    Establece el valor de una métrica gauge.
    
    Args:
        metric_name: Nombre de la métrica
        value: Valor a establecer
        tags: Tags adicionales
        
    Usage:
        set_gauge("active_connections", 42, tags={"server": "api1"})
    """
    from ..analytics.metrics import get_metrics_collector
    metrics = get_metrics_collector()
    metrics.gauge(metric_name, value, tags=tags)








