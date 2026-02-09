"""
Performance Monitor
===================

Monitor de rendimiento para funciones y métodos.
"""

import time
import functools
import logging
from typing import Callable, Any, Dict, Optional

from .metrics import get_metrics_collector

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """
    Monitor de rendimiento.
    """
    
    def __init__(self):
        """Inicializar monitor."""
        self.metrics = get_metrics_collector()
    
    def monitor_function(
        self,
        func: Callable,
        metric_name: Optional[str] = None
    ) -> Callable:
        """
        Decorator para monitorear función.
        
        Args:
            func: Función a monitorear
            metric_name: Nombre de métrica (opcional)
            
        Returns:
            Función decorada
        """
        name = metric_name or f"{func.__module__}.{func.__name__}"
        timer = self.metrics.timer(name)
        counter = self.metrics.counter(f"{name}_calls")
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            counter.inc()
            with timer:
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    error_counter = self.metrics.counter(f"{name}_errors")
                    error_counter.inc()
                    raise
        
        return wrapper
    
    def monitor_class_methods(self, cls: type):
        """
        Decorator para monitorear todos los métodos de una clase.
        
        Args:
            cls: Clase a monitorear
            
        Returns:
            Clase decorada
        """
        for attr_name in dir(cls):
            attr = getattr(cls, attr_name)
            if callable(attr) and not attr_name.startswith("_"):
                metric_name = f"{cls.__name__}.{attr_name}"
                setattr(cls, attr_name, self.monitor_function(attr, metric_name))
        
        return cls


def monitor_function(metric_name: Optional[str] = None):
    """
    Decorator para monitorear función.
    
    Args:
        metric_name: Nombre de métrica (opcional)
        
    Returns:
        Decorator
    """
    monitor = PerformanceMonitor()
    
    def decorator(func: Callable) -> Callable:
        return monitor.monitor_function(func, metric_name)
    
    return decorator


def monitor_class_methods(cls: type):
    """
    Decorator para monitorear todos los métodos de una clase.
    
    Args:
        cls: Clase a monitorear
        
    Returns:
        Clase decorada
    """
    monitor = PerformanceMonitor()
    return monitor.monitor_class_methods(cls)

