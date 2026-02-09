"""
Monitoring Utilities
====================
Utilidades para monitoreo y observabilidad.
"""

import time
from typing import Dict, Any, Optional
from datetime import datetime
from functools import wraps
from ...utils.logger import get_logger
from ...utils.metrics import request_duration_seconds

logger = get_logger(__name__)


class PerformanceMonitor:
    """Monitor de rendimiento."""
    
    def __init__(self):
        self.metrics: Dict[str, Any] = {
            "requests": 0,
            "errors": 0,
            "total_duration": 0.0,
            "start_time": datetime.now()
        }
    
    def record_request(self, duration: float, success: bool = True):
        """Registrar una request."""
        self.metrics["requests"] += 1
        self.metrics["total_duration"] += duration
        
        if not success:
            self.metrics["errors"] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        requests = self.metrics["requests"]
        avg_duration = (
            self.metrics["total_duration"] / requests
            if requests > 0 else 0.0
        )
        
        uptime = (datetime.now() - self.metrics["start_time"]).total_seconds()
        
        return {
            "total_requests": requests,
            "total_errors": self.metrics["errors"],
            "average_duration": avg_duration,
            "error_rate": (
                self.metrics["errors"] / requests * 100
                if requests > 0 else 0.0
            ),
            "uptime_seconds": uptime,
            "requests_per_second": requests / uptime if uptime > 0 else 0.0
        }


def track_performance(func_name: Optional[str] = None):
    """
    Decorador para trackear rendimiento de funciones.
    
    Args:
        func_name: Nombre de la función (opcional)
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            name = func_name or func.__name__
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Registrar métrica
                request_duration_seconds.labels(
                    endpoint=name,
                    method="async"
                ).observe(duration)
                
                logger.debug(
                    f"{name} completed in {duration:.3f}s",
                    duration=duration,
                    success=True
                )
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                    f"{name} failed after {duration:.3f}s",
                    duration=duration,
                    error=str(e),
                    success=False
                )
                raise
        
        return wrapper
    return decorator


def get_system_info() -> Dict[str, Any]:
    """
    Obtener información del sistema.
    
    Returns:
        Diccionario con información del sistema
    """
    import platform
    import sys
    import psutil
    
    try:
        return {
            "platform": platform.platform(),
            "python_version": sys.version,
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available,
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent
        }
    except ImportError:
        return {
            "platform": platform.platform(),
            "python_version": sys.version,
            "note": "psutil not available"
        }

