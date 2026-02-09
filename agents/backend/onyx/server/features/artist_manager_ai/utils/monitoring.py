"""
Monitoring Utilities
====================

Utilidades de monitoreo avanzado.
"""

import logging
import psutil
import time
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class SystemMonitor:
    """Monitor del sistema."""
    
    def __init__(self):
        """Inicializar monitor."""
        self._logger = logger
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """
        Obtener métricas del sistema.
        
        Returns:
            Métricas del sistema
        """
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                "cpu": {
                    "percent": cpu_percent,
                    "count": psutil.cpu_count()
                },
                "memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "percent": memory.percent,
                    "used": memory.used
                },
                "disk": {
                    "total": disk.total,
                    "used": disk.used,
                    "free": disk.free,
                    "percent": (disk.used / disk.total) * 100
                },
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self._logger.error(f"Error getting system metrics: {str(e)}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_process_metrics(self) -> Dict[str, Any]:
        """
        Obtener métricas del proceso actual.
        
        Returns:
            Métricas del proceso
        """
        try:
            process = psutil.Process()
            
            return {
                "pid": process.pid,
                "cpu_percent": process.cpu_percent(interval=0.1),
                "memory": {
                    "rss": process.memory_info().rss,
                    "vms": process.memory_info().vms,
                    "percent": process.memory_percent()
                },
                "num_threads": process.num_threads(),
                "open_files": len(process.open_files()),
                "connections": len(process.connections()),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self._logger.error(f"Error getting process metrics: {str(e)}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


class PerformanceMonitor:
    """Monitor de rendimiento de aplicación."""
    
    def __init__(self):
        """Inicializar monitor."""
        self.operation_times: Dict[str, list] = {}
        self._logger = logger
    
    def track_operation(self, operation_name: str):
        """
        Decorador para trackear operación.
        
        Args:
            operation_name: Nombre de la operación
        
        Returns:
            Decorador
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    elapsed = time.time() - start_time
                    self._record_time(operation_name, elapsed)
                    return result
                except Exception as e:
                    elapsed = time.time() - start_time
                    self._record_time(operation_name, elapsed, error=True)
                    raise
            
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    elapsed = time.time() - start_time
                    self._record_time(operation_name, elapsed)
                    return result
                except Exception as e:
                    elapsed = time.time() - start_time
                    self._record_time(operation_name, elapsed, error=True)
                    raise
            
            if hasattr(func, '__code__') and func.__code__.co_flags & 0x80:
                return async_wrapper
            return wrapper
        
        return decorator
    
    def _record_time(self, operation_name: str, elapsed: float, error: bool = False):
        """Registrar tiempo de operación."""
        if operation_name not in self.operation_times:
            self.operation_times[operation_name] = []
        
        self.operation_times[operation_name].append({
            "elapsed": elapsed,
            "error": error,
            "timestamp": datetime.now().isoformat()
        })
        
        # Mantener solo últimos 1000 registros
        if len(self.operation_times[operation_name]) > 1000:
            self.operation_times[operation_name] = self.operation_times[operation_name][-1000:]
    
    def get_operation_stats(self, operation_name: str) -> Dict[str, Any]:
        """
        Obtener estadísticas de operación.
        
        Args:
            operation_name: Nombre de la operación
        
        Returns:
            Estadísticas
        """
        if operation_name not in self.operation_times:
            return {}
        
        times = [t["elapsed"] for t in self.operation_times[operation_name]]
        errors = [t for t in self.operation_times[operation_name] if t["error"]]
        
        if not times:
            return {}
        
        return {
            "operation": operation_name,
            "count": len(times),
            "error_count": len(errors),
            "error_rate": len(errors) / len(times),
            "avg_time": sum(times) / len(times),
            "min_time": min(times),
            "max_time": max(times),
            "p50": sorted(times)[len(times) // 2],
            "p95": sorted(times)[int(len(times) * 0.95)],
            "p99": sorted(times)[int(len(times) * 0.99)]
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Obtener estadísticas de todas las operaciones."""
        return {
            name: self.get_operation_stats(name)
            for name in self.operation_times.keys()
        }




