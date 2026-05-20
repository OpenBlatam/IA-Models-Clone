"""
Performance Optimizer - Optimizador de Rendimiento
==================================================

Sistema avanzado de optimización de rendimiento.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
import statistics

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Métricas de rendimiento."""
    operation: str
    count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    avg_time: float = 0.0
    p50_time: float = 0.0
    p95_time: float = 0.0
    p99_time: float = 0.0
    recent_times: deque = field(default_factory=lambda: deque(maxlen=1000))


class PerformanceOptimizer:
    """Optimizador de rendimiento."""
    
    def __init__(self):
        self.metrics: Dict[str, PerformanceMetrics] = {}
        self._lock = asyncio.Lock()
        self.optimization_enabled = True
    
    async def measure(
        self,
        operation: str,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Medir tiempo de ejecución de una operación.
        
        Args:
            operation: Nombre de la operación
            func: Función a ejecutar
            *args, **kwargs: Argumentos para la función
        
        Returns:
            Resultado de la función
        """
        start_time = time.time()
        
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            execution_time = time.time() - start_time
            await self._record_metric(operation, execution_time)
            
            return result
        
        except Exception as e:
            execution_time = time.time() - start_time
            await self._record_metric(operation, execution_time, error=True)
            raise
    
    async def _record_metric(self, operation: str, execution_time: float, error: bool = False):
        """Registrar métrica."""
        async with self._lock:
            if operation not in self.metrics:
                self.metrics[operation] = PerformanceMetrics(operation=operation)
            
            metric = self.metrics[operation]
            metric.count += 1
            metric.total_time += execution_time
            metric.min_time = min(metric.min_time, execution_time)
            metric.max_time = max(metric.max_time, execution_time)
            metric.recent_times.append(execution_time)
            
            # Calcular estadísticas
            if metric.recent_times:
                times = list(metric.recent_times)
                metric.avg_time = statistics.mean(times)
                sorted_times = sorted(times)
                metric.p50_time = statistics.median(sorted_times)
                metric.p95_time = sorted_times[int(len(sorted_times) * 0.95)] if len(sorted_times) > 0 else 0
                metric.p99_time = sorted_times[int(len(sorted_times) * 0.99)] if len(sorted_times) > 0 else 0
    
    def get_metrics(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """Obtener métricas."""
        if operation:
            if operation in self.metrics:
                m = self.metrics[operation]
                return {
                    "operation": m.operation,
                    "count": m.count,
                    "total_time": m.total_time,
                    "min_time": m.min_time,
                    "max_time": m.max_time,
                    "avg_time": m.avg_time,
                    "p50_time": m.p50_time,
                    "p95_time": m.p95_time,
                    "p99_time": m.p99_time,
                }
            return None
        else:
            return {
                op: {
                    "count": m.count,
                    "avg_time": m.avg_time,
                    "p95_time": m.p95_time,
                }
                for op, m in self.metrics.items()
            }
    
    async def optimize_slow_operations(self, threshold: float = 1.0):
        """Identificar y optimizar operaciones lentas."""
        slow_operations = []
        
        async with self._lock:
            for op, metric in self.metrics.items():
                if metric.avg_time > threshold:
                    slow_operations.append({
                        "operation": op,
                        "avg_time": metric.avg_time,
                        "count": metric.count,
                    })
        
        if slow_operations:
            logger.warning(f"Slow operations detected: {slow_operations}")
        
        return slow_operations
    
    def reset_metrics(self, operation: Optional[str] = None):
        """Resetear métricas."""
        async with self._lock:
            if operation:
                if operation in self.metrics:
                    del self.metrics[operation]
            else:
                self.metrics.clear()


class ConnectionPool:
    """Pool de conexiones para optimización."""
    
    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self.pool: deque = deque(maxlen=max_size)
        self._lock = asyncio.Lock()
    
    async def get(self):
        """Obtener conexión del pool."""
        async with self._lock:
            if self.pool:
                return self.pool.popleft()
            return None
    
    async def put(self, connection):
        """Retornar conexión al pool."""
        async with self._lock:
            if len(self.pool) < self.max_size:
                self.pool.append(connection)
































