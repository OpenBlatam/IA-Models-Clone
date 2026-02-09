"""
Performance Monitor for Humanoid Devin Robot (Optimizado)
==========================================================

Monitor de rendimiento para operaciones del robot.
"""

import time
import logging
from typing import Dict, Any, List, Optional, Callable
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class OperationMetrics:
    """Métricas de una operación."""
    name: str
    count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    errors: int = 0
    last_execution: Optional[datetime] = None
    
    @property
    def average_time(self) -> float:
        """Tiempo promedio de ejecución."""
        return self.total_time / self.count if self.count > 0 else 0.0
    
    @property
    def success_rate(self) -> float:
        """Tasa de éxito."""
        total = self.count + self.errors
        return (self.count / total * 100) if total > 0 else 0.0


class PerformanceMonitor:
    """
    Monitor de rendimiento para operaciones del robot.
    
    Rastrea tiempos de ejecución, errores y estadísticas de operaciones.
    """
    
    def __init__(self, enabled: bool = True):
        """
        Inicializar monitor de rendimiento.
        
        Args:
            enabled: Habilitar o deshabilitar monitoreo
        """
        self.enabled = enabled
        self.metrics: Dict[str, OperationMetrics] = defaultdict(
            lambda: OperationMetrics(name="")
        )
        self.start_times: Dict[str, float] = {}
    
    def start_operation(self, operation_name: str) -> None:
        """
        Iniciar medición de operación.
        
        Args:
            operation_name: Nombre de la operación
        """
        if not self.enabled:
            return
        
        self.start_times[operation_name] = time.time()
    
    def end_operation(
        self, 
        operation_name: str, 
        success: bool = True,
        error: Optional[Exception] = None
    ) -> float:
        """
        Finalizar medición de operación.
        
        Args:
            operation_name: Nombre de la operación
            success: Si la operación fue exitosa
            error: Excepción si hubo error (opcional)
            
        Returns:
            Tiempo de ejecución en segundos
        """
        if not self.enabled:
            return 0.0
        
        if operation_name not in self.start_times:
            logger.warning(f"Operation {operation_name} ended without start")
            return 0.0
        
        duration = time.time() - self.start_times.pop(operation_name)
        
        # Actualizar métricas
        if operation_name not in self.metrics:
            self.metrics[operation_name] = OperationMetrics(name=operation_name)
        
        metric = self.metrics[operation_name]
        metric.name = operation_name
        
        if success:
            metric.count += 1
            metric.total_time += duration
            metric.min_time = min(metric.min_time, duration)
            metric.max_time = max(metric.max_time, duration)
        else:
            metric.errors += 1
        
        metric.last_execution = datetime.now()
        
        return duration
    
    def measure_operation(
        self, 
        operation_name: str,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Medir operación usando decorador funcional.
        
        Args:
            operation_name: Nombre de la operación
            func: Función a medir
            *args: Argumentos posicionales
            **kwargs: Argumentos con nombre
            
        Returns:
            Resultado de la función
        """
        self.start_operation(operation_name)
        
        try:
            result = func(*args, **kwargs)
            self.end_operation(operation_name, success=True)
            return result
        except Exception as e:
            self.end_operation(operation_name, success=False, error=e)
            raise
    
    async def measure_async_operation(
        self,
        operation_name: str,
        coro,
        *args,
        **kwargs
    ) -> Any:
        """
        Medir operación asíncrona.
        
        Args:
            operation_name: Nombre de la operación
            coro: Corrutina a medir
            *args: Argumentos posicionales
            **kwargs: Argumentos con nombre
            
        Returns:
            Resultado de la corrutina
        """
        self.start_operation(operation_name)
        
        try:
            if callable(coro):
                result = await coro(*args, **kwargs)
            else:
                result = await coro
            self.end_operation(operation_name, success=True)
            return result
        except Exception as e:
            self.end_operation(operation_name, success=False, error=e)
            raise
    
    def get_metrics(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Obtener métricas de operación(es).
        
        Args:
            operation_name: Nombre de operación específica (None para todas)
            
        Returns:
            Dict con métricas
        """
        if operation_name:
            if operation_name not in self.metrics:
                return {}
            
            metric = self.metrics[operation_name]
            return {
                "name": metric.name,
                "count": metric.count,
                "errors": metric.errors,
                "total_time": metric.total_time,
                "average_time": metric.average_time,
                "min_time": metric.min_time if metric.min_time != float('inf') else 0.0,
                "max_time": metric.max_time,
                "success_rate": metric.success_rate,
                "last_execution": metric.last_execution.isoformat() if metric.last_execution else None
            }
        else:
            return {
                name: {
                    "name": metric.name,
                    "count": metric.count,
                    "errors": metric.errors,
                    "total_time": metric.total_time,
                    "average_time": metric.average_time,
                    "min_time": metric.min_time if metric.min_time != float('inf') else 0.0,
                    "max_time": metric.max_time,
                    "success_rate": metric.success_rate,
                    "last_execution": metric.last_execution.isoformat() if metric.last_execution else None
                }
                for name, metric in self.metrics.items()
            }
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Obtener resumen de todas las métricas.
        
        Returns:
            Dict con resumen
        """
        if not self.metrics:
            return {"total_operations": 0}
        
        total_operations = sum(m.count + m.errors for m in self.metrics.values())
        total_time = sum(m.total_time for m in self.metrics.values())
        total_errors = sum(m.errors for m in self.metrics.values())
        
        return {
            "total_operations": total_operations,
            "total_time": total_time,
            "total_errors": total_errors,
            "success_rate": ((total_operations - total_errors) / total_operations * 100) if total_operations > 0 else 0.0,
            "operations": len(self.metrics),
            "average_time_per_operation": total_time / total_operations if total_operations > 0 else 0.0
        }
    
    def reset(self) -> None:
        """Resetear todas las métricas."""
        self.metrics.clear()
        self.start_times.clear()
        logger.info("Performance metrics reset")
    
    def print_summary(self) -> None:
        """Imprimir resumen de métricas."""
        summary = self.get_summary()
        print("\n" + "=" * 60)
        print("Performance Summary")
        print("=" * 60)
        print(f"Total Operations: {summary['total_operations']}")
        print(f"Total Time: {summary['total_time']:.3f}s")
        print(f"Total Errors: {summary['total_errors']}")
        print(f"Success Rate: {summary['success_rate']:.2f}%")
        print(f"Average Time: {summary['average_time_per_operation']*1000:.2f}ms")
        print("\nOperations:")
        
        for name, metric in sorted(self.metrics.items()):
            print(f"  {name}:")
            print(f"    Count: {metric.count}")
            print(f"    Avg Time: {metric.average_time*1000:.2f}ms")
            print(f"    Min/Max: {metric.min_time*1000:.2f}ms / {metric.max_time*1000:.2f}ms")
            print(f"    Success Rate: {metric.success_rate:.2f}%")
        
        print("=" * 60)

