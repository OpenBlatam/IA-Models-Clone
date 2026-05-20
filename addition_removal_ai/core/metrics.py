"""
Metrics - Sistema de métricas y monitoreo
"""

import logging
import time
from typing import Dict, Any, Optional
from datetime import datetime
from collections import defaultdict, deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class OperationMetrics:
    """Métricas de una operación"""
    operation_type: str
    duration: float
    success: bool
    content_length: int
    result_length: int
    timestamp: datetime = field(default_factory=datetime.utcnow)
    error: Optional[str] = None


class MetricsCollector:
    """Recolector de métricas del sistema"""

    def __init__(self, max_history: int = 1000):
        """
        Inicializar el recolector de métricas.

        Args:
            max_history: Número máximo de métricas a mantener
        """
        self.max_history = max_history
        self.operations: deque = deque(maxlen=max_history)
        self.counters: Dict[str, int] = defaultdict(int)
        self.timings: Dict[str, list] = defaultdict(list)
        self.errors: deque = deque(maxlen=100)

    def record_operation(
        self,
        operation_type: str,
        duration: float,
        success: bool,
        content_length: int,
        result_length: int,
        error: Optional[str] = None
    ):
        """
        Registrar una operación.

        Args:
            operation_type: Tipo de operación (add, remove, batch_add, etc.)
            duration: Duración en segundos
            success: Si la operación fue exitosa
            content_length: Longitud del contenido original
            result_length: Longitud del contenido resultante
            error: Mensaje de error si aplica
        """
        metric = OperationMetrics(
            operation_type=operation_type,
            duration=duration,
            success=success,
            content_length=content_length,
            result_length=result_length,
            error=error
        )
        
        self.operations.append(metric)
        self.counters[f"{operation_type}_total"] += 1
        
        if success:
            self.counters[f"{operation_type}_success"] += 1
        else:
            self.counters[f"{operation_type}_failed"] += 1
            if error:
                self.errors.append({
                    "operation": operation_type,
                    "error": error,
                    "timestamp": datetime.utcnow().isoformat()
                })
        
        self.timings[operation_type].append(duration)
        # Mantener solo las últimas 1000 mediciones por tipo
        if len(self.timings[operation_type]) > 1000:
            self.timings[operation_type] = self.timings[operation_type][-1000:]

    def get_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas generales.

        Returns:
            Diccionario con estadísticas
        """
        stats = {
            "total_operations": len(self.operations),
            "counters": dict(self.counters),
            "average_timings": {},
            "error_rate": {},
            "recent_errors": list(self.errors)[-10:]
        }
        
        # Calcular promedios de tiempo
        for op_type, timings in self.timings.items():
            if timings:
                stats["average_timings"][op_type] = {
                    "avg": sum(timings) / len(timings),
                    "min": min(timings),
                    "max": max(timings),
                    "count": len(timings)
                }
        
        # Calcular tasas de error
        for op_type in set(m.operation_type for m in self.operations):
            total = self.counters.get(f"{op_type}_total", 0)
            failed = self.counters.get(f"{op_type}_failed", 0)
            if total > 0:
                stats["error_rate"][op_type] = failed / total
        
        return stats

    def get_operation_stats(self, operation_type: str) -> Dict[str, Any]:
        """
        Obtener estadísticas de un tipo de operación específico.

        Args:
            operation_type: Tipo de operación

        Returns:
            Estadísticas del tipo de operación
        """
        operations = [m for m in self.operations if m.operation_type == operation_type]
        
        if not operations:
            return {"operation_type": operation_type, "count": 0}
        
        durations = [m.duration for m in operations]
        successes = [m for m in operations if m.success]
        
        return {
            "operation_type": operation_type,
            "count": len(operations),
            "success_count": len(successes),
            "failure_count": len(operations) - len(successes),
            "success_rate": len(successes) / len(operations) if operations else 0,
            "avg_duration": sum(durations) / len(durations),
            "min_duration": min(durations),
            "max_duration": max(durations),
            "avg_content_length": sum(m.content_length for m in operations) / len(operations),
            "avg_result_length": sum(m.result_length for m in operations) / len(operations)
        }

    def clear(self):
        """Limpiar todas las métricas"""
        self.operations.clear()
        self.counters.clear()
        self.timings.clear()
        self.errors.clear()


class PerformanceMonitor:
    """Monitor de rendimiento con decoradores"""

    def __init__(self, metrics_collector: MetricsCollector):
        """
        Inicializar el monitor.

        Args:
            metrics_collector: Recolector de métricas
        """
        self.metrics = metrics_collector

    def track_operation(self, operation_type: str):
        """
        Decorador para rastrear operaciones.

        Args:
            operation_type: Tipo de operación
        """
        def decorator(func):
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                success = False
                error = None
                content_length = 0
                result_length = 0
                
                try:
                    result = await func(*args, **kwargs)
                    success = True
                    
                    # Intentar obtener longitudes
                    if isinstance(result, dict):
                        if "content" in result:
                            result_length = len(str(result["content"]))
                        if "original" in kwargs:
                            content_length = len(str(kwargs["original"]))
                        elif len(args) > 0:
                            content_length = len(str(args[0]))
                    
                    return result
                except Exception as e:
                    error = str(e)
                    logger.error(f"Error en {operation_type}: {e}")
                    raise
                finally:
                    duration = time.time() - start_time
                    self.metrics.record_operation(
                        operation_type=operation_type,
                        duration=duration,
                        success=success,
                        content_length=content_length,
                        result_length=result_length,
                        error=error
                    )
            
            return wrapper
        return decorator






