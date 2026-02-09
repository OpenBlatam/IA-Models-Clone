"""
Advanced Logging Utilities
==========================
Utilidades avanzadas de logging.
"""

import logging
import asyncio
from typing import Any, Dict, Optional, List
from datetime import datetime
from contextlib import contextmanager
import traceback

from .logger import get_logger

logger = get_logger(__name__)


class LogContext:
    """Context manager para logging con contexto."""
    
    def __init__(self, **context):
        """
        Inicializar log context.
        
        Args:
            **context: Variables de contexto
        """
        self.context = context
        self.old_context = {}
    
    def __enter__(self):
        """Entrar al contexto."""
        # Guardar contexto anterior
        self.old_context = logger._context.copy() if hasattr(logger, '_context') else {}
        
        # Agregar nuevo contexto
        for key, value in self.context.items():
            logger.bind(**{key: value})
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Salir del contexto."""
        # Restaurar contexto anterior
        if self.old_context:
            logger._context = self.old_context


class PerformanceLogger:
    """Logger para performance."""
    
    def __init__(self, threshold_ms: float = 1000.0):
        """
        Inicializar performance logger.
        
        Args:
            threshold_ms: Threshold en milisegundos para logging
        """
        self.threshold_ms = threshold_ms
    
    @contextmanager
    def log_performance(self, operation: str, **context):
        """
        Context manager para logging de performance.
        
        Args:
            operation: Nombre de la operación
            **context: Contexto adicional
        """
        start = datetime.now()
        
        try:
            yield
        except Exception as e:
            logger.error(
                f"Operation {operation} failed",
                operation=operation,
                error=str(e),
                **context
            )
            raise
        finally:
            elapsed = (datetime.now() - start).total_seconds() * 1000
            
            if elapsed >= self.threshold_ms:
                logger.warning(
                    f"Slow operation: {operation}",
                    operation=operation,
                    duration_ms=elapsed,
                    threshold_ms=self.threshold_ms,
                    **context
                )
            else:
                logger.debug(
                    f"Operation {operation} completed",
                    operation=operation,
                    duration_ms=elapsed,
                    **context
                )


class StructuredLogger:
    """Logger estructurado avanzado."""
    
    def __init__(self, name: str):
        """
        Inicializar structured logger.
        
        Args:
            name: Nombre del logger
        """
        self.name = name
        self.logger = get_logger(name)
        self.metrics: Dict[str, Any] = {}
    
    def log_event(
        self,
        event_type: str,
        message: str,
        level: str = "info",
        **kwargs
    ):
        """
        Loggear evento estructurado.
        
        Args:
            event_type: Tipo de evento
            message: Mensaje
            level: Nivel de log
            **kwargs: Campos adicionales
        """
        log_method = getattr(self.logger, level.lower(), self.logger.info)
        
        log_method(
            message,
            event_type=event_type,
            timestamp=datetime.now().isoformat(),
            **kwargs
        )
    
    def log_metric(self, metric_name: str, value: float, **tags):
        """
        Loggear métrica.
        
        Args:
            metric_name: Nombre de la métrica
            value: Valor
            **tags: Tags adicionales
        """
        self.metrics[metric_name] = {
            "value": value,
            "timestamp": datetime.now().isoformat(),
            "tags": tags
        }
        
        self.logger.info(
            f"Metric: {metric_name}",
            metric_name=metric_name,
            metric_value=value,
            **tags
        )
    
    def log_error_with_context(
        self,
        error: Exception,
        context: Dict[str, Any],
        level: str = "error"
    ):
        """
        Loggear error con contexto.
        
        Args:
            error: Excepción
            context: Contexto adicional
            level: Nivel de log
        """
        log_method = getattr(self.logger, level.lower(), self.logger.error)
        
        log_method(
            f"Error: {str(error)}",
            error_type=type(error).__name__,
            error_message=str(error),
            traceback=traceback.format_exc(),
            **context
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Obtener métricas acumuladas."""
        return self.metrics.copy()


def create_log_context(**context) -> LogContext:
    """
    Crear contexto de logging.
    
    Args:
        **context: Variables de contexto
        
    Returns:
        LogContext
    """
    return LogContext(**context)


def log_performance(operation: str, threshold_ms: float = 1000.0):
    """
    Decorator para logging de performance.
    
    Args:
        operation: Nombre de la operación
        threshold_ms: Threshold en milisegundos
    """
    def decorator(func):
        perf_logger = PerformanceLogger(threshold_ms)
        
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                with perf_logger.log_performance(operation):
                    return await func(*args, **kwargs)
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                with perf_logger.log_performance(operation):
                    return func(*args, **kwargs)
            return sync_wrapper
    
    return decorator

