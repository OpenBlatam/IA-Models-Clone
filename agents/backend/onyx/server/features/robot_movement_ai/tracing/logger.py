"""
Structured Logger - Logger estructurado mejorado
"""
import json
import logging
import uuid
import traceback
from typing import Any, Dict, Optional
from datetime import datetime
from contextvars import ContextVar
from functools import wraps

request_id_var: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
operation_name_var: ContextVar[Optional[str]] = ContextVar('operation_name', default=None)


class StructuredLogger:
    """Logger estructurado mejorado con contexto y correlación"""
    
    def __init__(self, name: Optional[str] = None, include_stack_trace: bool = False):
        """
        Inicializar logger estructurado.
        
        Args:
            name: Nombre del logger
            include_stack_trace: Incluir stack trace en errores
        """
        self.logger = logging.getLogger(name or __name__)
        self.include_stack_trace = include_stack_trace
        self._context: Dict[str, Any] = {}
    
    def set_context(self, **kwargs):
        """Establecer contexto global para todos los logs."""
        self._context.update(kwargs)
    
    def clear_context(self):
        """Limpiar contexto global."""
        self._context.clear()
    
    def _get_context(self, **kwargs) -> Dict[str, Any]:
        """Obtener contexto completo para el log."""
        context = {
            'timestamp': datetime.utcnow().isoformat(),
            **self._context
        }
        
        request_id = request_id_var.get()
        if request_id:
            context['request_id'] = request_id
        
        operation_name = operation_name_var.get()
        if operation_name:
            context['operation'] = operation_name
        
        context.update(kwargs)
        return context
    
    def log(
        self,
        level: str,
        message: str,
        exc_info: Optional[Exception] = None,
        **kwargs
    ):
        """
        Registra un log estructurado.
        
        Args:
            level: Nivel de log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            message: Mensaje del log
            exc_info: Excepción opcional para incluir traceback
            **kwargs: Campos adicionales para el log
        """
        log_data = self._get_context(
            level=level.upper(),
            message=message,
            **kwargs
        )
        
        if exc_info:
            log_data['exception'] = {
                'type': exc_info.__class__.__name__,
                'message': str(exc_info)
            }
            if self.include_stack_trace:
                log_data['exception']['traceback'] = traceback.format_exc()
        
        log_message = json.dumps(log_data, default=str)
        
        log_level = getattr(logging, level.upper(), logging.INFO)
        self.logger.log(log_level, log_message, exc_info=exc_info)
    
    def debug(self, message: str, **kwargs):
        """Log de debug."""
        self.log('DEBUG', message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log de info."""
        self.log('INFO', message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log de warning."""
        self.log('WARNING', message, **kwargs)
    
    def error(self, message: str, exc_info: Optional[Exception] = None, **kwargs):
        """Log de error."""
        self.log('ERROR', message, exc_info=exc_info, **kwargs)
    
    def critical(self, message: str, exc_info: Optional[Exception] = None, **kwargs):
        """Log crítico."""
        self.log('CRITICAL', message, exc_info=exc_info, **kwargs)
    
    def log_performance(
        self,
        operation: str,
        duration_ms: float,
        **kwargs
    ):
        """
        Log de métrica de rendimiento.
        
        Args:
            operation: Nombre de la operación
            duration_ms: Duración en milisegundos
            **kwargs: Métricas adicionales
        """
        self.info(
            f"Performance: {operation}",
            operation=operation,
            duration_ms=duration_ms,
            metric_type='performance',
            **kwargs
        )
    
    def log_metric(
        self,
        name: str,
        value: float,
        unit: Optional[str] = None,
        **kwargs
    ):
        """
        Log de métrica.
        
        Args:
            name: Nombre de la métrica
            value: Valor de la métrica
            unit: Unidad de la métrica
            **kwargs: Tags adicionales
        """
        self.info(
            f"Metric: {name}={value}",
            metric_name=name,
            metric_value=value,
            metric_unit=unit,
            metric_type='metric',
            **kwargs
        )


def with_request_id(request_id: Optional[str] = None):
    """
    Decorador para establecer request ID en contexto.
    
    Args:
        request_id: ID de request (se genera si no se proporciona)
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            req_id = request_id or str(uuid.uuid4())
            token = request_id_var.set(req_id)
            try:
                return await func(*args, **kwargs)
            finally:
                request_id_var.reset(token)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            req_id = request_id or str(uuid.uuid4())
            token = request_id_var.set(req_id)
            try:
                return func(*args, **kwargs)
            finally:
                request_id_var.reset(token)
        
        if hasattr(func, '__code__') and func.__code__.co_flags & 0x80:
            return async_wrapper
        return sync_wrapper
    return decorator


def with_operation_name(operation: str):
    """
    Decorador para establecer nombre de operación en contexto.
    
    Args:
        operation: Nombre de la operación
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            token = operation_name_var.set(operation)
            try:
                return await func(*args, **kwargs)
            finally:
                operation_name_var.reset(token)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            token = operation_name_var.set(operation)
            try:
                return func(*args, **kwargs)
            finally:
                operation_name_var.reset(token)
        
        if hasattr(func, '__code__') and func.__code__.co_flags & 0x80:
            return async_wrapper
        return sync_wrapper
    return decorator

