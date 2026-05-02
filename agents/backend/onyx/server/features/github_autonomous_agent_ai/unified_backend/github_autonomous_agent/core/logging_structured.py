"""
Logging Estructurado con Contexto Enriquecido.
"""

import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from contextvars import ContextVar
from functools import wraps

from config.logging_config import get_logger

logger = get_logger(__name__)

# Context variables para logging estructurado
request_id_var: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
user_id_var: ContextVar[Optional[str]] = ContextVar('user_id', default=None)
correlation_id_var: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)


class StructuredFormatter(logging.Formatter):
    """Formatter para logging estructurado en JSON."""
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Formatear log record como JSON estructurado.
        
        Args:
            record: Log record
            
        Returns:
            String JSON formateado
        """
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Agregar contexto
        if request_id_var.get():
            log_data["request_id"] = request_id_var.get()
        if user_id_var.get():
            log_data["user_id"] = user_id_var.get()
        if correlation_id_var.get():
            log_data["correlation_id"] = correlation_id_var.get()
        
        # Agregar excepción si existe
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Agregar extra fields
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)
        
        return json.dumps(log_data, ensure_ascii=False)


class StructuredLogger:
    """Logger con soporte para contexto estructurado."""
    
    def __init__(self, name: str):
        """
        Inicializar logger estructurado.
        
        Args:
            name: Nombre del logger
        """
        self.logger = logging.getLogger(name)
        self.name = name
    
    def _log_with_context(
        self,
        level: int,
        message: str,
        *args,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        """
        Log con contexto adicional.
        
        Args:
            level: Nivel de log
            message: Mensaje
            *args: Argumentos adicionales
            context: Contexto adicional
            **kwargs: Keyword arguments
        """
        # Agregar contexto al record
        if context:
            if not hasattr(self.logger, 'extra_fields'):
                self.logger.extra_fields = {}
            self.logger.extra_fields.update(context)
        
        # Agregar contexto de context vars
        extra = kwargs.get('extra', {})
        if request_id_var.get():
            extra['request_id'] = request_id_var.get()
        if user_id_var.get():
            extra['user_id'] = user_id_var.get()
        if correlation_id_var.get():
            extra['correlation_id'] = correlation_id_var.get()
        
        kwargs['extra'] = extra
        self.logger.log(level, message, *args, **kwargs)
    
    def debug(self, message: str, context: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        """Log debug con contexto."""
        self._log_with_context(logging.DEBUG, message, context=context, **kwargs)
    
    def info(self, message: str, context: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        """Log info con contexto."""
        self._log_with_context(logging.INFO, message, context=context, **kwargs)
    
    def warning(self, message: str, context: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        """Log warning con contexto."""
        self._log_with_context(logging.WARNING, message, context=context, **kwargs)
    
    def error(self, message: str, context: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        """Log error con contexto."""
        self._log_with_context(logging.ERROR, message, context=context, **kwargs)
    
    def critical(self, message: str, context: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        """Log critical con contexto."""
        self._log_with_context(logging.CRITICAL, message, context=context, **kwargs)
    
    def with_context(self, **context):
        """
        Crear logger con contexto fijo.
        
        Args:
            **context: Contexto a agregar
            
        Returns:
            Logger con contexto
        """
        class ContextLogger:
            def __init__(self, base_logger, ctx):
                self.base_logger = base_logger
                self.ctx = ctx
            
            def _merge_context(self, additional_context):
                merged = self.ctx.copy()
                if additional_context:
                    merged.update(additional_context)
                return merged
            
            def debug(self, message: str, context: Optional[Dict[str, Any]] = None, **kwargs):
                self.base_logger.debug(message, context=self._merge_context(context), **kwargs)
            
            def info(self, message: str, context: Optional[Dict[str, Any]] = None, **kwargs):
                self.base_logger.info(message, context=self._merge_context(context), **kwargs)
            
            def warning(self, message: str, context: Optional[Dict[str, Any]] = None, **kwargs):
                self.base_logger.warning(message, context=self._merge_context(context), **kwargs)
            
            def error(self, message: str, context: Optional[Dict[str, Any]] = None, **kwargs):
                self.base_logger.error(message, context=self._merge_context(context), **kwargs)
            
            def critical(self, message: str, context: Optional[Dict[str, Any]] = None, **kwargs):
                self.base_logger.critical(message, context=self._merge_context(context), **kwargs)
        
        return ContextLogger(self, context)


def set_request_context(request_id: str, user_id: Optional[str] = None, correlation_id: Optional[str] = None) -> None:
    """
    Establecer contexto de request.
    
    Args:
        request_id: ID de la request
        user_id: ID del usuario (opcional)
        correlation_id: ID de correlación (opcional)
    """
    request_id_var.set(request_id)
    if user_id:
        user_id_var.set(user_id)
    if correlation_id:
        correlation_id_var.set(correlation_id)


def clear_request_context() -> None:
    """Limpiar contexto de request."""
    request_id_var.set(None)
    user_id_var.set(None)
    correlation_id_var.set(None)


def log_function_call(func):
    """
    Decorador para logging automático de llamadas a funciones.
    
    Args:
        func: Función a decorar
        
    Returns:
        Función decorada
    """
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        logger = StructuredLogger(func.__module__)
        logger.debug(
            f"Calling {func.__name__}",
            context={
                "function": func.__name__,
                "args_count": len(args),
                "kwargs_keys": list(kwargs.keys())
            }
        )
        try:
            result = await func(*args, **kwargs)
            logger.debug(
                f"Completed {func.__name__}",
                context={"function": func.__name__}
            )
            return result
        except Exception as e:
            logger.error(
                f"Error in {func.__name__}: {e}",
                context={"function": func.__name__},
                exc_info=True
            )
            raise
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        logger = StructuredLogger(func.__module__)
        logger.debug(
            f"Calling {func.__name__}",
            context={
                "function": func.__name__,
                "args_count": len(args),
                "kwargs_keys": list(kwargs.keys())
            }
        )
        try:
            result = func(*args, **kwargs)
            logger.debug(
                f"Completed {func.__name__}",
                context={"function": func.__name__}
            )
            return result
        except Exception as e:
            logger.error(
                f"Error in {func.__name__}: {e}",
                context={"function": func.__name__},
                exc_info=True
            )
            raise
    
    import asyncio
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


def get_structured_logger(name: str) -> StructuredLogger:
    """
    Obtener logger estructurado.
    
    Args:
        name: Nombre del logger
        
    Returns:
        StructuredLogger
    """
    return StructuredLogger(name)



