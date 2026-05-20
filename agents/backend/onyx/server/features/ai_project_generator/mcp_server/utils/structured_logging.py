"""
Structured Logging - Logging estructurado avanzado
===================================================

Utilidades para logging estructurado con contexto, métricas,
y formateo avanzado.
"""

import logging
import json
import sys
from typing import Any, Dict, Optional, List
from datetime import datetime
from contextvars import ContextVar
from pathlib import Path

logger = logging.getLogger(__name__)

# Context variable para contexto de logging
_log_context: ContextVar[Dict[str, Any]] = ContextVar('log_context', default={})


class StructuredFormatter(logging.Formatter):
    """
    Formatter para logging estructurado.
    
    Formatea logs como JSON estructurado con contexto y metadatos.
    """
    
    def __init__(
        self,
        include_context: bool = True,
        include_traceback: bool = True,
        extra_fields: Optional[Dict[str, Any]] = None
    ):
        """
        Inicializar formatter.
        
        Args:
            include_context: Si True, incluye contexto
            include_traceback: Si True, incluye traceback en errores
            extra_fields: Campos adicionales a incluir
        """
        super().__init__()
        self.include_context = include_context
        self.include_traceback = include_traceback
        self.extra_fields = extra_fields or {}
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Formatear log record.
        
        Args:
            record: Log record
        
        Returns:
            String JSON formateado
        """
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Agregar contexto
        if self.include_context:
            context = _log_context.get({})
            if context:
                log_data["context"] = context
        
        # Agregar campos extra
        if hasattr(record, 'extra'):
            log_data.update(record.extra)
        
        # Agregar campos adicionales
        log_data.update(self.extra_fields)
        
        # Agregar excepción si existe
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
            }
            if self.include_traceback:
                import traceback
                log_data["traceback"] = traceback.format_exception(*record.exc_info)
        
        return json.dumps(log_data, default=str)


class ContextLogger:
    """
    Logger con soporte para contexto.
    
    Permite agregar contexto a logs que persiste durante
    la ejecución de un bloque de código.
    """
    
    def __init__(self, logger: logging.Logger):
        """
        Inicializar context logger.
        
        Args:
            logger: Logger base
        """
        self.logger = logger
        self._context_token = None
    
    def set_context(self, **context: Any) -> None:
        """
        Establecer contexto de logging.
        
        Args:
            **context: Valores de contexto
        
        Example:
            logger.set_context(user_id="123", request_id="abc")
        """
        current = _log_context.get({}).copy()
        current.update(context)
        self._context_token = _log_context.set(current)
    
    def update_context(self, **context: Any) -> None:
        """
        Actualizar contexto existente.
        
        Args:
            **context: Valores de contexto a actualizar
        """
        current = _log_context.get({}).copy()
        current.update(context)
        _log_context.set(current)
    
    def clear_context(self) -> None:
        """Limpiar contexto."""
        if self._context_token:
            _log_context.reset(self._context_token)
            self._context_token = None
    
    def get_context(self) -> Dict[str, Any]:
        """
        Obtener contexto actual.
        
        Returns:
            Diccionario con contexto
        """
        return _log_context.get({}).copy()
    
    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug con contexto."""
        self.logger.debug(message, extra=kwargs)
    
    def info(self, message: str, **kwargs: Any) -> None:
        """Log info con contexto."""
        self.logger.info(message, extra=kwargs)
    
    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning con contexto."""
        self.logger.warning(message, extra=kwargs)
    
    def error(self, message: str, **kwargs: Any) -> None:
        """Log error con contexto."""
        self.logger.error(message, extra=kwargs)
    
    def critical(self, message: str, **kwargs: Any) -> None:
        """Log critical con contexto."""
        self.logger.critical(message, extra=kwargs)


def setup_structured_logging(
    level: str = "INFO",
    format_type: str = "json",
    output: Optional[Path] = None,
    include_context: bool = True
) -> logging.Logger:
    """
    Configurar logging estructurado.
    
    Args:
        level: Nivel de logging
        format_type: Tipo de formato ('json' o 'text')
        output: Archivo de salida (None = stdout)
        include_context: Si True, incluye contexto
    
    Returns:
        Logger configurado
    
    Example:
        logger = setup_structured_logging(level="DEBUG", format_type="json")
    """
    logger = logging.getLogger("mcp_server")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remover handlers existentes
    logger.handlers.clear()
    
    # Crear handler
    if output:
        handler = logging.FileHandler(output)
    else:
        handler = logging.StreamHandler(sys.stdout)
    
    # Configurar formatter
    if format_type == "json":
        formatter = StructuredFormatter(include_context=include_context)
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger


def log_with_context(**context: Any):
    """
    Decorador para agregar contexto a logs de función.
    
    Args:
        **context: Contexto a agregar
    
    Example:
        @log_with_context(component="api", version="1.0")
        def my_function():
            ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            ctx_logger = ContextLogger(logging.getLogger(func.__module__))
            ctx_logger.set_context(**context)
            try:
                return func(*args, **kwargs)
            finally:
                ctx_logger.clear_context()
        return wrapper
    return decorator


def create_logger(
    name: str,
    level: str = "INFO",
    structured: bool = True
) -> ContextLogger:
    """
    Crear logger con contexto.
    
    Args:
        name: Nombre del logger
        level: Nivel de logging
        structured: Si True, usa formato estructurado
    
    Returns:
        ContextLogger
    
    Example:
        logger = create_logger("my_module", level="DEBUG")
        logger.set_context(user_id="123")
        logger.info("User action")
    """
    base_logger = logging.getLogger(name)
    base_logger.setLevel(getattr(logging, level.upper()))
    
    if structured and not base_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(StructuredFormatter())
        base_logger.addHandler(handler)
    
    return ContextLogger(base_logger)


__all__ = [
    "StructuredFormatter",
    "ContextLogger",
    "setup_structured_logging",
    "log_with_context",
    "create_logger",
]

