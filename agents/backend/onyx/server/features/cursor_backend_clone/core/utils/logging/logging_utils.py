"""
Logging Utils - Utilidades de Logging Avanzadas
================================================

Utilidades avanzadas para logging estructurado y análisis de logs.
"""

import logging
import sys
from typing import Any, Dict, Optional, List, Callable
from datetime import datetime
from contextlib import contextmanager
from functools import wraps

logger = logging.getLogger(__name__)


class StructuredLogger:
    """
    Logger estructurado con contexto y metadata.
    """
    
    def __init__(self, name: str, logger: Optional[logging.Logger] = None):
        self.name = name
        self._logger = logger or logging.getLogger(name)
        self._context: Dict[str, Any] = {}
    
    def set_context(self, **context) -> None:
        """
        Establecer contexto para logs.
        
        Args:
            **context: Variables de contexto
        """
        self._context.update(context)
    
    def clear_context(self) -> None:
        """Limpiar contexto"""
        self._context.clear()
    
    def _format_message(self, message: str, **kwargs) -> str:
        """Formatear mensaje con contexto"""
        if self._context or kwargs:
            context_str = ", ".join(
                f"{k}={v}" for k, v in {**self._context, **kwargs}.items()
            )
            return f"{message} | {context_str}"
        return message
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug"""
        self._logger.debug(self._format_message(message, **kwargs))
    
    def info(self, message: str, **kwargs) -> None:
        """Log info"""
        self._logger.info(self._format_message(message, **kwargs))
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning"""
        self._logger.warning(self._format_message(message, **kwargs))
    
    def error(self, message: str, **kwargs) -> None:
        """Log error"""
        self._logger.error(self._format_message(message, **kwargs))
    
    def critical(self, message: str, **kwargs) -> None:
        """Log critical"""
        self._logger.critical(self._format_message(message, **kwargs))
    
    @contextmanager
    def context(self, **context):
        """
        Context manager para contexto temporal.
        
        Args:
            **context: Variables de contexto
        """
        old_context = self._context.copy()
        self.set_context(**context)
        try:
            yield
        finally:
            self._context = old_context


class LogFilter:
    """
    Filtro de logs personalizado.
    """
    
    def __init__(
        self,
        min_level: Optional[int] = None,
        max_level: Optional[int] = None,
        include_loggers: Optional[List[str]] = None,
        exclude_loggers: Optional[List[str]] = None,
        keyword_filter: Optional[str] = None
    ):
        """
        Inicializar filtro.
        
        Args:
            min_level: Nivel mínimo
            max_level: Nivel máximo
            include_loggers: Loggers a incluir
            exclude_loggers: Loggers a excluir
            keyword_filter: Palabra clave para filtrar
        """
        self.min_level = min_level
        self.max_level = max_level
        self.include_loggers = include_loggers or []
        self.exclude_loggers = exclude_loggers or []
        self.keyword_filter = keyword_filter
    
    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filtrar record.
        
        Args:
            record: Log record
            
        Returns:
            True si debe incluirse
        """
        # Filtrar por nivel
        if self.min_level and record.levelno < self.min_level:
            return False
        if self.max_level and record.levelno > self.max_level:
            return False
        
        # Filtrar por logger
        logger_name = record.name
        if self.include_loggers and logger_name not in self.include_loggers:
            return False
        if self.exclude_loggers and logger_name in self.exclude_loggers:
            return False
        
        # Filtrar por keyword
        if self.keyword_filter:
            message = record.getMessage()
            if self.keyword_filter.lower() not in message.lower():
                return False
        
        return True


class LogFormatter:
    """
    Formatter de logs personalizado.
    """
    
    def __init__(
        self,
        include_timestamp: bool = True,
        include_level: bool = True,
        include_logger: bool = True,
        include_module: bool = False,
        include_line: bool = False
    ):
        """
        Inicializar formatter.
        
        Args:
            include_timestamp: Incluir timestamp
            include_level: Incluir nivel
            include_logger: Incluir nombre de logger
            include_module: Incluir módulo
            include_line: Incluir número de línea
        """
        self.include_timestamp = include_timestamp
        self.include_level = include_level
        self.include_logger = include_logger
        self.include_module = include_module
        self.include_line = include_line
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Formatear record.
        
        Args:
            record: Log record
            
        Returns:
            Mensaje formateado
        """
        parts = []
        
        if self.include_timestamp:
            parts.append(f"[{datetime.fromtimestamp(record.created).isoformat()}]")
        
        if self.include_level:
            parts.append(f"[{record.levelname}]")
        
        if self.include_logger:
            parts.append(f"[{record.name}]")
        
        if self.include_module:
            parts.append(f"[{record.module}]")
        
        if self.include_line:
            parts.append(f"[L{record.lineno}]")
        
        parts.append(record.getMessage())
        
        return " ".join(parts)


def log_function_call(log_args: bool = False, log_result: bool = False):
    """
    Decorador para loggear llamadas a funciones.
    
    Args:
        log_args: Si loggear argumentos
        log_result: Si loggear resultado
        
    Returns:
        Decorador
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.debug(f"Calling {func.__name__}")
            
            if log_args:
                logger.debug(f"  Args: {args}")
                logger.debug(f"  Kwargs: {kwargs}")
            
            result = func(*args, **kwargs)
            
            if log_result:
                logger.debug(f"  Result: {result}")
            
            return result
        
        return wrapper
    
    return decorator


def log_async_function_call(log_args: bool = False, log_result: bool = False):
    """
    Decorador para loggear llamadas a funciones async.
    
    Args:
        log_args: Si loggear argumentos
        log_result: Si loggear resultado
        
    Returns:
        Decorador
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            logger.debug(f"Calling {func.__name__} (async)")
            
            if log_args:
                logger.debug(f"  Args: {args}")
                logger.debug(f"  Kwargs: {kwargs}")
            
            result = await func(*args, **kwargs)
            
            if log_result:
                logger.debug(f"  Result: {result}")
            
            return result
        
        return wrapper
    
    return decorator


@contextmanager
def log_execution_time(operation: str, level: int = logging.INFO):
    """
    Context manager para loggear tiempo de ejecución.
    
    Args:
        operation: Nombre de operación
        level: Nivel de log
    """
    import time
    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        logger.log(level, f"{operation} took {elapsed:.3f}s")


def setup_file_logging(
    log_file: str,
    level: int = logging.INFO,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Handler:
    """
    Configurar logging a archivo con rotación.
    
    Args:
        log_file: Ruta del archivo
        level: Nivel de logging
        max_bytes: Tamaño máximo antes de rotar
        backup_count: Número de backups
        
    Returns:
        Handler configurado
    """
    try:
        from logging.handlers import RotatingFileHandler
        
        handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        handler.setLevel(level)
        handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        )
        
        logger.addHandler(handler)
        return handler
    except Exception as e:
        logger.error(f"Error setting up file logging: {e}")
        return None


def setup_console_logging(
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> logging.Handler:
    """
    Configurar logging a consola.
    
    Args:
        level: Nivel de logging
        format_string: String de formato opcional
        
    Returns:
        Handler configurado
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    
    if format_string:
        handler.setFormatter(logging.Formatter(format_string))
    else:
        handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        )
    
    logger.addHandler(handler)
    return handler




