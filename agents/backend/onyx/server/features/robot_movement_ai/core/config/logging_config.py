"""
Structured Logging Configuration
==================================

Configuración de logging estructurado para el sistema.
"""

import logging
import sys
import json
from typing import Any, Dict, Optional
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager


class StructuredFormatter(logging.Formatter):
    """
    Formatter que produce logs estructurados en JSON.
    
    Útil para integración con sistemas de logging como ELK, Splunk, etc.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Formatear log record como JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Agregar excepción si existe
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Agregar campos extra si existen
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)
        
        return json.dumps(log_data)


class ColoredFormatter(logging.Formatter):
    """
    Formatter con colores para terminal.
    
    Facilita la lectura de logs en desarrollo.
    """
    
    COLORS = {
        "DEBUG": "\033[36m",      # Cyan
        "INFO": "\033[32m",       # Green
        "WARNING": "\033[33m",    # Yellow
        "ERROR": "\033[31m",      # Red
        "CRITICAL": "\033[35m",   # Magenta
    }
    RESET = "\033[0m"
    
    def format(self, record: logging.LogRecord) -> str:
        """Formatear log con colores."""
        color = self.COLORS.get(record.levelname, "")
        reset = self.RESET
        
        # Formato: [LEVEL] logger: message
        formatted = (
            f"{color}[{record.levelname:8s}]{reset} "
            f"{record.name}: {record.getMessage()}"
        )
        
        if record.exc_info:
            formatted += f"\n{self.formatException(record.exc_info)}"
        
        return formatted


def setup_logging(
    level: str = "INFO",
    structured: bool = False,
    colored: bool = True,
    log_file: str = None
) -> None:
    """
    Configurar logging del sistema.
    
    Args:
        level: Nivel de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        structured: Si True, usar formato JSON estructurado
        colored: Si True, usar colores en terminal (solo si no structured)
        log_file: Ruta opcional a archivo de log
    """
    # Configurar nivel
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Crear formatter
    if structured:
        formatter = StructuredFormatter()
    elif colored and sys.stdout.isatty():
        formatter = ColoredFormatter()
    else:
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)8s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    
    # Configurar handler de consola
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(numeric_level)
    
    # Configurar root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    
    # Configurar handler de archivo si se especifica
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(numeric_level)
        root_logger.addHandler(file_handler)
    
    # Configurar loggers específicos
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Obtener logger con nombre específico.
    
    Args:
        name: Nombre del logger
        
    Returns:
        Logger configurado
    """
    return logging.getLogger(name)


class LoggerAdapter(logging.LoggerAdapter):
    """
    Adapter para agregar contexto a logs.
    
    Usage:
        logger = LoggerAdapter(logging.getLogger(__name__), {"robot_id": "robot1"})
        logger.info("Message")  # Incluirá robot_id en logs estructurados
    """
    
    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """Procesar mensaje agregando contexto."""
        if "extra" not in kwargs:
            kwargs["extra"] = {}
        
        if "extra_fields" not in kwargs["extra"]:
            kwargs["extra"]["extra_fields"] = {}
        
        kwargs["extra"]["extra_fields"].update(self.extra)
        return msg, kwargs


def log_function_call(logger: logging.Logger):
    """
    Decorador para loggear llamadas a funciones.
    
    Usage:
        @log_function_call(logger)
        def my_function(arg1, arg2):
            ...
    """
    def decorator(func):
        import functools
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger.debug(
                f"Calling {func.__name__} with args={args}, kwargs={kwargs}"
            )
            try:
                result = func(*args, **kwargs)
                logger.debug(f"{func.__name__} returned: {result}")
                return result
            except Exception as e:
                logger.error(f"{func.__name__} raised {type(e).__name__}: {e}")
                raise
        
        return wrapper
    return decorator


def log_function_call_async(logger: logging.Logger):
    """
    Decorador para loggear llamadas a funciones async.
    
    Usage:
        @log_function_call_async(logger)
        async def my_async_function(arg1, arg2):
            ...
    """
    def decorator(func):
        import functools
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            logger.debug(
                f"Calling {func.__name__} (async) with args={args}, kwargs={kwargs}"
            )
            try:
                result = await func(*args, **kwargs)
                logger.debug(f"{func.__name__} (async) returned: {result}")
                return result
            except Exception as e:
                logger.error(f"{func.__name__} (async) raised {type(e).__name__}: {e}")
                raise
        
        return wrapper
    return decorator


@contextmanager
def temporary_log_level(level: str, logger_name: Optional[str] = None):
    """
    Context manager para cambiar temporalmente el nivel de logging.
    
    Args:
        level: Nivel de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        logger_name: Nombre del logger (None = root logger)
    
    Yields:
        None
    """
    logger = logging.getLogger(logger_name) if logger_name else logging.getLogger()
    original_level = logger.level
    
    try:
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        logger.setLevel(numeric_level)
        yield
    finally:
        logger.setLevel(original_level)


@contextmanager
def capture_logs(logger_name: str, level: str = "DEBUG"):
    """
    Context manager para capturar logs durante una operación.
    
    Args:
        logger_name: Nombre del logger
        level: Nivel mínimo a capturar
    
    Yields:
        Lista de mensajes de log
    """
    import io
    import logging
    
    logger = logging.getLogger(logger_name)
    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(getattr(logging, level.upper(), logging.DEBUG))
    
    logger.addHandler(handler)
    original_level = logger.level
    logger.setLevel(getattr(logging, level.upper(), logging.DEBUG))
    
    try:
        yield log_capture
    finally:
        logger.removeHandler(handler)
        logger.setLevel(original_level)
    
    return log_capture.getvalue().splitlines()



