"""
Logging Helpers - Utilidades mejoradas para logging
===================================================

Funciones helper para mejorar el logging en el servidor MCP.
"""

import logging
import json
import traceback
from typing import Dict, Any, Optional, Union
from datetime import datetime
from functools import wraps
from contextlib import contextmanager

logger = logging.getLogger(__name__)


def setup_logging(
    level: str = "INFO",
    format_string: Optional[str] = None,
    include_timestamp: bool = True,
    include_module: bool = True,
    structured: bool = False
) -> None:
    """
    Configurar logging básico para la aplicación.
    
    Args:
        level: Nivel de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Formato personalizado (opcional)
        include_timestamp: Incluir timestamp en logs
        include_module: Incluir nombre del módulo en logs
        structured: Usar formato JSON estructurado
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    if format_string is None:
        if structured:
            # Formato JSON estructurado
            format_string = '%(message)s'
        else:
            # Formato estándar
            parts = []
            if include_timestamp:
                parts.append("%(asctime)s")
            if include_module:
                parts.append("%(name)s")
            parts.extend(["%(levelname)s", "%(message)s"])
            format_string = " - ".join(parts)
    
    logging.basicConfig(
        level=log_level,
        format=format_string,
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def log_with_context(
    logger_instance: logging.Logger,
    level: int,
    message: str,
    **context
) -> None:
    """
    Registrar log con contexto adicional.
    
    Args:
        logger_instance: Logger instance
        level: Nivel de logging (logging.DEBUG, etc.)
        message: Mensaje principal
        **context: Contexto adicional a incluir
    """
    extra = {
        "context": context,
        "timestamp": datetime.utcnow().isoformat()
    }
    logger_instance.log(level, message, extra=extra)


def log_exception(
    logger_instance: logging.Logger,
    exception: Exception,
    message: Optional[str] = None,
    **context
) -> None:
    """
    Registrar excepción con contexto y traceback.
    
    Args:
        logger_instance: Logger instance
        exception: Excepción a registrar
        message: Mensaje adicional (opcional)
        **context: Contexto adicional
    """
    error_message = message or f"Exception occurred: {type(exception).__name__}"
    extra = {
        "exception_type": type(exception).__name__,
        "exception_message": str(exception),
        "traceback": traceback.format_exc(),
        "context": context,
        "timestamp": datetime.utcnow().isoformat()
    }
    logger_instance.error(error_message, exc_info=True, extra=extra)


def log_performance(
    logger_instance: logging.Logger,
    operation: str,
    duration: float,
    **metrics
) -> None:
    """
    Registrar métrica de rendimiento.
    
    Args:
        logger_instance: Logger instance
        operation: Nombre de la operación
        duration: Duración en segundos
        **metrics: Métricas adicionales
    """
    extra = {
        "operation": operation,
        "duration_seconds": duration,
        "duration_ms": duration * 1000,
        "metrics": metrics,
        "timestamp": datetime.utcnow().isoformat(),
        "type": "performance"
    }
    logger_instance.info(f"Performance: {operation} took {duration:.3f}s", extra=extra)


def log_request(
    logger_instance: logging.Logger,
    method: str,
    path: str,
    status_code: int,
    duration: float,
    request_id: Optional[str] = None,
    **context
) -> None:
    """
    Registrar request HTTP con contexto.
    
    Args:
        logger_instance: Logger instance
        method: Método HTTP
        path: Path del request
        status_code: Código de estado HTTP
        duration: Duración en segundos
        request_id: ID del request (opcional)
        **context: Contexto adicional
    """
    extra = {
        "method": method,
        "path": path,
        "status_code": status_code,
        "duration_seconds": duration,
        "duration_ms": duration * 1000,
        "request_id": request_id,
        "context": context,
        "timestamp": datetime.utcnow().isoformat(),
        "type": "http_request"
    }
    
    log_level = logging.INFO
    if status_code >= 500:
        log_level = logging.ERROR
    elif status_code >= 400:
        log_level = logging.WARNING
    
    logger_instance.log(
        log_level,
        f"HTTP {method} {path} - {status_code} ({duration:.3f}s)",
        extra=extra
    )


def log_function_call(
    logger_instance: logging.Logger,
    function_name: str,
    args: tuple = (),
    kwargs: Optional[Dict[str, Any]] = None,
    result: Any = None,
    duration: Optional[float] = None,
    level: int = logging.DEBUG
) -> None:
    """
    Registrar llamada a función con parámetros y resultado.
    
    Args:
        logger_instance: Logger instance
        function_name: Nombre de la función
        args: Argumentos posicionales
        kwargs: Argumentos nombrados (opcional)
        result: Resultado de la función (opcional)
        duration: Duración de la ejecución (opcional)
        level: Nivel de logging
    """
    extra = {
        "function": function_name,
        "args": str(args) if args else None,
        "kwargs": kwargs or {},
        "result": str(result) if result is not None else None,
        "duration_seconds": duration,
        "timestamp": datetime.utcnow().isoformat(),
        "type": "function_call"
    }
    logger_instance.log(level, f"Function call: {function_name}", extra=extra)


def log_with_metadata(
    logger_instance: logging.Logger,
    level: int,
    message: str,
    metadata: Optional[Dict[str, Any]] = None,
    **kwargs
) -> None:
    """
    Registrar log con metadata estructurada.
    
    Args:
        logger_instance: Logger instance
        level: Nivel de logging
        message: Mensaje principal
        metadata: Metadata adicional (opcional)
        **kwargs: Campos adicionales
    """
    extra = {
        "metadata": metadata or {},
        "timestamp": datetime.utcnow().isoformat(),
        **kwargs
    }
    logger_instance.log(level, message, extra=extra)


@contextmanager
def log_execution_time(
    logger_instance: logging.Logger,
    operation: str,
    level: int = logging.INFO
):
    """
    Context manager para registrar tiempo de ejecución.
    
    Args:
        logger_instance: Logger instance
        operation: Nombre de la operación
        level: Nivel de logging
    
    Yields:
        None
    """
    import time
    start_time = time.time()
    
    try:
        yield
    finally:
        duration = time.time() - start_time
        log_performance(logger_instance, operation, duration, level=level)


def logged_function(
    logger_instance: Optional[logging.Logger] = None,
    log_args: bool = True,
    log_result: bool = False,
    log_duration: bool = True,
    level: int = logging.DEBUG
):
    """
    Decorator para registrar automáticamente llamadas a funciones.
    
    Args:
        logger_instance: Logger instance (opcional, usa logger por defecto)
        log_args: Registrar argumentos
        log_result: Registrar resultado
        log_duration: Registrar duración
        level: Nivel de logging
    
    Returns:
        Decorator function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_logger = logger_instance or logging.getLogger(func.__module__)
            func_name = f"{func.__module__}.{func.__name__}"
            
            if log_args:
                log_function_call(
                    func_logger,
                    func_name,
                    args=args,
                    kwargs=kwargs,
                    level=level
                )
            
            start_time = None
            if log_duration:
                import time
                start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                
                if log_duration and start_time:
                    duration = time.time() - start_time
                    log_performance(func_logger, func_name, duration, level=level)
                
                if log_result:
                    log_function_call(
                        func_logger,
                        func_name,
                        result=result,
                        level=level
                    )
                
                return result
            except Exception as e:
                log_exception(func_logger, e, function=func_name)
                raise
        
        return wrapper
    return decorator


def create_structured_logger(
    name: str,
    level: str = "INFO",
    include_context: bool = True
) -> logging.Logger:
    """
    Crear logger con formato estructurado JSON.
    
    Args:
        name: Nombre del logger
        level: Nivel de logging
        include_context: Incluir contexto en logs
    
    Returns:
        Logger configurado
    """
    logger_instance = logging.getLogger(name)
    logger_instance.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # Handler con formato JSON
    handler = logging.StreamHandler()
    
    class JSONFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            log_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
            
            if include_context and hasattr(record, "context"):
                log_data["context"] = record.context
            
            if hasattr(record, "extra"):
                log_data.update(record.extra)
            
            return json.dumps(log_data)
    
    handler.setFormatter(JSONFormatter())
    logger_instance.addHandler(handler)
    
    return logger_instance

