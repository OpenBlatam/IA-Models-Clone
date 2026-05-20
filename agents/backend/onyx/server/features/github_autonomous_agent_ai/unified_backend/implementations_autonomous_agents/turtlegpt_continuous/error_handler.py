"""
Error Handler Module
====================

Manejo centralizado de errores y excepciones.
Proporciona decoradores y utilidades para manejo robusto de errores.
"""

import asyncio
import logging
import functools
from typing import Callable, Any, Optional, Type, Union, Tuple, Dict
from enum import Enum

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Niveles de severidad de errores."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AgentError(Exception):
    """Excepción base para errores del agente."""
    
    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.severity = severity
        self.context = context or {}
        self.original_error = original_error


class TaskProcessingError(AgentError):
    """Error al procesar una tarea."""
    pass


class StrategyExecutionError(AgentError):
    """Error al ejecutar una estrategia."""
    pass


class LLMError(AgentError):
    """Error en comunicación con LLM."""
    pass


class MemoryError(AgentError):
    """Error en operaciones de memoria."""
    pass


def handle_errors(
    default_return: Any = None,
    log_error: bool = True,
    reraise: bool = False,
    error_class: Type[AgentError] = AgentError,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
):
    """
    Decorador para manejo centralizado de errores.
    
    Args:
        default_return: Valor a retornar en caso de error
        log_error: Si se debe loguear el error
        reraise: Si se debe relanzar la excepción
        error_class: Clase de error a usar
        severity: Severidad del error
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                context = {
                    "function": func.__name__,
                    "args": str(args)[:200],
                    "kwargs": str(kwargs)[:200]
                }
                
                if log_error:
                    logger.error(
                        f"Error in {func.__name__}: {e}",
                        exc_info=True,
                        extra={"context": context, "severity": severity.value}
                    )
                
                if reraise:
                    if isinstance(e, AgentError):
                        raise
                    raise error_class(
                        f"Error in {func.__name__}: {str(e)}",
                        severity=severity,
                        context=context,
                        original_error=e
                    )
                
                return default_return
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = {
                    "function": func.__name__,
                    "args": str(args)[:200],
                    "kwargs": str(kwargs)[:200]
                }
                
                if log_error:
                    logger.error(
                        f"Error in {func.__name__}: {e}",
                        exc_info=True,
                        extra={"context": context, "severity": severity.value}
                    )
                
                if reraise:
                    if isinstance(e, AgentError):
                        raise
                    raise error_class(
                        f"Error in {func.__name__}: {str(e)}",
                        severity=severity,
                        context=context,
                        original_error=e
                    )
                
                return default_return
        
        # Detectar si es async
        if hasattr(func, '__code__'):
            co_flags = func.__code__.co_flags
            if co_flags & 0x80:  # CO_COROUTINE
                return async_wrapper
        
        return sync_wrapper
    
    return decorator


def safe_execute(
    func: Callable,
    *args,
    default_return: Any = None,
    error_class: Type[AgentError] = AgentError,
    **kwargs
) -> Tuple[Any, Optional[Exception]]:
    """
    Ejecutar función de forma segura retornando resultado y error.
    
    Args:
        func: Función a ejecutar
        *args: Argumentos posicionales
        default_return: Valor por defecto en caso de error
        error_class: Clase de error a usar
        **kwargs: Argumentos nombrados
        
    Returns:
        Tupla (resultado, error)
    """
    try:
        if asyncio.iscoroutinefunction(func):
            result = await func(*args, **kwargs)
        else:
            result = func(*args, **kwargs)
        return result, None
    except Exception as e:
        logger.error(f"Error executing {func.__name__}: {e}", exc_info=True)
        return default_return, e


class ErrorRecoveryStrategy:
    """Estrategias de recuperación de errores."""
    
    @staticmethod
    def retry_on_failure(
        max_retries: int = 3,
        backoff_factor: float = 1.0,
        retryable_errors: Optional[Tuple[Type[Exception], ...]] = None
    ):
        """
        Decorador para reintentar en caso de fallo.
        
        Args:
            max_retries: Número máximo de reintentos
            backoff_factor: Factor de espera exponencial
            retryable_errors: Tipos de errores que se pueden reintentar
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                last_error = None
                for attempt in range(max_retries + 1):
                    try:
                        return await func(*args, **kwargs)
                    except Exception as e:
                        last_error = e
                        
                        if retryable_errors and not isinstance(e, retryable_errors):
                            raise
                        
                        if attempt < max_retries:
                            wait_time = backoff_factor * (2 ** attempt)
                            logger.warning(
                                f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}. "
                                f"Retrying in {wait_time}s..."
                            )
                            await asyncio.sleep(wait_time)
                        else:
                            logger.error(f"All {max_retries + 1} attempts failed for {func.__name__}")
                            raise
                
                raise last_error
            
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                last_error = None
                for attempt in range(max_retries + 1):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        last_error = e
                        
                        if retryable_errors and not isinstance(e, retryable_errors):
                            raise
                        
                        if attempt < max_retries:
                            wait_time = backoff_factor * (2 ** attempt)
                            logger.warning(
                                f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}. "
                                f"Retrying in {wait_time}s..."
                            )
                            import time
                            time.sleep(wait_time)
                        else:
                            logger.error(f"All {max_retries + 1} attempts failed for {func.__name__}")
                            raise
                
                raise last_error
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            return sync_wrapper
        
        return decorator
    
    @staticmethod
    def fallback_to_default(default_value: Any):
        """
        Decorador para usar valor por defecto en caso de error.
        
        Args:
            default_value: Valor por defecto a retornar
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Error in {func.__name__}, using default: {e}")
                    return default_value
            
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Error in {func.__name__}, using default: {e}")
                    return default_value
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            return sync_wrapper
        
        return decorator


