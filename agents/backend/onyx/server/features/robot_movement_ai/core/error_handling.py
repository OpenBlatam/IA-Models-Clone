"""
Error Handling Utilities
=========================

Utilidades para manejo robusto de errores y validación.
"""

from typing import Any, Callable, Optional, TypeVar, Union
from functools import wraps
import logging
import traceback
from contextlib import contextmanager

from .exceptions import BaseRobotException

logger = logging.getLogger(__name__)

T = TypeVar('T')
F = TypeVar('F', bound=Callable)


def safe_execute(
    func: Callable[[], T],
    default: Optional[T] = None,
    log_error: bool = True,
    reraise: bool = False
) -> Optional[T]:
    """
    Ejecutar función de forma segura con manejo de errores.
    
    Args:
        func: Función a ejecutar
        default: Valor por defecto si falla
        log_error: Si debe loguear el error
        reraise: Si debe relanzar la excepción
    
    Returns:
        Resultado de la función o default
    """
    try:
        return func()
    except BaseRobotException:
        if reraise:
            raise
        if log_error:
            logger.error(f"Robot exception in safe_execute: {func.__name__}", exc_info=True)
        return default
    except Exception as e:
        if reraise:
            raise
        if log_error:
            logger.error(f"Unexpected error in safe_execute: {func.__name__}: {e}", exc_info=True)
        return default


async def safe_execute_async(
    func: Callable[[], T],
    default: Optional[T] = None,
    log_error: bool = True,
    reraise: bool = False
) -> Optional[T]:
    """
    Ejecutar función async de forma segura.
    
    Args:
        func: Función async a ejecutar
        default: Valor por defecto si falla
        log_error: Si debe loguear el error
        reraise: Si debe relanzar la excepción
    
    Returns:
        Resultado de la función o default
    """
    try:
        return await func()
    except BaseRobotException:
        if reraise:
            raise
        if log_error:
            logger.error(f"Robot exception in safe_execute_async: {func.__name__}", exc_info=True)
        return default
    except Exception as e:
        if reraise:
            raise
        if log_error:
            logger.error(f"Unexpected error in safe_execute_async: {func.__name__}: {e}", exc_info=True)
        return default


def handle_errors(
    default: Optional[Any] = None,
    log_error: bool = True,
    exception_type: type = BaseRobotException
):
    """
    Decorador para manejo de errores.
    
    Args:
        default: Valor por defecto si falla
        log_error: Si debe loguear el error
        exception_type: Tipo de excepción a capturar
    
    Returns:
        Decorador
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exception_type as e:
                if log_error:
                    logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
                return default
            except Exception as e:
                if log_error:
                    logger.error(f"Unexpected error in {func.__name__}: {e}", exc_info=True)
                return default
        return wrapper
    return decorator


def handle_errors_async(
    default: Optional[Any] = None,
    log_error: bool = True,
    exception_type: type = BaseRobotException
):
    """
    Decorador para manejo de errores en funciones async.
    
    Args:
        default: Valor por defecto si falla
        log_error: Si debe loguear el error
        exception_type: Tipo de excepción a capturar
    
    Returns:
        Decorador
    """
    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except exception_type as e:
                if log_error:
                    logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
                return default
            except Exception as e:
                if log_error:
                    logger.error(f"Unexpected error in {func.__name__}: {e}", exc_info=True)
                return default
        return wrapper
    return decorator


@contextmanager
def error_context(operation: str, log_error: bool = True):
    """
    Context manager para manejo de errores con contexto.
    
    Args:
        operation: Nombre de la operación
        log_error: Si debe loguear el error
    
    Yields:
        None
    """
    try:
        yield
    except BaseRobotException as e:
        if log_error:
            logger.error(f"Error in {operation}: {e}", exc_info=True)
        raise
    except Exception as e:
        if log_error:
            logger.error(f"Unexpected error in {operation}: {e}", exc_info=True)
        raise BaseRobotException(
            f"Error in {operation}: {str(e)}",
            error_code="UNEXPECTED_ERROR",
            cause=e
        )


def validate_not_none(value: Any, name: str = "value") -> None:
    """
    Validar que un valor no sea None.
    
    Args:
        value: Valor a validar
        name: Nombre del valor para el mensaje de error
    
    Raises:
        ValueError: Si el valor es None
    """
    if value is None:
        raise ValueError(f"{name} cannot be None")


def validate_type(value: Any, expected_type: type, name: str = "value") -> None:
    """
    Validar tipo de un valor.
    
    Args:
        value: Valor a validar
        expected_type: Tipo esperado
        name: Nombre del valor para el mensaje de error
    
    Raises:
        TypeError: Si el tipo no coincide
    """
    if not isinstance(value, expected_type):
        raise TypeError(
            f"{name} must be of type {expected_type.__name__}, "
            f"got {type(value).__name__}"
        )


def validate_range(
    value: Union[int, float],
    min_val: Optional[Union[int, float]] = None,
    max_val: Optional[Union[int, float]] = None,
    name: str = "value"
) -> None:
    """
    Validar que un valor esté en un rango.
    
    Args:
        value: Valor a validar
        min_val: Valor mínimo (inclusive)
        max_val: Valor máximo (inclusive)
        name: Nombre del valor para el mensaje de error
    
    Raises:
        ValueError: Si el valor está fuera del rango
    """
    if min_val is not None and value < min_val:
        raise ValueError(f"{name} must be >= {min_val}, got {value}")
    if max_val is not None and value > max_val:
        raise ValueError(f"{name} must be <= {max_val}, got {value}")

