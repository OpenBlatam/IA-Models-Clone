"""
Advanced error handling helpers for consistent error management.
Eliminates repetitive try-except patterns.
"""

import logging
from typing import Callable, TypeVar, Optional, Type, Tuple, Any
from functools import wraps
import asyncio

logger = logging.getLogger(__name__)

T = TypeVar('T')


def handle_errors(
    operation: str,
    error_types: Optional[Tuple[Type[Exception], ...]] = None,
    default_error: Optional[Type[Exception]] = None,
    log_error: bool = True,
    reraise: bool = True
):
    """
    Decorador para manejo consistente de errores.
    
    Args:
        operation: Nombre de la operación para logging
        error_types: Tipos de error específicos a manejar (None = todos)
        default_error: Tipo de error por defecto si no coincide con error_types
        log_error: Si hacer log del error (default: True)
        reraise: Si re-lanzar el error (default: True)
        
    Usage:
        @handle_errors("extract_profile", error_types=(ValueError, KeyError))
        def extract_profile(username: str):
            # código
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Verificar si el error es del tipo esperado
                if error_types and not isinstance(e, error_types):
                    if reraise:
                        raise
                    return None
                
                if log_error:
                    logger.error(
                        f"Error in {operation}: {e}",
                        exc_info=True
                    )
                
                if reraise:
                    if default_error:
                        raise default_error(f"Error in {operation}: {str(e)}") from e
                    raise
                
                return None
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if error_types and not isinstance(e, error_types):
                    if reraise:
                        raise
                    return None
                
                if log_error:
                    logger.error(
                        f"Error in {operation}: {e}",
                        exc_info=True
                    )
                
                if reraise:
                    if default_error:
                        raise default_error(f"Error in {operation}: {str(e)}") from e
                    raise
                
                return None
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def safe_execute(
    func: Callable[..., T],
    *args,
    default: Optional[T] = None,
    operation: Optional[str] = None,
    **kwargs
) -> Optional[T]:
    """
    Ejecuta una función de forma segura, retornando default si hay error.
    
    Args:
        func: Función a ejecutar
        *args: Argumentos posicionales
        default: Valor por defecto si hay error (default: None)
        operation: Nombre de la operación para logging
        **kwargs: Keyword arguments
        
    Returns:
        Resultado de la función o default si hay error
        
    Usage:
        result = safe_execute(
            extract_profile,
            username,
            default={},
            operation="extract_profile"
        )
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        op_name = operation or func.__name__
        logger.warning(f"Error in {op_name}: {e}")
        return default


async def safe_execute_async(
    func: Callable[..., T],
    *args,
    default: Optional[T] = None,
    operation: Optional[str] = None,
    **kwargs
) -> Optional[T]:
    """
    Ejecuta una función async de forma segura.
    
    Args:
        func: Función async a ejecutar
        *args: Argumentos posicionales
        default: Valor por defecto si hay error
        operation: Nombre de la operación para logging
        **kwargs: Keyword arguments
        
    Returns:
        Resultado de la función o default si hay error
    """
    try:
        return await func(*args, **kwargs)
    except Exception as e:
        op_name = operation or func.__name__
        logger.warning(f"Error in {op_name}: {e}")
        return default


def retry_on_failure(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
):
    """
    Decorador para retry automático en caso de fallo.
    
    Args:
        max_attempts: Número máximo de intentos (default: 3)
        delay: Delay inicial en segundos (default: 1.0)
        backoff: Factor de backoff exponencial (default: 2.0)
        exceptions: Tipos de excepción que activan retry (default: todas)
        
    Usage:
        @retry_on_failure(max_attempts=5, delay=2.0)
        def fetch_data():
            # código que puede fallar
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            import time
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_attempts}): {e}. "
                            f"Retrying in {current_delay}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"{func.__name__} failed after {max_attempts} attempts")
            
            raise last_exception
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            import asyncio
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_attempts}): {e}. "
                            f"Retrying in {current_delay}s..."
                        )
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"{func.__name__} failed after {max_attempts} attempts")
            
            raise last_exception
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator








