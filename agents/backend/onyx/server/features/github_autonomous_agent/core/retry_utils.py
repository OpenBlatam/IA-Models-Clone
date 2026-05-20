"""
Utilidades para retry logic y manejo de reintentos con logging mejorado.
"""

import logging
from functools import wraps
from typing import Callable, Type, Tuple, Optional, Any
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryError,
    before_sleep_log,
    after_log
)

from config.logging_config import get_logger

logger = get_logger(__name__)


def retry_on_github_error(
    max_attempts: int = 3,
    min_wait: float = 2.0,
    max_wait: float = 10.0,
    exceptions: Tuple[Type[Exception], ...] = None
):
    """
    Decorador para agregar retry logic a funciones que interactúan con GitHub.
    
    Args:
        max_attempts: Número máximo de intentos (default: 3)
        min_wait: Tiempo mínimo de espera en segundos (default: 2.0)
        max_wait: Tiempo máximo de espera en segundos (default: 10.0)
        exceptions: Tupla de excepciones que deben activar el retry
        
    Returns:
        Decorador con retry logic
        
    Example:
        @retry_on_github_error(max_attempts=5, min_wait=1.0, max_wait=20.0)
        def my_github_function():
            # código que puede fallar
            pass
    """
    if exceptions is None:
        from github.GithubException import GithubException
        exceptions = (GithubException, Exception)
    
    def decorator(func: Callable) -> Callable:
        @retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
            retry=retry_if_exception_type(exceptions),
            reraise=True,
            before_sleep=before_sleep_log(logger, logging.WARNING),
            after=after_log(logger, logging.DEBUG)
        )
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                logger.debug(f"Ejecutando {func.__name__} (máximo {max_attempts} intentos)")
                return func(*args, **kwargs)
            except RetryError as e:
                logger.error(
                    f"Error después de {max_attempts} intentos en {func.__name__}: "
                    f"{type(e.last_attempt.exception()).__name__}: {e.last_attempt.exception()}",
                    exc_info=True
                )
                raise e.last_attempt.exception() from e
            except exceptions as e:
                logger.warning(
                    f"Error en {func.__name__}: {type(e).__name__}: {e}. "
                    f"Reintentando (máximo {max_attempts} intentos)..."
                )
                raise
        
        return wrapper
    return decorator


def retry_async_on_github_error(
    max_attempts: int = 3,
    min_wait: float = 2.0,
    max_wait: float = 10.0,
    exceptions: Tuple[Type[Exception], ...] = None
):
    """
    Decorador para agregar retry logic a funciones async que interactúan con GitHub.
    
    Args:
        max_attempts: Número máximo de intentos (default: 3)
        min_wait: Tiempo mínimo de espera en segundos (default: 2.0)
        max_wait: Tiempo máximo de espera en segundos (default: 10.0)
        exceptions: Tupla de excepciones que deben activar el retry
        
    Returns:
        Decorador con retry logic para funciones async
        
    Example:
        @retry_async_on_github_error(max_attempts=5, min_wait=1.0, max_wait=20.0)
        async def my_async_github_function():
            # código async que puede fallar
            pass
    """
    if exceptions is None:
        from github.GithubException import GithubException
        exceptions = (GithubException, Exception)
    
    def decorator(func: Callable) -> Callable:
        @retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
            retry=retry_if_exception_type(exceptions),
            reraise=True,
            before_sleep=before_sleep_log(logger, logging.WARNING),
            after=after_log(logger, logging.DEBUG)
        )
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            try:
                logger.debug(f"Ejecutando {func.__name__} async (máximo {max_attempts} intentos)")
                return await func(*args, **kwargs)
            except RetryError as e:
                logger.error(
                    f"Error después de {max_attempts} intentos en {func.__name__}: "
                    f"{type(e.last_attempt.exception()).__name__}: {e.last_attempt.exception()}",
                    exc_info=True
                )
                raise e.last_attempt.exception() from e
            except exceptions as e:
                logger.warning(
                    f"Error en {func.__name__} async: {type(e).__name__}: {e}. "
                    f"Reintentando (máximo {max_attempts} intentos)..."
                )
                raise
        
        return wrapper
    return decorator

