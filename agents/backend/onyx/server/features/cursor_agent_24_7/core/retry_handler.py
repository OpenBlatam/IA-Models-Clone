"""
Retry Handler - Manejo de reintentos con tenacity
=================================================

Wrapper para usar tenacity de forma consistente en toda la aplicación.
"""

import logging
from typing import Callable, Any, Optional, Type, Union, List
from functools import wraps

logger = logging.getLogger(__name__)

# Intentar importar tenacity
try:
    from tenacity import (
        retry,
        stop_after_attempt,
        wait_exponential,
        retry_if_exception_type,
        RetryCallState,
        before_sleep_log,
        after_log
    )
    HAS_TENACITY = True
except ImportError:
    HAS_TENACITY = False
    logger.warning("tenacity not installed, retry functionality will be limited")


def retry_with_backoff(
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 60.0,
    exponential_base: float = 2.0,
    retry_on: Optional[Union[Type[Exception], List[Type[Exception]]]] = None,
    reraise: bool = True
):
    """
    Decorador para retry con backoff exponencial.
    
    Args:
        max_attempts: Número máximo de intentos.
        min_wait: Tiempo mínimo de espera en segundos.
        max_wait: Tiempo máximo de espera en segundos.
        exponential_base: Base exponencial para backoff.
        retry_on: Excepciones que deben causar retry (None = todas).
        reraise: Si True, re-lanza la excepción después de agotar intentos.
    
    Returns:
        Decorador.
    """
    if not HAS_TENACITY:
        # Fallback: no retry
        def no_retry_decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                return await func(*args, **kwargs)
            return wrapper
        return no_retry_decorator
    
    # Configurar condiciones de retry
    stop = stop_after_attempt(max_attempts)
    wait = wait_exponential(
        multiplier=min_wait,
        min=min_wait,
        max=max_wait,
        exp_base=exponential_base
    )
    
    # Configurar qué excepciones retry
    if retry_on:
        if isinstance(retry_on, list):
            retry_condition = retry_if_exception_type(tuple(retry_on))
        else:
            retry_condition = retry_if_exception_type(retry_on)
    else:
        retry_condition = retry_if_exception_type(Exception)
    
    # Logging callbacks
    def log_retry(state: RetryCallState):
        """Log antes de retry"""
        logger.warning(
            f"Retrying {state.fn.__name__} after {state.outcome.exception()} "
            f"(attempt {state.attempt_number}/{max_attempts})"
        )
    
    def log_final(state: RetryCallState):
        """Log después de retry"""
        if state.outcome.failed:
            logger.error(
                f"Failed {state.fn.__name__} after {max_attempts} attempts: "
                f"{state.outcome.exception()}"
            )
    
    return retry(
        stop=stop,
        wait=wait,
        retry=retry_condition,
        before_sleep=before_sleep_log(logger, logging.WARNING),
        after=after_log(logger, logging.ERROR),
        reraise=reraise
    )


def retry_async(
    func: Callable,
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 60.0,
    retry_on: Optional[Union[Type[Exception], List[Type[Exception]]]] = None
) -> Any:
    """
    Ejecutar función async con retry.
    
    Args:
        func: Función async a ejecutar.
        max_attempts: Número máximo de intentos.
        min_wait: Tiempo mínimo de espera.
        max_wait: Tiempo máximo de espera.
        retry_on: Excepciones que causan retry.
    
    Returns:
        Resultado de la función.
    
    Raises:
        Exception: Si falla después de todos los intentos.
    """
    import asyncio
    
    @retry_with_backoff(
        max_attempts=max_attempts,
        min_wait=min_wait,
        max_wait=max_wait,
        retry_on=retry_on
    )
    async def wrapper():
        return await func()
    
    return wrapper()


# Retry handlers específicos para servicios comunes

def retry_dynamodb(func: Callable) -> Callable:
    """Retry específico para operaciones DynamoDB."""
    if not HAS_TENACITY:
        return func
    
    from botocore.exceptions import ClientError, BotoCoreError
    
    @retry_with_backoff(
        max_attempts=5,
        min_wait=0.5,
        max_wait=10.0,
        retry_on=[ClientError, BotoCoreError]
    )
    @wraps(func)
    async def wrapper(*args, **kwargs):
        return await func(*args, **kwargs)
    
    return wrapper


def retry_redis(func: Callable) -> Callable:
    """Retry específico para operaciones Redis."""
    if not HAS_TENACITY:
        return func
    
    try:
        import redis
        redis_exceptions = (redis.ConnectionError, redis.TimeoutError)
    except ImportError:
        redis_exceptions = Exception
    
    @retry_with_backoff(
        max_attempts=3,
        min_wait=0.1,
        max_wait=2.0,
        retry_on=redis_exceptions
    )
    @wraps(func)
    async def wrapper(*args, **kwargs):
        return await func(*args, **kwargs)
    
    return wrapper


def retry_http(func: Callable) -> Callable:
    """Retry específico para requests HTTP."""
    if not HAS_TENACITY:
        return func
    
    try:
        import httpx
        http_exceptions = (httpx.RequestError, httpx.TimeoutException)
    except ImportError:
        http_exceptions = Exception
    
    @retry_with_backoff(
        max_attempts=3,
        min_wait=1.0,
        max_wait=10.0,
        retry_on=http_exceptions
    )
    @wraps(func)
    async def wrapper(*args, **kwargs):
        return await func(*args, **kwargs)
    
    return wrapper




