"""
Decoradores útiles para endpoints y funciones

Incluye decoradores para logging, validación, rate limiting, y más.
"""

import asyncio
import functools
import logging
import time
from typing import Callable, Any, Optional, Dict
from functools import wraps

from fastapi import Request, HTTPException, status

from .performance_monitor import measure_time
from .rate_limit_helpers import check_rate_limit
from .request_helpers import get_client_ip

logger = logging.getLogger(__name__)


def log_request(func: Callable) -> Callable:
    """
    Decorador para logging automático de requests.
    
    Registra información sobre el request y el tiempo de ejecución.
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Extraer request si está presente
        request = None
        for arg in args:
            if isinstance(arg, Request):
                request = arg
                break
        if not request:
            request = kwargs.get('request')
        
        start_time = time.time()
        client_ip = get_client_ip(request) if request else "unknown"
        
        logger.info(
            f"Request started: {func.__name__}",
            extra={
                "function": func.__name__,
                "client_ip": client_ip,
                "path": request.url.path if request else None
            }
        )
        
        try:
            result = await func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.info(
                f"Request completed: {func.__name__} ({elapsed:.3f}s)",
                extra={
                    "function": func.__name__,
                    "elapsed": elapsed,
                    "client_ip": client_ip
                }
            )
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(
                f"Request failed: {func.__name__} ({elapsed:.3f}s) - {str(e)}",
                exc_info=True,
                extra={
                    "function": func.__name__,
                    "elapsed": elapsed,
                    "client_ip": client_ip,
                    "error": str(e)
                }
            )
            raise
    
    return wrapper


def rate_limit_decorator(
    max_requests: int = 60,
    window_seconds: int = 60
) -> Callable:
    """
    Decorador para rate limiting a nivel de endpoint.
    
    Args:
        max_requests: Número máximo de requests permitidos
        window_seconds: Ventana de tiempo en segundos
        
    Returns:
        Decorador que aplica rate limiting
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extraer request
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            if not request:
                request = kwargs.get('request')
            
            if not request:
                # Si no hay request, ejecutar sin rate limiting
                return await func(*args, **kwargs)
            
            # Obtener identificador del cliente
            client_id = f"ip:{get_client_ip(request)}"
            user_id = request.headers.get("X-User-ID")
            if user_id:
                client_id = f"user:{user_id}"
            
            # Verificar rate limit
            is_allowed, retry_after = check_rate_limit(
                identifier=client_id,
                max_requests=max_requests,
                window_seconds=window_seconds
            )
            
            if not is_allowed:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Rate limit exceeded. Maximum {max_requests} requests per {window_seconds} seconds.",
                    headers={"Retry-After": str(retry_after) if retry_after else str(window_seconds)}
                )
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


def validate_request(func: Callable) -> Callable:
    """
    Decorador para validación automática de requests.
    
    Valida que el request tenga los headers necesarios y sea válido.
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Extraer request
        request = None
        for arg in args:
            if isinstance(arg, Request):
                request = arg
                break
        if not request:
            request = kwargs.get('request')
        
        if request:
            # Validar Content-Type si es POST/PUT/PATCH
            if request.method in ['POST', 'PUT', 'PATCH']:
                content_type = request.headers.get('Content-Type', '')
                if not content_type.startswith('application/json'):
                    logger.warning(
                        f"Invalid Content-Type: {content_type}",
                        extra={"path": request.url.path}
                    )
        
        return await func(*args, **kwargs)
    
    return wrapper


def cache_control(
    max_age: int = 3600,
    public: bool = True,
    must_revalidate: bool = False
) -> Callable:
    """
    Decorador para agregar headers de cache control.
    
    Args:
        max_age: Tiempo máximo de cache en segundos
        public: Si el cache es público
        must_revalidate: Si debe revalidar antes de usar cache
        
    Returns:
        Decorador que agrega headers de cache
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            from fastapi import Response
            
            result = await func(*args, **kwargs)
            
            # Si el resultado es una Response, agregar headers
            if isinstance(result, Response):
                cache_directive = f"{'public' if public else 'private'}, max-age={max_age}"
                if must_revalidate:
                    cache_directive += ", must-revalidate"
                result.headers["Cache-Control"] = cache_directive
            
            return result
        
        return wrapper
    return decorator


def retry_on_failure(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
) -> Callable:
    """
    Decorador para reintentar en caso de fallo.
    
    Args:
        max_retries: Número máximo de reintentos
        delay: Delay inicial en segundos
        backoff: Factor de multiplicación para el delay
        exceptions: Tupla de excepciones que deben causar reintento
        
    Returns:
        Decorador que reintenta en caso de fallo
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"Retry {attempt + 1}/{max_retries} for {func.__name__}: {str(e)}"
                        )
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            f"Max retries exceeded for {func.__name__}: {str(e)}"
                        )
            
            raise last_exception
        
        return wrapper
    return decorator


def measure_performance(func: Callable) -> Callable:
    """
    Decorador para medir el rendimiento de una función.
    
    Usa el performance monitor para registrar métricas.
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        with measure_time(func.__name__):
            return await func(*args, **kwargs)
    
    return wrapper


def require_auth(func: Callable) -> Callable:
    """
    Decorador para requerir autenticación.
    
    Valida que el request tenga un token de autenticación válido.
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Extraer request
        request = None
        for arg in args:
            if isinstance(arg, Request):
                request = arg
                break
        if not request:
            request = kwargs.get('request')
        
        if not request:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required"
            )
        
        # Verificar token (implementación básica)
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing authentication token",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        return await func(*args, **kwargs)
    
    return wrapper

