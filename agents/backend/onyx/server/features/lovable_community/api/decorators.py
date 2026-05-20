"""
Decoradores reutilizables para endpoints (optimizado)

Incluye decoradores para logging, medición de tiempo, validación, etc.
"""

import time
import logging
from functools import wraps
from typing import Callable, Any, Optional, TypeVar, ParamSpec
from fastapi import Request, HTTPException, status

logger = logging.getLogger(__name__)

P = ParamSpec('P')
R = TypeVar('R')

try:
    from ..exceptions import BaseCommunityException
except ImportError:
    BaseCommunityException = Exception


def log_request(func: Callable) -> Callable:
    """
    Decorador para logging de requests (optimizado).
    
    Registra información sobre cada request incluyendo:
    - Método HTTP
    - Path
    - Tiempo de procesamiento
    - Status code
    
    Returns:
        Decorador
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Intentar extraer request
        request = None
        for arg in args:
            if isinstance(arg, Request):
                request = arg
                break
        if not request:
            request = kwargs.get('request')
        
        start_time = time.time()
        
        try:
            result = await func(*args, **kwargs)
            process_time = time.time() - start_time
            
            if request:
                logger.info(
                    f"{request.method} {request.url.path} - "
                    f"Status: {getattr(result, 'status_code', 200)} - "
                    f"Time: {process_time:.3f}s"
                )
            else:
                logger.debug(f"{func.__name__} executed in {process_time:.3f}s")
            
            return result
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                f"Error in {func.__name__}: {e} - Time: {process_time:.3f}s",
                exc_info=True
            )
            raise
    
    return wrapper


def measure_time(func: Callable) -> Callable:
    """
    Decorador para medir tiempo de ejecución (optimizado).
    
    Agrega el tiempo de ejecución a la respuesta.
    
    Returns:
        Decorador
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        
        result = await func(*args, **kwargs)
        
        process_time = time.time() - start_time
        
        # Si la respuesta es un dict, agregar tiempo
        if isinstance(result, dict):
            result["_process_time"] = round(process_time, 3)
        
        return result
    
    return wrapper


def validate_request_data(required_fields: Optional[list] = None):
    """
    Decorador para validar datos del request (optimizado).
    
    Args:
        required_fields: Lista de campos requeridos
        
    Returns:
        Decorador
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if required_fields:
                # Intentar extraer request body
                request_body = None
                for arg in args:
                    if hasattr(arg, '__dict__'):
                        request_body = arg
                        break
                if not request_body:
                    request_body = kwargs.get('request') or kwargs.get('body')
                
                if request_body:
                    missing_fields = []
                    for field in required_fields:
                        if not hasattr(request_body, field) or getattr(request_body, field) is None:
                            missing_fields.append(field)
                    
                    if missing_fields:
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Missing required fields: {', '.join(missing_fields)}"
                        )
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


def handle_errors(func: Callable[P, R]) -> Callable[P, R]:
    """
    Decorador para manejo centralizado de errores (optimizado).
    
    Captura excepciones y las convierte en HTTPException apropiadas.
    Funciona con excepciones personalizadas de la comunidad.
    
    Returns:
        Decorador
    """
    @wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return await func(*args, **kwargs)
        except HTTPException:
            # Re-raise HTTPException as-is
            raise
        except BaseCommunityException as e:
            # Handle custom community exceptions
            status_code = getattr(e, 'status_code', status.HTTP_500_INTERNAL_SERVER_ERROR)
            detail = getattr(e, 'detail', str(e))
            
            log_level = logging.WARNING if status_code < 500 else logging.ERROR
            logger.log(
                log_level,
                f"API exception in {func.__name__}: {detail}",
                extra={
                    "exception_type": type(e).__name__,
                    "status_code": status_code
                }
            )
            raise HTTPException(
                status_code=status_code,
                detail=detail
            )
        except ValueError as e:
            # Handle validation errors
            logger.warning(f"Validation error in {func.__name__}: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
        except TypeError as e:
            # Handle type errors (often validation issues)
            logger.warning(f"Type error in {func.__name__}: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid parameter type: {str(e)}"
            )
        except Exception as e:
            # Handle unexpected errors
            logger.error(f"Unexpected error in {func.__name__}: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An unexpected error occurred"
            )
    
    return wrapper


def cache_control(max_age: int = 60):
    """
    Decorador para agregar headers de cache control (optimizado).
    
    Args:
        max_age: Tiempo máximo de cache en segundos
        
    Returns:
        Decorador
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            from fastapi import Response
            
            # Intentar extraer response
            response = None
            for arg in args:
                if isinstance(arg, Response):
                    response = arg
                    break
            if not response:
                response = kwargs.get('response')
            
            result = await func(*args, **kwargs)
            
            # Si hay response, agregar headers
            if response:
                response.headers["Cache-Control"] = f"public, max-age={max_age}"
                response.headers["X-Cache-TTL"] = str(max_age)
            
            return result
        
        return wrapper
    return decorator

