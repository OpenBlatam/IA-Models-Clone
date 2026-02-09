"""
Decoradores para endpoints de API
Elimina duplicación de código en manejo de errores y logging
"""

import logging
from functools import wraps
from typing import Callable, Any
from fastapi import HTTPException, status

logger = logging.getLogger(__name__)


def handle_api_errors(func: Callable) -> Callable:
    """
    Decorador para manejar errores en endpoints de API
    
    Proporciona:
    - Manejo consistente de errores HTTP
    - Logging estructurado
    - Respuestas de error consistentes
    
    Usage:
        @router.post("/endpoint")
        @handle_api_errors
        async def my_endpoint(request: Request):
            # código del endpoint
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except HTTPException:
            # Re-raise HTTP exceptions (ya están manejadas)
            raise
        except ValueError as e:
            logger.warning(f"Validation error in {func.__name__}: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
        except Exception as e:
            logger.error(
                f"Unexpected error in {func.__name__}: {e}",
                exc_info=True
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error interno: {str(e)}"
            )
    
    return wrapper


def log_endpoint_call(func: Callable) -> Callable:
    """
    Decorador para logging de llamadas a endpoints
    
    Proporciona:
    - Logging de entrada y salida
    - Medición de tiempo de ejecución
    
    Usage:
        @router.post("/endpoint")
        @log_endpoint_call
        async def my_endpoint(request: Request):
            # código del endpoint
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        import time
        start_time = time.time()
        
        logger.info(f"Calling endpoint: {func.__name__}")
        
        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start_time
            logger.info(
                f"Endpoint {func.__name__} completed in {duration:.3f}s"
            )
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                f"Endpoint {func.__name__} failed after {duration:.3f}s: {e}"
            )
            raise
    
    return wrapper


def cache_response(cache_key_func: Callable[[Any], str] = None):
    """
    Decorador para cachear respuestas de endpoints
    
    Args:
        cache_key_func: Función para generar clave de caché desde argumentos
    
    Usage:
        @router.get("/endpoint/{id}")
        @cache_response(lambda id: f"endpoint_{id}")
        async def my_endpoint(id: str):
            # código del endpoint
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Importar caché global
            from .routes import _response_cache, _cache_max_size
            import hashlib
            
            # Generar clave de caché
            if cache_key_func:
                cache_key_str = cache_key_func(*args, **kwargs)
            else:
                # Default: usar nombre de función + argumentos
                cache_key_str = f"{func.__name__}_{args}_{kwargs}"
            
            cache_key = hashlib.md5(cache_key_str.encode()).hexdigest()
            
            # Verificar caché
            if cache_key in _response_cache:
                logger.debug(f"Cache hit for {func.__name__}")
                return _response_cache[cache_key]
            
            # Ejecutar función
            result = await func(*args, **kwargs)
            
            # Guardar en caché
            if len(_response_cache) >= _cache_max_size:
                _response_cache.popitem(last=False)
            _response_cache[cache_key] = result
            
            return result
        
        return wrapper
    return decorator

