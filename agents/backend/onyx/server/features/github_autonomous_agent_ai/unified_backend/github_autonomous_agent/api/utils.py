"""
Utilidades para las rutas de la API.
"""

import asyncio
from functools import wraps
from typing import Callable, Any
from fastapi import HTTPException
from config.logging_config import get_logger
from core.constants import ErrorMessages

logger = get_logger(__name__)


def handle_api_errors(func: Callable) -> Callable:
    """
    Decorador para manejar errores en endpoints de la API.
    
    Convierte excepciones en HTTPException apropiadas.
    Soporta funciones síncronas y asíncronas.
    
    Args:
        func: Función del endpoint a decorar
        
    Returns:
        Función decorada con manejo de errores
    """
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except HTTPException:
            raise
        except ValueError as e:
            logger.error(f"Error de validación en {func.__name__}: {e}")
            raise HTTPException(status_code=400, detail=str(e))
        except KeyError as e:
            logger.error(f"Clave faltante en {func.__name__}: {e}")
            raise HTTPException(status_code=400, detail=f"Parámetro faltante: {e}")
        except Exception as e:
            logger.error(f"Error inesperado en {func.__name__}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except HTTPException:
            raise
        except ValueError as e:
            logger.error(f"Error de validación en {func.__name__}: {e}")
            raise HTTPException(status_code=400, detail=str(e))
        except KeyError as e:
            logger.error(f"Clave faltante en {func.__name__}: {e}")
            raise HTTPException(status_code=400, detail=f"Parámetro faltante: {e}")
        except Exception as e:
            logger.error(f"Error inesperado en {func.__name__}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


def validate_github_token() -> None:
    """
    Validar que el token de GitHub esté configurado.
    
    Raises:
        HTTPException: Si el token no está configurado
    """
    from config.settings import settings
    
    if not settings.GITHUB_TOKEN:
        raise HTTPException(
            status_code=400,
            detail=ErrorMessages.GITHUB_TOKEN_NOT_CONFIGURED
        )


def create_error_response(message: str, status_code: int = 500) -> dict:
    """
    Crear una respuesta de error estandarizada.
    
    Args:
        message: Mensaje de error
        status_code: Código de estado HTTP
        
    Returns:
        Diccionario con la respuesta de error
    """
    return {
        "error": True,
        "message": message,
        "status_code": status_code
    }

