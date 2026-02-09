"""
API Decorators - Decoradores para endpoints de API
===================================================
"""

import logging
from functools import wraps
from typing import Callable, Any
from fastapi import HTTPException

logger = logging.getLogger(__name__)


def handle_api_errors(default_status: int = 500):
    """
    Decorador para manejo consistente de errores en endpoints.
    
    Args:
        default_status: Código de estado HTTP por defecto para errores
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error en {func.__name__}: {e}", exc_info=True)
                raise HTTPException(
                    status_code=default_status,
                    detail=f"Error en {func.__name__}: {str(e)}"
                )
        return wrapper
    return decorator

