"""
Handlers de errores optimizados

Incluye funciones para manejar errores de forma consistente y user-friendly.
"""

import logging
from typing import Optional, Dict, Any
from fastapi import HTTPException, status

from ..exceptions import (
    BaseAPIException,
    SongNotFoundError,
    SongGenerationError,
    AudioProcessingError,
    InvalidInputError,
    DatabaseError
)

logger = logging.getLogger(__name__)


def handle_service_error(
    error: Exception,
    operation: str,
    context: Optional[Dict[str, Any]] = None
) -> HTTPException:
    """
    Maneja errores de servicios de forma consistente.
    
    Args:
        error: Excepción capturada
        operation: Nombre de la operación que falló
        context: Contexto adicional (opcional)
        
    Returns:
        HTTPException apropiada
    """
    error_msg = str(error)
    context_str = f" (context: {context})" if context else ""
    
    logger.error(f"Error in {operation}: {error_msg}{context_str}", exc_info=True)
    
    # Mapear tipos de error a HTTPExceptions
    if isinstance(error, BaseAPIException):
        return HTTPException(
            status_code=error.status_code,
            detail=error.detail
        )
    
    if isinstance(error, ValueError):
        return HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input for {operation}: {error_msg}"
        )
    
    if isinstance(error, FileNotFoundError):
        return HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File not found in {operation}: {error_msg}"
        )
    
    if isinstance(error, PermissionError):
        return HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Permission denied in {operation}: {error_msg}"
        )
    
    # Error genérico
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=f"Error in {operation}: {error_msg}"
    )


def safe_execute(
    func,
    *args,
    error_message: str = "Operation failed",
    default_return: Any = None,
    **kwargs
) -> Any:
    """
    Ejecuta una función de forma segura con manejo de errores.
    
    Args:
        func: Función a ejecutar
        *args: Argumentos posicionales
        error_message: Mensaje de error personalizado
        default_return: Valor a retornar en caso de error
        **kwargs: Argumentos nombrados
        
    Returns:
        Resultado de la función o default_return si falla
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.warning(f"{error_message}: {e}")
        return default_return


async def safe_execute_async(
    func,
    *args,
    error_message: str = "Operation failed",
    default_return: Any = None,
    **kwargs
) -> Any:
    """
    Ejecuta una función async de forma segura con manejo de errores.
    
    Args:
        func: Función async a ejecutar
        *args: Argumentos posicionales
        error_message: Mensaje de error personalizado
        default_return: Valor a retornar en caso de error
        **kwargs: Argumentos nombrados
        
    Returns:
        Resultado de la función o default_return si falla
    """
    try:
        return await func(*args, **kwargs)
    except Exception as e:
        logger.warning(f"{error_message}: {e}")
        return default_return

