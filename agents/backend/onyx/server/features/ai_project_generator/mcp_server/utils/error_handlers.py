"""
Error Handlers - Common error handling utilities
================================================

Utilidades para manejo de errores compartidas entre handlers y servicios.
"""

import logging
from typing import Any, Dict, Optional

from fastapi import HTTPException, status

from ..exceptions import (
    MCPError,
    MCPAuthorizationError,
    MCPResourceNotFoundError,
    MCPConnectorError,
    MCPOperationError,
    MCPValidationError,
    MCPRateLimitError,
    MCPContextLimitError,
)
from ..models import MCPResponse

logger = logging.getLogger(__name__)


def map_exception_to_http_status(exception: Exception) -> int:
    """
    Mapear excepción MCP a código de estado HTTP.
    
    Args:
        exception: Excepción a mapear (debe ser instancia de Exception)
        
    Returns:
        Código de estado HTTP apropiado
        
    Raises:
        TypeError: Si exception no es una instancia de Exception
    """
    if not isinstance(exception, Exception):
        raise TypeError(f"exception must be an instance of Exception, got {type(exception)}")
    
    if isinstance(exception, MCPAuthorizationError):
        return status.HTTP_403_FORBIDDEN
    elif isinstance(exception, MCPResourceNotFoundError):
        return status.HTTP_404_NOT_FOUND
    elif isinstance(exception, MCPValidationError):
        return status.HTTP_400_BAD_REQUEST
    elif isinstance(exception, MCPRateLimitError):
        return status.HTTP_429_TOO_MANY_REQUESTS
    elif isinstance(exception, (MCPConnectorError, MCPOperationError)):
        return status.HTTP_500_INTERNAL_SERVER_ERROR
    elif isinstance(exception, MCPContextLimitError):
        return status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
    elif isinstance(exception, MCPError):
        # Para otras excepciones MCP, usar 500 por defecto
        return status.HTTP_500_INTERNAL_SERVER_ERROR
    else:
        return status.HTTP_500_INTERNAL_SERVER_ERROR


def get_error_type(exception: Exception) -> str:
    """
    Obtiene el tipo de error como string.
    
    Args:
        exception: Excepción a analizar
        
    Returns:
        Nombre del tipo de error (snake_case)
        
    Raises:
        TypeError: Si exception no es una instancia de Exception
    """
    if not isinstance(exception, Exception):
        raise TypeError(f"exception must be an instance of Exception, got {type(exception)}")
    
    if isinstance(exception, MCPAuthorizationError):
        return "authorization_error"
    elif isinstance(exception, MCPResourceNotFoundError):
        return "resource_not_found"
    elif isinstance(exception, MCPConnectorError):
        return "connector_error"
    elif isinstance(exception, MCPOperationError):
        return "operation_error"
    elif isinstance(exception, MCPValidationError):
        return "validation_error"
    elif isinstance(exception, MCPRateLimitError):
        return "rate_limit_error"
    elif isinstance(exception, MCPContextLimitError):
        return "context_limit_error"
    else:
        return "unexpected_error"


def create_error_response(
    exception: Exception,
    resource_id: Optional[str] = None,
    operation: Optional[str] = None,
    include_details: bool = True
) -> MCPResponse:
    """
    Crear respuesta de error MCP desde una excepción.
    
    Args:
        exception: Excepción que causó el error (debe ser instancia de Exception)
        resource_id: ID del recurso (opcional)
        operation: Operación que falló (opcional)
        include_details: Si incluir detalles adicionales (default: True)
        
    Returns:
        MCPResponse con información del error
        
    Raises:
        TypeError: Si exception no es una instancia de Exception
    """
    if not isinstance(exception, Exception):
        raise TypeError(f"exception must be an instance of Exception, got {type(exception)}")
    
    error_type = get_error_type(exception)
    error_message = str(exception)
    
    # No incluir detalles sensibles en producción
    if not include_details and isinstance(exception, MCPError):
        error_message = f"{error_type.replace('_', ' ').title()}"
    
    metadata: Dict[str, Any] = {
        "error_type": error_type,
    }
    
    if resource_id:
        metadata["resource_id"] = resource_id
    if operation:
        metadata["operation"] = operation
    
    # Agregar detalles adicionales si la excepción los tiene
    if isinstance(exception, MCPError) and exception.details:
        metadata["details"] = exception.details
    if isinstance(exception, MCPError) and exception.error_code:
        metadata["error_code"] = exception.error_code
    
    return MCPResponse(
        success=False,
        error=error_message,
        metadata=metadata
    )


def handle_mcp_exception(
    exception: Exception,
    resource_id: Optional[str] = None,
    operation: Optional[str] = None,
    user_id: Optional[str] = None,
    log_level: str = "error"
) -> MCPResponse:
    """
    Manejar excepción MCP y crear respuesta apropiada.
    
    Args:
        exception: Excepción a manejar (debe ser instancia de Exception)
        resource_id: ID del recurso (opcional)
        operation: Operación que falló (opcional)
        user_id: ID del usuario (opcional)
        log_level: Nivel de logging ("error", "warning", "info", "debug")
        
    Returns:
        MCPResponse con información del error
        
    Raises:
        TypeError: Si exception no es una instancia de Exception
        ValueError: Si log_level es inválido
    """
    if not isinstance(exception, Exception):
        raise TypeError(f"exception must be an instance of Exception, got {type(exception)}")
    
    valid_log_levels = {"error", "warning", "info", "debug"}
    if log_level not in valid_log_levels:
        raise ValueError(f"log_level must be one of {valid_log_levels}, got {log_level}")
    
    error_type = get_error_type(exception)
    
    # Logging apropiado según el tipo de error
    log_message = (
        f"{error_type.replace('_', ' ').title()}: "
        f"{operation or 'operation'} on {resource_id or 'resource'}"
    )
    if user_id:
        log_message += f" for user {user_id}"
    
    if log_level == "warning":
        logger.warning(log_message, exc_info=False)
    elif log_level == "info":
        logger.info(log_message)
    else:
        logger.error(log_message, exc_info=True)
    
    return create_error_response(
        exception=exception,
        resource_id=resource_id,
        operation=operation,
        include_details=True
    )


def raise_http_exception_from_mcp(exception: Exception) -> None:
    """
    Convertir excepción MCP a HTTPException y lanzarla.
    
    Args:
        exception: Excepción MCP a convertir
        
    Raises:
        HTTPException: Excepción HTTP apropiada
    """
    http_status = map_exception_to_http_status(exception)
    error_type = get_error_type(exception)
    
    detail = str(exception)
    if isinstance(exception, MCPError) and exception.error_code:
        detail = f"[{exception.error_code}] {detail}"
    
    raise HTTPException(
        status_code=http_status,
        detail=detail,
        headers={
            "X-Error-Type": error_type,
        }
    )

