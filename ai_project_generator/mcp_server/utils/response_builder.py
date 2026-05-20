"""
Response Builder - Utilities for building MCP responses
=======================================================

Utilidades para construir respuestas MCP de forma consistente.
"""

import logging
from typing import Any, Dict, Optional
from datetime import datetime

from ..models import MCPResponse

logger = logging.getLogger(__name__)


def build_success_response(
    data: Any,
    metadata: Optional[Dict[str, Any]] = None,
    additional_metadata: Optional[Dict[str, Any]] = None
) -> MCPResponse:
    """
    Construir respuesta de éxito.
    
    Args:
        data: Datos de respuesta
        metadata: Metadata base (opcional)
        additional_metadata: Metadata adicional a combinar (opcional)
        
    Returns:
        MCPResponse con success=True
    """
    final_metadata = metadata or {}
    if additional_metadata:
        final_metadata = {**final_metadata, **additional_metadata}
    
    return MCPResponse.success_response(
        data=data,
        metadata=final_metadata
    )


def build_error_response(
    error: str,
    error_type: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    error_code: Optional[str] = None
) -> MCPResponse:
    """
    Construir respuesta de error.
    
    Args:
        error: Mensaje de error
        error_type: Tipo de error (opcional)
        metadata: Metadata adicional (opcional)
        error_code: Código de error (opcional)
        
    Returns:
        MCPResponse con success=False
    """
    final_metadata = metadata or {}
    if error_type:
        final_metadata["error_type"] = error_type
    if error_code:
        final_metadata["error_code"] = error_code
    
    return MCPResponse.error_response(
        error=error,
        metadata=final_metadata
    )


def build_response_from_exception(
    exception: Exception,
    resource_id: Optional[str] = None,
    operation: Optional[str] = None,
    include_details: bool = True
) -> MCPResponse:
    """
    Construir respuesta desde una excepción.
    
    Args:
        exception: Excepción a convertir
        resource_id: ID del recurso (opcional)
        operation: Operación que falló (opcional)
        include_details: Si incluir detalles de la excepción
        
    Returns:
        MCPResponse con información del error
    """
    from ..exceptions import MCPError
    from ..utils.error_handlers import get_error_type, create_error_response
    
    metadata: Dict[str, Any] = {}
    if resource_id:
        metadata["resource_id"] = resource_id
    if operation:
        metadata["operation"] = operation
    
    if isinstance(exception, MCPError):
        error_message = str(exception)
        error_type = get_error_type(exception)
        
        if exception.error_code:
            metadata["error_code"] = exception.error_code
        if include_details and exception.details:
            metadata["details"] = exception.details
        
        return build_error_response(
            error=error_message,
            error_type=error_type,
            metadata=metadata,
            error_code=exception.error_code
        )
    else:
        # Excepción genérica
        error_type = get_error_type(exception)
        error_message = str(exception) if include_details else "Internal server error"
        
        return build_error_response(
            error=error_message,
            error_type=error_type,
            metadata=metadata
        )

