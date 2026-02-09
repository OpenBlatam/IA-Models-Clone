"""
Helper functions for standardizing API responses.
Eliminates repetitive response dictionary construction.
"""

from typing import Any, Dict, Optional
from fastapi import status


def success_response(
    data: Any = None,
    message: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    status_code: int = status.HTTP_200_OK
) -> Dict[str, Any]:
    """
    Crea una respuesta de éxito estandarizada.
    
    Args:
        data: Datos principales de la respuesta
        message: Mensaje opcional de éxito
        metadata: Metadatos adicionales (stats, pagination, etc.)
        status_code: Código de estado HTTP (default: 200)
        
    Returns:
        Diccionario con estructura de respuesta estándar
        
    Examples:
        >>> success_response(
        ...     data={"identity": identity.model_dump()},
        ...     metadata={"count": 1}
        ... )
        {
            "success": True,
            "data": {"identity": {...}},
            "metadata": {"count": 1}
        }
    """
    response = {"success": True}
    
    if data is not None:
        response["data"] = data
    elif message:
        response["message"] = message
    
    if metadata:
        response["metadata"] = metadata
    
    return response


def error_response(
    message: str,
    error_code: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    status_code: int = status.HTTP_400_BAD_REQUEST
) -> Dict[str, Any]:
    """
    Crea una respuesta de error estandarizada.
    
    Args:
        message: Mensaje de error
        error_code: Código de error opcional
        details: Detalles adicionales del error
        status_code: Código de estado HTTP
        
    Returns:
        Diccionario con estructura de error estándar
    """
    response = {
        "success": False,
        "error": {
            "message": message
        }
    }
    
    if error_code:
        response["error"]["code"] = error_code
    
    if details:
        response["error"]["details"] = details
    
    return response


def paginated_response(
    items: list,
    page: int,
    page_size: int,
    total: Optional[int] = None,
    **additional_metadata
) -> Dict[str, Any]:
    """
    Crea una respuesta paginada estandarizada.
    
    Args:
        items: Lista de items en la página actual
        page: Número de página (1-indexed)
        page_size: Tamaño de página
        total: Total de items (opcional)
        **additional_metadata: Metadatos adicionales
        
    Returns:
        Diccionario con respuesta paginada
    """
    response = success_response(
        data=items,
        metadata={
            "pagination": {
                "page": page,
                "page_size": page_size,
                "count": len(items)
            },
            **additional_metadata
        }
    )
    
    if total is not None:
        response["metadata"]["pagination"]["total"] = total
        response["metadata"]["pagination"]["total_pages"] = (
            (total + page_size - 1) // page_size
        )
    
    return response








