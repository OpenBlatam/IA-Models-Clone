"""
Response Helpers
================
Utilidades para crear respuestas estandarizadas.
"""

from typing import Optional, Dict, Any
from datetime import datetime
from fastapi import status
from fastapi.responses import JSONResponse


def create_success_response(
    data: Any,
    message: Optional[str] = None,
    status_code: int = status.HTTP_200_OK,
    metadata: Optional[Dict[str, Any]] = None
) -> JSONResponse:
    """
    Crear respuesta de éxito estandarizada.
    
    Args:
        data: Datos de la respuesta
        message: Mensaje opcional
        status_code: Código de estado HTTP
        metadata: Metadatos adicionales
        
    Returns:
        JSONResponse con formato estándar
    """
    content: Dict[str, Any] = {
        "success": True,
        "data": data,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if message:
        content["message"] = message
    
    if metadata:
        content["metadata"] = metadata
    
    return JSONResponse(
        content=content,
        status_code=status_code,
        media_type="application/json"
    )


def create_error_response(
    message: str,
    error_code: Optional[str] = None,
    status_code: int = status.HTTP_400_BAD_REQUEST,
    details: Optional[Dict[str, Any]] = None,
    field: Optional[str] = None
) -> JSONResponse:
    """
    Crear respuesta de error estandarizada.
    
    Args:
        message: Mensaje de error
        error_code: Código de error opcional
        status_code: Código de estado HTTP
        details: Detalles adicionales
        field: Campo que causó el error
        
    Returns:
        JSONResponse con formato de error estándar
    """
    error_content: Dict[str, Any] = {
        "message": message,
        "code": error_code or "ERROR"
    }
    
    if field:
        error_content["field"] = field
    
    if details:
        error_content["details"] = details
    
    content = {
        "success": False,
        "error": error_content,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    return JSONResponse(
        content=content,
        status_code=status_code,
        media_type="application/json"
    )

