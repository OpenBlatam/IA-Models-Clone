"""
Helper functions for creating standardized HTTP exceptions.
Eliminates repetitive HTTPException patterns.
"""

from fastapi import HTTPException, status
from typing import Optional, Dict, Any


class APIException(HTTPException):
    """Excepción HTTP personalizada con estructura consistente"""
    
    def __init__(
        self,
        message: str,
        status_code: int = status.HTTP_400_BAD_REQUEST,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.error_code = error_code
        self.details = details or {}
        
        detail_dict = {"message": message}
        if error_code:
            detail_dict["error_code"] = error_code
        if details:
            detail_dict["details"] = details
        
        super().__init__(status_code=status_code, detail=detail_dict)


def not_found(resource: str, identifier: str) -> APIException:
    """
    Crea excepción 404 para recurso no encontrado.
    
    Args:
        resource: Tipo de recurso (ej: "Identidad", "Contenido")
        identifier: Identificador del recurso
        
    Returns:
        APIException con status 404
    """
    return APIException(
        message=f"{resource} no encontrado: {identifier}",
        status_code=status.HTTP_404_NOT_FOUND,
        error_code="RESOURCE_NOT_FOUND",
        details={"resource": resource, "identifier": identifier}
    )


def validation_error(message: str, field: Optional[str] = None) -> APIException:
    """
    Crea excepción 400 para error de validación.
    
    Args:
        message: Mensaje de error
        field: Campo que falló la validación (opcional)
        
    Returns:
        APIException con status 400
    """
    details = {}
    if field:
        details["field"] = field
    
    return APIException(
        message=message,
        status_code=status.HTTP_400_BAD_REQUEST,
        error_code="VALIDATION_ERROR",
        details=details
    )


def unauthorized(message: str = "No autorizado") -> APIException:
    """Crea excepción 401 para no autorizado."""
    return APIException(
        message=message,
        status_code=status.HTTP_401_UNAUTHORIZED,
        error_code="UNAUTHORIZED"
    )


def forbidden(message: str = "Acceso prohibido") -> APIException:
    """Crea excepción 403 para prohibido."""
    return APIException(
        message=message,
        status_code=status.HTTP_403_FORBIDDEN,
        error_code="FORBIDDEN"
    )


def internal_error(message: str = "Error interno del servidor") -> APIException:
    """Crea excepción 500 para error interno."""
    return APIException(
        message=message,
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_code="INTERNAL_ERROR"
    )








