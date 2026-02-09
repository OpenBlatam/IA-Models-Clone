"""
Custom error types and error factories for consistent error handling
"""

from typing import Optional, Dict, Any
from fastapi import HTTPException, status


class APIError(Exception):
    """Base API error class"""
    def __init__(
        self,
        message: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class ValidationError(APIError):
    """Validation error"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_400_BAD_REQUEST,
            error_code="VALIDATION_ERROR",
            details=details
        )


class NotFoundError(APIError):
    """Resource not found error"""
    def __init__(self, resource: str, resource_id: Optional[str] = None):
        message = f"{resource} no encontrado"
        if resource_id:
            message += f": {resource_id}"
        super().__init__(
            message=message,
            status_code=status.HTTP_404_NOT_FOUND,
            error_code="NOT_FOUND",
            details={"resource": resource, "resource_id": resource_id}
        )


class UnauthorizedError(APIError):
    """Unauthorized access error"""
    def __init__(self, message: str = "No autorizado"):
        super().__init__(
            message=message,
            status_code=status.HTTP_401_UNAUTHORIZED,
            error_code="UNAUTHORIZED"
        )


class ForbiddenError(APIError):
    """Forbidden access error"""
    def __init__(self, message: str = "Acceso prohibido"):
        super().__init__(
            message=message,
            status_code=status.HTTP_403_FORBIDDEN,
            error_code="FORBIDDEN"
        )


class ConflictError(APIError):
    """Resource conflict error"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_409_CONFLICT,
            error_code="CONFLICT",
            details=details
        )


def create_http_exception(error: APIError) -> HTTPException:
    """Convert APIError to HTTPException"""
    return HTTPException(
        status_code=error.status_code,
        detail={
            "message": error.message,
            "error_code": error.error_code,
            "details": error.details
        }
    )


def handle_service_error(error: Exception) -> HTTPException:
    """Handle service errors and convert to HTTPException"""
    if isinstance(error, APIError):
        return create_http_exception(error)
    
    # Log unexpected errors
    import logging
    logger = logging.getLogger(__name__)
    logger.error(f"Unexpected error: {str(error)}", exc_info=True)
    
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail={
            "message": "Error interno del servidor",
            "error_code": "INTERNAL_SERVER_ERROR"
        }
    )

