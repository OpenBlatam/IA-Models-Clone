"""
Custom exceptions for Physical Store Designer AI

This module defines a hierarchy of custom exceptions for consistent error handling
across the application. All exceptions inherit from PhysicalStoreDesignerError
and include status codes and structured error details.
"""

from typing import Optional, Dict, Any


class PhysicalStoreDesignerError(Exception):
    """
    Base exception for Physical Store Designer AI.
    
    All custom exceptions in the application inherit from this class.
    It provides structured error information including status codes,
    error codes, and detailed context.
    
    Attributes:
        message: Human-readable error message
        status_code: HTTP status code (default: 500)
        error_code: Machine-readable error code
        details: Additional error context and details
        
    Example:
        ```python
        raise PhysicalStoreDesignerError(
            message="Operation failed",
            status_code=400,
            error_code="OPERATION_FAILED",
            details={"operation": "create_design", "reason": "Invalid input"}
        )
        ```
    """
    
    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize exception.
        
        Args:
            message: Human-readable error message
            status_code: HTTP status code (default: 500)
            error_code: Machine-readable error code (defaults to class name)
            details: Additional error context
        """
        self.message = message
        self.status_code = status_code
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert exception to dictionary for API responses.
        
        Returns:
            Dictionary with error information suitable for JSON responses
            
        Example:
            ```python
            error_dict = exception.to_dict()
            # Returns: {
            #     "error": "VALIDATION_ERROR",
            #     "message": "Invalid input",
            #     "details": {"field": "name", "reason": "Required"}
            # }
            ```
        """
        return {
            "error": self.error_code,
            "message": self.message,
            "details": self.details
        }


class ValidationError(PhysicalStoreDesignerError):
    """Validation error"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=400,
            error_code="VALIDATION_ERROR",
            details=details
        )


class NotFoundError(PhysicalStoreDesignerError):
    """Resource not found error"""
    
    def __init__(self, resource: str, resource_id: Optional[str] = None):
        message = f"{resource} no encontrado"
        if resource_id:
            message += f": {resource_id}"
        super().__init__(
            message=message,
            status_code=404,
            error_code="NOT_FOUND",
            details={"resource": resource, "resource_id": resource_id}
        )


class StorageError(PhysicalStoreDesignerError):
    """Storage operation error"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=500,
            error_code="STORAGE_ERROR",
            details=details
        )


class ServiceError(PhysicalStoreDesignerError):
    """Service operation error"""
    
    def __init__(self, service: str, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"Error en {service}: {message}",
            status_code=500,
            error_code="SERVICE_ERROR",
            details={"service": service, **details} if details else {"service": service}
        )


class ExternalAPIError(PhysicalStoreDesignerError):
    """External API error"""
    
    def __init__(self, api: str, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"Error en API externa {api}: {message}",
            status_code=502,
            error_code="EXTERNAL_API_ERROR",
            details={"api": api, **details} if details else {"api": api}
        )


class RateLimitError(PhysicalStoreDesignerError):
    """Rate limit exceeded error"""
    
    def __init__(self, message: str = "Límite de requests excedido", retry_after: Optional[int] = None):
        details = {}
        if retry_after:
            details["retry_after"] = retry_after
        super().__init__(
            message=message,
            status_code=429,
            error_code="RATE_LIMIT_EXCEEDED",
            details=details
        )


class AuthenticationError(PhysicalStoreDesignerError):
    """Authentication error"""
    
    def __init__(self, message: str = "No autenticado"):
        super().__init__(
            message=message,
            status_code=401,
            error_code="AUTHENTICATION_ERROR"
        )


class AuthorizationError(PhysicalStoreDesignerError):
    """Authorization error"""
    
    def __init__(self, message: str = "No autorizado"):
        super().__init__(
            message=message,
            status_code=403,
            error_code="AUTHORIZATION_ERROR"
        )


class TimeoutError(PhysicalStoreDesignerError):
    """Operation timeout error"""
    
    def __init__(self, operation: str, timeout_seconds: Optional[float] = None):
        message = f"Operación '{operation}' excedió el tiempo límite"
        if timeout_seconds:
            message += f" ({timeout_seconds}s)"
        super().__init__(
            message=message,
            status_code=504,
            error_code="TIMEOUT_ERROR",
            details={"operation": operation, "timeout_seconds": timeout_seconds}
        )


class ConflictError(PhysicalStoreDesignerError):
    """Resource conflict error"""
    
    def __init__(self, resource: str, message: Optional[str] = None):
        error_message = message or f"Conflicto con recurso: {resource}"
        super().__init__(
            message=error_message,
            status_code=409,
            error_code="CONFLICT_ERROR",
            details={"resource": resource}
        )


class TooManyRequestsError(PhysicalStoreDesignerError):
    """Too many requests error (alternative to RateLimitError)"""
    
    def __init__(self, message: str = "Demasiadas solicitudes", retry_after: Optional[int] = None):
        details = {}
        if retry_after:
            details["retry_after"] = retry_after
        super().__init__(
            message=message,
            status_code=429,
            error_code="TOO_MANY_REQUESTS",
            details=details
        )
