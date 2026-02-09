"""
Custom exceptions for Lovable Community SAM3.
"""

from typing import Optional, Dict, Any


class LovableException(Exception):
    """Base exception for Lovable Community SAM3."""
    
    def __init__(
        self,
        message: str,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize exception."""
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary."""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "status_code": self.status_code,
            "details": self.details
        }


class NotFoundError(LovableException):
    """Exception raised when a resource is not found."""
    
    def __init__(self, resource: str, resource_id: Optional[str] = None):
        """Initialize not found error."""
        message = f"{resource} not found"
        if resource_id:
            message += f": {resource_id}"
        super().__init__(message, status_code=404, details={"resource": resource, "resource_id": resource_id})


class ValidationError(LovableException):
    """Exception raised when validation fails."""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Optional[Any] = None):
        """Initialize validation error."""
        details = {}
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = str(value)
        super().__init__(message, status_code=400, details=details)


class AuthorizationError(LovableException):
    """Exception raised when authorization fails."""
    
    def __init__(self, message: str = "Not authorized"):
        """Initialize authorization error."""
        super().__init__(message, status_code=403)


class ConflictError(LovableException):
    """Exception raised when a conflict occurs (e.g., duplicate resource)."""
    
    def __init__(self, message: str, resource: Optional[str] = None):
        """Initialize conflict error."""
        details = {}
        if resource:
            details["resource"] = resource
        super().__init__(message, status_code=409, details=details)


class ServiceUnavailableError(LovableException):
    """Exception raised when a service is unavailable."""
    
    def __init__(self, service: str, message: Optional[str] = None):
        """Initialize service unavailable error."""
        if not message:
            message = f"Service {service} is currently unavailable"
        super().__init__(message, status_code=503, details={"service": service})






