"""Custom exception classes for better error handling."""
from fastapi import HTTPException
from typing import Any, Dict, Optional


class APIError(HTTPException):
    """Base API error with structured details."""
    
    def __init__(
        self,
        status_code: int,
        detail: str,
        error_code: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        super().__init__(status_code=status_code, detail=detail)
        self.error_code = error_code
        self.metadata = metadata or {}


class ValidationError(APIError):
    """Input validation error."""
    
    def __init__(self, detail: str, field: Optional[str] = None):
        super().__init__(
            status_code=422,
            detail=detail,
            error_code="VALIDATION_ERROR",
            metadata={"field": field} if field else {}
        )


class NotFoundError(APIError):
    """Resource not found error."""
    
    def __init__(self, resource_type: str, resource_id: str):
        super().__init__(
            status_code=404,
            detail=f"{resource_type} not found: {resource_id}",
            error_code="NOT_FOUND",
            metadata={"resource_type": resource_type, "resource_id": resource_id}
        )


class RateLimitError(APIError):
    """Rate limit exceeded error."""
    
    def __init__(self, retry_after: Optional[int] = None):
        super().__init__(
            status_code=429,
            detail="Rate limit exceeded",
            error_code="RATE_LIMIT_EXCEEDED",
            metadata={"retry_after": retry_after} if retry_after else {}
        )


class ServiceUnavailableError(APIError):
    """Service unavailable error (circuit breaker, etc.)."""
    
    def __init__(self, service_name: str, reason: Optional[str] = None):
        super().__init__(
            status_code=503,
            detail=f"Service unavailable: {service_name}",
            error_code="SERVICE_UNAVAILABLE",
            metadata={"service_name": service_name, "reason": reason}
        )
