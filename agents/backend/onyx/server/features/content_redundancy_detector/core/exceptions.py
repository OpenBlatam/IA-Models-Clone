"""
Custom Exception Classes
Centralized exception handling with proper error codes and messages
"""

from typing import Optional, Dict, Any
import time


class APIException(Exception):
    """Base API exception class"""
    
    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_type: str = "APIError",
        detail: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.status_code = status_code
        self.error_type = error_type
        self.detail = detail or {}
        self.timestamp = time.time()
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary format"""
        return {
            "success": False,
            "data": None,
            "error": {
                "message": self.message,
                "status_code": self.status_code,
                "type": self.error_type,
                "detail": self.detail,
                "timestamp": self.timestamp
            },
            "timestamp": self.timestamp
        }


class ValidationError(APIException):
    """Validation error exception"""
    
    def __init__(
        self,
        message: str = "Validation error",
        detail: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            status_code=400,
            error_type="ValidationError",
            detail=detail
        )


class AuthenticationError(APIException):
    """Authentication error exception"""
    
    def __init__(
        self,
        message: str = "Authentication required",
        detail: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            status_code=401,
            error_type="AuthenticationError",
            detail=detail
        )


class AuthorizationError(APIException):
    """Authorization error exception"""
    
    def __init__(
        self,
        message: str = "Insufficient permissions",
        detail: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            status_code=403,
            error_type="AuthorizationError",
            detail=detail
        )


class NotFoundError(APIException):
    """Not found error exception"""
    
    def __init__(
        self,
        message: str = "Resource not found",
        detail: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            status_code=404,
            error_type="NotFoundError",
            detail=detail
        )


class RateLimitError(APIException):
    """Rate limit error exception"""
    
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: int = 60,
        detail: Optional[Dict[str, Any]] = None
    ):
        error_detail = detail or {}
        error_detail["retry_after"] = retry_after
        
        super().__init__(
            message=message,
            status_code=429,
            error_type="RateLimitError",
            detail=error_detail
        )


class InternalServerError(APIException):
    """Internal server error exception"""
    
    def __init__(
        self,
        message: str = "Internal server error",
        detail: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            status_code=500,
            error_type="InternalServerError",
            detail=detail
        )


class ServiceUnavailableError(APIException):
    """Service unavailable error exception"""
    
    def __init__(
        self,
        message: str = "Service temporarily unavailable",
        detail: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            status_code=503,
            error_type="ServiceUnavailableError",
            detail=detail
        )





