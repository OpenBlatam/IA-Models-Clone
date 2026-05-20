"""
PDF Variantes API - Enhanced Error Handling
Comprehensive error handling with recovery strategies
"""

import logging
import traceback
from typing import Any, Dict, Optional, Type
from datetime import datetime
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import ValidationError

logger = logging.getLogger(__name__)


class APIError(Exception):
    """Base API error with structured information"""
    
    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        retryable: bool = False
    ):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code or f"ERR_{status_code}"
        self.details = details or {}
        self.retryable = retryable
        super().__init__(self.message)


class ValidationError(APIError):
    """Validation error"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=400,
            error_code="VALIDATION_ERROR",
            details=details
        )


class NotFoundError(APIError):
    """Resource not found error"""
    def __init__(self, resource: str, resource_id: str):
        super().__init__(
            message=f"{resource} not found: {resource_id}",
            status_code=404,
            error_code="NOT_FOUND",
            details={"resource": resource, "resource_id": resource_id}
        )


class RateLimitError(APIError):
    """Rate limit exceeded error"""
    def __init__(self, retry_after: int = 60):
        super().__init__(
            message="Rate limit exceeded",
            status_code=429,
            error_code="RATE_LIMIT_EXCEEDED",
            details={"retry_after": retry_after},
            retryable=True
        )


class ServiceUnavailableError(APIError):
    """Service unavailable error"""
    def __init__(self, service: str, retry_after: Optional[int] = None):
        super().__init__(
            message=f"Service unavailable: {service}",
            status_code=503,
            error_code="SERVICE_UNAVAILABLE",
            details={"service": service, "retry_after": retry_after},
            retryable=True
        )


class ErrorHandler:
    """Centralized error handler"""
    
    ERROR_HANDLERS: Dict[Type[Exception], Callable] = {}
    
    @classmethod
    def register_handler(cls, exception_type: Type[Exception], handler: Callable):
        """Register error handler for exception type"""
        cls.ERROR_HANDLERS[exception_type] = handler
    
    @classmethod
    async def handle_error(
        cls,
        request: Request,
        exc: Exception,
        include_traceback: bool = False
    ) -> JSONResponse:
        """Handle exception and return appropriate response"""
        import os
        
        # Determine if we're in development
        dev_mode = os.getenv("ENVIRONMENT", "development").lower() == "development"
        
        # Check for registered handler
        if type(exc) in cls.ERROR_HANDLERS:
            return await cls.ERROR_HANDLERS[type(exc)](request, exc)
        
        # Handle API errors
        if isinstance(exc, APIError):
            return cls._handle_api_error(request, exc, dev_mode)
        
        # Handle HTTP exceptions
        if isinstance(exc, HTTPException):
            return cls._handle_http_exception(request, exc, dev_mode)
        
        # Handle validation errors
        if isinstance(exc, ValidationError):
            return cls._handle_validation_error(request, exc, dev_mode)
        
        # Handle generic exceptions
        return cls._handle_generic_error(request, exc, dev_mode)
    
    @classmethod
    def _handle_api_error(
        cls,
        request: Request,
        exc: APIError,
        dev_mode: bool
    ) -> JSONResponse:
        """Handle APIError"""
        error_response = {
            "success": False,
            "error": {
                "message": exc.message,
                "code": exc.error_code,
                "status_code": exc.status_code,
                "retryable": exc.retryable
            },
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": getattr(request.state, 'request_id', None)
        }
        
        if exc.details:
            error_response["error"]["details"] = exc.details
        
        if exc.retryable:
            error_response["error"]["retry_after"] = exc.details.get("retry_after", 60)
        
        # Add traceback in dev mode
        if dev_mode:
            error_response["error"]["traceback"] = traceback.format_exc().split("\n")
        
        logger.error(f"API Error: {exc.error_code} - {exc.message}")
        
        return JSONResponse(
            status_code=exc.status_code,
            content=error_response,
            headers={"Retry-After": str(exc.details.get("retry_after", 60))} if exc.retryable else {}
        )
    
    @classmethod
    def _handle_http_exception(
        cls,
        request: Request,
        exc: HTTPException,
        dev_mode: bool
    ) -> JSONResponse:
        """Handle HTTPException"""
        error_response = {
            "success": False,
            "error": {
                "message": exc.detail,
                "code": f"HTTP_{exc.status_code}",
                "status_code": exc.status_code
            },
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": getattr(request.state, 'request_id', None)
        }
        
        if dev_mode:
            error_response["error"]["traceback"] = traceback.format_exc().split("\n")
        
        return JSONResponse(
            status_code=exc.status_code,
            content=error_response
        )
    
    @classmethod
    def _handle_validation_error(
        cls,
        request: Request,
        exc: ValidationError,
        dev_mode: bool
    ) -> JSONResponse:
        """Handle ValidationError"""
        errors = []
        if hasattr(exc, "errors"):
            errors = exc.errors()
        
        error_response = {
            "success": False,
            "error": {
                "message": "Validation error",
                "code": "VALIDATION_ERROR",
                "status_code": 400,
                "details": errors
            },
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": getattr(request.state, 'request_id', None)
        }
        
        return JSONResponse(
            status_code=400,
            content=error_response
        )
    
    @classmethod
    def _handle_generic_error(
        cls,
        request: Request,
        exc: Exception,
        dev_mode: bool
    ) -> JSONResponse:
        """Handle generic exceptions"""
        error_response = {
            "success": False,
            "error": {
                "message": str(exc) if dev_mode else "An unexpected error occurred",
                "code": "INTERNAL_ERROR",
                "status_code": 500,
                "type": type(exc).__name__
            },
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": getattr(request.state, 'request_id', None)
        }
        
        if dev_mode:
            error_response["error"]["traceback"] = traceback.format_exc().split("\n")
        
        logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
        
        return JSONResponse(
            status_code=500,
            content=error_response
        )


# Exception handlers for FastAPI
async def robust_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """FastAPI exception handler using ErrorHandler"""
    return await ErrorHandler.handle_error(request, exc)






