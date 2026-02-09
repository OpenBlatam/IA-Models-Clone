"""
Error Handler Middleware
========================

Advanced error handling middleware following FastAPI best practices.
"""

from __future__ import annotations
import logging
import traceback
from typing import Any, Dict, Optional, Union
from fastapi import Request, Response, HTTPException, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from ..services.audit_service import log_security_event, AuditEventType, AuditStatus
from ..services.metrics_collector import increment_counter, observe_histogram
from ..utils.helpers import DateTimeHelpers


logger = logging.getLogger(__name__)


class ErrorResponse:
    """Standardized error response"""
    
    def __init__(
        self,
        error_code: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        status_code: int = 500,
        request_id: Optional[str] = None
    ):
        self.error_code = error_code
        self.message = message
        self.details = details or {}
        self.status_code = status_code
        self.request_id = request_id
        self.timestamp = DateTimeHelpers.now_utc().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "error": {
                "code": self.error_code,
                "message": self.message,
                "details": self.details,
                "request_id": self.request_id,
                "timestamp": self.timestamp
            }
        }


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """Advanced error handling middleware"""
    
    def __init__(
        self,
        app: ASGIApp,
        enable_logging: bool = True,
        enable_metrics: bool = True,
        enable_audit: bool = True,
        include_traceback: bool = False
    ):
        super().__init__(app)
        self.enable_logging = enable_logging
        self.enable_metrics = enable_metrics
        self.enable_audit = enable_audit
        self.include_traceback = include_traceback
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request with error handling"""
        request_id = getattr(request.state, 'request_id', None)
        
        try:
            response = await call_next(request)
            
            # Log successful requests
            if self.enable_logging:
                logger.info(f"Request completed successfully: {request.method} {request.url.path}")
            
            # Record success metrics
            if self.enable_metrics:
                increment_counter("http_requests_success_total", labels={
                    "method": request.method,
                    "endpoint": self._normalize_endpoint(request.url.path),
                    "status_code": str(response.status_code)
                })
            
            return response
        
        except HTTPException as e:
            return await self._handle_http_exception(e, request, request_id)
        
        except RequestValidationError as e:
            return await self._handle_validation_error(e, request, request_id)
        
        except StarletteHTTPException as e:
            return await self._handle_starlette_http_exception(e, request, request_id)
        
        except Exception as e:
            return await self._handle_unexpected_error(e, request, request_id)
    
    async def _handle_http_exception(
        self, 
        exc: HTTPException, 
        request: Request, 
        request_id: Optional[str]
    ) -> JSONResponse:
        """Handle HTTP exceptions"""
        error_response = ErrorResponse(
            error_code="HTTP_ERROR",
            message=exc.detail,
            status_code=exc.status_code,
            request_id=request_id
        )
        
        # Log error
        if self.enable_logging:
            logger.warning(f"HTTP error: {exc.status_code} - {exc.detail}")
        
        # Record metrics
        if self.enable_metrics:
            increment_counter("http_errors_total", labels={
                "method": request.method,
                "endpoint": self._normalize_endpoint(request.url.path),
                "status_code": str(exc.status_code),
                "error_type": "http_exception"
            })
        
        # Audit security-related errors
        if self.enable_audit and exc.status_code in [401, 403, 429]:
            await self._audit_security_error(exc, request)
        
        return JSONResponse(
            status_code=exc.status_code,
            content=error_response.to_dict()
        )
    
    async def _handle_validation_error(
        self, 
        exc: RequestValidationError, 
        request: Request, 
        request_id: Optional[str]
    ) -> JSONResponse:
        """Handle validation errors"""
        error_response = ErrorResponse(
            error_code="VALIDATION_ERROR",
            message="Request validation failed",
            details={"errors": exc.errors()},
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            request_id=request_id
        )
        
        # Log error
        if self.enable_logging:
            logger.warning(f"Validation error: {exc.errors()}")
        
        # Record metrics
        if self.enable_metrics:
            increment_counter("http_errors_total", labels={
                "method": request.method,
                "endpoint": self._normalize_endpoint(request.url.path),
                "status_code": "422",
                "error_type": "validation_error"
            })
        
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=error_response.to_dict()
        )
    
    async def _handle_starlette_http_exception(
        self, 
        exc: StarletteHTTPException, 
        request: Request, 
        request_id: Optional[str]
    ) -> JSONResponse:
        """Handle Starlette HTTP exceptions"""
        error_response = ErrorResponse(
            error_code="STARLETTE_HTTP_ERROR",
            message=exc.detail,
            status_code=exc.status_code,
            request_id=request_id
        )
        
        # Log error
        if self.enable_logging:
            logger.warning(f"Starlette HTTP error: {exc.status_code} - {exc.detail}")
        
        # Record metrics
        if self.enable_metrics:
            increment_counter("http_errors_total", labels={
                "method": request.method,
                "endpoint": self._normalize_endpoint(request.url.path),
                "status_code": str(exc.status_code),
                "error_type": "starlette_http_exception"
            })
        
        return JSONResponse(
            status_code=exc.status_code,
            content=error_response.to_dict()
        )
    
    async def _handle_unexpected_error(
        self, 
        exc: Exception, 
        request: Request, 
        request_id: Optional[str]
    ) -> JSONResponse:
        """Handle unexpected errors"""
        # Log error with full traceback
        if self.enable_logging:
            logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
        
        # Record metrics
        if self.enable_metrics:
            increment_counter("http_errors_total", labels={
                "method": request.method,
                "endpoint": self._normalize_endpoint(request.url.path),
                "status_code": "500",
                "error_type": "unexpected_error"
            })
        
        # Audit critical errors
        if self.enable_audit:
            await self._audit_critical_error(exc, request)
        
        # Prepare error response
        error_details = {
            "error_type": type(exc).__name__,
            "error_message": str(exc)
        }
        
        if self.include_traceback:
            error_details["traceback"] = traceback.format_exc()
        
        error_response = ErrorResponse(
            error_code="INTERNAL_SERVER_ERROR",
            message="An internal server error occurred",
            details=error_details,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            request_id=request_id
        )
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_response.to_dict()
        )
    
    async def _audit_security_error(self, exc: HTTPException, request: Request) -> None:
        """Audit security-related errors"""
        try:
            client_ip = self._get_client_ip(request)
            user_agent = request.headers.get("user-agent", "unknown")
            
            audit_status = AuditStatus.FAILURE
            if exc.status_code == 401:
                event_type = AuditEventType.UNAUTHORIZED_ACCESS
            elif exc.status_code == 403:
                event_type = AuditEventType.PERMISSION_DENIED
            elif exc.status_code == 429:
                event_type = AuditEventType.SECURITY_VIOLATION
            else:
                return
            
            log_security_event(
                event_type=event_type,
                user_id=None,
                ip_address=client_ip,
                user_agent=user_agent,
                status=audit_status,
                details={
                    "error_code": exc.status_code,
                    "error_message": exc.detail,
                    "endpoint": request.url.path,
                    "method": request.method
                }
            )
        except Exception as e:
            logger.error(f"Failed to audit security error: {e}")
    
    async def _audit_critical_error(self, exc: Exception, request: Request) -> None:
        """Audit critical errors"""
        try:
            client_ip = self._get_client_ip(request)
            user_agent = request.headers.get("user-agent", "unknown")
            
            log_security_event(
                event_type=AuditEventType.SECURITY_VIOLATION,
                user_id=None,
                ip_address=client_ip,
                user_agent=user_agent,
                status=AuditStatus.FAILURE,
                details={
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "endpoint": request.url.path,
                    "method": request.method,
                    "critical": True
                }
            )
        except Exception as e:
            logger.error(f"Failed to audit critical error: {e}")
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address"""
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        if hasattr(request.client, 'host'):
            return request.client.host
        
        return "unknown"
    
    def _normalize_endpoint(self, url_path: str) -> str:
        """Normalize endpoint path for metrics"""
        import re
        
        # Remove query parameters
        if "?" in url_path:
            url_path = url_path.split("?")[0]
        
        # Replace dynamic segments with placeholders
        url_path = re.sub(r'/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', '/{uuid}', url_path)
        url_path = re.sub(r'/\d+', '/{id}', url_path)
        url_path = re.sub(r'/[a-zA-Z0-9_-]{20,}', '/{token}', url_path)
        
        return url_path


# Custom exception classes
class BusinessLogicError(Exception):
    """Business logic error"""
    
    def __init__(self, message: str, error_code: str = "BUSINESS_LOGIC_ERROR", details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(message)


class ValidationError(Exception):
    """Validation error"""
    
    def __init__(self, message: str, field: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.field = field
        self.details = details or {}
        super().__init__(message)


class ResourceNotFoundError(Exception):
    """Resource not found error"""
    
    def __init__(self, resource_type: str, resource_id: str, details: Optional[Dict[str, Any]] = None):
        self.resource_type = resource_type
        self.resource_id = resource_id
        self.details = details or {}
        message = f"{resource_type} with ID {resource_id} not found"
        super().__init__(message)


class PermissionDeniedError(Exception):
    """Permission denied error"""
    
    def __init__(self, action: str, resource: str, details: Optional[Dict[str, Any]] = None):
        self.action = action
        self.resource = resource
        self.details = details or {}
        message = f"Permission denied for action '{action}' on resource '{resource}'"
        super().__init__(message)


class RateLimitExceededError(Exception):
    """Rate limit exceeded error"""
    
    def __init__(self, limit: int, window: int, retry_after: int, details: Optional[Dict[str, Any]] = None):
        self.limit = limit
        self.window = window
        self.retry_after = retry_after
        self.details = details or {}
        message = f"Rate limit exceeded: {limit} requests per {window} seconds"
        super().__init__(message)


# Error handler functions
def handle_business_logic_error(exc: BusinessLogicError, request: Request) -> JSONResponse:
    """Handle business logic errors"""
    error_response = ErrorResponse(
        error_code=exc.error_code,
        message=exc.message,
        details=exc.details,
        status_code=status.HTTP_400_BAD_REQUEST
    )
    
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=error_response.to_dict()
    )


def handle_validation_error(exc: ValidationError, request: Request) -> JSONResponse:
    """Handle validation errors"""
    error_response = ErrorResponse(
        error_code="VALIDATION_ERROR",
        message=exc.message,
        details={
            "field": exc.field,
            **exc.details
        },
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=error_response.to_dict()
    )


def handle_resource_not_found_error(exc: ResourceNotFoundError, request: Request) -> JSONResponse:
    """Handle resource not found errors"""
    error_response = ErrorResponse(
        error_code="RESOURCE_NOT_FOUND",
        message=str(exc),
        details={
            "resource_type": exc.resource_type,
            "resource_id": exc.resource_id,
            **exc.details
        },
        status_code=status.HTTP_404_NOT_FOUND
    )
    
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content=error_response.to_dict()
    )


def handle_permission_denied_error(exc: PermissionDeniedError, request: Request) -> JSONResponse:
    """Handle permission denied errors"""
    error_response = ErrorResponse(
        error_code="PERMISSION_DENIED",
        message=str(exc),
        details={
            "action": exc.action,
            "resource": exc.resource,
            **exc.details
        },
        status_code=status.HTTP_403_FORBIDDEN
    )
    
    return JSONResponse(
        status_code=status.HTTP_403_FORBIDDEN,
        content=error_response.to_dict()
    )


def handle_rate_limit_exceeded_error(exc: RateLimitExceededError, request: Request) -> JSONResponse:
    """Handle rate limit exceeded errors"""
    error_response = ErrorResponse(
        error_code="RATE_LIMIT_EXCEEDED",
        message=str(exc),
        details={
            "limit": exc.limit,
            "window": exc.window,
            "retry_after": exc.retry_after,
            **exc.details
        },
        status_code=status.HTTP_429_TOO_MANY_REQUESTS
    )
    
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content=error_response.to_dict(),
        headers={"Retry-After": str(exc.retry_after)}
    )


# Utility functions
def create_error_response(
    error_code: str,
    message: str,
    status_code: int = 500,
    details: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None
) -> JSONResponse:
    """Create standardized error response"""
    error_response = ErrorResponse(
        error_code=error_code,
        message=message,
        details=details,
        status_code=status_code,
        request_id=request_id
    )
    
    return JSONResponse(
        status_code=status_code,
        content=error_response.to_dict()
    )


def log_error(error: Exception, request: Request, context: Optional[Dict[str, Any]] = None) -> None:
    """Log error with context"""
    context = context or {}
    
    logger.error(
        f"Error in {request.method} {request.url.path}: {str(error)}",
        extra={
            "error_type": type(error).__name__,
            "error_message": str(error),
            "request_method": request.method,
            "request_path": request.url.path,
            "client_ip": request.client.host if request.client else "unknown",
            "user_agent": request.headers.get("user-agent", "unknown"),
            **context
        },
        exc_info=True
    )




