from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

from typing import Dict, Any, Optional, Union
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import structlog
from datetime import datetime, timezone
from .http_exceptions import (
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
    from fastapi import FastAPI
    from .exceptions.exception_handlers import register_exception_handlers, RequestIDMiddleware
    from .exceptions.http_exceptions import (
from typing import Any, List, Dict, Optional
import logging
import asyncio
#!/usr/bin/env python3
"""
Exception Handlers for HeyGen AI API
Comprehensive exception handling with structured responses and logging.
"""


    BaseHTTPException, ValidationError, AuthenticationError,
    AuthorizationError, NotFoundError, RateLimitError,
    ExternalServiceError, ResourceConflictError, PayloadTooLargeError,
    UnsupportedMediaError, BusinessLogicError, InternalServerError,
    ErrorResponse, log_exception, create_error_response,
    ErrorCategory, ErrorSeverity
)

logger = structlog.get_logger()

# =============================================================================
# Exception Handler Functions
# =============================================================================

async async def http_exception_handler(
    request: Request,
    exc: BaseHTTPException
) -> JSONResponse:
    """Handle custom HTTP exceptions."""
    request_id = request.headers.get("X-Request-ID")
    
    # Log the exception
    log_exception(exc, request_id)
    
    # Create error response
    error_response = create_error_response(exc, request_id)
    
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response,
        headers=exc.headers
    )

async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError
) -> JSONResponse:
    """Handle validation exceptions."""
    request_id = request.headers.get("X-Request-ID")
    
    # Convert validation errors to our format
    details = []
    for error in exc.errors():
        detail = {
            "field": " -> ".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "value": error.get("input"),
            "suggestion": get_validation_suggestion(error)
        }
        details.append(detail)
    
    # Create validation error
    validation_error = ValidationError(
        message="Request validation failed",
        details=details,
        request_id=request_id
    )
    
    # Log the exception
    log_exception(validation_error, request_id)
    
    # Create error response
    error_response = create_error_response(validation_error, request_id)
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=error_response
    )

async async def starlette_http_exception_handler(
    request: Request,
    exc: StarletteHTTPException
) -> JSONResponse:
    """Handle Starlette HTTP exceptions."""
    request_id = request.headers.get("X-Request-ID")
    
    # Convert to our format
    error_response = ErrorResponse(
        error_code=f"HTTP_{exc.status_code}",
        message=exc.detail,
        category=get_error_category(exc.status_code),
        severity=get_error_severity(exc.status_code),
        timestamp=datetime.now(timezone.utc),
        request_id=request_id
    )
    
    # Log the exception
    logger.error(
        "Starlette HTTP exception",
        status_code=exc.status_code,
        detail=exc.detail,
        request_id=request_id
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.dict(),
        headers=exc.headers
    )

async def general_exception_handler(
    request: Request,
    exc: Exception
) -> JSONResponse:
    """Handle general exceptions."""
    request_id = request.headers.get("X-Request-ID")
    
    # Create internal server error
    internal_error = InternalServerError(
        message="An unexpected error occurred",
        request_id=request_id
    )
    
    # Log the exception with full details
    logger.error(
        "Unhandled exception",
        exception_type=type(exc).__name__,
        exception_message=str(exc),
        request_id=request_id,
        exc_info=True
    )
    
    # Create error response
    error_response = create_error_response(internal_error, request_id)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response
    )

# =============================================================================
# Utility Functions
# =============================================================================

def get_validation_suggestion(error: Dict[str, Any]) -> Optional[str]:
    """Get suggestion for validation error."""
    error_type = error.get("type")
    
    suggestions = {
        "missing": "This field is required",
        "value_error.missing": "This field is required",
        "type_error.none.not_allowed": "This field cannot be null",
        "type_error.integer": "This field must be an integer",
        "type_error.float": "This field must be a number",
        "type_error.string": "This field must be a string",
        "type_error.boolean": "This field must be a boolean",
        "value_error.any_str.min_length": "This field is too short",
        "value_error.any_str.max_length": "This field is too long",
        "value_error.number.not_gt": "This value must be greater than the minimum",
        "value_error.number.not_lt": "This value must be less than the maximum",
        "value_error.email": "Please provide a valid email address",
        "value_error.url": "Please provide a valid URL",
        "value_error.uuid": "Please provide a valid UUID",
        "value_error.date": "Please provide a valid date",
        "value_error.datetime": "Please provide a valid datetime",
    }
    
    return suggestions.get(error_type, "Please check the field value")

def get_error_category(status_code: int) -> ErrorCategory:
    """Get error category from status code."""
    if status_code == 400:
        return ErrorCategory.VALIDATION
    elif status_code == 401:
        return ErrorCategory.AUTHENTICATION
    elif status_code == 403:
        return ErrorCategory.AUTHORIZATION
    elif status_code == 404:
        return ErrorCategory.NOT_FOUND
    elif status_code == 409:
        return ErrorCategory.RESOURCE_CONFLICT
    elif status_code == 413:
        return ErrorCategory.PAYLOAD_TOO_LARGE
    elif status_code == 415:
        return ErrorCategory.UNSUPPORTED_MEDIA
    elif status_code == 422:
        return ErrorCategory.BUSINESS_LOGIC
    elif status_code == 429:
        return ErrorCategory.RATE_LIMIT
    elif status_code >= 500:
        return ErrorCategory.EXTERNAL_SERVICE
    else:
        return ErrorCategory.INTERNAL_ERROR

def get_error_severity(status_code: int) -> ErrorSeverity:
    """Get error severity from status code."""
    if status_code in [400, 404, 409, 413, 415, 422, 429]:
        return ErrorSeverity.MEDIUM
    elif status_code in [401, 403]:
        return ErrorSeverity.HIGH
    elif status_code >= 500:
        return ErrorSeverity.CRITICAL
    else:
        return ErrorSeverity.MEDIUM

# =============================================================================
# Exception Handler Registration
# =============================================================================

def register_exception_handlers(app: FastAPI):
    """Register all exception handlers with FastAPI app."""
    
    # Register custom HTTP exception handler
    app.add_exception_handler(BaseHTTPException, http_exception_handler)
    
    # Register validation exception handler
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    
    # Register Starlette HTTP exception handler
    app.add_exception_handler(StarletteHTTPException, starlette_http_exception_handler)
    
    # Register general exception handler (must be last)
    app.add_exception_handler(Exception, general_exception_handler)
    
    logger.info("Exception handlers registered successfully")

# =============================================================================
# Middleware for Request ID
# =============================================================================


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware to add request ID to all requests."""
    
    async def dispatch(self, request: Request, call_next):
        
    """dispatch function."""
# Generate request ID if not present
        request_id = request.headers.get("X-Request-ID")
        if not request_id:
            request_id = str(uuid.uuid4())
        
        # Add request ID to request state
        request.state.request_id = request_id
        
        # Process request
        response = await call_next(request)
        
        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        
        return response

# =============================================================================
# Error Response Examples
# =============================================================================

ERROR_RESPONSE_EXAMPLES = {
    "validation_error": {
        "error_code": "VALIDATION_ERROR",
        "message": "Request validation failed",
        "category": "validation",
        "severity": "medium",
        "timestamp": "2024-01-01T12:00:00Z",
        "request_id": "123e4567-e89b-12d3-a456-426614174000",
        "details": [
            {
                "field": "body -> email",
                "message": "value is not a valid email address",
                "value": "invalid-email",
                "suggestion": "Please provide a valid email address"
            }
        ],
        "documentation_url": None,
        "retry_after": None
    },
    
    "authentication_error": {
        "error_code": "INVALID_CREDENTIALS",
        "message": "Invalid credentials provided",
        "category": "authentication",
        "severity": "high",
        "timestamp": "2024-01-01T12:00:00Z",
        "request_id": "123e4567-e89b-12d3-a456-426614174000",
        "details": None,
        "documentation_url": "https://docs.example.com/authentication",
        "retry_after": None
    },
    
    "rate_limit_error": {
        "error_code": "RATE_LIMIT_EXCEEDED",
        "message": "Rate limit exceeded",
        "category": "rate_limit",
        "severity": "medium",
        "timestamp": "2024-01-01T12:00:00Z",
        "request_id": "123e4567-e89b-12d3-a456-426614174000",
        "details": None,
        "documentation_url": "https://docs.example.com/rate-limits",
        "retry_after": 60
    },
    
    "not_found_error": {
        "error_code": "VIDEO_NOT_FOUND",
        "message": "Video not found",
        "category": "not_found",
        "severity": "medium",
        "timestamp": "2024-01-01T12:00:00Z",
        "request_id": "123e4567-e89b-12d3-a456-426614174000",
        "details": None,
        "documentation_url": "https://docs.example.com/videos",
        "retry_after": None
    },
    
    "external_service_error": {
        "error_code": "HEYGEN_API_ERROR",
        "message": "HeyGen AI service is temporarily unavailable",
        "category": "external_service",
        "severity": "high",
        "timestamp": "2024-01-01T12:00:00Z",
        "request_id": "123e4567-e89b-12d3-a456-426614174000",
        "details": None,
        "documentation_url": "https://docs.example.com/troubleshooting",
        "retry_after": 30
    },
    
    "internal_server_error": {
        "error_code": "INTERNAL_SERVER_ERROR",
        "message": "Internal server error occurred",
        "category": "internal_error",
        "severity": "critical",
        "timestamp": "2024-01-01T12:00:00Z",
        "request_id": "123e4567-e89b-12d3-a456-426614174000",
        "details": None,
        "documentation_url": "https://docs.example.com/support",
        "retry_after": None
    }
}

# =============================================================================
# Usage Examples
# =============================================================================

def example_usage():
    """Example of how to use the exception system."""
    
    # In your FastAPI app
    
    app = FastAPI()
    
    # Register exception handlers
    register_exception_handlers(app)
    
    # Add request ID middleware
    app.add_middleware(RequestIDMiddleware)
    
    # In your route handlers
        ValidationError, UserNotFoundError, RateLimitError,
        ExceptionFactory
    )
    
    @app.get("/users/{user_id}")
    async def get_user(user_id: int):
        
    """get_user function."""
# Example: User not found
        if user_id > 1000:
            raise UserNotFoundError(
                message=f"User with ID {user_id} not found",
                request_id="123e4567-e89b-12d3-a456-426614174000"
            )
        
        # Example: Rate limit error
        if user_id == 999:
            raise RateLimitError(
                message="Too many requests for this user",
                retry_after=60
            )
        
        return {"user_id": user_id, "name": "John Doe"}
    
    @app.post("/videos")
    async def create_video(video_data: dict):
        
    """create_video function."""
# Example: Validation error
        if not video_data.get("script"):
            raise ValidationError(
                message="Script is required for video creation",
                details=[
                    {
                        "field": "script",
                        "message": "Script is required",
                        "value": None,
                        "suggestion": "Please provide a script for the video"
                    }
                ]
            )
        
        # Example: Using exception factory
        if len(video_data.get("script", "")) > 1000:
            raise ExceptionFactory.create_validation_error(
                field="script",
                message="Script is too long",
                value=video_data["script"],
                suggestion="Script must be less than 1000 characters"
            )
        
        return {"video_id": 123, "status": "created"}

# =============================================================================
# Export
# =============================================================================

__all__ = [
    "http_exception_handler",
    "validation_exception_handler",
    "starlette_http_exception_handler",
    "general_exception_handler",
    "get_validation_suggestion",
    "get_error_category",
    "get_error_severity",
    "register_exception_handlers",
    "RequestIDMiddleware",
    "ERROR_RESPONSE_EXAMPLES",
    "example_usage",
] 